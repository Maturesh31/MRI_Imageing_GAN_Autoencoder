import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from utils import set_seed, save_metrics, plot_training_curves, plot_confusion_matrix, plot_roc_curve
from metrics import compute_all_metrics, print_metrics_table
from data_loader import get_data_loaders
from cnn_model import SimpleCNN
from ae_model import ConvAutoencoder
from gan_model import ConditionalGenerator, ConditionalDiscriminator

class TensorDatasetWrapper(Dataset):
    """General dataset wrapper for image tensors and labels in RAM."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_or_train_autoencoder(train_loader, device, epochs=6, lr=0.0005):
    """Load existing trained Autoencoder or train if not found."""
    ae_path = os.path.join(Config.RESULTS_DIR, "e2_autoencoder.pth")
    ae = ConvAutoencoder().to(device)

    if os.path.exists(ae_path):
        print(f"[INFO] Loading pre-trained Autoencoder checkpoint from: {ae_path}")
        ae.load_state_dict(torch.load(ae_path, map_location=device))
    else:
        print("\n[INFO] Training Convolutional Autoencoder for E3...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(1, epochs + 1):
            ae.train()
            running_loss = 0.0
            total_samples = 0
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                reconstructed = ae(imgs)
                loss = criterion(reconstructed, imgs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)
            print(f"AE Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {running_loss/total_samples:.6f}")
        torch.save(ae.state_dict(), ae_path)
    return ae

def get_or_train_gan(train_loader, device, epochs=6, latent_dim=128, lr=0.0002):
    """Load existing trained GAN Generator or train if not found."""
    gen_path = os.path.join(Config.RESULTS_DIR, "e1_gan_generator.pth")
    generator = ConditionalGenerator(latent_dim=latent_dim).to(device)

    if os.path.exists(gen_path):
        print(f"[INFO] Loading pre-trained GAN Generator checkpoint from: {gen_path}")
        generator.load_state_dict(torch.load(gen_path, map_location=device))
    else:
        print("\n[INFO] Training Conditional GAN for E3...")
        discriminator = ConditionalDiscriminator().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(1, epochs + 1):
            for real_imgs, labels in train_loader:
                b_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                labels = labels.long().to(device)
                real_target = torch.ones(b_size, device=device) * 0.9
                fake_target = torch.zeros(b_size, device=device)

                # Train D
                optimizer_D.zero_grad()
                d_real = criterion(discriminator(real_imgs, labels), real_target)
                z = torch.randn(b_size, latent_dim, device=device)
                gen_labels = torch.randint(0, 2, (b_size,), device=device)
                fake_imgs = generator(z, gen_labels)
                d_fake = criterion(discriminator(fake_imgs.detach(), gen_labels), fake_target)
                (d_real + d_fake).backward()
                optimizer_D.step()

                # Train G
                optimizer_G.zero_grad()
                g_loss = criterion(discriminator(fake_imgs, gen_labels), torch.ones(b_size, device=device))
                g_loss.backward()
                optimizer_G.step()
        torch.save(generator.state_dict(), gen_path)

    return generator

def enhance_dataset(dataloader, autoencoder, device):
    """Pass dataloader through autoencoder to generate enhanced tensors in memory."""
    autoencoder.eval()
    enhanced_imgs = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            enhanced = autoencoder(imgs).cpu()
            for img, lbl in zip(enhanced, lbls):
                enhanced_imgs.append(img)
                labels_list.append(lbl)
    return enhanced_imgs, labels_list

def generate_enhanced_synthetic(generator, autoencoder, num_samples_per_class=200, latent_dim=128, device=Config.DEVICE):
    """Generate synthetic samples using GAN and enhance them with Autoencoder."""
    generator.eval()
    autoencoder.eval()
    print(f"[INFO] Synthesizing and enhancing {num_samples_per_class * 2} GAN samples...")

    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for class_idx in [0, 1]:
            generated = 0
            while generated < num_samples_per_class:
                curr_b = min(64, num_samples_per_class - generated)
                z = torch.randn(curr_b, latent_dim, device=device)
                labels = torch.full((curr_b,), class_idx, dtype=torch.long, device=device)
                fakes = generator(z, labels)
                # Enhance synthetic slices via Autoencoder
                enhanced_fakes = autoencoder(fakes).cpu()

                for img in enhanced_fakes:
                    synthetic_images.append(img)
                    synthetic_labels.append(torch.tensor(class_idx, dtype=torch.float32))

                generated += curr_b

    return synthetic_images, synthetic_labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / total
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    eval_acc = np.mean(all_preds == all_labels)

    return eval_loss, eval_acc, all_labels, all_probs

def run_e3(cnn_epochs=15, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE, aug_samples=400):
    print("=" * 70)
    print("  EXPERIMENT 3 (E3): Combined Autoencoder Enhancement + GAN Augmentation")
    print("=" * 70)

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"[INFO] Using Device: {device}")

    # Load baseline split data
    train_loader, test_loader, split_data = get_data_loaders(
        batch_size=batch_size,
        cache_in_memory=Config.CACHE_IN_MEMORY,
        num_workers=Config.NUM_WORKERS
    )

    # 1. Load / Prepare Autoencoder & GAN
    autoencoder = get_or_train_autoencoder(train_loader, device)
    generator = get_or_train_gan(train_loader, device)

    # 2. Enhance original training slices via Autoencoder
    print("\n[INFO] Enhancing original training slices with Autoencoder...")
    orig_enhanced_imgs, orig_enhanced_lbls = enhance_dataset(train_loader, autoencoder, device)

    # 3. Enhance test slices via Autoencoder
    print("[INFO] Enhancing test slices with Autoencoder...")
    test_enhanced_imgs, test_enhanced_lbls = enhance_dataset(test_loader, autoencoder, device)
    test_dataset = TensorDatasetWrapper(test_enhanced_imgs, test_enhanced_lbls)
    test_loader_e3 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Generate GAN synthetic slices and enhance them via Autoencoder
    synth_imgs, synth_lbls = generate_enhanced_synthetic(
        generator,
        autoencoder,
        num_samples_per_class=aug_samples // 2,
        device=device
    )

    # 5. Combine original enhanced slices + synthetic enhanced slices
    all_train_imgs = orig_enhanced_imgs + synth_imgs
    all_train_lbls = orig_enhanced_lbls + synth_lbls
    combined_train_dataset = TensorDatasetWrapper(all_train_imgs, all_train_lbls)
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"\n[INFO] Original Enhanced Train Slices : {len(orig_enhanced_imgs)}")
    print(f"[INFO] GAN Synthetic Enhanced Slices : {len(synth_imgs)}")
    print(f"[INFO] Total Combined Train Slices    : {len(combined_train_dataset)}")
    print(f"[INFO] Untouched Enhanced Test Slices : {len(test_dataset)}")

    # 6. Train SimpleCNN Classifier on Combined (AE + GAN) Data
    print("\n" + "=" * 70)
    print("  PHASE 2: Training SimpleCNN on Combined (AE + GAN) Data")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e3_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 8
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, combined_train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader_e3, criterion, device)
        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        time_epoch = time.time() - t0

        score = (val_acc * 100.0) - val_loss
        is_best = score > best_score
        if is_best:
            best_score = score
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_marker = f" [BEST SAVED - Acc: {val_acc*100:.2f}%]"
        else:
            patience_counter += 1
            best_marker = f" (no improvement for {patience_counter}/{patience})"

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{cnn_epochs:02d}] ({time_epoch:.1f}s, lr={current_lr:.6f}) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%{best_marker}")

        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping triggered after {patience} epochs.")
            break

    total_training_time = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] CNN Training completed in {total_training_time / 60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

    # Load best model checkpoint for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_acc, y_true, y_prob = evaluate(model, test_loader_e3, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob)
    metrics["experiment"] = "E3_Combined_AE_GAN"
    metrics["cnn_epochs_trained"] = epoch
    metrics["synthetic_samples_added"] = aug_samples
    metrics["total_train_samples"] = len(combined_train_dataset)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E3 - Combined Autoencoder + GAN + Simple CNN", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E3_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E3_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E3_confusion_matrix.png")
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E3_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E3 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E3 Combined Experiment")
    parser.add_argument("--cnn_epochs", type=int, default=15, help="Number of CNN training epochs")
    parser.add_argument("--aug_samples", type=int, default=400, help="Number of GAN synthetic samples to add")
    args = parser.parse_args()

    run_e3(cnn_epochs=args.cnn_epochs, aug_samples=args.aug_samples)
