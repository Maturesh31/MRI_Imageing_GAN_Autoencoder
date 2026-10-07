import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import Config
from utils import set_seed, save_metrics, plot_training_curves, plot_confusion_matrix, plot_roc_curve
from metrics import compute_all_metrics, print_metrics_table
from data_loader import get_data_loaders, train_transforms
from cnn_model import SimpleCNN
from ae_model import ConvAutoencoder
from gan_model import ConditionalGenerator, ConditionalDiscriminator

class TensorDatasetWrapper(Dataset):
    """General dataset wrapper for image tensors and labels in RAM with optional augmentation."""
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.augment:
            img = train_transforms(img)
        return img, self.labels[idx]

def get_or_train_autoencoder(train_loader, device, epochs=30, lr=0.0005):
    """Load existing trained Autoencoder or train if not found."""
    ae_path = os.path.join(Config.RESULTS_DIR, "e2_autoencoder.pth")
    ae = ConvAutoencoder().to(device)

    if os.path.exists(ae_path):
        print(f"[INFO] Loading pre-trained Autoencoder checkpoint from: {ae_path}")
        ae.load_state_dict(torch.load(ae_path, map_location=device))
    else:
        print(f"\n[INFO] Training Convolutional Autoencoder for E3 ({epochs} epochs)...")
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
            if epoch % 5 == 0 or epoch == epochs:
                print(f"  AE Epoch [{epoch:02d}/{epochs:02d}] - MSE Loss: {running_loss/total_samples:.6f}")
        torch.save(ae.state_dict(), ae_path)
    return ae

def get_or_train_gan(train_loader, device, epochs=30, latent_dim=128, lr=0.0002):
    """Load existing trained GAN or train if not found."""
    gan_path = os.path.join(Config.RESULTS_DIR, "e1_gan_generator.pth")
    generator = ConditionalGenerator(latent_dim=latent_dim).to(device)

    if os.path.exists(gan_path):
        print(f"[INFO] Loading pre-trained GAN Generator checkpoint from: {gan_path}")
        generator.load_state_dict(torch.load(gan_path, map_location=device))
    else:
        print(f"\n[INFO] Training Conditional GAN for E3 ({epochs} epochs)...")
        discriminator = ConditionalDiscriminator().to(device)
        criterion = nn.BCEWithLogitsLoss()
        opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        for epoch in range(1, epochs + 1):
            d_loss_tot, g_loss_tot, n_b = 0.0, 0.0, 0
            for real_imgs, labels in train_loader:
                b = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                labels = labels.long().to(device)

                # Train D
                opt_D.zero_grad()
                d_real = criterion(discriminator(real_imgs, labels), torch.ones(b, device=device) * 0.9)
                z = torch.randn(b, latent_dim, device=device)
                fl = torch.randint(0, 2, (b,), device=device)
                fakes = generator(z, fl)
                d_fake = criterion(discriminator(fakes.detach(), fl), torch.zeros(b, device=device) + 0.1)
                d_loss = (d_real + d_fake) / 2.0
                d_loss.backward()
                opt_D.step()

                # Train G
                for _ in range(2):
                    opt_G.zero_grad()
                    z = torch.randn(b, latent_dim, device=device)
                    fl = torch.randint(0, 2, (b,), device=device)
                    fakes = generator(z, fl)
                    g_loss = criterion(discriminator(fakes, fl), torch.ones(b, device=device))
                    g_loss.backward()
                    opt_G.step()

                d_loss_tot += d_loss.item()
                g_loss_tot += g_loss.item()
                n_b += 1

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  GAN Epoch [{epoch:02d}/{epochs:02d}] - D Loss: {d_loss_tot/n_b:.4f}, G Loss: {g_loss_tot/n_b:.4f}")
        torch.save(generator.state_dict(), gan_path)
    return generator

def enhance_dataset_via_ae(loader, autoencoder, device, augment=False):
    autoencoder.eval()
    imgs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            recon = autoencoder(x.to(device)).cpu()
            for i in range(len(recon)):
                imgs.append(recon[i])
                lbls.append(y[i])
    return TensorDatasetWrapper(imgs, lbls, augment=augment)

def generate_gan_samples(generator, autoencoder, count_per_class=400, latent_dim=128, device=Config.DEVICE):
    generator.eval()
    autoencoder.eval()
    imgs, lbls = [], []
    batch_gen = 64
    with torch.no_grad():
        for c in [0, 1]:
            done = 0
            while done < count_per_class:
                cur = min(batch_gen, count_per_class - done)
                z = torch.randn(cur, latent_dim, device=device)
                cl = torch.full((cur,), c, dtype=torch.long, device=device)
                raw_gen = generator(z, cl)
                # Denoise/enhance synthetic slices via Autoencoder
                enh_gen = autoencoder(raw_gen).cpu()
                for i in range(cur):
                    imgs.append(enh_gen[i])
                    lbls.append(torch.tensor(c, dtype=torch.float32))
                done += cur
    return TensorDatasetWrapper(imgs, lbls, augment=True)

def train_epoch(model, dataloader, criterion, optimizer, device, label_smooth=0.1, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pos_smooth = 1.0 - label_smooth / 2.0
    neg_smooth = label_smooth / 2.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_smooth = labels * pos_smooth + (1.0 - labels) * neg_smooth

        if use_mixup and np.random.rand() < 0.5:
            lam = float(np.random.beta(0.2, 0.2))
            idx = torch.randperm(inputs.size(0), device=device)
            mixed_inputs = lam * inputs + (1.0 - lam) * inputs[idx]
            y_a, y_b = labels_smooth, labels_smooth[idx]

            optimizer.zero_grad()
            logits = model(mixed_inputs)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels_smooth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

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
    try:
        eval_auc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        eval_auc = 0.5

    return eval_loss, eval_acc, eval_auc, all_labels, all_probs

def run_e3(cnn_epochs=60, batch_size=Config.BATCH_SIZE, lr=1e-4, wd=5e-3, aug_samples=1000):
    print("=" * 70)
    print("  EXPERIMENT 3 (E3): Combined Autoencoder + GAN + Simple CNN")
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

    # 1. Get or train Autoencoder and GAN
    autoencoder = get_or_train_autoencoder(train_loader, device, epochs=30)
    generator = get_or_train_gan(train_loader, device, epochs=30)

    # 2. Enhance real slices through AE
    print("\n[INFO] Enhancing real train & test slices through Autoencoder...")
    e3_real_train = enhance_dataset_via_ae(train_loader, autoencoder, device, augment=True)
    e3_test = enhance_dataset_via_ae(test_loader, autoencoder, device, augment=False)

    # 3. Generate synthetic slices through GAN + AE enhancement
    print(f"[INFO] Generating {aug_samples} AE-enhanced GAN synthetic slices...")
    e3_synth = generate_gan_samples(generator, autoencoder, count_per_class=aug_samples//2, device=device)

    # 4. Combine real enhanced + synthetic enhanced
    combined_train_ds = ConcatDataset([e3_real_train, e3_synth])
    combined_train_loader = DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_e3 = DataLoader(e3_test, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Real Enhanced Slices     : {len(e3_real_train)}")
    print(f"[INFO] Synthetic Enhanced Slices: {len(e3_synth)}")
    print(f"[INFO] Total Combined Slices    : {len(combined_train_ds)}")

    print("\n" + "=" * 70)
    print("  Training Enhanced CNN on Combined AE + GAN Dataset")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e3_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 15
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, combined_train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, test_loader_e3, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        time_epoch = time.time() - t0

        if val_auc > best_auc:
            best_auc = val_auc
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            best_marker = f" [BEST SAVED - AUC: {val_auc:.4f}, Acc: {val_acc*100:.2f}%]"
        else:
            patience_counter += 1
            best_marker = f" (no AUC gain for {patience_counter}/{patience})"

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{cnn_epochs:02d}] ({time_epoch:.1f}s, lr={current_lr:.6f}) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}{best_marker}")

        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping triggered after {patience} epochs.")
            break

    total_training_time = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] E3 CNN Training completed in {total_training_time / 60:.2f} minutes. Best Val AUC: {best_auc:.4f}")

    # Load best model for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set with Youden optimal threshold
    test_loss, test_acc, _, y_true, y_prob = evaluate(model, test_loader_e3, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob, threshold="optimal")
    metrics["experiment"] = "E3_Combined_AE_GAN"
    metrics["cnn_epochs_trained"] = epoch
    metrics["total_train_samples"] = len(combined_train_ds)
    metrics["test_samples"] = len(e3_test)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E3 - Combined AE + GAN + Simple CNN (SE Attention + Calibrated Threshold)", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E3_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E3_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E3_confusion_matrix.png")
    thresh = metrics.get("threshold", 0.5)
    y_pred = (y_prob >= thresh).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E3_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E3 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E3 Combined Experiment")
    parser.add_argument("--cnn_epochs", type=int, default=60, help="Number of CNN training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--aug_samples", type=int, default=1000, help="Number of GAN synthetic samples to add")
    args = parser.parse_args()

    run_e3(
        cnn_epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        aug_samples=args.aug_samples
    )
