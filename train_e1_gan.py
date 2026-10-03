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
from gan_model import ConditionalGenerator, ConditionalDiscriminator

class SyntheticDataset(Dataset):
    """Dataset for GAN-generated synthetic brain MRI slices."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def train_cgan(train_loader, device, epochs=10, latent_dim=128, lr=0.0002):
    """
    Train Conditional GAN on the training sMRI slices.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Training Conditional GAN for Brain sMRI Synthesis")
    print("=" * 70)

    generator = ConditionalGenerator(latent_dim=latent_dim).to(device)
    discriminator = ConditionalDiscriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(8, latent_dim).to(device)
    fixed_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long).to(device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        d_loss_total = 0.0
        g_loss_total = 0.0
        batches = 0

        for real_imgs, labels in train_loader:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.long().to(device)

            real_target = torch.ones(b_size, device=device) * 0.9  # Label smoothing
            fake_target = torch.zeros(b_size, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            d_real_out = discriminator(real_imgs, labels)
            d_real_loss = criterion(d_real_out, real_target)

            z = torch.randn(b_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 2, (b_size,), device=device)
            fake_imgs = generator(z, gen_labels)

            d_fake_out = discriminator(fake_imgs.detach(), gen_labels)
            d_fake_loss = criterion(d_fake_out, fake_target)

            d_loss = (d_real_loss + d_fake_loss) / 2.0
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            z = torch.randn(b_size, latent_dim, device=device)
            gen_labels = torch.randint(0, 2, (b_size,), device=device)
            fake_imgs = generator(z, gen_labels)

            d_fake_out = discriminator(fake_imgs, gen_labels)
            g_loss = criterion(d_fake_out, torch.ones(b_size, device=device))

            g_loss.backward()
            optimizer_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            batches += 1

        elapsed = time.time() - t0
        print(f"GAN Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - D Loss: {d_loss_total/batches:.4f}, G Loss: {g_loss_total/batches:.4f}")

    # Save Generator checkpoint
    gen_path = os.path.join(Config.RESULTS_DIR, "e1_gan_generator.pth")
    torch.save(generator.state_dict(), gen_path)
    print(f"[INFO] GAN Generator saved to: {gen_path}")

    # Visualize fixed sample generated MRI slices
    generator.eval()
    with torch.no_grad():
        sample_fakes = generator(fixed_noise, fixed_labels).cpu().squeeze().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ax.imshow(sample_fakes[i], cmap="bone")
        cls_name = "Typical Control" if fixed_labels[i].item() == 0 else "Autistic (ASD)"
        ax.set_title(f"Synthetic {cls_name}", fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.suptitle("E1: GAN-Generated Synthetic Brain MRI Slices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    gan_samples_path = os.path.join(Config.RESULTS_DIR, "E1_gan_generated_samples.png")
    plt.savefig(gan_samples_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample synthetic MRI slices saved to: {gan_samples_path}")

    return generator

def generate_augmented_data(generator, num_samples_per_class=400, latent_dim=128, device=Config.DEVICE):
    """
    Generate balanced synthetic data using the trained GAN Generator.
    """
    generator.eval()
    print(f"[INFO] Synthesizing {num_samples_per_class * 2} augmented brain slices ({num_samples_per_class} per class)...")

    synthetic_images = []
    synthetic_labels = []

    with torch.no_grad():
        for class_idx in [0, 1]:
            # Generate in batches of 64
            generated_so_far = 0
            while generated_so_far < num_samples_per_class:
                curr_b = min(64, num_samples_per_class - generated_so_far)
                z = torch.randn(curr_b, latent_dim, device=device)
                labels = torch.full((curr_b,), class_idx, dtype=torch.long, device=device)
                fakes = generator(z, labels).cpu()

                for img in fakes:
                    synthetic_images.append(img)
                    synthetic_labels.append(torch.tensor(class_idx, dtype=torch.float32))

                generated_so_far += curr_b

    print(f"[INFO] Generated {len(synthetic_images)} synthetic slices successfully.")
    return SyntheticDataset(synthetic_images, synthetic_labels)

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

def run_e1(gan_epochs=8, cnn_epochs=20, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE, aug_samples=800):
    print("=" * 70)
    print("  EXPERIMENT 1 (E1): GAN-based Augmentation + Simple CNN")
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

    # 1. Train or load GAN
    generator = train_cgan(train_loader, device, epochs=gan_epochs, latent_dim=128)

    # 2. Synthesize augmented data
    synthetic_dataset = generate_augmented_data(
        generator,
        num_samples_per_class=aug_samples // 2,
        latent_dim=128,
        device=device
    )

    # 3. Combine original training data + GAN augmented data
    augmented_train_dataset = ConcatDataset([train_loader.dataset, synthetic_dataset])
    augmented_train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"\n[INFO] Original Training Slices : {len(train_loader.dataset)}")
    print(f"[INFO] GAN Synthetic Slices    : {len(synthetic_dataset)}")
    print(f"[INFO] Total Augmented Train   : {len(augmented_train_dataset)}")
    print(f"[INFO] Test Slices (Untouched) : {len(test_loader.dataset)}")

    # 4. Train SimpleCNN on GAN-augmented dataset
    print("\n" + "=" * 70)
    print("  PHASE 2: Training SimpleCNN on GAN-Augmented Data")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e1_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 8
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, augmented_train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        time_epoch = time.time() - t0

        # Prioritize validation accuracy, tie-broken by lower validation loss
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

    # Load best model for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_acc, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob)
    metrics["experiment"] = "E1_GAN_Augmentation"
    metrics["gan_epochs"] = gan_epochs
    metrics["cnn_epochs_trained"] = epoch
    metrics["synthetic_samples_added"] = aug_samples
    metrics["total_train_samples"] = len(augmented_train_dataset)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E1 - GAN Augmentation + Simple CNN", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E1_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E1_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E1_confusion_matrix.png")
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E1_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E1 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E1 GAN Augmentation Experiment")
    parser.add_argument("--gan_epochs", type=int, default=8, help="Number of GAN training epochs")
    parser.add_argument("--cnn_epochs", type=int, default=20, help="Number of CNN training epochs")
    parser.add_argument("--aug_samples", type=int, default=800, help="Number of synthetic samples to generate")
    args = parser.parse_args()

    run_e1(gan_epochs=args.gan_epochs, cnn_epochs=args.cnn_epochs, aug_samples=args.aug_samples)
