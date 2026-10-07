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
from gan_model import ConditionalGenerator, ConditionalDiscriminator

class SyntheticDataset(Dataset):
    """Dataset for GAN-generated synthetic brain MRI slices with optional augmentation."""
    def __init__(self, images, labels, augment=True):
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

def train_cgan(train_loader, device, epochs=30, latent_dim=128, lr=0.0002):
    """
    Train Conditional GAN on the training sMRI slices with 2:1 G/D ratio and label smoothing.
    """
    print("\n" + "=" * 70)
    print(f"  PHASE 1: Training Conditional GAN for Brain sMRI Synthesis ({epochs} epochs)")
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

            real_target = torch.ones(b_size, device=device) * 0.9
            fake_target = torch.zeros(b_size, device=device) + 0.1

            # Train Discriminator
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

            # Train Generator (2 steps for better convergence)
            for _ in range(2):
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
        avg_d = d_loss_total / batches
        avg_g = g_loss_total / batches

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  GAN Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - D Loss: {avg_d:.4f}, G Loss: {avg_g:.4f}")

    # Save GAN checkpoint
    gan_path = os.path.join(Config.RESULTS_DIR, "e1_gan_generator.pth")
    torch.save(generator.state_dict(), gan_path)
    print(f"[INFO] GAN Generator saved to: {gan_path}")

    # Save sample visualizations
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise, fixed_labels).cpu()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0], cmap="gray")
        lbl = "Autistic" if fixed_labels[i] == 1 else "Control"
        ax.set_title(f"Synthetic {lbl}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "E1_gan_generated_samples.png"), dpi=200)
    plt.close()

    return generator

def generate_augmented_data(generator, num_samples_per_class=400, latent_dim=128, device=Config.DEVICE):
    print(f"\n[INFO] Generating {num_samples_per_class * 2} synthetic slices ({num_samples_per_class} per class)...")
    generator.eval()
    synthetic_images = []
    synthetic_labels = []

    batch_gen = 64
    with torch.no_grad():
        for class_idx in [0, 1]:
            generated_so_far = 0
            while generated_so_far < num_samples_per_class:
                curr_b = min(batch_gen, num_samples_per_class - generated_so_far)
                z = torch.randn(curr_b, latent_dim, device=device)
                labels = torch.full((curr_b,), class_idx, dtype=torch.long, device=device)
                fakes = generator(z, labels).cpu()

                for img in fakes:
                    synthetic_images.append(img)
                    synthetic_labels.append(torch.tensor(class_idx, dtype=torch.float32))

                generated_so_far += curr_b

    print(f"[INFO] Generated {len(synthetic_images)} synthetic slices successfully.")
    return SyntheticDataset(synthetic_images, synthetic_labels, augment=True)

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

def run_e1(gan_epochs=30, cnn_epochs=60, batch_size=Config.BATCH_SIZE, lr=1e-4, wd=5e-3, aug_samples=1000):
    print("=" * 70)
    print("  EXPERIMENT 1 (E1): GAN-based Augmentation + Enhanced Simple CNN")
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
    print(f"[INFO] Total Training Slices   : {len(augmented_train_dataset)}")

    print("\n" + "=" * 70)
    print("  PHASE 2: Training Enhanced CNN on Real + GAN Augmented Dataset")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e1_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 15
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, augmented_train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, test_loader, criterion, device)
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
    print(f"[INFO] Training completed in {total_training_time / 60:.2f} minutes. Best Val AUC: {best_auc:.4f}")

    # Load best model for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set with Youden's J optimal threshold
    test_loss, test_acc, _, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob, threshold="optimal")
    metrics["experiment"] = "E1_GAN_Augmentation"
    metrics["gan_epochs"] = gan_epochs
    metrics["cnn_epochs_trained"] = epoch
    metrics["synthetic_samples_added"] = aug_samples
    metrics["total_train_samples"] = len(augmented_train_dataset)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E1 - GAN Augmentation + Simple CNN (SE Attention + Calibrated Cutoff)", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E1_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E1_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E1_confusion_matrix.png")
    thresh = metrics.get("threshold", 0.5)
    y_pred = (y_prob >= thresh).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E1_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E1 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E1 GAN Augmentation Experiment")
    parser.add_argument("--gan_epochs", type=int, default=30, help="Number of GAN training epochs")
    parser.add_argument("--cnn_epochs", type=int, default=60, help="Number of CNN training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-3, help="Weight decay")
    parser.add_argument("--aug_samples", type=int, default=1000, help="Number of GAN synthetic samples to add")
    args = parser.parse_args()

    run_e1(
        gan_epochs=args.gan_epochs,
        cnn_epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        aug_samples=args.aug_samples
    )
