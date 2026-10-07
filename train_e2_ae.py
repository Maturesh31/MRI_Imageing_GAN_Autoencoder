import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import Config
from utils import set_seed, save_metrics, plot_training_curves, plot_confusion_matrix, plot_roc_curve
from metrics import compute_all_metrics, print_metrics_table
from data_loader import get_data_loaders, train_transforms
from cnn_model import SimpleCNN
from ae_model import ConvAutoencoder

class EnhancedDataset(Dataset):
    """Dataset for Autoencoder enhanced brain MRI slices."""
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

def train_autoencoder(train_loader, device, epochs=30, lr=5e-4, wd=1e-5):
    print("\n" + "=" * 70)
    print(f"  PHASE 1: Training Convolutional Autoencoder ({epochs} epochs)")
    print("=" * 70)

    autoencoder = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        autoencoder.train()
        running_loss = 0.0
        total_samples = 0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_samples
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  AE Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - Reconstruction MSE Loss: {epoch_loss:.6f}")

    ae_path = os.path.join(Config.RESULTS_DIR, "e2_autoencoder.pth")
    torch.save(autoencoder.state_dict(), ae_path)
    print(f"[INFO] Autoencoder model saved to: {ae_path}")

    # Save visual reconstructions
    autoencoder.eval()
    with torch.no_grad():
        for sample_imgs, _ in train_loader:
            sample_imgs = sample_imgs[:4].to(device)
            recons = autoencoder(sample_imgs).cpu()
            break

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(sample_imgs[i, 0].cpu(), cmap="gray")
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i, 0], cmap="gray")
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, "E2_ae_reconstructions.png"), dpi=200)
    plt.close()

    return autoencoder

def create_enhanced_dataset(data_loader, autoencoder, device, augment=False):
    autoencoder.eval()
    enhanced_images = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            enhanced = autoencoder(images).cpu()
            for i in range(images.size(0)):
                enhanced_images.append(enhanced[i])
                labels_list.append(labels[i])

    return EnhancedDataset(enhanced_images, labels_list, augment=augment)

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

def run_e2(ae_epochs=30, cnn_epochs=60, batch_size=Config.BATCH_SIZE, lr=1e-4, wd=5e-3):
    print("=" * 70)
    print("  EXPERIMENT 2 (E2): Autoencoder Enhancement + Simple CNN (SE Attention)")
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

    # 1. Train Autoencoder
    autoencoder = train_autoencoder(train_loader, device, epochs=ae_epochs)

    # 2. Enhance both train and test slices using trained Autoencoder
    print("\n[INFO] Enhancing training slices with Autoencoder...")
    enhanced_train_dataset = create_enhanced_dataset(train_loader, autoencoder, device, augment=True)
    enhanced_train_loader = DataLoader(enhanced_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("[INFO] Enhancing test slices with Autoencoder...")
    enhanced_test_dataset = create_enhanced_dataset(test_loader, autoencoder, device, augment=False)
    enhanced_test_loader = DataLoader(enhanced_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Enhanced Train Slices: {len(enhanced_train_dataset)} | Enhanced Test Slices: {len(enhanced_test_dataset)}")

    print("\n" + "=" * 70)
    print("  PHASE 2: Training Enhanced CNN on AE-Enhanced Data")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e2_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 15
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, enhanced_train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, enhanced_test_loader, criterion, device)
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
    print(f"[INFO] CNN Training completed in {total_training_time / 60:.2f} minutes. Best Val AUC: {best_auc:.4f}")

    # Load best model for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set with Youden optimal threshold
    test_loss, test_acc, _, y_true, y_prob = evaluate(model, enhanced_test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob, threshold="optimal")
    metrics["experiment"] = "E2_Autoencoder_Enhancement"
    metrics["ae_epochs"] = ae_epochs
    metrics["cnn_epochs_trained"] = epoch
    metrics["train_samples"] = len(enhanced_train_dataset)
    metrics["test_samples"] = len(enhanced_test_dataset)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E2 - Autoencoder Enhancement + Simple CNN (SE Attention + Calibrated Threshold)", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E2_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E2_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E2_confusion_matrix.png")
    thresh = metrics.get("threshold", 0.5)
    y_pred = (y_prob >= thresh).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E2_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E2 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E2 Autoencoder Enhancement Experiment")
    parser.add_argument("--ae_epochs", type=int, default=30, help="Number of Autoencoder training epochs")
    parser.add_argument("--cnn_epochs", type=int, default=60, help="Number of CNN training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-3, help="Weight decay")
    args = parser.parse_args()

    run_e2(
        ae_epochs=args.ae_epochs,
        cnn_epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd
    )
