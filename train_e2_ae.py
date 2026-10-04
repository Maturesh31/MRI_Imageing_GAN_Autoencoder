import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from utils import set_seed, save_metrics, plot_training_curves, plot_confusion_matrix, plot_roc_curve
from metrics import compute_all_metrics, print_metrics_table
from data_loader import get_data_loaders
from cnn_model import SimpleCNN
from ae_model import ConvAutoencoder

class EnhancedDataset(Dataset):
    """Dataset holding Autoencoder-enhanced sMRI brain slices."""
    def __init__(self, enhanced_images, labels):
        self.images = enhanced_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def train_autoencoder(train_loader, device, epochs=8, lr=0.0005):
    """
    Train the Convolutional Autoencoder on training sMRI slices.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: Training Convolutional Autoencoder for Image Enhancement")
    print("=" * 70)

    ae = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=1e-5)

    ae_losses = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
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

        epoch_loss = running_loss / total_samples
        ae_losses.append(epoch_loss)
        elapsed = time.time() - t0
        print(f"AE Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - Reconstruction Loss (MSE): {epoch_loss:.6f}")

    # Save Autoencoder checkpoint
    ae_path = os.path.join(Config.RESULTS_DIR, "e2_autoencoder.pth")
    torch.save(ae.state_dict(), ae_path)
    print(f"[INFO] Autoencoder checkpoint saved to: {ae_path}")

    # Visualize original vs reconstructed enhanced slices
    ae.eval()
    test_batch, _ = next(iter(train_loader))
    test_batch = test_batch[:4].to(device)
    with torch.no_grad():
        recon_batch = ae(test_batch).cpu().squeeze().numpy()
    orig_batch = test_batch.cpu().squeeze().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(4):
        axes[0, i].imshow(orig_batch[i], cmap="bone")
        axes[0, i].set_title(f"Original Slice #{i+1}", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_batch[i], cmap="bone")
        axes[1, i].set_title(f"AE Enhanced #{i+1}", fontsize=11, fontweight="bold", color="#1565c0")
        axes[1, i].axis("off")

    plt.suptitle("E2: Autoencoder Enhancement Comparison (Original vs. Reconstructed)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    recons_path = os.path.join(Config.RESULTS_DIR, "E2_ae_reconstructions.png")
    plt.savefig(recons_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] AE Reconstruction comparison saved to: {recons_path}")

    return ae

def create_enhanced_dataset(dataloader, autoencoder, device):
    """
    Pass all slices in a dataloader through the trained Autoencoder to produce enhanced slices.
    """
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

    return EnhancedDataset(enhanced_imgs, labels_list)

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

def run_e2(ae_epochs=6, cnn_epochs=15, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE):
    print("=" * 70)
    print("  EXPERIMENT 2 (E2): Autoencoder Enhancement + Simple CNN")
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
    enhanced_train_dataset = create_enhanced_dataset(train_loader, autoencoder, device)
    enhanced_train_loader = DataLoader(enhanced_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("[INFO] Enhancing test slices with Autoencoder...")
    enhanced_test_dataset = create_enhanced_dataset(test_loader, autoencoder, device)
    enhanced_test_loader = DataLoader(enhanced_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Enhanced Train Slices: {len(enhanced_train_dataset)} | Enhanced Test Slices: {len(enhanced_test_dataset)}")

    # 3. Train SimpleCNN on Autoencoder-Enhanced Data
    print("\n" + "=" * 70)
    print("  PHASE 2: Training SimpleCNN on AE-Enhanced Data")
    print("=" * 70)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e2_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    patience = 8
    patience_counter = 0

    for epoch in range(1, cnn_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, enhanced_train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, enhanced_test_loader, criterion, device)
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

    # Load best model for evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_acc, y_true, y_prob = evaluate(model, enhanced_test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob)
    metrics["experiment"] = "E2_Autoencoder_Enhancement"
    metrics["ae_epochs"] = ae_epochs
    metrics["cnn_epochs_trained"] = epoch
    metrics["train_samples"] = len(enhanced_train_dataset)
    metrics["test_samples"] = len(enhanced_test_dataset)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E2 - Autoencoder Enhancement + Simple CNN", metrics)

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
    parser.add_argument("--ae_epochs", type=int, default=6, help="Number of Autoencoder training epochs")
    parser.add_argument("--cnn_epochs", type=int, default=15, help="Number of CNN training epochs")
    args = parser.parse_args()

    run_e2(ae_epochs=args.ae_epochs, cnn_epochs=args.cnn_epochs)
