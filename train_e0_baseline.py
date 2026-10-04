import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from config import Config
from cnn_model import SimpleCNN
from data_loader import get_data_loaders
from metrics import compute_all_metrics, print_metrics_table
from utils import set_seed, plot_training_curves, plot_confusion_matrix, plot_roc_curve, save_metrics

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
    try:
        eval_auc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        eval_auc = 0.5
    return eval_loss, eval_acc, eval_auc, all_labels, all_probs

def run_e0(epochs=60, batch_size=Config.BATCH_SIZE, lr=1e-4, wd=5e-3):
    set_seed(Config.SEED)
    device = Config.DEVICE

    print("\n" + "=" * 70)
    print("  EXPERIMENT 0 (E0): Enhanced Baseline CNN with SE Attention")
    print("=" * 70)
    print(f"[INFO] Using Device: {device}")

    # Load data with automatic augmentation on train set
    train_loader, test_loader, split_data = get_data_loaders(
        batch_size=batch_size,
        cache_in_memory=Config.CACHE_IN_MEMORY,
        num_workers=Config.NUM_WORKERS
    )

    print(f"[INFO] Training samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # Initialize model with SE blocks, AdamW, Cosine scheduler
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    best_auc = 0.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e0_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    print("\nStarting Training...\n" + "-" * 70)

    patience = 15
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({time_epoch:.1f}s, lr={current_lr:.6f}) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, AUC: {val_auc:.4f}{best_marker}")

        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping triggered after {patience} epochs with no validation AUC improvement.")
            break

    total_training_time = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] Training completed in {total_training_time / 60:.2f} minutes. Best Val AUC: {best_auc:.4f}")

    # Load best model for final evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set with Youden's J optimal threshold calibration
    test_loss, test_acc, _, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob, threshold="optimal")
    metrics["experiment"] = "E0_Baseline_CNN"
    metrics["epochs_trained"] = epochs
    metrics["batch_size"] = batch_size
    metrics["learning_rate"] = lr
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E0 - Baseline Simple CNN (SE Attention + Calibrated Threshold)", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E0_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots using the optimal threshold
    curves_path = os.path.join(Config.RESULTS_DIR, "E0_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E0_confusion_matrix.png")
    thresh = metrics.get("threshold", 0.5)
    y_pred = (y_prob >= thresh).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E0_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E0 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E0 Baseline Experiment")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-3, help="Weight decay")
    args = parser.parse_args()

    run_e0(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, wd=args.wd)
