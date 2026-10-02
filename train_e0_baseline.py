import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config
from utils import set_seed, save_metrics, plot_training_curves, plot_confusion_matrix, plot_roc_curve
from metrics import compute_all_metrics, print_metrics_table
from data_loader import get_data_loaders
from cnn_model import SimpleCNN

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

def run_e0(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE):
    print("=" * 70)
    print("  EXPERIMENT 0 (E0): Baseline CNN (No Augmentation, No Enhancement)")
    print("=" * 70)

    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"[INFO] Using Device: {device}")

    # Load data
    train_loader, test_loader, split_data = get_data_loaders(
        batch_size=batch_size,
        cache_in_memory=Config.CACHE_IN_MEMORY,
        num_workers=Config.NUM_WORKERS
    )

    print(f"[INFO] Training samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # Initialize model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -float("inf")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = os.path.join(Config.RESULTS_DIR, "e0_best_model.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    print("\nStarting Training...\n" + "-" * 70)

    patience = 10
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({time_epoch:.1f}s, lr={current_lr:.6f}) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%{best_marker}")

        if patience_counter >= patience:
            print(f"\n[INFO] Early stopping triggered after {patience} epochs with no validation improvement.")
            break

    total_training_time = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] Training completed in {total_training_time / 60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

    # Load best model for final evaluation
    print(f"[INFO] Loading best model checkpoint from {best_model_path} for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Evaluate on test set
    test_loss, test_acc, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_all_metrics(y_true, y_prob)
    metrics["experiment"] = "E0_Baseline_CNN"
    metrics["epochs_trained"] = epochs
    metrics["batch_size"] = batch_size
    metrics["learning_rate"] = lr
    metrics["best_val_loss"] = round(best_val_loss, 4)
    metrics["training_time_seconds"] = round(total_training_time, 2)

    # Print results
    print_metrics_table("E0 - Baseline Simple CNN", metrics)

    # Save metrics JSON
    metrics_path = os.path.join(Config.RESULTS_DIR, "E0_metrics.json")
    save_metrics(metrics, metrics_path)

    # Generate and save plots
    curves_path = os.path.join(Config.RESULTS_DIR, "E0_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    cm_path = os.path.join(Config.RESULTS_DIR, "E0_confusion_matrix.png")
    y_pred = (y_prob >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, cm_path)

    roc_path = os.path.join(Config.RESULTS_DIR, "E0_roc_curve.png")
    plot_roc_curve(y_true, y_prob, metrics["auc_roc"], roc_path)

    print("\n[SUCCESS] E0 Experiment Completed Successfully!")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E0 Baseline Experiment")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    run_e0(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
