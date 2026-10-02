import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

def set_seed(seed=42):
    """Fix random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_metrics(metrics_dict, filepath):
    """Save metrics dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[INFO] Metrics saved to {filepath}")

def load_metrics(filepath):
    """Load metrics dictionary from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot Loss and Accuracy curves across training epochs."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, val_losses, "r-s", label="Val / Test Loss", markersize=3)
    ax1.set_title("Loss vs. Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in train_accs], "b-o", label="Train Acc (%)", markersize=3)
    ax2.plot(epochs, [a * 100 for a in val_accs], "r-s", label="Val / Test Acc (%)", markersize=3)
    ax2.set_title("Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training curves plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=["Control", "ASD"]):
    """Plot publication-quality confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")

def plot_roc_curve(y_true, y_prob, auc_score, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] ROC curve saved to {save_path}")
