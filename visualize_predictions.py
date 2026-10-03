import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from cnn_model import SimpleCNN
from data_loader import get_or_create_split, ABIDEDataset

def visualize_sample_predictions(model_name="e1_best_model.pth", save_name="E1_sample_predictions.png", title="E1 GAN-Augmented Model - Sample Test Predictions", num_samples=12):
    device = Config.DEVICE
    print(f"[INFO] Using Device: {device}")

    # 1. Load split data
    split_data = get_or_create_split()
    test_files = split_data["test_files"]
    test_labels = split_data["test_labels"]

    # 2. Load model
    model_path = os.path.join(Config.RESULTS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Create test dataset without full memory cache to run fast
    test_dataset = ABIDEDataset(test_files, test_labels, cache_in_memory=False)

    # 4. Predict on a diverse selection of test samples
    np.random.seed(Config.SEED)
    sample_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)

    class_names = {0: "Typical Control", 1: "Autistic (ASD)"}

    images = []
    y_trues = []
    y_preds = []
    y_probs = []

    print(f"\n{'='*75}")
    print(f"{'Index':<6} | {'True Label':<18} | {'Predicted Label':<18} | {'Confidence':<10} | {'Status'}")
    print(f"{'='*75}")

    with torch.no_grad():
        for idx in sample_indices:
            img_tensor, label = test_dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= 0.5 else 0
            true_lbl = int(label.item())

            images.append(img_tensor.squeeze().numpy())
            y_trues.append(true_lbl)
            y_preds.append(pred)
            y_probs.append(prob)

            status = "CORRECT" if pred == true_lbl else "INCORRECT"
            conf = prob if pred == 1 else (1.0 - prob)
            print(f"{idx:<6} | {class_names[true_lbl]:<18} | {class_names[pred]:<18} | {conf*100:>5.1f}%     | {status}")

    print(f"{'='*75}\n")

    # 5. Plot grid of images with annotations
    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        img = images[i]
        true_lbl = y_trues[i]
        pred_lbl = y_preds[i]
        prob = y_probs[i]

        is_correct = (true_lbl == pred_lbl)
        color = "#1b5e20" if is_correct else "#b71c1c"  # Dark green / dark red
        status_text = "CORRECT" if is_correct else "WRONG"

        ax.imshow(img, cmap="bone")
        ax.set_title(
            f"True: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]} ({prob*100:.1f}% ASD)\n[{status_text}]",
            fontsize=11,
            fontweight="bold",
            color=color,
            pad=8
        )
        ax.axis("off")

    # Turn off unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    save_path = os.path.join(Config.RESULTS_DIR, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Visual predictions saved to: {save_path}")

if __name__ == "__main__":
    visualize_sample_predictions(model_name="e1_best_model.pth", save_name="E1_sample_predictions.png", title="E1 GAN-Augmented Model - Sample Test Predictions")
