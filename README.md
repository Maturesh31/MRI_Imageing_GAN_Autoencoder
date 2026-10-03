# Brain sMRI Classification: GAN Augmentation & Autoencoder Enhancement Ablation Study

This repository contains the complete implementation and ablation study of deep learning models for Autism Spectrum Disorder (ASD) classification from structural MRI (sMRI) brain slices using the ABIDE dataset.

## Ablation Study Overview

The study systematically evaluates the individual and combined impact of GAN-based data augmentation and Autoencoder feature enhancement:

| Experiment | Image Enhancement | Data Augmentation | Classifier | Status |
| :---: | :---: | :---: | :---: | :---: |
| **E0** | None | None | Simple CNN | ✅ Completed |
| **E1** | None | GAN | Simple CNN | ✅ Completed |
| **E2** | Autoencoder | None | Simple CNN | ✅ Completed |
| **E3** | Autoencoder | GAN | Simple CNN | ✅ Completed |

---

## Complete Experimental Results Comparison

| Metric | E0 (Baseline CNN) | E1 (GAN + CNN) | E2 (Autoencoder + CNN) | E3 (AE + GAN + CNN) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **62.92%** | 59.55% | **62.92%** | 61.12% |
| **Specificity** | 65.67% | 67.81% | 47.21% | **69.96%** *(Highest)* |
| **Recall / Sensitivity** | 59.91% | 50.47% | **80.19%** *(Highest)* | 51.42% |
| **F1-Score** | 0.6062 | 0.5431 | **0.6733** *(Highest)* | 0.5575 |
| **AUC-ROC** | 0.6522 | 0.6223 | **0.6665** *(Highest)* | 0.6129 |
| **Precision** | **61.35%** | 58.79% | 58.02% | 60.89% |

### Key Experimental Insights:
- **E0 (Baseline):** 62.92% accuracy, serving as the foundational baseline on raw preprocessed sMRI slices.
- **E1 (GAN Augmentation):** Increased specificity to 67.81%, reducing false positive diagnoses.
- **E2 (Autoencoder Enhancement):** Dramatically elevated Recall/Sensitivity to **80.19%** and F1-score to **0.6733**, correctly identifying 170 autistic cases with only 42 false negatives.
- **E3 (Combined AE + GAN):** Achieved the highest specificity (**69.96%**), maximizing true negative discrimination.

---

## Project Structure

```text
├── cnn_model.py                # 2D SimpleCNN architecture definition
├── gan_model.py                # Conditional Generator & Discriminator for sMRI synthesis
├── ae_model.py                 # Convolutional Autoencoder for image enhancement
├── config.py                   # Global configuration & hyperparameters
├── data_loader.py              # ABIDE dataset loader & stratified train/test split
├── metrics.py                  # Evaluation metrics computation
├── train_e0_baseline.py        # E0: Baseline CNN training pipeline
├── train_e1_gan.py             # E1: GAN training & augmented CNN training
├── train_e2_ae.py              # E2: Autoencoder training & enhanced CNN training
├── train_e3_combined.py        # E3: Combined AE enhancement + GAN augmentation
├── utils.py                    # Plotting and utility helper functions
├── visualize_predictions.py    # Visual prediction generator on test MRI slices
├── split_indices.pkl           # Exact stratified train/test split indices
├── requirements.txt            # Project dependencies
└── results/                    # All checkpoints, JSON metrics, and visual comparison plots
```

---

## Usage Instructions

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
# E0: Baseline CNN
python train_e0_baseline.py --epochs 25

# E1: GAN-Augmented CNN
python train_e1_gan.py --gan_epochs 6 --cnn_epochs 15 --aug_samples 400

# E2: Autoencoder-Enhanced CNN
python train_e2_ae.py --ae_epochs 6 --cnn_epochs 15

# E3: Combined AE + GAN CNN
python train_e3_combined.py --cnn_epochs 15 --aug_samples 400
```

### 3. Generate Visual Predictions
```bash
python visualize_predictions.py
```
