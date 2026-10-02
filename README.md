# Brain sMRI Classification: GAN Augmentation & Autoencoder Enhancement Ablation Study

This repository contains the implementation of deep learning models for Autism Spectrum Disorder (ASD) classification from structural MRI (sMRI) brain slices using the ABIDE dataset.

## Ablation Study Overview

The study systematically evaluates the individual and combined impact of GAN-based data augmentation and Autoencoder feature enhancement:

- **E0 (Baseline):** Simple 2D CNN on preprocessed raw sMRI slices (No Augmentation, No Enhancement)
- **E1:** Simple CNN with **GAN-based Augmentation**
- **E2:** Simple CNN with **Autoencoder Enhancement**
- **E3:** Simple CNN with **Combined GAN + Autoencoder**

---

## E0 Baseline Results

- **Accuracy:** 62.92%
- **Precision:** 61.35%
- **Recall / Sensitivity:** 59.91%
- **Specificity:** 65.67%
- **F1-Score:** 0.6062
- **AUC-ROC:** 0.6522

### Visual Results
- Confusion Matrix: `results/E0_confusion_matrix.png`
- Training Curves: `results/E0_training_curves.png`
- ROC Curve: `results/E0_roc_curve.png`
- Sample Visual Predictions: `results/E0_sample_predictions.png`

---

## Project Structure

```text
├── cnn_model.py                # 2D SimpleCNN architecture definition
├── config.py                   # Global configuration & hyperparameters
├── data_loader.py              # ABIDE dataset loader & stratified train/test split
├── metrics.py                  # Evaluation metrics computation
├── train_e0_baseline.py        # Training and evaluation script for E0 baseline
├── utils.py                    # Plotting and utility helper functions
├── visualize_predictions.py    # Visual prediction generator on test MRI slices
├── split_indices.pkl           # Saved stratified train/test split indices
├── requirements.txt            # Project dependencies
└── results/                    # Experiment checkpoints, JSON metrics, and visual plots
```

---

## Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run E0 Baseline Training
```bash
python train_e0_baseline.py --epochs 25
```

### 3. Generate Visual Test Predictions
```bash
python visualize_predictions.py
```
