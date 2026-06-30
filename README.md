# EEG-Based Cognitive Workload Estimation and Surgical Performance Prediction
### BCI Lab, MANIT Bhopal

---

## Overview
This project extends the base paper (Powar & Atheef, Scientific Reports 2026)
by applying deep learning to EEG-based cognitive workload estimation in
robot-assisted surgery and introducing the first surgical performance
prediction model on this dataset.

---

## Base Paper
> Powar, O.S. & Atheef, M.G.A. (2026).
> *Estimating cognitive workload in robot assisted surgery using time and
> frequency features from EEG epochs with random forest regression.*
> Scientific Reports, 16, 7624.
> DOI: 10.1038/s41598-026-35986-5

---

## Dataset
- **Name:** Electroencephalogram and Eye-Gaze Datasets for
  Robot-Assisted Surgery Performance Evaluation
- **Source:** PhysioNet (Shafiei et al. 2023)
- **DOI:** 10.13026/qj5m-n649
- **Link:** https://physionet.org/content/eeg-eye-gaze-data/1.0.0/
- **Subjects used:** 8 of 25 (S10, S13, S14, S16, S17, S18, S20, S24)
- **EEG channels:** 124 at 500 Hz
- **Total epochs:** 5159 (2-second epochs)

---

## Novel Contributions

| # | Contribution | Result |
|---|---|---|
| 1 | First EEGNet on this dataset (base paper task) | Beats RFR in all 4 brain regions |
| 2 | 500 Hz vs 128 Hz systematic comparison | 500 Hz wins 3/4 regions |
| 3 | Cross-subject LOSO gap analysis | ~0.93 R² gap exposed in base paper |
| 4 | First surgical performance prediction from EEG | Pooled R²=+0.2518, Acc=43.63% |

---

## Experiment Pipeline

| Cell | Description | Key Output |
|---|---|---|
| Cell 8R | Runtime restore after Colab disconnect | All variables reloaded |
| Cell 9A | Feature extraction (500Hz + 128Hz) | 11 features per region |
| Cell 9B | RFR base paper replication + XGBoost | R²=0.86–0.88 (10CV) |
| Cell 9C | EEGNet on base paper task | Beats RFR all 4 regions |
| Cell 10 | Pure LOSO gap analysis | ~0.93 generalization gap |
| Cell 11 | Proposed EEGNet pct prediction | Pooled R²=+0.2518, Acc=43.63% |
| Cell 12 | Final comparison + IEEE figures | 9 publication-ready figures |
| Cell 13 | Training curves (loss, accuracy, LR) | Epoch-wise training analysis |
| Cell 14 | Final summary table | LaTeX-ready results table |

---

## Models Used

| Model | Cell | Task | Validation | R² |
|---|---|---|---|---|
| RFR | 9B | EEG activation | Intra 10CV | 0.86–0.88 |
| SVR | 9B | EEG activation | Intra 10CV | 0.69–0.71 |
| Linear Regression | 9B | EEG activation | Intra 10CV | 0.62–0.76 |
| XGBoost | 9B | EEG activation | Intra 10CV | 0.84–0.87 |
| EEGNet500 | 9C | EEG activation | Intra 80/20 | -0.15 to -0.19 |
| RFR LOSO | 10 | EEG activation | Pure LOSO | -0.04 to -0.09 |
| EEGNet+SABN | 11 | pct score | Strat LOSO | **+0.2518 pooled** |

---

## Key Results

### Base Paper Replication (Cell 9B) — 500 Hz
| Region | RFR R² (10CV) | 500Hz vs 128Hz |
|---|---|---|
| Frontal | 0.8649 ± 0.0440 | 500Hz ✓ |
| Temporal | 0.8602 ± 0.0322 | 128Hz |
| Parietal | 0.8765 ± 0.0297 | 500Hz ✓ |
| Occipital | 0.8675 ± 0.0329 | 500Hz ✓ |

### EEGNet vs RFR (Cell 9C)
| Region | EEGNet R² | RFR R² | Winner |
|---|---|---|---|
| Frontal | -0.1905 | -0.2086 | EEGNet ✓ |
| Temporal | -0.1892 | -0.2085 | EEGNet ✓ |
| Parietal | -0.1597 | -0.2085 | EEGNet ✓ |
| Occipital | -0.1483 | -0.2086 | EEGNet ✓ |

### Generalization Gap (Cell 10)
| Region | Intra R² | LOSO R² | Gap |
|---|---|---|---|
| Frontal | 0.8649 | -0.0855 | +0.9504 |
| Temporal | 0.8602 | -0.0629 | +0.9231 |
| Parietal | 0.8765 | -0.0673 | +0.9438 |
| Occipital | 0.8675 | -0.0359 | +0.9034 |

### Proposed Model (Cell 11)
| Metric | Value |
|---|---|
| Primary Accuracy (6 normal folds) | **43.63%** |
| Pooled R² | **+0.2518** |
| Chance baseline | 33.33% |
| Best fold (S20) | 63.15% accuracy |

---

## Architecture — Proposed EEGNet + SABN
Input (Batch, 33 channels, 1000 samples)
→ Temporal Conv (kernel=125, captures 0.25s at 500Hz)
→ Depthwise Spatial Conv + SubjectAdaptiveBN  ← novel
→ AvgPool (1000 → 125)
→ Separable Conv (kernel=31)
→ AvgPool (125 → 8)
→ Flatten (112) → Linear (1)
→ pct score output

**Subject-Adaptive BN:** Learns per-subject gamma and beta via
nn.Embedding — normalizes inter-subject amplitude differences
without using any label information.
Published precedent: Kobler et al. IEEE TNSRE 2021.

---

## Features Extracted

**11 base paper features per region:**
Mean, Variance, Skewness, Kurtosis, RMS, Zero-crossings,
Delta, Theta, Alpha, Beta, Gamma (Welch PSD)

**5 novel ratio features (Cell 11):**

| Feature | Formula | Basis |
|---|---|---|
| Frontal theta/alpha | θ_F / α_F | Primary workload index |
| Frontal theta/beta | θ_F / β_F | Attention vs fatigue |
| (theta+alpha)/beta | (θ+α) / β | Combined engagement |
| Frontal theta/Parietal alpha | θ_F / α_P | Cross-region workload |
| Frontal alpha asymmetry | F4α − F3α | Hemispheric balance |

---

## Requirements

```bash
pip install mne numpy scipy scikit-learn torch matplotlib pickle
```

### Full Environment
Python     : 3.12
MNE        : EEG loading and preprocessing
PyTorch    : EEGNet implementation
scikit-learn: RFR, SVR, XGBoost, StandardScaler
SciPy      : Welch PSD, signal resampling
Matplotlib : All figures
Platform   : Google Colab T4 GPU

---

## How to Run

1. Open notebook in Google Colab
2. Mount Google Drive
3. Download dataset from PhysioNet:
   https://physionet.org/content/eeg-eye-gaze-data/1.0.0/
4. Set BASE path to your Drive folder
5. Run cells in order: 8R → 9A → 9B → 9C → 10 → 11 → 12 → 13 → 14

> ⚠ Cell 9A takes ~33 minutes (feature extraction)
> ⚠ Cell 11 takes ~60–90 minutes (EEGNet LOSO training on GPU)
> ⚠ Always run Cell 8R after any Colab disconnect

---

## Saved Files (Google Drive)

| File | Size | Description |
|---|---|---|
| X.npy | 2440 MB | EEG epochs (5159, 124, 1000) |
| pct_target.npy | 0.02 MB | Performance scores |
| cell9a_features.pkl | 1.89 MB | Extracted features |
| cell9b_rfr_results.pkl | 0.16 MB | RFR replication results |
| cell9c_eegnet_results.pkl | — | EEGNet base task results |
| cell10_loso_gap.pkl | — | LOSO gap analysis |
| cell11_proposed_results.pkl | — | Proposed model results |
| eegnet11_ckpt/ | — | 24 per-fold checkpoints |

---

## Author
**Name:** Kaya Kushwah
**Enrollment:** EN23CS301503
**Lab:** BCI Lab, MANIT Bhopal
**Guide:** Dr.Mitul Kumar Ahirwal
**Date:** June 2026

---

