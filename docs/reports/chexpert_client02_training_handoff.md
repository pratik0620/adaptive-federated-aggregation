# CheXpert Client 02: Model Training Handoff Summary

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Client Identifier:** `client_02_CheXpert` (Stanford Hospital, Stanford, CA, USA)  
**Assigned Task:** Binary Pleural Effusion Classification on Chest Radiographs  
**Handoff Date:** October 27, 2026  
**Author:** CheXpert Client 02 Team Member  

---

## 1. Scope and Responsibilities

This document provides the formal training and evaluation handoff for **CheXpert Client 02**.

> [!IMPORTANT]
> **Scope Boundary:** This work strictly encompasses local model training, validation verification, diagnostic threshold analysis, and baseline test evaluation for Client 02. It does **not** include the implementation or orchestration of the central adaptive federated aggregation algorithm, which falls under the purview of the central coordinator / project lead.

---

## 2. Training Run & Checkpoint Specification

- **Training Run Directory:** `outputs/chexpert/chexpert_resnet50_20261027_225232`
- **Best Model Checkpoint:** `outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt` (Epoch 3)
- **Model Architecture:** ResNet-50 (`torchvision.models.resnet50`) with an ImageNet-1k pretrained backbone and a single-logit linear classification head (`nn.Linear(2048, 1)`, dropout = 0.0). Total parameters: 23,510,081 across 320 tensors.
- **Hardware & Execution Environment:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), CUDA 12.8, PyTorch 2.13.0+cu130 with automatic mixed precision (AMP fp16).
- **Optimization:** AdamW optimizer ($\text{lr} = 1.0 \times 10^{-4}$, $\text{weight decay} = 1.0 \times 10^{-4}$), cosine annealing learning rate schedule across 5 epochs, batch size 16. Total training runtime: 364.73 seconds (~6.1 minutes).
- **Training Cohort:** 500 unique patients selected via reproducible seeded sampling (`seed = 42`), yielding 1,241 frontal radiographs (915 positive, 326 negative; 73.73% positive prevalence).

---

## 3. Best Epoch and Model Selection

Model selection was governed by validation Area Under the ROC Curve (AUROC) on the full, leak-free validation set of 10,024 frontal images:

- **Optimal Checkpoint:** **Epoch 3** of 5.
- **Validation AUROC at Best Epoch:** **0.89735** (`0.8973476016`).
- **Validation Loss at Best Epoch:** **0.34589** (lowest across all 5 epochs).
- **Overfitting Prevention:** Validation loss began increasing at Epoch 4 (0.3643) and Epoch 5 (0.3637) despite continuous drops in training loss (0.1899 $\to$ 0.0946), confirming Epoch 3 as the optimal generalization checkpoint.

---

## 4. Validation Parity and Patient Isolation Audits

A dedicated, standalone validation diagnostic ([`scripts/chexpert_client02_validation_diagnostic.py`](file:///D:/DS%20CP/adaptive-federated-aggregation/scripts/chexpert_client02_validation_diagnostic.py)) was executed over the full validation split:

1. **Patient Isolation:** **PASSED.**
   - Training cohort: 500 unique patient IDs.
   - Validation cohort: 4,081 unique patient IDs.
   - Overlap: Exactly **0 overlapping patient IDs** between training and validation.
2. **Dataset Accessibility:** **PASSED.**
   - All 10,024 of 10,024 validation images resolved and verified on disk (100.0% coverage, `is_complete = True`).
3. **Metric Parity Gate at Threshold 0.50:** **PASSED.**
   - Recomputed metrics matched saved baseline values (`val_metrics.json`) with an absolute difference $\le 5.26 \times 10^{-8}$ on continuous metrics (tolerance $1.0 \times 10^{-4}$).
   - Confusion-matrix counts matched **exactly**: $\text{TN} = 1,721, \text{FP} = 802, \text{FN} = 547, \text{TP} = 6,954$.

---

## 5. Performance Summary Across Splits

### 5.1 Validation Performance at Baseline Threshold ($\tau = 0.50$) vs. Candidate ($\tau = 0.70$)
*Evaluated on 10,024 frontal validation images (Prevalence = 74.83%; 7,501 positive, 2,523 negative):*

| Metric | Baseline ($\tau = 0.50$) | Candidate ($\tau = 0.70$)\* | Delta ($\Delta$) | Clinical / Operational Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **AUROC** | 0.89735 | 0.89735 | — | Robust threshold-independent discriminative capacity. |
| **PR-AUC** | 0.95191 | 0.95191 | — | Precision-recall area under high-prevalence validation data. |
| **Balanced Accuracy** | 0.80460 | **0.83074** | $\mathbf{+2.61\text{ pp}}$ | **Peak balanced accuracy across all 13 candidate thresholds.** |
| **Youden's J Index** | 0.60920 | **0.66147** | $\mathbf{+0.05227}$ | **Peak informedness across candidate thresholds.** |
| **Sensitivity** | 0.92708 | 0.87788 | $-4.92\text{ pp}$ | True Positive rate ($6,954 \to 6,585$; $+369$ false negatives). |
| **Specificity** | 0.68212 | **0.78359** | $\mathbf{+10.15\text{ pp}}$ | True Negative rate ($1,721 \to 1,977$; **256 false alarms eliminated**). |
| **Precision (PPV)** | 0.89660 | 0.92343 | $+2.68\text{ pp}$ | Fraction of positive flags representing true effusions. |
| **F1-Score** | 0.91158 | 0.90008 | $-1.15\text{ pp}$ | Harmonic mean favoring positive recall. |
| **Confusion Matrix** | $\text{TN}=1721, \text{FP}=802$<br>$\text{FN}=547, \text{TP}=6954$ | $\text{TN}=1977, \text{FP}=546$<br>$\text{FN}=916, \text{TP}=6585$ | $\Delta\text{TN}=+256$<br>$\Delta\text{FN}=+369$ | 31.9% reduction in false alarms at the cost of 5.3% relative sensitivity drop. |

*\*Candidate threshold $\tau = 0.70$ was selected strictly from exploratory validation data. It has not been used or tuned on test data.*

### 5.2 Held-Out Benchmark Test Performance at Baseline ($\tau = 0.50$)
*Evaluated on 202 frontal radiographs from the Stanford radiologist consensus benchmark (`valid.csv` in raw data, treated as held-out test in project schema; Prevalence = 31.68%; 64 positive, 138 negative):*

- **AUROC:** **0.83967**
- **PR-AUC:** **0.67338**
- **Sensitivity:** **0.93750** ($60 / 64$ detected; 4 false negatives)
- **Specificity:** **0.55797** ($77 / 138$ identified; 61 false positives)
- **Precision (PPV):** **0.49587** ($60 / 121$ positive predictions)
- **Balanced Accuracy:** **0.74774**
- **F1-Score:** **0.64865**
- **Confusion Matrix:** $\text{TN} = 77, \text{FP} = 61, \text{FN} = 4, \text{TP} = 60$

> [!NOTE]
> **Bayesian Precision Shift:** While Sensitivity remained consistent ($92.71\%$ validation vs $93.75\%$ test), Precision dropped from **89.66% on validation to 49.59% on test**. This shift is mathematically explained by Bayes' theorem under the base-rate decline from $74.83\%$ validation prevalence to $31.68\%$ test prevalence.

---

## 6. Key Limitations & Risk Summary

1. **Small Test Cohort Sample Size:**
   The test benchmark contains only 202 images across 200 patients (64 positives, 138 negatives). A single misclassification alters sensitivity by $1.56\%$ and specificity by $0.72\%$. Performance estimates on this split carry wide statistical uncertainty.
2. **Image-Level vs. Patient-Level Performance:**
   Validation images average $\approx 2.46$ radiographs per patient. Diagnostic treatment occurs at the patient level; image-level AUROC does not prove patient-level diagnostic accuracy.
3. **Frontal-Only Limitation:**
   Lateral views (15.94% of CheXpert) were excluded to ensure cross-institution interoperability with NIH (which lacks lateral views), missing shallow fluid accumulations ($<175\text{ mL}$) that are only visible in posterior sulci on lateral projections.
4. **Non-Clinical Research Milestone:**
   These models are local academic benchmarks and are not approved for clinical decision-making.

---

## 7. Local Artifact Directory & Replication Manifest

The local run artifacts reside on drive `D:` and are not hosted on GitHub due to repository size limits and `.gitignore` policies:

| Artifact Description | Local File Path | Size |
| :--- | :--- | :---: |
| **Model Checkpoint (Best Weights)** | `outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt` | 282.6 MB |
| **Final Checkpoint (Epoch 5)** | `outputs/chexpert/chexpert_resnet50_20261027_225232/final_checkpoint.pt` | 282.6 MB |
| **Validation Diagnostic Report** | `outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_diagnostic_report.json` | 7.1 KB |
| **Validation Threshold Sweep Table** | `outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_threshold_sweep.csv` | 1.9 KB |
| **Training Run Configuration** | `outputs/chexpert/chexpert_resnet50_20261027_225232/config.json` | 1.1 KB |
| **Training History (Epoch CSV)** | `outputs/chexpert/chexpert_resnet50_20261027_225232/history.csv` | 1.2 KB |
| **Training Run Metadata** | `outputs/chexpert/chexpert_resnet50_20261027_225232/run_metadata.json` | 1.6 KB |
| **Patient Cohort Manifest** | `outputs/chexpert/chexpert_resnet50_20261027_225232/patient_manifest.json` | 11.4 KB |
| **Path Resolution Audit Report** | `outputs/chexpert/chexpert_resnet50_20261027_225232/path_resolution_report.json` | 8.9 KB |
| **Saved Validation Metrics** | `outputs/chexpert/chexpert_resnet50_20261027_225232/val_metrics.json` | 795 B |
| **Saved Test Metrics** | `outputs/chexpert/chexpert_resnet50_20261027_225232/test_metrics.json` | 766 B |

---

## 8. Handoff Instructions for the Project Lead

To integrate CheXpert Client 02 into the multi-client federated aggregation experiments:

1. **Obtain the Model Checkpoint:**  
   The binary weight checkpoint `best_checkpoint.pt` (282.6 MB) is stored in the local run directory specified above. Per repository instructions ([`docs/traning_guide/CHEXPERT_TRAINING_GUIDE.md`](file:///D:/DS%20CP/adaptive-federated-aggregation/docs/traning_guide/CHEXPERT_TRAINING_GUIDE.md) Section 6), please obtain the zipped run folder via our shared team storage (Google Drive / OneDrive).
2. **Re-run Validation Diagnostic:**  
   To reproduce the parity gate and threshold sweep independently:
   ```powershell
   python scripts/chexpert_client02_validation_diagnostic.py
   ```
3. **Detailed Model Evaluation Reference:**  
   Consult [`docs/reports/chexpert_client02_model_evaluation.md`](file:///D:/DS%20CP/adaptive-federated-aggregation/docs/reports/chexpert_client02_model_evaluation.md) for full statistical derivations, distribution analyses, and clinical trade-off curves.
