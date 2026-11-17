# CheXpert Client 02: Federated Learning Aggregated Checkpoint Evaluation Report

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Client Node:** Client 02 — CheXpert (Stanford Hospital, Stanford, CA, USA)  
**Disease Target:** Pleural Effusion (Binary Classification)  
**Report Date:** November 17, 2026 (13:00 UTC)  
**Deliverable Directory:** `chexpert_evaluation_results/`  

---

> [!CAUTION]
> ### Mandatory Research Limitation Notice
> **Exploratory post-hoc checkpoint aggregation evaluation – not a validated multi-round Federated Learning experiment.**  
> These aggregated checkpoints were created by post-hoc averaging of independently trained client checkpoints. The original client models were not initialized from a verified common shared checkpoint, nor did they participate in synchronous, multi-round federated training with server-orchestrated weight broadcasting.  
> **Do not claim that Capped FedAvg improves disease prediction.**

---

## 1. Executive Summary & Audit Overview

This evaluation report documents the formal validation audit conducted by the CheXpert Client node on two centralized post-hoc aggregated model checkpoints:
1. **Standard FedAvg (`fedavg_oneshot_checkpoint.pt`)**
2. **Capped FedAvg with $c = 0.40$ (`capped_fedavg_c40_oneshot_checkpoint.pt`)**

In strict accordance with the project's scientific integrity guidelines:
- Evaluation was restricted exclusively to the **CheXpert validation dataset** ($N = 10,024$ frontal chest radiographs).
- **Test Set Protection:** The held-out test split (`data/processed/CheXpert/test.csv`) was **strictly never accessed, loaded, or evaluated**.
- **No Retraining or Tuning:** Model weights were neither modified, trained, nor fine-tuned.
- **Strict Anti-Fabrication Rule:** Missing metrics are explicitly reported as `N/A (Failed to load)` rather than imputed or fabricated.

---

## 2. Checkpoint Loading & File Availability Status

| Checkpoint Name | Prescribed Relative Path | File Existence | Model Load Status | Error Description / Reason |
| :--- | :--- | :---: | :---: | :--- |
| **FedAvg (One-Shot)** | `outputs/final_research_experiments/checkpoints/fedavg_oneshot_checkpoint.pt` | **MISSING** | `FAILED_TO_LOAD` | `FileNotFoundError`: Checkpoint file not deposited on local disk. |
| **Capped FedAvg ($c=0.40$)** | `outputs/final_research_experiments/checkpoints/capped_fedavg_c40_oneshot_checkpoint.pt` | **MISSING** | `FAILED_TO_LOAD` | `FileNotFoundError`: Checkpoint file not deposited on local disk. |

### Technical Investigation of Absence
1. **Local Filesystem Audit:** A recursive search of the workspace (`D:\DS CP\adaptive-federated-aggregation\`) confirmed that the directory `outputs/final_research_experiments/checkpoints/` does not exist on disk.
2. **Git Repository Status:** Git version history across all remote and local branches (`main`, `feature/federated-learning-baseline`, and `origin/Piyush`) was examined. Model weight binaries (`*.pt`) and the `outputs/` directory are excluded from version control by `.gitignore` (consistent with GitHub's 100 MB hard file limit).
3. **Client Workflow Separation:** The CheXpert client successfully trained and handed off its local model (`outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt`, Val AUROC = **0.89735**) to the central researcher via `CheXpert_Client02_Handoff.zip`. The central researcher performed post-hoc aggregation on their central host, but the resulting aggregated files have not yet been transferred or placed into this client environment.

---

## 3. Dataset Configuration & Evaluation Protocol

- **Dataset Source:** Stanford CheXpert (`data/processed/CheXpert/val.csv`).
- **Split Evaluated:** Validation split only ($N = 10,024$ frontal images).
- **View Filter Policy:** Frontal views only (Anteroposterior `AP` and Posteroanterior `PA`). Lateral and costal views were strictly excluded.
- **Patient Isolation:** Zero patient overlap with the 500-patient training cohort (confirmed via `patient_manifest.json` intersection check: $\text{Train} \cap \text{Val} = \emptyset$).
- **Image Preprocessing:** Deterministic evaluation pipeline:
  - Input resolution: $224 \times 224$ pixels.
  - Channels: 3-channel RGB (grayscale replicated across channels).
  - Normalization: Standard ImageNet constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Decision Threshold:** Fixed standard benchmark threshold $\tau = 0.50$.
- **Inference Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU using PyTorch 2.13.0+cu130 on Windows-11-10.0.26300-SP0.

---

## 4. Evaluation Metrics & Comparative Performance

| Metric | CheXpert Local Baseline (Epoch 3 Client Model) | Aggregated FedAvg (One-Shot) | Aggregated Capped FedAvg ($c = 0.40$) |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.89735** | N/A (Failed to load) | N/A (Failed to load) |
| **PR-AUC** | **0.95191** | N/A (Failed to load) | N/A (Failed to load) |
| **Sensitivity (Recall)** | **0.92708** | N/A (Failed to load) | N/A (Failed to load) |
| **Specificity** | **0.68212** | N/A (Failed to load) | N/A (Failed to load) |
| **Precision** | **0.89660** | N/A (Failed to load) | N/A (Failed to load) |
| **F1-Score** | **0.91158** | N/A (Failed to load) | N/A (Failed to load) |
| **Balanced Accuracy** | **0.80460** | N/A (Failed to load) | N/A (Failed to load) |
| **Confusion Matrix** | TN=1721, FP=802, FN=547, TP=6954 | N/A | N/A |
| **Status** | **VERIFIED (On Disk)** | **FAILED_TO_LOAD** | **FAILED_TO_LOAD** |

*Note: The CheXpert Local Baseline represents the verified 500-patient ResNet-50 model trained on Client 02 and validated on the exact same 10,024 images.*

---

## 5. Model Comparison & Analysis

Because the aggregated checkpoint files `fedavg_oneshot_checkpoint.pt` and `capped_fedavg_c40_oneshot_checkpoint.pt` were not present in `outputs/final_research_experiments/checkpoints/`, numerical inference could not be executed for the aggregated models.

### Expected Theoretical Dynamics Under Post-Hoc One-Shot Aggregation
1. **Weight Dominance in Standard FedAvg:**
   Under standard FedAvg, aggregation weights are proportional to client sample counts ($p_k = n_k / N$). If CheXpert contributed a disproportionately large training subset, FedAvg weights would heavily skew toward CheXpert's parameter distribution.
2. **Impact of Capped FedAvg ($c = 0.40$):**
   By capping the maximum client weight fraction at $0.40$ and renormalizing, Capped FedAvg artificially elevates the relative representation of minority clients (e.g., NIH ChestX-ray14 and PadChest).
3. **Weight Permutation & Symmetry Incompatibility (Critical Flaw):**
   Because the three client models were trained **independently without a common initialization checkpoint**, standard neural network permutation symmetries (channel permutations and non-convex saddle points) guarantee that direct parameter averaging will cause catastrophic destructive interference. One-shot parameter averaging of independently initialized deep ResNets typically collapses intermediate representations, resulting in near-random or degraded diagnostic predictions.

---

## 6. Confusion Matrix Images Status

- **FedAvg Confusion Matrix Image:** Not generated (`fedavg_confusion_matrix.png` unsupported due to absence of model predictions).
- **Capped FedAvg Confusion Matrix Image:** Not generated (`capped_fedavg_c40_confusion_matrix.png` unsupported due to absence of model predictions).
- **Infrastructure Status:** The plotting pipeline (`matplotlib` / `pyplot`) is verified and fully functional in `scripts/evaluate_aggregated_checkpoints.py`. Once the `.pt` files are deposited, running the evaluation script will immediately generate and save high-resolution confusion matrix plots.

---

## 7. Action Items for Central Researcher

To complete the numerical validation run:
1. Deposit the two aggregated checkpoint files into the client workstation:
   - `outputs/final_research_experiments/checkpoints/fedavg_oneshot_checkpoint.pt`
   - `outputs/final_research_experiments/checkpoints/capped_fedavg_c40_oneshot_checkpoint.pt`
2. Execute the verified evaluation harness:
   ```bash
   python scripts/evaluate_aggregated_checkpoints.py --output_dir chexpert_evaluation_results
   ```
3. All outputs (`fedavg_validation_metrics.json`, `capped_fedavg_validation_metrics.json`, `comparison.csv`, `evaluation_report.md`, and confusion matrix PNGs) will be automatically recomputed and populated.
