# PadChest Client 03 — Official Training & Federated Integration Handoff Guide

> **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
> **Client Identifier:** `client_03_PadChest`  
> **Target Disease:** Pleural Effusion (Binary Classification: `1` = Present, `0` = Absent)  
> **Shared Model:** ResNet-50 (`ResNet50Binary`, Single-Logit Output, BCEWithLogitsLoss)  
> **Target Audience:** Teammates responsible for downloading official PadChest data, running Client 03 training, and generating checkpoints for federated aggregation.

---

## 1. Executive Summary & Objective

This repository contains the complete preprocessing, path-resolution, preflight guard, ResNet-50 training pipeline, and federated client adapters for **PadChest** (Client 03).

### What Has Already Been Built & Verified
1. **Official 60K Project Cohort:** The official PadChest 160K metadata (`PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`, 160,861 rows) has been ingested and partitioned into an agreed **60,243-record** project cohort (`data/processed/PadChest/`) across 33,412 unique patients with **strictly zero patient leakage** and **zero study leakage**.
2. **Frontal Policy:** The cohort applies the project-wide frontal-view policy (`PA`, `AP`, `AP_horizontal`), excluding Lateral (`L`) and Costal rib views, yielding **34,367 training frontal images**, **4,278 validation frontal images**, and **4,243 test frontal images**.
3. **Empty-Validation Safety:** Validation metrics and model selection cleanly handle empty loaders without crashing or fabricating `0.0` accuracy scores.
4. **Automated Preflight:** `scripts/preflight_padchest.py` audits system dependencies, metadata integrity, archive path resolution, and class balance before training begins.
5. **One-Command Workflow:** Standardized runners in `scripts/` driven by `configs/padchest.yaml`.
6. **Full Automated Test Suite:** 71 test cases passing across client pipelines, preflight checks, FedAvg aggregation, and label mapping.

### What You (The Teammate) Need to Do
1. Follow the prerequisites below to activate your virtual environment.
2. Request and download the official PadChest image archives from BIMCV.
3. Extract the downloaded archives into `data/raw/PadChest/images/`.
4. Execute the verification, preflight, and training commands.
5. Send back the generated training artifacts and checkpoint to the lead repository coordinator.

---

## 2. Prerequisites & System Requirements

### A. Software & Environment
- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13 (Python 3.13 64-bit tested and verified).
- **Operating System:** Linux, macOS, or Windows 10/11 (PowerShell or Bash).
- **Virtual Environment:** Use a dedicated `.venv` in the repository root.
- **Git Branch:** `feature/padchest-preprocessing-smoke-test` (or merged `main`).

```powershell
# Windows PowerShell setup:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS Bash setup:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### B. Hardware Recommendations
- **GPU (Recommended):** NVIDIA GPU with >= 8 GB VRAM (CUDA 11.8+ or 12.1+).
- **CPU (Supported Fallback):** Multi-core x86_64 CPU (training time will be significantly longer; reduce batch size or use pilot mode).
- **RAM:** 16 GB minimum, 32 GB recommended.
- **Storage:**
  - **Full 160K archive:** ~1.1 TB uncompressed (or ~100–150 GB compressed tar archives).
  - **Pilot / Subset:** ~10–25 GB depending on number of chunks extracted.

To verify your compute device in Python:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 3. Official Dataset Access & Download Instructions

> [!IMPORTANT]
> **Data Licensing & Privacy Compliance**  
> PadChest is clinical radiograph data managed by the **Valencian Institute of Oncology (IVO)** and **Biomedical Imaging Center (BIMCV)**. The dataset is distributed under a **Data Use Agreement for Non-Commercial Academic Research**.  
> You must register, submit your institutional credentials, and accept their license terms directly. **Do not redistribute raw images or share download links.**

### A. Access Registration
1. Navigate to the official PadChest project portal:
   👉 **BIMCV-PadChest Official Portal:** [https://bimcv.cipf.es/bimcv-projects/padchest/](https://bimcv.cipf.es/bimcv-projects/padchest/)
2. Request access following their documented procedure. You will receive authorized download credentials / links.

### B. Required Files
1. **Metadata CSV:**  
   `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`  
   *Note: This file is already pre-staged in your workspace at `data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`.*
2. **Image Archives:**  
   The image collection is packaged into 50+ compressed archives (numbered `0.tar.gz`, `1.tar.gz`, ... `50.tar.gz` or `images-0.zip` etc.).
   - For **Full Official Training:** Download all archives.
   - For **Development / Pilot Testing:** Download archives `0`, `1`, `2`, `3`, `4`, and `5` (~5,000–10,000 images).

### C. Extraction Procedure
Extract the archives into `data/raw/PadChest/images/` such that numbered subdirectories (`0/`, `1/`, etc.) or images are placed inside.

```bash
# Example extraction on Linux/macOS:
mkdir -p data/raw/PadChest/images
tar -xzf 0.tar.gz -C data/raw/PadChest/images/
tar -xzf 1.tar.gz -C data/raw/PadChest/images/

# Example extraction on Windows PowerShell (using 7-Zip or tar):
tar -xzf 0.tar.gz -C data/raw/PadChest/images/
```

The pipeline's `PadChestPathResolver` scans up to 6 levels deep and automatically indexes both numbered subdirectories (`images/0/<ImageID>.png`) and flat structures (`images/<ImageID>.png`), while strictly ignoring macOS metadata sidecars (`._*`).

---

## 4. Expected Directory Structure

Ensure your project filesystem matches this structure before running preflight:

```text
Project Repo/
├── configs/
│   ├── padchest.yaml            <-- Official full training configuration
│   └── padchest_pilot.yaml      <-- 100-patient pilot configuration
├── data/
│   ├── processed/
│   │   └── PadChest/
│   │       ├── train.csv        <-- 48,340 records (34,367 frontal)
│   │       ├── val.csv          <-- 5,977 records (4,278 frontal)
│   │       ├── test.csv         <-- 5,926 records (4,243 frontal)
│   │       └── summary.json     <-- 60,243 cohort summary
│   └── raw/
│       └── PadChest/
│           ├── metadata/
│           │   ├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
│           │   └── chest_x_ray_images_labels_sample.csv (optional sample)
│           └── images/          <-- EXTRACTED ARCHIVES GO HERE
│               ├── 0/
│               │   └── <ImageID>.png
│               ├── 1/
│               │   └── <ImageID>.png
│               ├── ...
│               └── 50/
│                   └── <ImageID>.png
├── scripts/
│   ├── prepare_padchest.py      <-- Dataset ingestion and split audit
│   ├── preflight_padchest.py    <-- Pre-training dataset readiness guard
│   └── train_padchest.py        <-- Training executor
├── src/
│   ├── data/
│   │   └── padchest_dataset.py  <-- Resolver, transforms, view filter
│   ├── federated/
│   │   ├── adapters.py          <-- build_padchest_client
│   │   └── preflight.py         <-- Experiment readiness audit
│   ├── models/
│   │   └── resnet.py            <-- Shared ResNet50Binary builder
│   ├── preprocess_padchest.py   <-- Preprocessing logic
│   ├── train_padchest.py        <-- Local training orchestration
│   └── trainer.py               <-- ClientTrainer engine
└── outputs/
    └── padchest/                <-- Generated checkpoints & logs
```

---

## 5. Step-by-Step Command Workflow

Run the following commands in order from the repository root:

### Step 1: Verify the Existing Cohort Splits
Verify that the 60K project cohort manifests in `data/processed/PadChest/` are present and confirm zero patient leakage:
```bash
python scripts/prepare_padchest.py --config configs/padchest.yaml --verify_only
```
*Expected Result:* `[SUCCESS] All existing PadChest cohort splits verified successfully.`

### Step 2: Run Dataset Preflight Readiness Audit
Once images are extracted into `data/raw/PadChest/images/`, execute preflight to audit image resolution and verify class balance:
```bash
# For full training run:
python scripts/preflight_padchest.py --config configs/padchest.yaml

# Or for small sample / pilot testing:
python scripts/preflight_padchest.py --config configs/padchest_pilot.yaml --smoke_test
```
*Expected Result:*
- Checks 1 to 5 report `[PASSED]`.
- Summary concludes: `PREFLIGHT RESULT: PASSED - PadChest client is fully ready for training.`

> [!CAUTION]
> If preflight fails, **do not proceed to training**. Read the bulleted blocker issues printed to the terminal, extract the missing image archives, and rerun preflight until it passes.

### Step 3 (Optional): Run Dry-Run Pipeline Validation
Verify DataLoaders, model initialization, and artifact paths without training:
```bash
python scripts/train_padchest.py --config configs/padchest.yaml --dry_run
```

### Step 4: Execute PadChest Model Training
Launch the official ResNet-50 training run:
```bash
# Official full run (uses parameters in configs/padchest.yaml: 10 epochs, batch size 32, lr 1e-4):
python scripts/train_padchest.py --config configs/padchest.yaml

# Or run on a pilot subset (100 patients, 2 epochs) if testing hardware:
python scripts/train_padchest.py --config configs/padchest_pilot.yaml
```

**Custom CLI Overrides (optional):**
```bash
# Run with CUDA GPU explicitly and custom batch size:
python scripts/train_padchest.py --config configs/padchest.yaml --device cuda --batch_size 64

# Run with mixed precision (AMP fp16) on NVIDIA GPU:
python scripts/train_padchest.py --config configs/padchest.yaml --device cuda --use_amp
```

### Step 5: Checkpoint Verification
The training runner automatically verifies the generated `best_checkpoint.pt` at the end of training using strict PyTorch state dictionary validation. You can also re-verify manually:
```bash
python -c "
import torch
from src.models.resnet import build_resnet50_binary, load_checkpoint
model = build_resnet50_binary(pretrained=False, device='cpu')
ckpt = load_checkpoint('outputs/padchest/padchest_client03_resnet50/best_checkpoint.pt', model=model, strict=True)
print('[SUCCESS] Checkpoint loaded cleanly. Best epoch:', ckpt['epoch'], '| Val AUROC:', ckpt['best_val_metric'])
"
```

### Step 6: Verify Federated Client Compatibility
Confirm that the federated adapter can initialize and synchronize parameters for federated learning:
```bash
python -c "
from src.federated.adapters import build_padchest_client
client = build_padchest_client(batch_size=16, device='cpu')
print('[SUCCESS] PadChest FederatedClient ready. Accessible train samples:', client.get_sample_count())
"
```

---

## 6. Expected Output Artifacts

All outputs are written to `outputs/padchest/<run_name>/`:

| File | Description | Purpose |
|---|---|---|
| `best_checkpoint.pt` | Model weights with highest validation AUROC | Shared with team for FedAvg aggregation |
| `final_checkpoint.pt` | Model weights after the final epoch | Training completion record |
| `history.csv` | Epoch-by-epoch loss, AUROC, PR-AUC, learning rate | Convergence and learning curve analysis |
| `val_metrics.json` | Comprehensive validation metrics (sensitivity, specificity, F1) | Model selection benchmark |
| `test_metrics.json` | Held-out test split evaluation metrics | Final performance benchmark |
| `config.json` | Exact hyperparameters and hardware configuration | Reproducibility record |
| `run_metadata.json` | Hardware, OS, PyTorch version, and execution duration | Environment audit |
| `patient_manifest.json` | Selected patient IDs, class distribution, and slicing info | Patient isolation audit |
| `path_resolution_report.json` | Image resolution statistics and archive availability | Dataset coverage audit |

---

## 7. Troubleshooting & Common Issues

### Issue 1: "PREFLIGHT BLOCKED: Zero accessible PadChest training images found on disk"
- **Cause:** Radiograph files are not in `data/raw/PadChest/images/` or were extracted into an unexpected path.
- **Solution:** Verify the path. Run `dir data\raw\PadChest\images` (Windows) or `ls data/raw/PadChest/images` (Linux). Ensure image files (`.png`) or numbered subfolders (`0/`, `1/`, etc.) exist inside.

### Issue 2: "Only 24 images found / 7 frontal images passing"
- **Cause:** You are running against the tiny Kaggle sample metadata instead of the official 160K dataset.
- **Solution:** Ensure `configs/padchest.yaml` points to `data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`.

### Issue 3: "CUDA Out of Memory (OOM)"
- **Cause:** GPU VRAM exceeded by large batch size.
- **Solution:** Reduce batch size via command line:
  `python scripts/train_padchest.py --config configs/padchest.yaml --batch_size 16` (or `--batch_size 8`).
  You can also enable mixed precision: `--use_amp`.

### Issue 4: "Empty validation split / val_loss is N/A"
- **Cause:** Preprocessing or path resolution resolved 0 frontal images in the validation split.
- **Solution:** Ensure you extracted multiple image archive chunks so that validation patient images are present. In pilot runs with small subsets, use `--sample_from_accessible` so sampling only selects patients with images available on disk.

### Issue 5: "Missing required dependencies: torchvision, pillow"
- **Cause:** Virtual environment lacks required packages.
- **Solution:** Run: `pip install -r requirements.txt`.

---

## 8. What to Deliver to the Project Lead

After training completes, compress and deliver the run directory:
📁 `outputs/padchest/padchest_client03_resnet50/`

### Exact Delivery Checklist:
- [ ] `best_checkpoint.pt`
- [ ] `val_metrics.json`
- [ ] `test_metrics.json` (if test evaluation was run)
- [ ] `history.csv`
- [ ] `config.json`
- [ ] `run_metadata.json`
- [ ] `path_resolution_report.json`
- [ ] Terminal log from `python scripts/preflight_padchest.py --config configs/padchest.yaml`

> [!CAUTION]
> **DO NOT commit or upload raw medical images (`.png`), dataset tar archives, or model checkpoints to GitHub.**  
> Share the zipped `outputs/padchest/<run_name>/` folder via Google Drive, OneDrive, or the team's secure cloud storage.

---

## 9. Instructions for the Teammate's Coding Agent

> [!NOTE]
> If you are an AI coding assistant (e.g. Antigravity, Claude Code, Cursor, Copilot) assisting the teammate, you MUST adhere strictly to the following rules:

1. **Read This Document First:** Do not guess file paths, dataset schemas, or model architectures. Everything is specified here.
2. **Preserve Shared Architectures:** Do NOT modify `ResNet50Binary` in `src/models/resnet.py`. All three federated clients (NIH, CheXpert, PadChest) share this exact 320-parameter state dictionary with classification head `classifier` (`[1, 2048]`). Changing parameter names or shapes breaks FedAvg aggregation.
3. **Preserve the 60K Project Cohort:** Do NOT regenerate `data/processed/PadChest/` unless explicitly instructed. The existing manifests (`train.csv`: 48,340, `val.csv`: 5,977, `test.csv`: 5,926) are already partitioned with zero patient leakage and agreed upon by the team.
4. **Enforce Binary Label Policy:**
   - Positive (`target = 1`): Must contain `pleural effusion`.
   - Negative (`target = 0`): Must strictly be `['normal']`.
   - Ambiguous findings (e.g. `costophrenic angle blunting`, `pleural thickening`, `pericardial effusion`) must remain excluded. Do NOT fabricate negative labels for unmentioned findings.
5. **Enforce Frontal Policy:** Radiographs must be filtered for `PA`, `AP`, or `AP_horizontal`. Lateral (`L`) and Costal rib views must be excluded.
6. **Preflight Guard Is Mandatory:** Always run `python scripts/preflight_padchest.py` before launching any model training. Do not bypass preflight errors by commenting out assertions.
7. **No Git Data Commits:** Never commit `.png`, `.pt`, `.tar.gz`, or raw CSV files to Git. Respect `.gitignore`.
8. **Report Actual Metrics:** If validation or test metrics are undefined due to an incomplete subset, report them truthfully as partial/uninformative rather than modifying filters to fabricate numbers.
