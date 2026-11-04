# PadChest (Client 03) Training & Federated Handoff Guide

> **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
> **Client ID:** `client_03_PadChest`  
> **Target Disease:** Pleural Effusion (Binary Classification: `1` = Effusion Present, `0` = Absent/Normal)  
> **Shared Model:** ResNet-50 (`ResNet50Binary`, single-logit binary classification head, BCEWithLogitsLoss)  
> **Target Audience:** Teammate responsible for obtaining initial PadChest archives (`0.zip`, `1.zip`), running preflight verification, executing Client 03 ResNet-50 training, and packaging deliverables for federated aggregation.

---

> [!CAUTION]
> **SAMPLE METADATA STRICTLY FORBIDDEN:**  
> The 24-record Kaggle sample CSV (`chest_x_ray_images_labels_sample.csv`) has been completely removed and invalidated. It must **never** be used for cohort generation, preflight checks, or model training.
> The automated pipeline will reject any file containing `"sample"` in its name or any metadata file with fewer than 100,000 records.
> The single source of truth for PadChest (Client 03) is the official full metadata file:  
> `data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`

---

## 1. Project & Client 03 Overview

Our research develops an adaptive federated learning algorithm across three distinct clinical radiographic populations:
1. **Client 01:** NIH ChestX-ray14 (Hospital-based clinical dataset)
2. **Client 02:** CheXpert (Stanford University Medical Center cohort)
3. **Client 03:** PadChest (Hospital Universitario San Juan de Alicante, Spain)

Each federated client trains a private local **ResNet-50 binary classifier** predicting **Pleural Effusion** from chest radiographs. Local checkpoints are periodically communicated to the central federated coordinator for parameter aggregation (FedAvg / Capped FedAvg).

### Key Architectural & Preprocessing Standards
- **Model Backbone:** Standard ResNet-50 with single-logit linear classification head (`classifier`: `in_features=2048, out_features=1`).
- **Input Dimensions:** 3-channel RGB image tensor of size `(3, 224, 224)`.
- **Normalization:** ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- **Binary Loss:** `BCEWithLogitsLoss`.
- **Evaluation Metric:** Area Under the Receiver Operating Characteristic (AUROC) and Precision-Recall AUC (PR-AUC).
- **Patient Isolation:** Strictly patient-level train, validation, and test splits with **zero patient leakage** and **zero study leakage**.

---

## 2. Verified PadChest Metadata Schema & Label Ontology

The official PadChest metadata file is:
`PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz` (or uncompressed `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`).

It contains **160,861 records** across 36 columns. Below are the verified columns utilized by our ingestion and training pipelines:

| Column Name | Data Type | Description & Pipeline Usage |
|---|---|---|
| `ImageID` | String | Unique radiograph filename (e.g. `20536686640136348236148679891455886468_k6ga29.png`). Used for filesystem resolution and dataset indexing. |
| `ImageDir` | Int / Float | Archive chunk number (e.g. `0`, `1`, ..., `50`). Identifies which `.zip` or `.tar.gz` archive chunk contains the image file. |
| `PatientID` | String / Object | Unique anonymized patient identifier (e.g. `839860488694292331637988235681460987`). Mandatory for patient-level split partitioning. |
| `StudyID` | String / Object | Clinical examination identifier (e.g. `20536686640136348236148679891455886468`). Enforces multi-image study grouping and zero study leakage. |
| `Projection` | String | Projection angle: `PA`, `AP`, `AP_horizontal`, `L` (Lateral), `COSTAL`, `EXCLUDE`, `UNK`. |
| `ViewPosition_DICOM` | String | DICOM acquisition view: `POSTEROANTERIOR`, `LATERAL`, `LL`, `RL`, `ANTEROPOSTERIOR`. Used as secondary lateral check. |
| `Labels` | String (repr list) | Radiologist finding labels formatted as stringified Python list, e.g. `"['normal']"`, `"['pleural effusion']"`, `"['pulmonary fibrosis', 'chronic changes']"`. |
| `PatientBirth` | Float | Year of birth (e.g. `1930.0`), used with `StudyDate_DICOM` to derive patient age. |
| `StudyDate_DICOM` | Int / String | Examination date (e.g. `20140915`). |
| `PatientSex_DICOM` | String | Biological sex (`M`, `F`). |
| `Rows_DICOM`, `Columns_DICOM` | Int | Native image height and width. |
| `Manufacturer_DICOM` | String | X-ray scanner manufacturer (e.g. `ImagingDynamicsCompanyLtd`, `PhilipsMedicalSystems`). |

### Research Label Policy (Strict Binary Classification)
- **Positive (`target = 1`):** Radiograph label list explicitly contains `'pleural effusion'`. (e.g. `['pleural effusion']` or `['pleural effusion', 'pneumonia']`).
- **Negative (`target = 0`):** Radiograph label list is strictly and solely `['normal']` (no other co-occurring findings).
- **Excluded Cases:** Any non-effusion pathology (e.g. `COPD signs`, `cardiomegaly`, `aortic elongation`), ambiguous findings (`costophrenic angle blunting`, `pleural thickening`), or empty/missing labels. Substring pitfalls like `pericardial effusion` are strictly excluded from matching Pleural Effusion.
- **Frontal View Policy:** Retains only `PA`, `AP`, and `AP_horizontal`. Excludes Lateral (`L`), Costal rib views, and contradicting lateral DICOM view positions.

---

## 3. Data Ingestion & Dataset Concept Distinctions

To avoid any confusion during execution, understand these distinct concepts:

```text
[ 1. Full Official Metadata ]  ==> 160,861 raw records (36 columns)
            │
            ▼ (Apply binary label policy: effusion=1, normal=0, exclude ambiguous)
[ 2. Full Eligible Cohort ]   ==>  60,243 records (54,408 normal, 5,835 effusion)
            │
            ▼ (Apply frontal projection filter: PA, AP, AP_horizontal)
[ 3. Frontal Cohort ]         ==>  42,888 frontal images (35,946 negative, 6,942 positive)
            │
            ▼ (Intersect with downloaded archives: 0.zip & 1.zip)
[ 4. Archive Matches (0+1) ]  ==>   6,028 images in chunks 0 and 1
            │
            ▼ (Apply binary labels & frontal view filter to chunk 0+1)
[ 5. Usable Chunk 0+1 Images ] ==>  1,534 accessible frontal images (1,334 negative, 200 positive)
            │
            ▼ (Partition by agreed patient splits)
[ 6. Accessible Per Split ]   ==> Train: 1,234 | Val: 160 | Test: 140
```

- **Metadata Records:** Total entries in the official CSV (160,861).
- **Archive Matches:** Rows in the metadata whose `ImageDir` matches archives you downloaded (`0` and `1`).
- **Extracted Images:** Radiograph `.png` files unzipped onto your local hard drive in `data/raw/PadChest/images/`.
- **Usable Training Images:** Radiographs that exist physically on disk, belong to the assigned patient split, pass frontal-view filters, and carry an unambiguous binary label.

---

## 4. Exact Directory Structure

Set up your local directory tree relative to the repository root:

```text
Project Repo/
│
├── configs/
│   ├── padchest.yaml                    <-- Official full dataset configuration
│   ├── padchest_partial.yaml            <-- Initial partial archive configuration (0.zip & 1.zip)
│   └── padchest_pilot.yaml              <-- 100-patient pilot configuration
│
├── data/
│   ├── raw/
│   │   └── PadChest/
│   │       ├── archives/                <-- STORE DOWNLOADED ZIP FILES HERE
│   │       │   ├── 0.zip
│   │       │   └── 1.zip
│   │       │
│   │       ├── archive_manifests/       <-- STORE ARCHIVE LISTING FILES HERE
│   │       │   ├── 0.zip.unzip-l.txt
│   │       │   └── 1.zip.unzip-l.txt
│   │       │
│   │       ├── metadata/                <-- OFFICIAL 160K METADATA ONLY
│   │       │   └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
│   │       │       (or PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv)
│   │       │
│   │       └── images/                  <-- EXTRACTED RADIOGRAPHS GO HERE
│   │           ├── 0/ (or flat)
│   │           └── 1/
│   │
│   └── processed/
│       └── PadChest/
│           ├── train.csv                <-- 48,263 official cohort records (34,369 frontal)
│           ├── val.csv                  <--  6,090 official cohort records (4,321 frontal)
│           ├── test.csv                 <--  5,890 official cohort records (4,198 frontal)
│           ├── summary.json             <-- Verified 60,243 cohort summary with SHA256
│           ├── padchest_cohort.csv      <-- Consolidated cohort manifest
│           ├── train_cohort.csv         <-- Matching train manifest
│           ├── val_cohort.csv           <-- Matching validation manifest
│           ├── test_cohort.csv          <-- Matching test manifest
│           └── cohort_summary.json      <-- Matching cohort summary
│
├── scripts/
│   ├── prepare_padchest.py              <-- Ingestion and split verification runner
│   ├── preflight_padchest.py            <-- Dataset readiness preflight guard
│   └── train_padchest.py                <-- Client 03 ResNet-50 training runner
│
├── src/
│   ├── data/
│   │   └── padchest_dataset.py          <-- PadChestPathResolver, transforms, frontal filter
│   ├── federated/
│   │   ├── adapters.py                  <-- build_padchest_client
│   │   └── preflight.py                 <-- Multi-client experiment preflight
│   ├── models/
│   │   └── resnet.py                    <-- ResNet50Binary model architecture
│   ├── config_loader.py                 <-- Pure-Python YAML/JSON parser
│   ├── preprocess_padchest.py           <-- Preprocessing logic & validation guard
│   ├── train_padchest.py                <-- Local training orchestration
│   └── trainer.py                       <-- ClientTrainer engine
│
└── outputs/
    └── padchest/                        <-- Generated run outputs & checkpoints
```

---

## 5. Dataset Download & Windows Extraction Instructions

### A. Dataset Source & Agreement
PadChest is managed by the Valencian Institute of Oncology (IVO) and Biomedical Imaging Center (BIMCV). Access requires submitting a Non-Commercial Academic Research Agreement:
- **Official BIMCV Portal:** [https://bimcv.cipf.es/bimcv-projects/padchest/](https://bimcv.cipf.es/bimcv-projects/padchest/)
- **Academic Download Host:** B2Drop / BIMCV Cloud.

### B. Files to Download Initially
1. Metadata file: `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`  
   *(Pre-staged in your workspace at `data/raw/PadChest/metadata/`)*
2. First two image archives: `0.zip` and `1.zip` (~1.5–3 GB each).

Place `0.zip` and `1.zip` in:
📁 `data/raw/PadChest/archives/`

### C. Remove Any Stale Sample Metadata Safely
If any old sample CSV files exist from earlier testing, remove them:
```powershell
Remove-Item -Path "data/raw/PadChest/metadata/*sample*.csv" -Force -ErrorAction SilentlyContinue
```

### D. Generate Archive Manifests
Before extracting, generate listing files so archive contents can be audited without opening the zip files:

```powershell
# Generate listing for 0.zip
tar -tf "data/raw/PadChest/archives/0.zip" > "data/raw/PadChest/archive_manifests/0.zip.unzip-l.txt"

# Generate listing for 1.zip
tar -tf "data/raw/PadChest/archives/1.zip" > "data/raw/PadChest/archive_manifests/1.zip.unzip-l.txt"
```
*(If `tar` is not available, use 7-Zip: `7z l data/raw/PadChest/archives/0.zip > data/raw/PadChest/archive_manifests/0.zip.unzip-l.txt`)*

### E. Extract Image Archives
Extract `0.zip` and `1.zip` into `data/raw/PadChest/images/`:

```powershell
# Create destination directory if not present:
New-Item -ItemType Directory -Path "data/raw/PadChest/images" -Force

# Extract 0.zip:
tar -xf "data/raw/PadChest/archives/0.zip" -C "data/raw/PadChest/images"

# Extract 1.zip:
tar -xf "data/raw/PadChest/archives/1.zip" -C "data/raw/PadChest/images"
```

### F. Verify Extraction Without Re-Extracting
To verify that radiographs are extracted and avoid re-extracting:

```powershell
# Count extracted PNG files:
(Get-ChildItem -Path "data/raw/PadChest/images" -Recurse -Filter "*.png").Count
```
*Expected Result:* Approximately **6,028 images** across archives `0` and `1`.

---

## 6. Step-by-Step Execution Workflow

Execute these commands in order from the repository root:

### Step 1: Verify Existing Official Cohort Splits
Verify that the official 60K project cohort manifests in `data/processed/PadChest/` are present and confirm zero patient leakage:
```bash
python scripts/prepare_padchest.py --config configs/padchest_partial.yaml --verify_only
```
*Expected Output:*
```text
=== Existing PadChest Split Verification ===
Directory: .../data/processed/PadChest
Total Cohort: 60,243 images across 33,412 unique patients
Verified Split Manifests:
  Split      Images   Patients   Studies   Positive   Negative
  ------------------------------------------------------------
  train      48,263     26,729     34084      7,963     40,300
  val         6,090      3,341      4300        971      5,119
  test        5,890      3,342      4183        919      4,971
  Patient Leakage: PASSED (Zero patient overlap across Train, Val, Test)
  Study Leakage:   PASSED (Zero study overlap across Train, Val, Test)
  Source Metadata: PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz (SHA256: 83f82ed54df4fc46...)
============================================
[SUCCESS] All existing PadChest cohort splits verified successfully.
```

*(If you ever need to regenerate the manifests from scratch, run `python scripts/prepare_padchest.py --config configs/padchest.yaml`)*.

### Step 2: Run Dataset Preflight Verification Guard
Execute the preflight audit to index the extracted images and verify accessibility:
```bash
python scripts/preflight_padchest.py --config configs/padchest_partial.yaml --allow_partial
```
*Expected Report Fields:*
```text
==============================================================================
  PADCHEST FEDERATED CLIENT PREFLIGHT VERIFICATION GUARD
==============================================================================
Execution Mode:       PARTIAL ARCHIVE SUBSET (0.zip & 1.zip)
Config File:          configs/padchest_partial.yaml
------------------------------------------------------------------------------
[Check 1/6] Runtime Dependencies:
  - torch           : INSTALLED
  - torchvision     : INSTALLED
  - pillow          : INSTALLED
  - resnet50_model  : INSTALLED

[Check 2/6] Metadata File Integrity:
  [PASSED] Metadata file exists: PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz (16.4 MB)

[Check 3/6] Processed Manifests & Patient Isolation:
  [PASSED] Patient isolation verified (Zero leakage across 33,412 patients)
  - train :  48,263 images | 26,729 pts | Pos: 7,963 | Neg: 40,300
  - val   :   6,090 images |  3,341 pts | Pos:   971 | Neg: 5,119
  - test  :   5,890 images |  3,342 pts | Pos:   919 | Neg: 4,971

[Check 4/6] Image Path Resolution & Archive Indexing:
  Indexed on disk: 6,028 image files (macOS ._* sidecars excluded)
  Frontal Train: Accessible: 1,234 / 34,369 (3.6%)
  Frontal Val:   Accessible:   160 /  4,321 (3.7%)
  Frontal Test:  Accessible:   140 /  4,198 (3.3%)

[Check 5/6] Federated Client Adapter Readiness:
  [PASSED] Federated adapter compatible (1,234 train images)

==============================================================================
PREFLIGHT RESULT: PASSED (PARTIAL ARCHIVE MODE)
  Notice: Evaluated on partial archive download (0.zip and 1.zip).
  Training will run on physically available frontal radiographs.
  Evaluation status will be marked: PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL.
==============================================================================
```

### Step 3: Optional Dry-Run Check
Verify DataLoaders, model tensor shapes, and manifest generation without starting gradient updates:
```bash
python scripts/train_padchest.py --config configs/padchest_partial.yaml --dry_run
```

### Step 4: Quick Pilot Run (100 Patients, 2 Epochs)
Run a fast 2-epoch pilot test to verify gradient backpropagation, CUDA/CPU operation, and metric calculation:
```bash
python scripts/train_padchest.py --config configs/padchest_pilot.yaml
```

### Step 5: Full Partial-Archive Training Run (Archives 0 & 1)
Train ResNet-50 across all physically accessible frontal radiographs in archives 0 and 1:
```bash
python scripts/train_padchest.py --config configs/padchest_partial.yaml
```
*(On GPU, this will train for 10 epochs on 1,234 training frontal images, validating on 160 validation frontal images with full validation AUROC model selection).*

**Optional Hardware Overrides:**
```bash
# Force CUDA device explicitly:
python scripts/train_padchest.py --config configs/padchest_partial.yaml --device cuda

# Enable Automatic Mixed Precision (AMP fp16) on NVIDIA GPU:
python scripts/train_padchest.py --config configs/padchest_partial.yaml --device cuda --use_amp

# Adjust mini-batch size (e.g. if VRAM is constrained):
python scripts/train_padchest.py --config configs/padchest_partial.yaml --batch_size 16
```

### Step 6: Post-Training Checkpoint & Federated Server Verification
Verify that `best_checkpoint.pt` loads strictly into the shared architecture and confirms federated client readiness:
```bash
python -c "
import torch
from src.models.resnet import build_resnet50_binary, load_checkpoint
from src.federated.adapters import build_padchest_client

# 1. Strict state dict verification
model = build_resnet50_binary(pretrained=False, device='cpu')
ckpt = load_checkpoint('outputs/padchest/padchest_partial_0_1_resnet50/best_checkpoint.pt', model=model, strict=True)
print('[PASSED] Checkpoint loaded cleanly. Best Epoch:', ckpt['epoch'], '| Best Val AUROC:', ckpt['best_val_metric'])

# 2. Federated adapter verification
client = build_padchest_client(batch_size=16, device='cpu')
print('[PASSED] PadChest FederatedClient initialized with', client.get_sample_count(), 'accessible train images.')
"
```

---

## 7. Understanding Partial Subset vs. Full Benchmark Status

> [!WARNING]
> **RESEARCH INTEGRITY & BENCHMARK CLASSIFICATION**  
> Because you are training on `0.zip` and `1.zip` (~1,534 frontal images) rather than the complete 160K PadChest dataset (~42,888 frontal images across 50 archives):
> - **Evaluation Status:** The pipeline explicitly tags all metrics as `PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL`.
> - **Coverage Percentage:** Reported as ~3.6% of the official training cohort.
> - **Scientific Reporting:** These results validate client convergence, parameter structure, and federated integration plumbing. They must **never** be cited or reported in papers as full-dataset disease prediction benchmarks. Full benchmark experiments will be run once all 50 archives are downloaded.

---

## 8. Expected Output Artifacts

All training artifacts are saved to:
📁 `outputs/padchest/padchest_partial_0_1_resnet50/`

| Output File | Contents | Significance |
|---|---|---|
| `best_checkpoint.pt` | Model state dictionary from the epoch with highest validation AUROC | Shared with lead for federated aggregation |
| `final_checkpoint.pt` | Model state dictionary after the final epoch | Training completion record |
| `history.csv` | Epoch-by-epoch loss, AUROC, PR-AUC, sensitivity, specificity, and learning rate | Learning curves and convergence analysis |
| `val_metrics.json` | Comprehensive validation set metrics from the best epoch | Validation performance benchmark |
| `test_metrics.json` | Held-out test set evaluation metrics | Final benchmark evaluation |
| `config.json` | Consolidated hyperparameters, paths, and hardware settings | Run reproducibility record |
| `run_metadata.json` | Hardware specs, PyTorch version, date, and execution duration | Compute environment record |
| `patient_manifest.json` | List of selected patient IDs, image counts, and class prevalence | Patient isolation audit |
| `path_resolution_report.json` | Number of indexed images, resolved images, and missing files | Archive coverage record |

---

## 9. Troubleshooting & Solutions

| Symptom / Error | Cause | Resolution |
|---|---|---|
| `Sample PadChest metadata file ... is explicitly rejected` | Preprocessing or training was configured with a sample CSV. | Change `metadata` path in YAML config to `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`. |
| `Zero accessible PadChest training images found on disk` | Images were not extracted to `data/raw/PadChest/images/` or path is misspelled. | Check directory with `Get-ChildItem data/raw/PadChest/images`. Ensure image files are present. |
| `CUDA out of memory (OOM)` | Mini-batch size too large for GPU VRAM. | Reduce batch size: `--batch_size 16` or `--batch_size 8`. Enable mixed precision: `--use_amp`. |
| `Missing required Python dependencies: torchvision, pillow` | Virtual environment packages not installed. | Run `pip install -r requirements.txt`. |
| `Preflight blocked: Only 1,234 training images resolved on disk` | Running in full mode without `--allow_partial`. | Pass `--allow_partial` on CLI or use `configs/padchest_partial.yaml`. |
| `val_loss is N/A / best_checkpoint.pt not updated` | Empty validation loader (0 accessible validation images). | Ensure both `0.zip` and `1.zip` are extracted so validation patients are present. |

---

## 10. Deliverables Checklist: What to Send to the Project Lead

### A. Push to GitHub
Commit and push only code, configurations, and documentation:
- [ ] Updated scripts in `scripts/`
- [ ] Updated configurations in `configs/`
- [ ] Unit tests in `tests/`
- [ ] Documentation (`docs/PADCHEST_TRAINING.md`)
- [ ] Sanitized summaries

> [!CAUTION]
> **NEVER COMMIT TO GITHUB:**
> - `0.zip`, `1.zip`, or any archive files.
> - Raw radiograph `.png` images.
> - Patient-identifiable raw metadata or un-hashed patient manifests.
> - Model checkpoint files (`*.pt`, `*.pth`).
> - Local `.venv` directories.

### B. Package as ZIP (`PadChest_Client03_Training_Results.zip`)
Create a zip archive containing your training artifacts:

```text
PadChest_Client03_Training_Results/
│
├── README.md
│
├── reports/
│   ├── dataset_audit_report.json        <-- from outputs/padchest/.../path_resolution_report.json
│   ├── preflight_report.json            <-- terminal output of preflight run
│   ├── training_summary.json            <-- from outputs/padchest/.../config.json
│   └── split_summary.json               <-- from outputs/padchest/.../patient_manifest.json
│
├── metrics/
│   ├── training_history.csv             <-- from outputs/padchest/.../history.csv
│   ├── validation_metrics.json          <-- from outputs/padchest/.../val_metrics.json
│   └── test_metrics.json                <-- from outputs/padchest/.../test_metrics.json
│
├── checkpoints/
│   └── best_checkpoint.pt               <-- from outputs/padchest/.../best_checkpoint.pt
│
├── configs/
│   └── padchest_training_config.yaml    <-- configs/padchest_partial.yaml
│
└── logs/
    └── training.log                     <-- captured terminal output
```

Share `PadChest_Client03_Training_Results.zip` via the team's shared Google Drive or OneDrive folder.

### C. Send as a Short Message / Email Summary
Provide this concise summary to the lead:
1. **Archives Used:** `0.zip` and `1.zip`.
2. **Resolved Images:** 1,234 train frontal images, 160 val frontal images, 140 test frontal images.
3. **Class Counts:** Train (166 pos, 1,068 neg), Val (15 pos, 145 neg), Test (19 pos, 121 neg).
4. **Unique Patients:** Train (1,165 pts), Val (154 pts), Test (132 pts) — Zero leakage.
5. **Training Epochs:** 10 epochs.
6. **Best Epoch & Validation AUROC:** (e.g. Epoch 7, Val AUROC: 0.8124).
7. **Test AUROC:** (e.g. 0.8015).
8. **Checkpoint Verification:** `best_checkpoint.pt` verified with strict state dict loading into `build_resnet50_binary`.
9. **Deviations / Notes:** Partial archive run (3.6% archive coverage).

---

## 11. Instructions for the Teammate's Coding Agent

> [!IMPORTANT]
> If you are an AI assistant (Antigravity, Claude Code, Cursor, Copilot) assisting the teammate, you must strictly follow these directives:

1. **Read This Document First:** Follow the exact commands and directory paths detailed here. Do not improvise arbitrary directories or invent CLI flags.
2. **Preserve the Shared Model Architecture:**
   - Do NOT modify `ResNet50Binary` in `src/models/resnet.py`.
   - All three federated clients share this exact 320-parameter state dictionary with classification head `classifier: Linear(in_features=2048, out_features=1)`. Renaming parameters breaks FedAvg parameter aggregation.
3. **Preserve the 60K Project Cohort:**
   - Do NOT delete or regenerate `data/processed/PadChest/` CSV files unless specifically instructed.
   - The cohort (`train.csv`: 48,263, `val.csv`: 6,090, `test.csv`: 5,890) was constructed from the official 160K metadata with strict patient isolation. Always verify it using `python scripts/prepare_padchest.py --verify_only`.
4. **Enforce Binary Label & View Policies:**
   - Positive (`target = 1`): Must contain `'pleural effusion'`.
   - Negative (`target = 0`): Must strictly be `['normal']`.
   - Ambiguous findings (`costophrenic angle blunting`, `pleural thickening`, `pericardial effusion`) must remain excluded. Do NOT fabricate negative labels for unmentioned findings.
   - Only frontal radiographs (`PA`, `AP`, `AP_horizontal`) are retained. Lateral and Costal views must remain excluded.
5. **Run Preflight Before Training:**
   - Always execute `python scripts/preflight_padchest.py --config configs/padchest_partial.yaml --allow_partial` before training.
   - Do NOT bypass preflight errors by editing assertions or deleting checks.
6. **No Data in Git:**
   - Ensure `0.zip`, `1.zip`, extracted `.png` files, and `best_checkpoint.pt` are NOT staged or committed to Git. Respect `.gitignore`.
7. **Report Actual Metrics Truthfully:**
   - If validation or test metrics are computed on a partial archive, label them clearly as `PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL`. Never alter filters merely to inflate image counts.
