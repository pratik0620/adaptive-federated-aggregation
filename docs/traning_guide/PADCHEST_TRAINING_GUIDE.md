# PadChest Training Guide (Client 03)

## 1. Project Overview & Your Role

- **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data.
- **Task:** Train a binary image classifier to predict **Pleural Effusion** from chest X-rays. Each teammate trains the same model architecture on a different dataset. The trained model checkpoint and evaluation results will be shared with the project lead for federated aggregation experiments.
- **Your Responsibility:** Obtain the PadChest dataset, place the images in the expected folder, run the training commands below, and return the generated output artifacts.

---

## 2. Dataset Download Link

Download sample images from Kaggle:

👉 **[Kaggle: PadChest Sample Dataset](https://www.kaggle.com/datasets/raddar/padchest-chest-xrays-sample)**

> [!WARNING]
> **SAMPLE DATASET NOTICE:**  
> The Kaggle link above provides a **sample dataset** (~20,000 images), **NOT the complete PadChest collection** (which contains >160,000 images). This sample is intended for code verification and development. It does not contain enough images for full planned experiments. The full PadChest dataset must be requested separately from the official Medical Image Analysis Group (BIMCV / University of Alicante) following their official data use agreement.

---

## 3. How to Download and Extract

1. Open the [Kaggle PadChest sample link](https://www.kaggle.com/datasets/raddar/padchest-chest-xrays-sample) and sign in.
2. Download the dataset archive (or use Kaggle CLI: `kaggle datasets download -d raddar/padchest-chest-xrays-sample`).
3. Extract the downloaded zip file on your computer.
4. Place the extracted image files or numbered subfolders inside `data/raw/PadChest/images/`.
5. **Keep the existing processed CSV files** in `data/processed/PadChest/` unchanged. Do not regenerate or modify them.
6. Run the preflight dry-run command (Section 5) to verify that your images are detected before starting training.

---

## 4. Expected Directory Structure

Place your extracted files so the directory looks like this:

```text
Project Repo/
├── data/
│   ├── processed/
│   │   └── PadChest/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   └── raw/
│       └── PadChest/
│           ├── images/
│           │   ├── 0/
│           │   │   └── <ImageID>.png
│           │   ├── 1/
│           │   │   └── <ImageID>.png
│           │   └── ...
│           └── metadata/
└── src/
```

> [!TIP]
> Images can be stored in numbered release folders (for example, `images/0/` through `images/49/`) or placed directly inside `data/raw/PadChest/images/`. The path resolver automatically searches subfolders up to 6 levels deep. The pipeline automatically filters for frontal projections (`PA`, `AP`, `AP_horizontal`) and safely excludes Lateral and Costal views.

---

## 5. Training Commands

Open a terminal at the root of the repository and activate your Python environment:

```bash
# Activate your virtual environment (Windows PowerShell):
.\.venv\Scripts\Activate.ps1
# Or on Linux / macOS:
# source .venv/bin/activate

# Ensure required dependencies are installed:
pip install -r requirements.txt
pip install torchvision pillow
```

### Command 1: Dataset Preprocessing (Generate Split CSVs)
To generate or regenerate the standard 17-column split CSVs (`train.csv`, `val.csv`, `test.csv`) and `summary.json`:

```bash
# Standard preprocessing (auto-detects full 160K or Kaggle sample CSV):
python src/preprocess_padchest.py

# Or dry-run without writing files:
python src/preprocess_padchest.py --dry_run
```

> [!NOTE]
> **24-Image Sample Limitation:**  
> When running against the 24-image Kaggle sample metadata, the pipeline filters for frontal views and valid binary labels, yielding **8 train, 1 validation, and 1 test** records. Frontal filtering retains **6 train images (1 positive, 5 negative), 0 validation images (the 1 record is Lateral and excluded), and 1 test image (negative)**. This sample is strictly for mechanical smoke testing; evaluation metrics (AUROC) are undefined or non-informative.

### Command 2: Preflight Verification (No Training)
Checks that your dataset paths are resolved correctly, audits image availability, and applies frontal filtering without training a model:

```bash
python src/train_padchest.py --pilot --dry_run
```

### Command 3: Pipeline Smoke Test (Existing 24 Sample Images)
Runs a 2-epoch CPU smoke test validating the entire training, dataloading, and export pipeline:

```bash
python src/train_padchest.py --pilot --sample_from_accessible --device cpu --metric_to_optimize val_loss
```

> [!IMPORTANT]
> `--metric_to_optimize val_loss` is used here purely as a smoke-test plumbing flag so that `best_checkpoint.pt` is generated despite the validation loader having 0 frontal samples (`val_loss = 0.0`). It is **not** a valid measure of model generalization.

### Command 4: Pilot Training Run (Larger Downloaded Dataset)
Trains a quick model on 100 patients for 2 epochs once more frontal images are downloaded:

```bash
python src/train_padchest.py --pilot --sample_from_accessible --device cpu
```
*(Replace `cpu` with `cuda` if you have an NVIDIA GPU).*

### Command 5: Full Client Training Run (Official Experiments)
Trains on a configurable number of unique training patients (e.g. 500 patients for 5 epochs) with full validation AUROC model selection:

```bash
python src/train_padchest.py --patient_limit 500 --epochs 5 --batch_size 32 --device cpu
```

---

## 6. Outputs to Share with Project Lead

After training finishes, all outputs are saved in `outputs/padchest/<run_name>/`. Please zip and share this folder:

- `best_checkpoint.pt` — Best model weights (compatible with federated aggregation).
- `final_checkpoint.pt` — Final epoch model checkpoint.
- `history.csv` — Training loss, validation loss, and AUROC for each epoch.
- `val_metrics.json` & `test_metrics.json` — Evaluated performance metrics.
- `config.json` & `run_metadata.json` — Training settings and hardware metadata.
- `patient_manifest.json` — Selected patient IDs and class distribution.
- `path_resolution_report.json` — Image availability and missing file audit.

> [!CAUTION]
> **Do NOT commit or upload images, dataset archives, or model checkpoints to GitHub.** Share the `outputs/padchest/<run_name>/` folder via Google Drive, OneDrive, or the team's shared storage.
