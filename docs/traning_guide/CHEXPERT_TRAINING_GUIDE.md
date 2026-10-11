# CheXpert Training Guide (Client 02)

## 1. Project Overview & Your Role

- **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data.
- **Task:** Train a binary image classifier to predict **Pleural Effusion** from chest X-rays. Each teammate trains the same model architecture on a different dataset. The trained model checkpoint and evaluation results will be shared with the project lead for federated aggregation experiments.
- **Your Responsibility:** Download the CheXpert dataset, place the images in the expected folder, run the training commands below, and return the generated output artifacts.

---

## 2. Dataset Download Link

Download the images from Kaggle:

👉 **[Kaggle: CheXpert Dataset (CheXpert Small)](https://www.kaggle.com/datasets/ashery/chexpert?select=train)**

> [!IMPORTANT]
> Download the **CheXpert Small** version (`CheXpert-v1.0-small.zip`, ~11 GB) including the training images and validation files. **Do NOT download the 439 GB full-resolution dataset.**

---

## 3. How to Download and Extract

1. Open the [Kaggle CheXpert link](https://www.kaggle.com/datasets/ashery/chexpert?select=train) and sign in.
2. Download the `CheXpert-v1.0-small` archive (or use Kaggle CLI: `kaggle datasets download -d ashery/chexpert`).
3. Extract the downloaded zip file on your computer.
4. Place the extracted patient folders inside `data/raw/CheXpert/images/` (or keep the `CheXpert-v1.0-small/` folder inside `data/raw/CheXpert/`).
5. **Keep the existing processed CSV files** in `data/processed/CheXpert/` unchanged. Do not regenerate or modify them.
6. Run the preflight dry-run command (Section 5) to verify that your images are detected before starting training.

---

## 4. Expected Directory Structure

Place your extracted files so the directory looks like this:

```text
Project Repo/
├── data/
│   ├── processed/
│   │   └── CheXpert/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   └── raw/
│       └── CheXpert/
│           ├── images/
│           │   └── patient00001/
│           │       └── study1/
│           │           ├── view1_frontal.jpg
│           │           └── view2_lateral.jpg
│           └── metadata/
└── src/
```

> [!TIP]
> The path resolver also automatically recognizes `data/raw/CheXpert/CheXpert-v1.0-small/train/patientXXXXX/...` and `valid/patientXXXXX/...`. The pipeline automatically filters for frontal radiographs (AP/PA) and safely ignores macOS metadata files (`._*`).

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

### Command 1: Preflight Verification (No Training)
Checks that your dataset paths are resolved correctly, audits image availability, and applies frontal filtering without training a model:

```bash
python src/train_chexpert.py --pilot --dry_run
```

### Command 2: Small Pilot Run (**Actually Trains Model**)
Trains a quick model on 100 patients for 2 epochs. If you only have a partial subset of patients downloaded, add `--sample_from_accessible`:

```bash
python src/train_chexpert.py --pilot --sample_from_accessible --device cpu
```
*(Replace `cpu` with `cuda` if you have an NVIDIA GPU).*

### Command 3: Larger Training Run (**Actually Trains Model**)
Trains on a configurable number of unique training patients (e.g. 500 patients for 5 epochs):

```bash
python src/train_chexpert.py --patient_limit 500 --epochs 5 --batch_size 32 --device cpu
```

---

## 6. Outputs to Share with Project Lead

After training finishes, all outputs are saved in `outputs/chexpert/<run_name>/`. Please zip and share this folder:

- `best_checkpoint.pt` — Best model weights based on validation AUROC.
- `final_checkpoint.pt` — Final epoch model checkpoint.
- `history.csv` — Training loss, validation loss, and AUROC for each epoch.
- `val_metrics.json` & `test_metrics.json` — Evaluated performance metrics.
- `config.json` & `run_metadata.json` — Training settings and hardware metadata.
- `patient_manifest.json` — Selected patient IDs and class distribution.
- `path_resolution_report.json` — Image availability and missing file audit.

> [!CAUTION]
> **Do NOT commit or upload images, dataset archives, or model checkpoints to GitHub.** Share the `outputs/chexpert/<run_name>/` folder via Google Drive, OneDrive, or the team's shared storage.
