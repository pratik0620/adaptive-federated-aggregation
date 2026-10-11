# NIH ChestX-ray14 Training Guide (Client 01)

## 1. Project Overview & Your Role

- **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data.
- **Task:** Train a binary image classifier to predict **Pleural Effusion** from chest X-rays. Each teammate trains the same model architecture on a different dataset. The trained model checkpoint and evaluation results will be shared with the project lead for federated aggregation experiments.
- **Your Responsibility:** Download the NIH ChestX-ray14 dataset, place the images in the expected folder, run the training commands below, and return the generated output artifacts.

---

## 2. Dataset Download Link

Download the images from Kaggle:

👉 **[Kaggle: NIH Chest X-rays (images_001)](https://www.kaggle.com/datasets/nih-chest-xrays/data/data?select=images_001)**

> [!NOTE]
> `images_001` is the first of 12 image archives in the full NIH dataset. It is sufficient for local pilots and initial testing (~4,999 images). If you are running larger experiments, additional archives (`images_002` through `images_012`) can be downloaded from the same Kaggle page.

---

## 3. How to Download and Extract

1. Open the [Kaggle dataset link](https://www.kaggle.com/datasets/nih-chest-xrays/data/data?select=images_001) and sign in.
2. Download the archive `images_001.zip` (or use the Kaggle CLI if preferred: `kaggle datasets download -d nih-chest-xrays/data`).
3. Extract the downloaded zip file on your computer.
4. Place the extracted image folder inside `data/raw/NIH/images/` in this repository.
5. **Keep the existing processed CSV files** in `data/processed/NIH/` unchanged. Do not edit or recreate them.
6. Run the preflight dry-run command (Section 5) to verify that your images are detected before starting training.

---

## 4. Expected Directory Structure

Place your extracted files so the directory looks like this:

```text
Project Repo/
├── data/
│   ├── processed/
│   │   └── NIH/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   └── raw/
│       └── NIH/
│           ├── images/
│           │   └── images_001/
│           │       └── images/
│           │           ├── 00000001_000.png
│           │           └── ...
│           └── metadata/
└── src/
```

> [!TIP]
> Each downloaded archive may have its own nested `images/` folder (for example, `images_001/images/*.png`). The built-in path resolver automatically searches subfolders up to 4 levels deep, so any additional archives (`images_002`, etc.) placed inside `data/raw/NIH/images/` will be discovered automatically.

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
Checks that your dataset paths are resolved correctly and audits image availability without training a model:

```bash
python src/train_nih.py --pilot --dry_run
```

### Command 2: Small Pilot Run (**Actually Trains Model**)
Trains a quick model on 100 patients for 2 epochs. If you only downloaded `images_001`, add `--sample_from_accessible` so it selects from locally available images:

```bash
python src/train_nih.py --pilot --sample_from_accessible --device cpu
```
*(Replace `cpu` with `cuda` if you have an NVIDIA GPU).*

### Command 3: Larger Training Run (**Actually Trains Model**)
Trains on a configurable number of unique training patients (e.g. 500 patients for 5 epochs):

```bash
python src/train_nih.py --patient_limit 500 --epochs 5 --batch_size 32 --device cpu
```

---

## 6. Outputs to Share with Project Lead

After training finishes, all outputs are saved in `outputs/nih/<run_name>/`. Please zip and share this folder:

- `best_checkpoint.pt` — Best model weights based on validation AUROC.
- `final_checkpoint.pt` — Final epoch model checkpoint.
- `history.csv` — Training loss, validation loss, and AUROC for each epoch.
- `val_metrics.json` & `test_metrics.json` — Evaluated performance metrics.
- `config.json` & `run_metadata.json` — Training settings and hardware metadata.
- `patient_manifest.json` — Selected patient IDs and class distribution.
- `path_resolution_report.json` — Image availability and missing file audit.

> [!CAUTION]
> **Do NOT commit or upload images, dataset archives, or model checkpoints to GitHub.** Share the `outputs/nih/<run_name>/` folder via Google Drive, OneDrive, or the team's shared storage.
