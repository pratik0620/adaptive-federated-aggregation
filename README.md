# Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data

This repository prepares three public chest X-ray metadata sources as separate, leakage-safe federated clients. It does not train a model, implement FedAvg/FedProx, or redistribute medical images.

## Objective

The current clean binary experiment targets `Pleural Effusion`. A target can be changed in notebook configuration without rewriting the pipeline. Positive means the verified target label is present; negative means an explicitly clean negative label is present. Uncertain, missing, mixed, and otherwise ambiguous cases are excluded rather than silently treated as negative.

## Datasets and access

- NIH ChestX-ray14: obtain the images and metadata from the official NIH source.
- CheXpert: obtain the dataset and access terms from Stanford ML Group.
- PadChest: obtain the metadata/images from the official BCN/AQUA repository and verify its current terms.

Access, download, and usage restrictions remain the user's responsibility. This repository contains preprocessing code and metadata structure only. Never commit downloaded images, archives, or restricted source files.

## Repository structure

```text
src/
  config.py
  dataset_utils.py
  preprocessing_utils.py
notebooks/
  01_NIH_preprocessing.ipynb
  02_CheXpert_preprocessing.ipynb
  03_PadChest_preprocessing.ipynb
  04_validate_federated_dataset.ipynb
```

Each preprocessing notebook writes metadata CSVs to a separate directory under `data/processed/`:
`data/processed/NIH`, `data/processed/CheXpert`, and `data/processed/PadChest`. The `image_path` column points to the original mounted image; files are not copied.

## Google Colab

1. Open a notebook in Colab and mount Google Drive if the data is stored there.
2. Clone this repository, or upload the repository files, and change the configuration paths in the first code cell.
3. Install dependencies:

```python
%pip install -r requirements.txt
```

4. Run the notebook from top to bottom. Set `PROJECT_ROOT` to the cloned repository and use `pathlib.Path` for dataset paths.
5. Run the validation notebook after all three client directories exist.

The notebooks do not download restricted datasets. They raise an actionable error when a configured metadata file or image directory is missing.

## Leakage and reproducibility

Splits are made at patient level with seed `42`; no patient can appear in more than one split. Study-level overlap is checked where study IDs exist. Counts are reported before filtering and sampling. Sampling is optional through `MAX_POSITIVE_SAMPLES` and `MAX_NEGATIVE_SAMPLES`, both defaulting to `None`.

## Standardized client schema

All clients contain `image_id`, `patient_id`, `study_id`, `source`, `target`, `age`, `sex`, `view_position`, `projection`, `image_width`, `image_height`, `pixel_spacing_x`, `pixel_spacing_y`, `scanner_manufacturer`, and `image_path`. Dataset-specific metadata may be retained as additional columns. Missing source fields remain `NaN`.

## Dataset-specific verification

Before a real run, verify the exact local column names and label spelling in each downloaded release. PadChest label vocabularies and serialized list formatting require explicit inspection; the notebook prints matching vocabulary examples and stops if the requested target label is not found. NIH official split information is not assumed because releases may differ; patient-level deterministic splits are used unless an official split mapping is supplied.
