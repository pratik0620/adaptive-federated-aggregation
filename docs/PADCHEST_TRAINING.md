# PadChest Training Guide (Client 03)

## 1. Project Overview & Client Role

- **Project:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data.
- **Client Node:** PadChest (Client 03).
- **Target Disease:** Pleural Effusion (Binary Classification).
- **Model Backbone:** Shared ResNet-50 (`torchvision.models.resnet50`).
- **Input Spec:** 224 × 224 RGB, ImageNet normalization.
- **Classification Head:** Single logit linear projection (`fc.weight: [1, 2048]`, `fc.bias: [1]`).
- **Loss Function:** Binary Cross-Entropy with Logits (`torch.nn.BCEWithLogitsLoss`).
- **View Policy:** Frontal radiographs only (`PA`, `AP`, `AP_horizontal`), excluding Lateral and Costal views.
- **Splits:** Patient-level 80/10/10 train/val/test splits with zero patient/study leakage.

---

## 2. Partial Dataset Specification (Archives 0 & 1)

For this partial client experiment, two ZIP archives and their manifests are utilized:
- `data/raw/PadChest/archives/0.zip`
- `data/raw/PadChest/archives/1.zip`
- `data/raw/PadChest/archive_manifests/0.zip.unzip-l.txt`
- `data/raw/PadChest/archive_manifests/1.zip.unzip-l.txt`
- Metadata: `data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`

---

## 3. Directory Structure

```text
data/
├── processed/
│   └── PadChest/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── summary.json
└── raw/
    └── PadChest/
        ├── archives/
        │   ├── 0.zip
        │   └── 1.zip
        ├── archive_manifests/
        │   ├── 0.zip.unzip-l.txt
        │   └── 1.zip.unzip-l.txt
        ├── metadata/
        │   └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz
        └── images/
            └── [extracted PNG images]
```

---

## 4. Hardware Configuration & Execution

- **CPU:** Intel Core i7-1355U (13th Gen)
- **RAM:** 16 GB
- **Device:** CPU
- **Batch Size:** 8
- **Workers:** 2 (or 0 for Windows interactive stability)
- **Image Size:** 224 × 224
- **Checkpoints:** Saved after every epoch to `outputs/padchest/<run_name>/`
