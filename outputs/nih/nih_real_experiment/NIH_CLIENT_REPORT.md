# NIH Client Report — Real 200-Image Pilot

## Outcome

The existing ResNet50Binary model was trained and executed successfully on 200 genuine NIH ChestX-ray14 images. This is a small balanced engineering pilot, not a full-dataset experiment or clinical result.

## Dataset and split

- Original dataset: NIH ChestX-ray14, official metadata version `Data_Entry_2017_v2020.csv`.
- Image delivery: public 224×224 mirror of the NIH images to avoid downloading the 45 GB archive set.
- Label: `Effusion` token present = 1; otherwise = 0. NIH ChestX-ray14 does not use uncertain labels.
- 200 images from 200 distinct patients; one image per patient.
- Train: 120 images (60 positive, 60 negative).
- Validation: 40 images (20 positive, 20 negative).
- Test: 40 images (20 positive, 20 negative).
- Patient overlap between splits: zero.

## Training

- Architecture: ResNet50Binary, single-logit classifier.
- Initialization: ImageNet-pretrained torchvision ResNet-50. The central initial checkpoint was unavailable, so this is not synchronized federated initialization.
- Input: 224×224 RGB with ImageNet normalization.
- Loss: BCEWithLogitsLoss; optimizer: AdamW; seed: 42; epochs: 3.
- Best validation epoch: 3.

## Validation

- AUROC: 0.7225
- PR-AUC: 0.7488296565870096
- Sensitivity: 1.0
- Specificity: 0.15
- F1: 0.7017543859649124
- Balanced accuracy: 0.575
- Threshold: 0.5

## Held-out test

- AUROC: 0.6925
- PR-AUC: 0.7421008711497842
- Sensitivity: 0.9
- Specificity: 0.05
- F1: 0.631578947368421
- Balanced accuracy: 0.47500000000000003
- Threshold fixed before test evaluation: 0.5

## Compatibility and limitations

The checkpoints use the unchanged project ResNet50Binary state-dict format and can be loaded by the central server. No aggregation was performed. The intentionally balanced 200-image subset is too small and non-representative for scientific or clinical claims; metrics have high variance and must not be compared with full-dataset benchmarks.
