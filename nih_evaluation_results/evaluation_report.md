# NIH Real Validation Evaluation and Checkpoint Audit

## Outcome

Both aggregated checkpoints loaded strictly into the unchanged `ResNet50Binary` architecture and were evaluated on the identical real NIH validation cohort. No training, fine-tuning, test-set access, or threshold tuning occurred.

## Cohort and preprocessing

- Real NIH ChestX-ray14 pilot validation subset: 40 images from 40 distinct patients.
- Labels: 20 Effusion-positive and 20 Effusion-negative.
- Input: 224×224 RGB.
- Normalization: ImageNet mean/std through `src.data.nih_dataset.get_nih_transforms(split="val")`.
- Decision threshold: fixed at 0.5 before evaluation.
- This is a deliberately balanced 40-image pilot subset, not the official full NIH validation split and not a prevalence-representative benchmark.

## Strict checkpoint audit

| Method | Strict load | State keys | SHA-256 |
|---|---:|---:|---|
| FedAvg | Pass | 320 | `3c1216ddea00340b2ce87ac0edd1f4136c5b99f80a84a70cce55727c06d1a7fe` |
| Capped FedAvg (c=0.40) | Pass | 320 | `aa9fcfc26acabd41d9695ec997e264ac5799bd2b3278736c549a1d8549c4359a` |

## Validation metrics

| Method | AUROC | PR-AUC | Sensitivity | Specificity | F1 | Balanced accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| FedAvg | 0.710000 | 0.725064 | 0.000000 | 1.000000 | 0.000000 | 0.500000 | TN=20, FP=0, FN=20, TP=0 |
| Capped FedAvg | 0.707500 | 0.722591 | 0.000000 | 1.000000 | 0.000000 | 0.500000 | TN=20, FP=0, FN=20, TP=0 |

## Provenance and limitations

The supplied aggregation metadata states that these are one-shot post-hoc averages of independently trained CheXpert and PadChest checkpoints. NIH did not contribute to the aggregation. The client checkpoints did not have verified common initialization. Therefore this is exploratory cross-dataset evaluation, not evidence of valid synchronous or multi-round federated learning.

The aggregation weights are nearly identical: FedAvg uses 50.0403% / 49.9597%, whereas the capped result uses 50% / 50%. Any numerical difference is expected to be very small. These 40-image pilot results cannot establish that Capped FedAvg is superior.
