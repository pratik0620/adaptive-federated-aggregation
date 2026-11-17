from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(r"C:\Users\acer\Documents\Codex\2026-11-17\asking-for-the-output-which-is")
REPO = ROOT / "work" / "adaptive-federated-aggregation-piyush"
CHECKPOINT_DIR = ROOT / "outputs" / "final_research_experiments" / "checkpoints"
PILOT_DIR = ROOT / "outputs" / "nih" / "nih_real_experiment"
OUTPUT_DIR = ROOT / "outputs" / "nih_evaluation_results"
THRESHOLD = 0.5

sys.path.insert(0, str(REPO))
from src.data.nih_dataset import NIHChestXRayDataset, get_nih_transforms
from src.metrics import compute_binary_metrics
from src.models.resnet import build_resnet50_binary, load_checkpoint


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validation_frame() -> pd.DataFrame:
    frame = pd.read_csv(PILOT_DIR / "selected_images.csv")
    frame = frame.loc[frame["split"].eq("validation")].copy()
    frame = frame.rename(columns={"Image Index": "image_id", "Patient ID": "patient_id"})
    # Use only the existing validation subset. Never read test rows or a test loader.
    if "local_path" not in frame.columns:
        raise KeyError("selected_images.csv has no local_path column")
    frame["resolved_path"] = frame["local_path"].map(str)
    missing = [p for p in frame["resolved_path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} validation images are missing; first={missing[0]}")
    if len(frame) != 40 or int(frame.target.sum()) != 20:
        raise RuntimeError("Unexpected validation cohort; expected 40 images with 20 positives")
    if frame.patient_id.nunique() != 40:
        raise RuntimeError("Validation cohort must contain 40 distinct patients")
    return frame


@torch.no_grad()
def evaluate(checkpoint_path: Path, loader: DataLoader) -> tuple[dict, list[float]]:
    model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
    checkpoint = load_checkpoint(checkpoint_path, model=model, device="cpu", strict=True)
    model.eval()
    y_true: list[int] = []
    y_prob: list[float] = []
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for images, targets in loader:
        logits = model(images).squeeze(-1)
        total_loss += loss_fn(logits, targets).item() * len(targets)
        y_prob.extend(torch.sigmoid(logits).cpu().tolist())
        y_true.extend(targets.int().cpu().tolist())
    metrics = compute_binary_metrics(y_true, y_prob, threshold=THRESHOLD)
    metrics.update({
        "loss": total_loss / len(y_true),
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "strict_load": True,
        "state_dict_key_count": len(checkpoint["model_state_dict"]),
        "evaluation_status": "UNOFFICIAL_REAL_NIH_40_IMAGE_PILOT",
        "research_limitation": (
            "Exploratory cross-dataset evaluation of a post-hoc two-client aggregation; "
            "not evidence of valid multi-round federated learning."
        ),
    })
    return metrics, y_prob


def plot_confusion(metrics: dict, title: str, path: Path) -> None:
    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title + " (threshold 0.5)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = validation_frame()
    dataset = NIHChestXRayDataset(frame, transform=get_nih_transforms(split="val", img_size=224))
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, drop_last=False)
    checkpoints = {
        "FedAvg": CHECKPOINT_DIR / "fedavg_oneshot_checkpoint.pt",
        "Capped FedAvg (c=0.40)": CHECKPOINT_DIR / "capped_fedavg_c40_oneshot_checkpoint.pt",
    }
    results: dict[str, dict] = {}
    probabilities: dict[str, list[float]] = {}
    for method, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics, probs = evaluate(path, loader)
        results[method] = metrics
        probabilities[method] = probs

    (OUTPUT_DIR / "fedavg_validation_metrics.json").write_text(
        json.dumps(results["FedAvg"], indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "capped_fedavg_validation_metrics.json").write_text(
        json.dumps(results["Capped FedAvg (c=0.40)"], indent=2), encoding="utf-8"
    )
    plot_confusion(results["FedAvg"], "FedAvg — NIH validation", OUTPUT_DIR / "fedavg_confusion_matrix.png")
    plot_confusion(results["Capped FedAvg (c=0.40)"], "Capped FedAvg — NIH validation", OUTPUT_DIR / "capped_fedavg_confusion_matrix.png")

    rows = []
    for method, metrics in results.items():
        cm = metrics["confusion_matrix"]
        rows.append({
            "method": method,
            "evaluated_images": metrics["total_samples"],
            "positive_labels": metrics["positive_count"],
            "negative_labels": metrics["negative_count"],
            "threshold": metrics["threshold"],
            "auroc": metrics["auroc"],
            "pr_auc": metrics["pr_auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    pred = frame[["image_id", "patient_id", "target"]].reset_index(drop=True)
    pred["fedavg_probability"] = probabilities["FedAvg"]
    pred["capped_fedavg_probability"] = probabilities["Capped FedAvg (c=0.40)"]
    pred.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    fed = results["FedAvg"]
    cap = results["Capped FedAvg (c=0.40)"]
    report = f"""# NIH Real Validation Evaluation and Checkpoint Audit

## Outcome

Both aggregated checkpoints loaded strictly into the unchanged `ResNet50Binary` architecture and were evaluated on the identical real NIH validation cohort. No training, fine-tuning, test-set access, or threshold tuning occurred.

## Cohort and preprocessing

- Real NIH ChestX-ray14 pilot validation subset: 40 images from 40 distinct patients.
- Labels: 20 Effusion-positive and 20 Effusion-negative.
- Input: 224×224 RGB.
- Normalization: ImageNet mean/std through `src.data.nih_dataset.get_nih_transforms(split=\"val\")`.
- Decision threshold: fixed at 0.5 before evaluation.
- This is a deliberately balanced 40-image pilot subset, not the official full NIH validation split and not a prevalence-representative benchmark.

## Strict checkpoint audit

| Method | Strict load | State keys | SHA-256 |
|---|---:|---:|---|
| FedAvg | Pass | {fed['state_dict_key_count']} | `{fed['checkpoint_sha256']}` |
| Capped FedAvg (c=0.40) | Pass | {cap['state_dict_key_count']} | `{cap['checkpoint_sha256']}` |

## Validation metrics

| Method | AUROC | PR-AUC | Sensitivity | Specificity | F1 | Balanced accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| FedAvg | {fed['auroc']:.6f} | {fed['pr_auc']:.6f} | {fed['sensitivity']:.6f} | {fed['specificity']:.6f} | {fed['f1']:.6f} | {fed['balanced_accuracy']:.6f} | TN={fed['confusion_matrix']['tn']}, FP={fed['confusion_matrix']['fp']}, FN={fed['confusion_matrix']['fn']}, TP={fed['confusion_matrix']['tp']} |
| Capped FedAvg | {cap['auroc']:.6f} | {cap['pr_auc']:.6f} | {cap['sensitivity']:.6f} | {cap['specificity']:.6f} | {cap['f1']:.6f} | {cap['balanced_accuracy']:.6f} | TN={cap['confusion_matrix']['tn']}, FP={cap['confusion_matrix']['fp']}, FN={cap['confusion_matrix']['fn']}, TP={cap['confusion_matrix']['tp']} |

## Provenance and limitations

The supplied aggregation metadata states that these are one-shot post-hoc averages of independently trained CheXpert and PadChest checkpoints. NIH did not contribute to the aggregation. The client checkpoints did not have verified common initialization. Therefore this is exploratory cross-dataset evaluation, not evidence of valid synchronous or multi-round federated learning.

The aggregation weights are nearly identical: FedAvg uses 50.0403% / 49.9597%, whereas the capped result uses 50% / 50%. Any numerical difference is expected to be very small. These 40-image pilot results cannot establish that Capped FedAvg is superior.
"""
    (OUTPUT_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")

    env = f"""NIH VALIDATION-ONLY CHECKPOINT EVALUATION
Date: 2026-11-17 (Asia/Calcutta)
Repository commit: 8ed909d63f521e4389cdaa6329b1783253935197
Platform: {platform.platform()}
Python: {sys.version.replace(chr(10), ' ')}
PyTorch: {torch.__version__}
CUDA available: {torch.cuda.is_available()}
Device: CPU
Validation images: 40
Positive labels: 20
Negative labels: 20
Threshold: 0.5 (fixed)

Exact command:
py -3.13 work\\evaluate_nih_aggregated.py

Pipeline interfaces used:
- src.data.nih_dataset.NIHChestXRayDataset
- src.data.nih_dataset.get_nih_transforms(split=\"val\", img_size=224)
- src.models.resnet.build_resnet50_binary(pretrained=False)
- src.models.resnet.load_checkpoint(..., strict=True)
- src.metrics.compute_binary_metrics(..., threshold=0.5)

Safeguards:
- Validation rows only; no test dataframe or test loader read.
- No optimizer, backward pass, training loop, or state-dict modification.
- Identical deterministic loader and sample order for both checkpoints.
"""
    (OUTPUT_DIR / "execution_environment.txt").write_text(env, encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in r.items() if m in ("auroc", "pr_auc", "sensitivity", "specificity", "f1", "balanced_accuracy", "confusion_matrix")} for k, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
