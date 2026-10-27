"""Validation-Only Evaluation Diagnostic for CheXpert Pilot Checkpoint.

Pure read-only execution:
- Loads best_checkpoint.pt without downloading weights (pretrained=False).
- Audits patient isolation against patient_manifest.json (halts if leakage detected).
- Resolves paths using verified search roots on validation data only.
- Uses exact evaluation batch size (16), resolution (224), and ImageNet normalization.
- Validates prediction count (10,024), finiteness, and binary targets before metric evaluation.
- Validates saved confusion matrix integrity (explicitly rejecting booleans) before exact comparison.
- Verifies numerical match against saved val_metrics.json (HALTS BEFORE SWEEP on mismatch).
- On parity success only, computes class-conditional distributions and threshold sweep.
- NEVER reads or evaluates test.csv.
- Does NOT alter Git state, edit source files, or download weights.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 1. Ensure repository root is on sys.path even when running from Temp
REPO_ROOT = Path(r"D:\DS CP\adaptive-federated-aggregation").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.chexpert_dataset import (
    CheXpertDataset,
    CheXpertPathResolver,
    filter_chexpert_frontal_views,
    get_chexpert_transforms,
)
from src.metrics import compute_binary_metrics
from src.models.resnet import build_resnet50_binary, load_checkpoint


def run_diagnostic() -> None:
    start_time = time.time()
    pilot_dir = REPO_ROOT / "outputs" / "chexpert" / "chexpert_pilot_resnet50_20261015_231055"
    ckpt_path = pilot_dir / "best_checkpoint.pt"
    manifest_path = pilot_dir / "patient_manifest.json"
    saved_val_metrics_path = pilot_dir / "val_metrics.json"
    val_csv_path = REPO_ROOT / "data" / "processed" / "CheXpert" / "val.csv"

    # Pre-flight existence checks
    for p, name in [
        (ckpt_path, "Checkpoint best_checkpoint.pt"),
        (manifest_path, "Patient Manifest"),
        (saved_val_metrics_path, "Saved val_metrics.json"),
        (val_csv_path, "Processed val.csv"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found at: {p}")

    print("=" * 80)
    print("  CHEXPERT CLIENT 02: VALIDATION-ONLY PILOT DIAGNOSTIC AUDIT")
    print(f"  Repository: {REPO_ROOT}")
    print(f"  Checkpoint: {ckpt_path.name}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECK 1: Strict Patient Isolation Verification
    # -------------------------------------------------------------------------
    print("\n[1/6] Verifying patient isolation against patient_manifest.json...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_patient_ids = set(manifest.get("selected_patient_ids", []))
    if len(train_patient_ids) != 100:
        raise ValueError(
            f"Validation constraint failed: Expected exactly 100 unique selected "
            f"training patient IDs, found {len(train_patient_ids)}"
        )

    val_df_raw = pd.read_csv(val_csv_path, low_memory=False)
    val_df, val_frontal_report = filter_chexpert_frontal_views(val_df_raw, split_name="val")
    val_patient_ids = set(val_df["patient_id"].unique())

    overlap = train_patient_ids.intersection(val_patient_ids)
    print(f"  Pilot training patients:   {len(train_patient_ids):,}")
    print(f"  Validation patients:       {len(val_patient_ids):,}")
    print(f"  Overlapping patient count: {len(overlap)}")

    if len(overlap) > 0:
        sample_leakage = sorted(list(overlap))[:5]
        raise RuntimeError(
            f"ABORTING EXECUTION: Strict patient isolation FAILED! "
            f"{len(overlap)} overlapping patient IDs detected between train and val: {sample_leakage}"
        )
    print("  --> Patient isolation check: PASSED (Zero leakage).")

    # -------------------------------------------------------------------------
    # CHECK 2: Path Resolution using Verified Training Search Roots
    # -------------------------------------------------------------------------
    print("\n[2/6] Resolving validation image paths using verified search roots...")
    search_roots = [
        REPO_ROOT / "data" / "raw" / "CheXpert" / "images",
        REPO_ROOT / "data" / "raw" / "CheXpert" / "CheXpert-v1.0-small",
        REPO_ROOT / "data" / "raw" / "CheXpert" / "CheXpert-v1.0",
        REPO_ROOT / "data" / "raw" / "CheXpert",
        REPO_ROOT / "data" / "raw",
    ]
    resolver = CheXpertPathResolver(search_roots=search_roots)
    num_indexed = resolver.build_index()
    print(f"  Total valid radiographs indexed on disk: {num_indexed:,}")

    if num_indexed == 0:
        raise RuntimeError("Path resolution failed: Zero valid radiographs indexed on disk!")

    resolved_val_df, val_path_report = resolver.resolve_dataframe(
        val_df, split_name="val", strict=False, filter_missing=True
    )
    print(
        f"  Validation images resolved: {len(resolved_val_df):,} / {len(val_df):,} "
        f"(Coverage: {val_path_report.get('accessible_fraction', 0.0):.1%})"
    )

    is_complete = bool(val_path_report.get("is_complete", False))
    if len(resolved_val_df) != 10024 or not is_complete:
        raise RuntimeError(
            f"Validation constraint failed: Expected exactly 10,024 complete resolved validation images, "
            f"found {len(resolved_val_df):,} (split complete: {is_complete})"
        )

    # -------------------------------------------------------------------------
    # CHECK 3: Model Instantiation (pretrained=False) and Strict Checkpoint Load
    # -------------------------------------------------------------------------
    print("\n[3/6] Instantiating ResNet-50 (pretrained=False) & loading checkpoint strictly...")
    device = torch.device("cpu")
    model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device=device)

    # load_checkpoint executes raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ckpt_data = load_checkpoint(ckpt_path, model=model, device=device, strict=True)
    model.eval()

    # Safely extract optional metadata tolerant of missing, null, or non-dict config
    epoch_val = None
    best_metric_val = None
    metric_name = "val_metric"

    if isinstance(ckpt_data, dict):
        epoch_val = ckpt_data.get("epoch")
        best_metric_val = ckpt_data.get("best_val_metric")

        if ckpt_data.get("metric_name"):
            metric_name = str(ckpt_data["metric_name"])
        else:
            cfg = ckpt_data.get("config")
            if isinstance(cfg, dict) and cfg.get("metric_to_optimize"):
                metric_name = str(cfg["metric_to_optimize"])

    epoch_str = str(epoch_val) if epoch_val is not None else "N/A"
    if isinstance(best_metric_val, (int, float)) and np.isfinite(best_metric_val):
        metric_str = f"{best_metric_val:.6f}"
    elif best_metric_val is not None:
        metric_str = str(best_metric_val)
    else:
        metric_str = "N/A"

    print("  --> Model instantiated with weights=None (zero weight downloads).")
    print("  --> Strict state_dict load: COMPLETED (strict=True succeeded).")
    if epoch_val is not None or best_metric_val is not None:
        print(f"  --> Checkpoint metadata: Epoch {epoch_str}, Best {metric_name}: {metric_str}")
    else:
        print("  --> Checkpoint metadata: Optional epoch/metric fields not present in checkpoint dictionary.")

    # -------------------------------------------------------------------------
    # CHECK 4: Native DataLoader & Inference Loop (batch_size=16)
    # -------------------------------------------------------------------------
    print("\n[4/6] Executing validation inference (batch_size=16, device=CPU)...")
    val_transform = get_chexpert_transforms(split="val", img_size=224)
    val_dataset = CheXpertDataset(resolved_val_df, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)

    all_logits: list[float] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader, start=1):
            images = images.to(device)
            logits = model(images).squeeze(-1)
            all_logits.extend(logits.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().astype(int).tolist())

            if batch_idx % 125 == 0 or batch_idx == len(val_loader):
                processed = min(batch_idx * 16, len(val_dataset))
                print(f"  Processed {processed:,} / {len(val_dataset):,} images ({processed / len(val_dataset):.1%})...")

    logits_arr = np.asarray(all_logits, dtype=np.float64)
    targets_arr = np.asarray(all_targets, dtype=np.int32)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))

    # Pre-metric evaluation sanity verification
    num_logits = len(logits_arr)
    num_targets = len(targets_arr)
    if num_logits != 10024 or num_targets != 10024:
        raise RuntimeError(
            f"Inference output count mismatch: Expected exactly 10,024 predictions and targets, "
            f"but produced {num_logits:,} logits and {num_targets:,} targets."
        )

    if not np.all(np.isfinite(logits_arr)):
        non_finite_cnt = int(np.sum(~np.isfinite(logits_arr)))
        raise RuntimeError(f"Inference sanity failure: Found {non_finite_cnt} non-finite (NaN/Inf) logit values.")

    if not np.all(np.isfinite(probs_arr)):
        non_finite_cnt = int(np.sum(~np.isfinite(probs_arr)))
        raise RuntimeError(f"Inference sanity failure: Found {non_finite_cnt} non-finite (NaN/Inf) probability values.")

    if np.any(probs_arr < 0.0) or np.any(probs_arr > 1.0):
        raise RuntimeError("Inference sanity failure: Probabilities out of valid range [0, 1].")

    unique_targets = set(np.unique(targets_arr).tolist())
    if not unique_targets.issubset({0, 1}):
        raise RuntimeError(
            f"Label integrity failure: Ground truth targets must be strictly binary {{0, 1}}, "
            f"found unique labels: {unique_targets}"
        )

    if 0 not in unique_targets or 1 not in unique_targets:
        raise RuntimeError(f"Label integrity failure: Both classes {{0, 1}} must be present, found: {unique_targets}")

    # -------------------------------------------------------------------------
    # CHECK 5: Mandatory Saved-Metric Parity Gate vs val_metrics.json
    # -------------------------------------------------------------------------
    print("\n[5/6] Verifying recomputed metrics against saved val_metrics.json...")
    base_metrics = compute_binary_metrics(targets_arr, probs_arr, threshold=0.50)

    with open(saved_val_metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    # 1. Verify presence and finiteness of required continuous comparison metrics
    required_keys = ["auroc", "pr_auc", "sensitivity", "specificity"]
    for k in required_keys:
        if k not in saved_metrics or saved_metrics[k] is None or not np.isfinite(saved_metrics[k]):
            raise RuntimeError(f"Parity check failed: Saved metric '{k}' is missing or non-finite in {saved_val_metrics_path.name}")
        if k not in base_metrics or base_metrics[k] is None or not np.isfinite(base_metrics[k]):
            raise RuntimeError(f"Parity check failed: Recomputed metric '{k}' is missing or non-finite")

    # 2. Validate saved confusion-matrix structural integrity, explicitly rejecting Booleans
    saved_cm = saved_metrics.get("confusion_matrix")
    if not isinstance(saved_cm, dict):
        raise RuntimeError("Parity check failed: Saved val_metrics.json 'confusion_matrix' is missing or not a dictionary.")

    for cm_key in ["tn", "fp", "fn", "tp"]:
        if cm_key not in saved_cm:
            raise RuntimeError(f"Parity check failed: Saved confusion matrix key '{cm_key}' missing from val_metrics.json.")
        val = saved_cm[cm_key]
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < 0:
            raise RuntimeError(
                f"Parity check failed: Saved confusion matrix key '{cm_key}' must be a non-negative integer "
                f"(Booleans strictly rejected), found {val!r} of type {type(val).__name__}."
            )

    for cnt_key in ["total_samples", "positive_count", "negative_count"]:
        if cnt_key not in saved_metrics:
            raise RuntimeError(f"Parity check failed: Key '{cnt_key}' missing from val_metrics.json.")
        val = saved_metrics[cnt_key]
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)) or int(val) < 0:
            raise RuntimeError(
                f"Parity check failed: Saved metric '{cnt_key}' must be a non-negative integer "
                f"(Booleans strictly rejected), found {val!r} of type {type(val).__name__}."
            )

    saved_tn = int(saved_cm["tn"])
    saved_fp = int(saved_cm["fp"])
    saved_fn = int(saved_cm["fn"])
    saved_tp = int(saved_cm["tp"])

    saved_total = int(saved_metrics["total_samples"])
    saved_pos = int(saved_metrics["positive_count"])
    saved_neg = int(saved_metrics["negative_count"])

    if (saved_tn + saved_fp + saved_fn + saved_tp) != 10024 or saved_total != 10024:
        raise RuntimeError(
            f"Parity check failed: Saved confusion matrix sum ({saved_tn + saved_fp + saved_fn + saved_tp}) "
            f"or total_samples ({saved_total}) does not equal 10,024."
        )

    if (saved_tn + saved_fp) != saved_neg or (saved_fn + saved_tp) != saved_pos:
        raise RuntimeError(
            f"Parity check failed: Saved confusion matrix does not partition class totals "
            f"(TN+FP={saved_tn + saved_fp} vs Neg={saved_neg}; FN+TP={saved_fn + saved_tp} vs Pos={saved_pos})."
        )

    # 3. Numerical comparisons with tolerance 1e-4
    tolerance = 1e-4
    saved_auroc = float(saved_metrics["auroc"])
    saved_prauc = float(saved_metrics["pr_auc"])
    saved_sens = float(saved_metrics["sensitivity"])
    saved_spec = float(saved_metrics["specificity"])

    recomp_auroc = float(base_metrics["auroc"])
    recomp_prauc = float(base_metrics["pr_auc"])
    recomp_sens = float(base_metrics["sensitivity"])
    recomp_spec = float(base_metrics["specificity"])

    auroc_diff = abs(recomp_auroc - saved_auroc)
    prauc_diff = abs(recomp_prauc - saved_prauc)
    sens_diff = abs(recomp_sens - saved_sens)
    spec_diff = abs(recomp_spec - saved_spec)

    auroc_match = auroc_diff <= tolerance
    prauc_match = prauc_diff <= tolerance
    sens_match = sens_diff <= tolerance
    spec_match = spec_diff <= tolerance

    recomp_cm = base_metrics.get("confusion_matrix", {})
    cm_match = (
        int(recomp_cm.get("tn", -1)) == saved_tn
        and int(recomp_cm.get("fp", -1)) == saved_fp
        and int(recomp_cm.get("fn", -1)) == saved_fn
        and int(recomp_cm.get("tp", -1)) == saved_tp
    )

    print(f"  AUROC:       Saved={saved_auroc:.6f} | Recomputed={recomp_auroc:.6f} (diff={auroc_diff:.2e}) -> {'MATCH' if auroc_match else 'FAIL'}")
    print(f"  PR-AUC:      Saved={saved_prauc:.6f} | Recomputed={recomp_prauc:.6f} (diff={prauc_diff:.2e}) -> {'MATCH' if prauc_match else 'FAIL'}")
    print(f"  Sensitivity: Saved={saved_sens:.6f} | Recomputed={recomp_sens:.6f} (diff={sens_diff:.2e}) -> {'MATCH' if sens_match else 'FAIL'}")
    print(f"  Specificity: Saved={saved_spec:.6f} | Recomputed={recomp_spec:.6f} (diff={spec_diff:.2e}) -> {'MATCH' if spec_match else 'FAIL'}")
    print(f"  Conf Matrix: Saved=TN={saved_tn}, FP={saved_fp}, FN={saved_fn}, TP={saved_tp}")
    print(f"               Recomp=TN={recomp_cm.get('tn')}, FP={recomp_cm.get('fp')}, FN={recomp_cm.get('fn')}, TP={recomp_cm.get('tp')} -> {'MATCH' if cm_match else 'FAIL'}")

    # MANDATORY GATE: Halt execution immediately before score breakdown or threshold sweep if parity fails
    if not (auroc_match and prauc_match and sens_match and spec_match and cm_match):
        error_msg = (
            "PARITY GATE FAILURE: Recomputed validation metrics do not match saved pilot metrics!\n"
            "Halting execution before score distribution or threshold analysis.\n"
            "Note: Small numerical differences across CPU architectures/PyTorch versions near the 0.50 decision\n"
            "boundary can shift predicted logits and change discrete classification counts (TN, FP, FN, TP).\n"
            "Further investigation into image decoding or floating-point variations is required before proceeding."
        )
        print(f"\n[ERROR] {error_msg}")
        raise RuntimeError(error_msg)

    print("  --> Metric equivalence verification: PASSED (Exact match within tolerance).")

    # -------------------------------------------------------------------------
    # STEP 6: Detailed Score Distribution & Exploratory Threshold Sweep
    # -------------------------------------------------------------------------
    print("\n[6/6] Generating Class-Conditional Distribution & Validation-Only Sweep Report...")
    pos_mask = (targets_arr == 1)
    neg_mask = (targets_arr == 0)

    print("\n" + "=" * 80)
    print("  SCORE DISTRIBUTION & SEPARATION SUMMARY (VALIDATION ONLY)")
    print("=" * 80)
    print(f"Cohort: Total={len(targets_arr):,} | Positive={np.sum(pos_mask):,} ({np.mean(pos_mask):.2%}) | Negative={np.sum(neg_mask):,} ({np.mean(neg_mask):.2%})")

    print("\n[Logits Distribution]")
    print(f"  Overall:  Min={np.min(logits_arr):8.4f} | 25%={np.percentile(logits_arr, 25):8.4f} | Median={np.median(logits_arr):8.4f} | Mean={np.mean(logits_arr):8.4f} | 75%={np.percentile(logits_arr, 75):8.4f} | Max={np.max(logits_arr):8.4f} | Std={np.std(logits_arr):8.4f}")
    print(f"  Class 0:  Min={np.min(logits_arr[neg_mask]):8.4f} | 25%={np.percentile(logits_arr[neg_mask], 25):8.4f} | Median={np.median(logits_arr[neg_mask]):8.4f} | Mean={np.mean(logits_arr[neg_mask]):8.4f} | 75%={np.percentile(logits_arr[neg_mask], 75):8.4f} | Max={np.max(logits_arr[neg_mask]):8.4f}")
    print(f"  Class 1:  Min={np.min(logits_arr[pos_mask]):8.4f} | 25%={np.percentile(logits_arr[pos_mask], 25):8.4f} | Median={np.median(logits_arr[pos_mask]):8.4f} | Mean={np.mean(logits_arr[pos_mask]):8.4f} | 75%={np.percentile(logits_arr[pos_mask], 75):8.4f} | Max={np.max(logits_arr[pos_mask]):8.4f}")

    print("\n[Predicted Probability Distribution P(Y=1)]")
    print(f"  Overall:  Min={np.min(probs_arr):8.6f} | 25%={np.percentile(probs_arr, 25):8.6f} | Median={np.median(probs_arr):8.6f} | Mean={np.mean(probs_arr):8.6f} | 75%={np.percentile(probs_arr, 75):8.6f} | Max={np.max(probs_arr):8.6f} | Std={np.std(probs_arr):8.6f}")
    print(f"  Class 0:  Min={np.min(probs_arr[neg_mask]):8.6f} | 25%={np.percentile(probs_arr[neg_mask], 25):8.6f} | Median={np.median(probs_arr[neg_mask]):8.6f} | Mean={np.mean(probs_arr[neg_mask]):8.6f} | 75%={np.percentile(probs_arr[neg_mask], 75):8.6f} | Max={np.max(probs_arr[neg_mask]):8.6f}")
    print(f"  Class 1:  Min={np.min(probs_arr[pos_mask]):8.6f} | 25%={np.percentile(probs_arr[pos_mask], 25):8.6f} | Median={np.median(probs_arr[pos_mask]):8.6f} | Mean={np.mean(probs_arr[pos_mask]):8.6f} | 75%={np.percentile(probs_arr[pos_mask], 75):8.6f} | Max={np.max(probs_arr[pos_mask]):8.6f}")

    print(f"\n  Images predicted positive at default threshold 0.50: {np.sum(probs_arr >= 0.50):,} / {len(probs_arr):,} ({np.mean(probs_arr >= 0.50):.4%})")

    print("\n" + "=" * 80)
    print("  VALIDATION-ONLY EXPLORATORY THRESHOLD SWEEP")
    print("=" * 80)
    print(f"{'Threshold':<10} | {'Sens':<8} | {'Spec':<8} | {'Bal Acc':<9} | {'Youden J':<9} | {'Precision':<9} | {'F1':<8} | {'TN':<5} | {'FP':<5} | {'FN':<4} | {'TP':<5}")
    print("-" * 102)

    sweep_thresholds = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    best_bal_acc = -1.0
    best_t_bacc = 0.50
    best_youden = -2.0
    best_t_youden = 0.50

    for t in sweep_thresholds:
        m = compute_binary_metrics(targets_arr, probs_arr, threshold=t)
        cm = m["confusion_matrix"]
        sens = m["sensitivity"] if m["sensitivity"] is not None else 0.0
        spec = m["specificity"] if m["specificity"] is not None else 0.0
        bacc = m["balanced_accuracy"] if m["balanced_accuracy"] is not None else 0.0
        prec = m["precision"] if m["precision"] is not None else 0.0
        f1 = m["f1"] if m["f1"] is not None else 0.0
        youden_j = sens + spec - 1.0

        if bacc > best_bal_acc:
            best_bal_acc = bacc
            best_t_bacc = t
        if youden_j > best_youden:
            best_youden = youden_j
            best_t_youden = t

        print(f"{t:<10.2f} | {sens:<8.4f} | {spec:<8.4f} | {bacc:<9.4f} | {youden_j:<9.4f} | {prec:<9.4f} | {f1:<8.4f} | {cm['tn']:<5d} | {cm['fp']:<5d} | {cm['fn']:<4d} | {cm['tp']:<5d}")

    print("-" * 102)
    print(f"Best Validation Balanced Accuracy: {best_bal_acc:.4f} at Threshold {best_t_bacc:.2f}")
    print(f"Best Validation Youden J:          {best_youden:.4f} at Threshold {best_t_youden:.2f}")
    print("\n[MANDATORY DISCLAIMER] These operating points are exploratory and derived exclusively on the")
    print("                       validation split. Test data was NOT accessed or evaluated.")
    print(f"Diagnostic audit finished in {time.time() - start_time:.1f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic()

