"""Comprehensive evaluation metrics for binary thoracic disease classification.

Computes AUROC (primary metric), PR-AUC, Sensitivity, Specificity, F1 score,
Balanced Accuracy, Confusion Matrix, and class distribution counts, with explicit
graceful handling for edge cases (e.g. single-class evaluation splits).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_binary_metrics(
    y_true: np.ndarray | list[int | float],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute comprehensive binary classification evaluation metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (in [0, 1]).
        threshold: Decision threshold for discrete class predictions.

    Returns:
        Dictionary of metrics, confusion matrix, sample counts, and warnings.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    total_samples = int(len(y_true))
    if total_samples == 0:
        return {
            "total_samples": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_fraction": 0.0,
            "auroc": None,
            "pr_auc": None,
            "sensitivity": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "threshold": threshold,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "warnings": ["Evaluation split is empty (0 samples)."],
        }

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    positive_fraction = float(positive_count / total_samples)

    warnings: list[str] = []

    # AUROC & PR-AUC require at least one sample of each class
    auroc: float | None = None
    pr_auc: float | None = None

    if positive_count == 0 or negative_count == 0:
        single_class = 1 if positive_count > 0 else 0
        warnings.append(
            f"Only a single class (class {single_class}) is present in the split. "
            "AUROC and PR-AUC are undefined."
        )
    else:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError as err:
            warnings.append(f"AUROC calculation failed: {err}")
            auroc = None

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError as err:
            warnings.append(f"PR-AUC calculation failed: {err}")
            pr_auc = None

    # Discrete predictions
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Sensitivity / Recall (TP / (TP + FN))
    if (tp + fn) > 0:
        sensitivity: float | None = float(tp / (tp + fn))
    else:
        sensitivity = None
        warnings.append("No true positive samples present; Sensitivity is undefined.")

    # Specificity (TN / (TN + FP))
    if (tn + fp) > 0:
        specificity: float | None = float(tn / (tn + fp))
    else:
        specificity = None
        warnings.append("No true negative samples present; Specificity is undefined.")

    # Precision / PPV (TP / (TP + FP))
    if (tp + fp) > 0:
        precision: float | None = float(tp / (tp + fp))
    else:
        precision = None

    # F1 Score
    if sensitivity is not None and precision is not None and (precision + sensitivity) > 0:
        f1: float | None = float(2 * (precision * sensitivity) / (precision + sensitivity))
    elif sensitivity is not None and sensitivity == 0.0:
        f1 = 0.0
    else:
        f1 = None

    # Balanced Accuracy ( (Sensitivity + Specificity) / 2 )
    if sensitivity is not None and specificity is not None:
        balanced_accuracy: float | None = float((sensitivity + specificity) / 2.0)
    elif sensitivity is not None:
        balanced_accuracy = sensitivity
    elif specificity is not None:
        balanced_accuracy = specificity
    else:
        balanced_accuracy = None

    return {
        "total_samples": total_samples,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": positive_fraction,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "warnings": warnings,
    }


def format_metrics_summary(metrics: dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Format metrics dictionary into a readable multi-line summary string."""
    cm = metrics.get("confusion_matrix", {})
    auroc_str = f"{metrics['auroc']:.4f}" if metrics.get("auroc") is not None else "N/A (undefined)"
    prauc_str = f"{metrics['pr_auc']:.4f}" if metrics.get("pr_auc") is not None else "N/A (undefined)"
    sens_str = f"{metrics['sensitivity']:.4f}" if metrics.get("sensitivity") is not None else "N/A"
    spec_str = f"{metrics['specificity']:.4f}" if metrics.get("specificity") is not None else "N/A"
    f1_str = f"{metrics['f1']:.4f}" if metrics.get("f1") is not None else "N/A"
    bacc_str = f"{metrics['balanced_accuracy']:.4f}" if metrics.get("balanced_accuracy") is not None else "N/A"

    lines = []

    # Highlight evaluation integrity status if present
    is_official = metrics.get("is_official_full_test")
    eval_status = metrics.get("evaluation_status")
    if is_official is False:
        lines.append("*******************************************************************************")
        lines.append("  [NOTICE: PARTIAL ARCHIVE SUBSET EVALUATION — NOT OFFICIAL BENCHMARK]")
        eval_cnt = metrics.get("evaluated_samples", metrics.get("total_samples", 0))
        tot_cnt = metrics.get("total_official_split_samples", 0)
        cov = metrics.get("coverage_percentage", 0.0)
        lines.append(f"  Coverage: {eval_cnt:,} / {tot_cnt:,} images ({cov:.2f}% of official split)")
        lines.append("  These metrics MUST NOT be described or cited as official full-test results.")
        lines.append("*******************************************************************************")
    elif is_official is True:
        lines.append("===============================================================================")
        lines.append(f"  [OFFICIAL FULL TEST BENCHMARK — 100% SPLIT COVERAGE: {metrics.get('total_samples', 0):,} IMAGES]")
        lines.append("===============================================================================")

    lines.extend([
        f"=== {title} ===",
        f"Evaluation Status:   {eval_status or 'Standard'}",
        f"Total Samples:       {metrics.get('total_samples', 0)}",
        f"Positive Samples:    {metrics.get('positive_count', 0)} ({metrics.get('positive_fraction', 0.0):.2%})",
        f"Negative Samples:    {metrics.get('negative_count', 0)}",
        f"AUROC (Primary):     {auroc_str}",
        f"PR-AUC:              {prauc_str}",
        f"Sensitivity:         {sens_str}",
        f"Specificity:         {spec_str}",
        f"F1 Score:            {f1_str}",
        f"Balanced Accuracy:   {bacc_str}",
        f"Confusion Matrix:    TN={cm.get('tn', 0)}, FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)}, TP={cm.get('tp', 0)}",
    ])
    warnings = metrics.get("warnings", [])
    if warnings:
        lines.append("Warnings / Disclaimers:")
        for w in warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
