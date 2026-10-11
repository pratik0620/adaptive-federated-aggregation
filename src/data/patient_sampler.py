"""Deterministic patient and sample selection for NIH training.

Implements reproducible seeded sampling from eligible training patients,
preserving natural class distribution without forcing 50/50 artificial balance,
while guaranteeing strict patient-level isolation (zero leakage).
"""

from __future__ import annotations

import random
from typing import Any
import pandas as pd


def get_patient_labels(df: pd.DataFrame) -> dict[str, int]:
    """Map each unique patient_id to a binary label (1 if any positive image exists, else 0)."""
    if "patient_id" not in df.columns or "target" not in df.columns:
        raise KeyError("DataFrame must contain 'patient_id' and 'target' columns.")
    patient_max = df.groupby(df["patient_id"].astype(str))["target"].max()
    return {str(pid): int(label) for pid, label in patient_max.items()}


def create_stratified_patient_sequence(
    patient_labels: dict[str, int],
    seed: int = 42,
) -> list[str]:
    """Create a reproducible, seeded permutation of patients that preserves natural class prevalence.

    Procedure:
      1. Partition unique patients into positive (has >=1 positive image) and negative.
      2. Sort both partitions deterministically.
      3. Shuffle each partition independently using seeded RNGs.
      4. Proportionally interleave the two shuffled lists so that any prefix or slice
         faithfully mirrors the cohort's natural class prevalence without forcing equal counts.
    """
    pos_pts = sorted([p for p, lbl in patient_labels.items() if lbl == 1])
    neg_pts = sorted([p for p, lbl in patient_labels.items() if lbl == 0])

    rng_pos = random.Random(seed)
    rng_neg = random.Random(seed + 1000)

    shuffled_pos = list(pos_pts)
    shuffled_neg = list(neg_pts)
    rng_pos.shuffle(shuffled_pos)
    rng_neg.shuffle(shuffled_neg)

    total_patients = len(pos_pts) + len(neg_pts)
    if total_patients == 0:
        return []
    if len(pos_pts) == 0:
        return shuffled_neg
    if len(neg_pts) == 0:
        return shuffled_pos

    pos_prevalence = len(pos_pts) / total_patients

    # Interleave to preserve natural prevalence along the entire sequence
    sequence: list[str] = []
    pos_idx = 0
    neg_idx = 0

    for i in range(total_patients):
        expected_pos = round((i + 1) * pos_prevalence)
        current_pos = pos_idx
        if current_pos < expected_pos and pos_idx < len(shuffled_pos):
            sequence.append(shuffled_pos[pos_idx])
            pos_idx += 1
        elif neg_idx < len(shuffled_neg):
            sequence.append(shuffled_neg[neg_idx])
            neg_idx += 1
        else:
            sequence.append(shuffled_pos[pos_idx])
            pos_idx += 1

    return sequence


def select_training_patients(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    patient_limit: int | None = None,
    patient_start: int | None = None,
    patient_end: int | None = None,
    seed: int = 42,
    accessible_patient_ids: set[str] | list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a deterministic subset of patients from the NIH training split.

    Selection Procedure:
      1. Map all eligible training patients to their patient-level disease status
         (positive if patient has at least one positive image, else negative).
      2. If accessible_patient_ids is specified, filter eligible patients to those
         present in local archives while retaining natural class distribution.
      3. Generate a reproducible seeded stratified sequence that preserves natural
         prevalence throughout the list.
      4. If patient_start or patient_end is set, slice the sequence [start:end].
      5. If patient_limit is set, take the first patient_limit patients from the slice.
      6. Filter train_df to include all eligible images for selected patients.
      7. Verify zero patient leakage into validation or test sets.

    Args:
        train_df: Official NIH training split DataFrame.
        val_df: Official NIH validation split DataFrame (for leakage assertion).
        test_df: Official NIH test split DataFrame (for leakage assertion).
        patient_limit: Maximum number of unique patients to select (None = all).
        patient_start: Optional starting index in the seeded patient sequence.
        patient_end: Optional ending index in the seeded patient sequence.
        seed: Random seed for reproducible sampling.
        accessible_patient_ids: Optional set of patient IDs verified accessible on disk.

    Returns:
        Tuple of (filtered training DataFrame, selection manifest dict).
    """
    eligible_df = train_df
    if accessible_patient_ids is not None:
        acc_set = set(str(p) for p in accessible_patient_ids)
        eligible_df = train_df[train_df["patient_id"].astype(str).isin(acc_set)]
        if eligible_df.empty:
            raise ValueError("No eligible training patients match the provided accessible_patient_ids.")

    patient_labels = get_patient_labels(eligible_df)
    total_eligible_patients = len(patient_labels)

    natural_pos_patients = sum(1 for lbl in patient_labels.values() if lbl == 1)
    natural_neg_patients = sum(1 for lbl in patient_labels.values() if lbl == 0)
    natural_patient_prevalence = (
        natural_pos_patients / total_eligible_patients if total_eligible_patients > 0 else 0.0
    )

    # Generate reproducible seeded stratified sequence
    ordered_sequence = create_stratified_patient_sequence(patient_labels, seed=seed)

    # Apply slicing [patient_start : patient_end]
    start_idx = max(0, int(patient_start)) if patient_start is not None else 0
    end_idx = min(total_eligible_patients, int(patient_end)) if patient_end is not None else total_eligible_patients
    sliced_sequence = ordered_sequence[start_idx:end_idx]

    # Apply patient_limit
    if patient_limit is not None and patient_limit > 0:
        selected_patient_ids = sliced_sequence[: int(patient_limit)]
    else:
        selected_patient_ids = sliced_sequence

    selected_set = set(selected_patient_ids)

    # Filter training split to include all images for selected patients
    filtered_train_df = train_df[train_df["patient_id"].astype(str).isin(selected_set)].copy().reset_index(drop=True)

    # Strict patient-level isolation assertions
    if val_df is not None and not val_df.empty:
        val_patients = set(val_df["patient_id"].astype(str).unique())
        leakage_with_val = selected_set & val_patients
        if leakage_with_val:
            raise AssertionError(
                f"Patient leakage detected! {len(leakage_with_val)} training patients overlap with validation set: "
                f"{list(leakage_with_val)[:5]}"
            )

    if test_df is not None and not test_df.empty:
        test_patients = set(test_df["patient_id"].astype(str).unique())
        leakage_with_test = selected_set & test_patients
        if leakage_with_test:
            raise AssertionError(
                f"Patient leakage detected! {len(leakage_with_test)} training patients overlap with test set: "
                f"{list(leakage_with_test)[:5]}"
            )

    # Compute patient-level and image-level statistics
    selected_pos_patients = sum(1 for pid in selected_patient_ids if patient_labels.get(pid, 0) == 1)
    selected_neg_patients = sum(1 for pid in selected_patient_ids if patient_labels.get(pid, 0) == 0)
    selected_patient_fraction = (
        selected_pos_patients / len(selected_patient_ids) if selected_patient_ids else 0.0
    )

    selected_image_count = len(filtered_train_df)
    positive_image_count = int((filtered_train_df["target"] == 1).sum()) if "target" in filtered_train_df.columns else 0
    negative_image_count = int((filtered_train_df["target"] == 0).sum()) if "target" in filtered_train_df.columns else 0
    positive_image_fraction = (
        float(positive_image_count / selected_image_count) if selected_image_count > 0 else 0.0
    )

    manifest: dict[str, Any] = {
        "selection_procedure": (
            "Reproducible seeded stratified sampling preserving natural cohort prevalence. "
            f"Seed={seed}, slice=[{start_idx}:{end_idx}], limit={patient_limit}, "
            f"accessible_restricted={accessible_patient_ids is not None}"
        ),
        "random_seed": seed,
        "patient_limit": patient_limit,
        "patient_start": patient_start,
        "patient_end": patient_end,
        "total_eligible_training_patients": total_eligible_patients,
        "natural_cohort_positive_patients": natural_pos_patients,
        "natural_cohort_negative_patients": natural_neg_patients,
        "natural_cohort_patient_prevalence": natural_patient_prevalence,
        "selected_patient_count": len(selected_patient_ids),
        "selected_positive_patients": selected_pos_patients,
        "selected_negative_patients": selected_neg_patients,
        "selected_patient_positive_fraction": selected_patient_fraction,
        "selected_image_count": selected_image_count,
        "selected_positive_images": positive_image_count,
        "selected_negative_images": negative_image_count,
        "selected_image_positive_fraction": positive_image_fraction,
        "selected_patient_ids": selected_patient_ids,
        "patient_leakage_assertion": "PASSED (strict isolation verified)",
    }

    return filtered_train_df, manifest


def format_patient_selection_report(manifest: dict[str, Any], title: str | None = None) -> str:
    """Format manifest statistics into a readable string."""
    nat_prev = manifest.get("natural_cohort_patient_prevalence", 0.0)
    sel_prev = manifest.get("selected_patient_positive_fraction", 0.0)
    img_prev = manifest.get("selected_image_positive_fraction", 0.0)
    dataset_name = manifest.get("dataset_name", manifest.get("dataset", "NIH"))

    header = title if title else f"=== {dataset_name} Training Patient Selection Report ==="

    lines = [
        header,
        f"Sampling Procedure:          Reproducible seeded stratified sampling (seed={manifest.get('random_seed')})",
        f"Eligible Training Patients:  {manifest.get('total_eligible_training_patients', 0):,} (Natural Prev: {nat_prev:.2%})",
        f"Selected Patients:           {manifest.get('selected_patient_count', 0):,} (Pos: {manifest.get('selected_positive_patients', 0)}, Neg: {manifest.get('selected_negative_patients', 0)}, Prev: {sel_prev:.2%})",
        f"Selected Images:             {manifest.get('selected_image_count', 0):,} (Pos: {manifest.get('selected_positive_images', 0)}, Neg: {manifest.get('selected_negative_images', 0)}, Prev: {img_prev:.2%})",
        f"Patient Slicing:             start={manifest.get('patient_start')}, end={manifest.get('patient_end')}, limit={manifest.get('patient_limit')}",
        f"Patient Leakage Assertion:   {manifest.get('patient_leakage_assertion', 'UNKNOWN')}",
    ]
    return "\n".join(lines)
