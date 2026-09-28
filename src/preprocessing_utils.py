"""Reusable, dataset-agnostic helpers for metadata-only preprocessing."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import STANDARD_COLUMNS
except ImportError:
    from config import STANDARD_COLUMNS


def set_reproducibility(seed: int = 42) -> None:
    """Seed the random generators used by preprocessing and sampling."""
    random.seed(seed)
    np.random.seed(seed)


def require_file(path: Path, description: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} was not found at {path}. "
            "Update the notebook configuration; datasets are not downloaded automatically."
        )
    return path


def require_directory(path: Path, description: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} was not found at {path}. Update the notebook configuration."
        )
    return path


def parse_label_list(value: object, separator: str = "|") -> list[str]:
    """Parse pipe-separated labels or PadChest string/list representations."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
    return [item.strip() for item in text.split(separator) if item.strip()]


def normalize_label(label: object) -> str:
    return " ".join(str(label).strip().casefold().split())


def parse_pixel_spacing(value: object) -> tuple[float, float]:
    """Parse common DICOM spacing forms; unknown values remain NaN."""
    if pd.isna(value):
        return (np.nan, np.nan)
    values = value if isinstance(value, (list, tuple)) else str(value).replace("\\", ",").split(",")
    try:
        numbers = [float(str(item).strip()) for item in values if str(item).strip()]
    except ValueError:
        return (np.nan, np.nan)
    if len(numbers) == 1:
        return (numbers[0], np.nan)
    return (numbers[0], numbers[1]) if len(numbers) >= 2 else (np.nan, np.nan)


def first_existing_column(frame: pd.DataFrame, candidates: Sequence[str], default: object = np.nan) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series(default, index=frame.index)


def sample_clean_cases(
    frame: pd.DataFrame,
    max_positive: int | None = None,
    max_negative: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Optionally sample already-filtered cases without changing class semantics."""
    parts = []
    for target, limit in ((1, max_positive), (0, max_negative)):
        subset = frame[frame["target"] == target]
        if limit is not None:
            subset = subset.sample(n=min(limit, len(subset)), random_state=seed)
        parts.append(subset)
    if not parts:
        return frame.iloc[0:0].copy()
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_by_patient(
    frame: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Split whole patients and whole studies, ensuring no patient or study leakage."""
    if not 0 < train_size < 1 or not 0 < val_size < 1 or train_size + val_size >= 1:
        raise ValueError("train_size and val_size must be positive and leave room for test")

    df_copy = frame.copy()
    patient_ids = df_copy["patient_id"].astype(str)

    # If study_id exists, group patients that share any study to guarantee zero study leakage
    if "study_id" in df_copy.columns and df_copy["study_id"].notna().any():
        study_ids = df_copy["study_id"].dropna().astype(str)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        study_to_pts = {}
        for p, s in zip(patient_ids, study_ids):
            study_to_pts.setdefault(s, []).append(p)

        for s, pts in study_to_pts.items():
            first_p = pts[0]
            for other_p in pts[1:]:
                union(first_p, other_p)

        group_col = patient_ids.map(find)
    else:
        group_col = patient_ids

    df_copy["_split_group"] = group_col
    dominant_target = df_copy.groupby("_split_group")["target"].mean() >= 0.5
    group_table = pd.DataFrame({"_split_group": df_copy["_split_group"].unique()})
    group_table["group_target"] = group_table["_split_group"].map(dominant_target)
    stratify = group_table["group_target"] if group_table["group_target"].nunique() > 1 else None

    try:
        train_groups, remainder = train_test_split(
            group_table["_split_group"], test_size=1 - train_size, random_state=seed, stratify=stratify
        )
    except ValueError:
        train_groups, remainder = train_test_split(
            group_table["_split_group"], test_size=1 - train_size, random_state=seed
        )

    remainder_table = group_table[group_table["_split_group"].isin(remainder)]
    remainder_stratify = remainder_table["group_target"] if remainder_table["group_target"].nunique() > 1 else None
    relative_val = val_size / (1 - train_size)

    try:
        val_groups, test_groups = train_test_split(
            remainder_table["_split_group"], test_size=1 - relative_val, random_state=seed, stratify=remainder_stratify
        )
    except ValueError:
        val_groups, test_groups = train_test_split(
            remainder_table["_split_group"], test_size=1 - relative_val, random_state=seed
        )

    assignments = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }

    assert assignments["train"].isdisjoint(assignments["val"])
    assert assignments["train"].isdisjoint(assignments["test"])
    assert assignments["val"].isdisjoint(assignments["test"])

    result = {}
    for split, groups in assignments.items():
        subset = df_copy[df_copy["_split_group"].isin(groups)].copy().reset_index(drop=True)
        subset.drop(columns=["_split_group"], inplace=True, errors="ignore")
        result[split] = subset

    return result


def split_train_val_by_patient(
    frame: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Split a frame into train and validation without creating a new test set."""
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    patient_table = frame[["patient_id", "target"]].drop_duplicates("patient_id").copy()
    patient_table["patient_id"] = patient_table["patient_id"].astype(str)
    patient_table["patient_target"] = patient_table.groupby("patient_id")["target"].transform("mean") >= 0.5
    stratify = patient_table["patient_target"] if patient_table["patient_target"].nunique() > 1 else None
    try:
        train_patients, val_patients = train_test_split(
            patient_table["patient_id"], test_size=val_size, random_state=seed, stratify=stratify
        )
    except ValueError:
        train_patients, val_patients = train_test_split(
            patient_table["patient_id"], test_size=val_size, random_state=seed
        )
    assignments = {"train": set(train_patients), "val": set(val_patients)}
    assert assignments["train"].isdisjoint(assignments["val"])
    return {
        split: frame[frame["patient_id"].astype(str).isin(patients)].copy().reset_index(drop=True)
        for split, patients in assignments.items()
    }


def assert_no_leakage(splits: dict[str, pd.DataFrame], column: str = "patient_id") -> None:
    sets = {name: set(df[column].dropna().astype(str)) for name, df in splits.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert sets[left].isdisjoint(sets[right]), f"{column} leakage: {left} overlaps {right}"


def assert_no_study_leakage(splits: dict[str, pd.DataFrame]) -> None:
    available = {name: set(df["study_id"].dropna().astype(str)) for name, df in splits.items()}
    available = {name: values for name, values in available.items() if values}
    names = list(available)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert available[left].isdisjoint(available[right]), f"study_id leakage: {left} overlaps {right}"


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)


def build_summary(splits: dict[str, pd.DataFrame], source: str, parameters: dict) -> dict:
    all_rows = pd.concat(splits.values(), ignore_index=True) if splits else pd.DataFrame()
    summary = {
        "source": source,
        "parameters": parameters,
        "splits": {
            name: {
                "images": int(len(df)),
                "patients": int(df["patient_id"].nunique(dropna=True)),
                "studies": int(df["study_id"].nunique(dropna=True)),
                "positive": int((df["target"] == 1).sum()),
                "negative": int((df["target"] == 0).sum()),
            }
            for name, df in splits.items()
        },
        "total_images": int(len(all_rows)),
        "total_patients": int(all_rows["patient_id"].nunique(dropna=True)) if not all_rows.empty else 0,
    }
    return summary


def save_summary(summary: dict, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def ensure_standard_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    ordered_cols = [c for c in STANDARD_COLUMNS if c in result.columns] + [
        c for c in result.columns if c not in STANDARD_COLUMNS
    ]
    return result[ordered_cols]
