"""PadChest preprocessing pipeline — CLI entry point.

Converts the raw PadChest metadata CSV into train/val/test split CSVs that
are fully compatible with the existing training script (train_padchest.py).

Pipeline steps
--------------
1. Validate required metadata columns.
2. Apply the existing binary-label policy:
   - Positive (target=1): Labels list explicitly contains 'pleural effusion'.
   - Negative (target=0): Labels list is exactly ['normal'] (sole entry).
   - Excluded: everything else (ambiguous, missing, multi-finding, unlabeled).
3. Standardize to the project's 17-column federated schema.
4. Perform patient-level 80/10/10 train/val/test split with leakage assertions.
5. Write train.csv, val.csv, test.csv to the output directory.
6. Write summary.json with record counts, class distributions, and warnings.

Small-sample behaviour
----------------------
When the eligible pool is too small for meaningful splits (fewer than 4 unique
patients OR either class is absent), the script issues explicit WARNINGS in the
terminal and in summary.json.  It still writes the CSVs so downstream tools
can validate path resolution and schema compatibility — but the output is
explicitly flagged as unsuitable for model evaluation.

Usage
-----
  # Full 160K metadata (from repo root):
  python src/preprocess_padchest.py

  # Explicit paths:
  python src/preprocess_padchest.py \\
      --metadata   data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv \\
      --output_dir data/processed/PadChest \\
      --seed       42

  # Override split ratios:
  python src/preprocess_padchest.py --train_size 0.8 --val_size 0.1

  # Dry-run (print summary only, do not write files):
  python src/preprocess_padchest.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository path bootstrapping
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.config import RANDOM_SEED, STANDARD_COLUMNS, TARGET_DISEASE
    from src.dataset_utils import padchest_clean_binary_labels, standardize_padchest
    from src.preprocessing_utils import (
        assert_no_leakage,
        assert_no_study_leakage,
        build_summary,
        save_splits,
        save_summary,
        set_reproducibility,
        split_by_patient,
    )
except ImportError:
    from config import RANDOM_SEED, STANDARD_COLUMNS, TARGET_DISEASE
    from dataset_utils import padchest_clean_binary_labels, standardize_padchest
    from preprocessing_utils import (
        assert_no_leakage,
        assert_no_study_leakage,
        build_summary,
        save_splits,
        save_summary,
        set_reproducibility,
        split_by_patient,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_LABEL = "pleural effusion"
NEGATIVE_LABEL = "normal"
MIN_PATIENTS_PER_CLASS_PER_SPLIT = 2
REQUIRED_COLUMNS = ["ImageID", "Labels"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _locate_metadata(repo_root: Path) -> tuple[Path | None, bool]:
    """Return (path, is_sample). Full 160K CSV takes priority over sample."""
    full_csv = (
        repo_root
        / "data"
        / "raw"
        / "PadChest"
        / "metadata"
        / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    )
    sample_csv = (
        repo_root
        / "data"
        / "raw"
        / "PadChest"
        / "metadata"
        / "chest_x_ray_images_labels_sample.csv"
    )
    if full_csv.exists():
        return full_csv, False
    if sample_csv.exists():
        return sample_csv, True
    return None, False


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"PadChest metadata at {path} is missing required columns: {missing}.\n"
            "Supported formats: full 160K CSV or Kaggle sample CSV.  "
            "Both must contain at least 'ImageID' and 'Labels' columns."
        )


def _compute_split_warnings(splits: dict[str, pd.DataFrame]) -> list[str]:
    """Return human-readable warning strings for degenerate splits."""
    warn: list[str] = []
    all_patients: set[str] = set()
    for df in splits.values():
        all_patients |= set(df["patient_id"].astype(str).unique())
    total_patients = len(all_patients)

    if total_patients < 4:
        warn.append(
            f"INSUFFICIENT_DATA: Only {total_patients} unique eligible patient(s) across all "
            "splits. At least 4 patients are needed for a minimal train/val/test split."
        )

    for split_name, df in splits.items():
        n_pos = int((df["target"] == 1).sum())
        n_neg = int((df["target"] == 0).sum())
        n_pts = df["patient_id"].nunique()

        if len(df) == 0:
            warn.append(f"EMPTY_SPLIT: '{split_name}' contains 0 images.")
        if n_pos == 0:
            warn.append(
                f"NO_POSITIVES: '{split_name}' has no Pleural Effusion examples (target=1). "
                "AUROC is undefined for this split."
            )
        if n_neg == 0:
            warn.append(
                f"NO_NEGATIVES: '{split_name}' has no Normal examples (target=0). "
                "AUROC is undefined for this split."
            )
        if 0 < n_pos < MIN_PATIENTS_PER_CLASS_PER_SPLIT:
            warn.append(
                f"FEW_POSITIVES: '{split_name}' has only {n_pos} positive image(s) from "
                f"{n_pts} patient(s). Evaluation metrics will be unreliable."
            )

    return warn


def _print_split_table(splits: dict[str, pd.DataFrame], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    header = f"  {'Split':<8} {'Images':>8} {'Patients':>10} {'Studies':>9} {'Positive':>10} {'Negative':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for split_name, df in splits.items():
        n_img = len(df)
        n_pts = df["patient_id"].nunique()
        n_std = df["study_id"].nunique() if "study_id" in df.columns else "N/A"
        n_pos = int((df["target"] == 1).sum())
        n_neg = int((df["target"] == 0).sum())
        print(
            f"  {split_name:<8} {n_img:>8,} {n_pts:>10,} {n_std:>9} {n_pos:>10,} {n_neg:>10,}"
        )


def _fallback_split(
    standardized: pd.DataFrame, train_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Unstratified patient-level split for very small datasets."""
    from sklearn.model_selection import train_test_split as _tts

    patient_ids = standardized["patient_id"].astype(str).unique().tolist()
    if len(patient_ids) < 3:
        return {
            "train": standardized.copy().reset_index(drop=True),
            "val": standardized.iloc[0:0].copy().reset_index(drop=True),
            "test": standardized.iloc[0:0].copy().reset_index(drop=True),
        }

    train_pts, rem_pts = _tts(patient_ids, test_size=1 - train_size, random_state=seed)
    if len(rem_pts) < 2:
        val_pts, test_pts = rem_pts, []
    else:
        val_pts, test_pts = _tts(rem_pts, test_size=0.5, random_state=seed)

    def _filter(pts: list[str]) -> pd.DataFrame:
        return (
            standardized[standardized["patient_id"].astype(str).isin(pts)]
            .copy()
            .reset_index(drop=True)
        )

    return {"train": _filter(train_pts), "val": _filter(val_pts), "test": _filter(test_pts)}


# ---------------------------------------------------------------------------
# Main preprocessing logic (importable for testing)
# ---------------------------------------------------------------------------

def run_padchest_preprocessing(
    metadata_path: Path,
    output_dir: Path,
    seed: int = RANDOM_SEED,
    train_size: float = 0.8,
    val_size: float = 0.1,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the full PadChest preprocessing pipeline.

    Parameters
    ----------
    metadata_path : Path
        Path to the raw PadChest metadata CSV.
    output_dir : Path
        Directory where train.csv, val.csv, test.csv, and summary.json are written.
    seed : int
        Random seed for deterministic splitting.
    train_size : float
        Fraction of patients in the training split (0 < train_size < 1).
    val_size : float
        Fraction of patients in the validation split (0 < val_size < 1,
        train_size + val_size < 1).
    dry_run : bool
        If True, compute everything but do not write any files.
    verbose : bool
        If True, print progress to stdout.

    Returns
    -------
    dict
        Summary dictionary identical to the one written to summary.json.
    """
    set_reproducibility(seed)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # 1. Load and validate metadata
    # ------------------------------------------------------------------
    log(f"\n[1/6] Loading PadChest metadata from: {metadata_path}")
    df_raw = pd.read_csv(metadata_path, low_memory=False)
    _validate_columns(df_raw, metadata_path)

    is_sample = "chest_x_ray_images_labels_sample" in metadata_path.name

    log(f"      Rows: {len(df_raw):,}  |  Columns: {len(df_raw.columns)}")
    if "PatientID" in df_raw.columns:
        log(f"      Unique patients: {df_raw['PatientID'].nunique():,}")
    if "StudyID" in df_raw.columns:
        log(f"      Unique studies:  {df_raw['StudyID'].nunique():,}")
    log(f"      Unique images:   {df_raw['ImageID'].nunique():,}")

    # ------------------------------------------------------------------
    # 2. Apply binary label policy
    # ------------------------------------------------------------------
    log(
        f"\n[2/6] Applying label policy "
        f"(positive='{TARGET_LABEL}', negative='{NEGATIVE_LABEL}')..."
    )
    clean_df, label_counts = padchest_clean_binary_labels(
        df_raw,
        target_label=TARGET_LABEL,
        labels_column="Labels",
        negative_label=NEGATIVE_LABEL,
    )
    n_pos_raw = label_counts["positive_before_filter"]
    n_neg_raw = label_counts["negative_before_filter"]
    n_excl = label_counts["excluded_ambiguous_before_filter"]
    log(f"      Positive (pleural effusion): {n_pos_raw:,}")
    log(f"      Negative (normal only):      {n_neg_raw:,}")
    log(f"      Excluded (ambiguous/other):  {n_excl:,}")
    log(f"      Eligible total:              {len(clean_df):,}")

    if len(clean_df) == 0:
        raise ValueError(
            "No eligible images remain after applying the label policy. "
            "Verify the metadata file contains 'Labels' entries matching "
            f"'{TARGET_LABEL}' (positive) or '{NEGATIVE_LABEL}' (negative)."
        )

    # ------------------------------------------------------------------
    # 3. Standardize to 17-column schema
    # ------------------------------------------------------------------
    log("\n[3/6] Standardizing to 17-column federated schema...")
    standardized = standardize_padchest(clean_df)
    n_cols = len(standardized.columns)
    log(f"      Schema shape: {standardized.shape}  |  Columns: {n_cols}")

    # ------------------------------------------------------------------
    # 4. Patient-level train/val/test split
    # ------------------------------------------------------------------
    log(
        f"\n[4/6] Splitting by patient "
        f"(train={train_size:.0%} / val={val_size:.0%} / "
        f"test={(1 - train_size - val_size):.0%}, seed={seed})..."
    )

    try:
        splits = split_by_patient(
            standardized, seed=seed, train_size=train_size, val_size=val_size
        )
    except (ValueError, Exception) as exc:
        warnings.warn(
            f"split_by_patient raised: {exc}. "
            "Falling back to unstratified split — dataset is too small for stratification.",
            stacklevel=2,
        )
        splits = _fallback_split(standardized, train_size=train_size, seed=seed)

    # ------------------------------------------------------------------
    # 5. Leakage assertions
    # ------------------------------------------------------------------
    log("\n[5/6] Verifying patient and study leakage isolation...")
    try:
        assert_no_leakage(splits)
        assert_no_study_leakage(splits)
        leakage_status = "PASSED"
        log("      Patient leakage assertion: PASSED")
        log("      Study leakage assertion:   PASSED")
    except AssertionError as exc:
        leakage_status = f"FAILED: {exc}"
        log(f"      [ERROR] Leakage check FAILED: {exc}")

    # ------------------------------------------------------------------
    # 6. Build summary and diagnostics
    # ------------------------------------------------------------------
    warnings_list = _compute_split_warnings(splits)
    if is_sample:
        warnings_list.insert(
            0,
            "SAMPLE_METADATA: Preprocessing ran against the Kaggle sample CSV, NOT the full "
            "160K PadChest dataset. Results are NOT suitable for model training or "
            "federated experiments.",
        )

    _print_split_table(splits, title="Split Summary:")

    if warnings_list:
        log("\n  [WARNINGS]")
        for w in warnings_list:
            log(f"  WARNING: {w}")

    summary = build_summary(
        splits,
        source="PadChest",
        parameters={
            "target_disease": TARGET_DISEASE,
            "source_target_label": TARGET_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "random_seed": seed,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": round(1 - train_size - val_size, 6),
            "split_strategy": "patient_level_proportional",
            "is_sample_metadata": is_sample,
            "metadata_file": metadata_path.name,
            "label_counts_raw": label_counts,
            "leakage_assertion": leakage_status,
            "warnings": warnings_list,
        },
    )

    # ------------------------------------------------------------------
    # 7. Write files (skip if dry_run)
    # ------------------------------------------------------------------
    if dry_run:
        log("\n[DRY RUN] --dry_run flag set. No files written.")
    else:
        log(f"\n[6/6] Writing CSVs to: {output_dir}")
        save_splits(splits, output_dir)
        save_summary(summary, output_dir)
        saved = sorted([p.name for p in output_dir.glob("*")])
        log(f"      Saved: {saved}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PadChest preprocessing pipeline — generates train/val/test CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Path to the raw PadChest metadata CSV. "
            "Defaults to auto-detect: full 160K CSV first, then Kaggle sample CSV."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/PadChest",
        help="Directory for output train.csv, val.csv, test.csv, and summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic patient-level splitting.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Fraction of patients assigned to the training split.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of patients assigned to the validation split.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute preprocessing but do not write output files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve metadata path
    if args.metadata:
        metadata_path = Path(args.metadata).expanduser()
        if not metadata_path.is_file():
            print(f"[ERROR] Metadata file not found: {metadata_path}", file=sys.stderr)
            sys.exit(1)
        is_auto = False
    else:
        metadata_path, _ = _locate_metadata(REPO_ROOT)
        is_auto = True
        if metadata_path is None:
            print(
                "[ERROR] No PadChest metadata CSV found.  Searched:\n"
                f"  {REPO_ROOT / 'data' / 'raw' / 'PadChest' / 'metadata' / 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'}\n"
                f"  {REPO_ROOT / 'data' / 'raw' / 'PadChest' / 'metadata' / 'chest_x_ray_images_labels_sample.csv'}\n"
                "Provide --metadata <path> or place the CSV at the expected location.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    print("=" * 70)
    print("  PadChest Preprocessing Pipeline")
    print("=" * 70)
    print(f"  Metadata:   {metadata_path}{'  [auto-detected]' if is_auto else ''}")
    print(f"  Output dir: {output_dir}")
    print(f"  Seed:       {args.seed}")
    print(
        f"  Split:      train={args.train_size:.0%} / val={args.val_size:.0%} / "
        f"test={(1 - args.train_size - args.val_size):.0%}"
    )
    print(f"  Dry run:    {args.dry_run}")
    print("=" * 70)

    try:
        summary = run_padchest_preprocessing(
            metadata_path=metadata_path,
            output_dir=output_dir,
            seed=args.seed,
            train_size=args.train_size,
            val_size=args.val_size,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    except (ValueError, KeyError) as exc:
        print(f"\n[ERROR] Preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Final status
    warnings_found = summary.get("parameters", {}).get("warnings", [])
    print("\n" + "=" * 70)
    if warnings_found:
        print("  Preprocessing completed WITH WARNINGS (see above).")
        print(f"  {len(warnings_found)} warning(s) recorded in summary.json.")
    else:
        print("  Preprocessing completed successfully.")
    if not args.dry_run:
        print(f"  Output directory: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
