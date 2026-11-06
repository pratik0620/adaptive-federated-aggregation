"""PadChest preprocessing pipeline — CLI entry point.

Converts the raw PadChest metadata CSV into train/val/test split CSVs that
are fully compatible with the existing training script (train_padchest.py).

Pipeline steps
--------------
1. Validate required metadata columns and report dataset health metrics.
2. Apply the binary-label policy:
   - Positive (target=1): Labels list contains 'pleural effusion'.
   - Negative (target=0): Labels list is exactly ['normal'] (sole entry).
   - Excluded: everything else (ambiguous, missing, multi-finding, unlabeled).
3. Standardize to the project's 17-column federated schema.
4. Perform patient-level 80/10/10 train/val/test split with leakage assertions.
5. Write train.csv, val.csv, test.csv to the output directory.
6. Write summary.json with record counts, class distributions, and warnings.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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
    from src.config_loader import load_config
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
    from config_loader import load_config
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
MIN_OFFICIAL_RECORDS = 100000
OFFICIAL_REQUIRED_COLUMNS = [
    "ImageID",
    "ImageDir",
    "PatientID",
    "StudyID",
    "Projection",
    "Labels",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hexadecimal digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _locate_metadata(repo_root: Path) -> tuple[Path | None, bool]:
    """Return (path, is_sample). Official full .csv.gz takes priority over uncompressed .csv.

    Sample CSVs are never returned as valid official metadata.
    """
    candidates = [
        repo_root / "data" / "raw" / "PadChest" / "metadata" / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
        repo_root / "data" / "raw" / "PadChest" / "metadata" / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c, False
    meta_dir = repo_root / "data" / "raw" / "PadChest" / "metadata"
    if meta_dir.is_dir():
        for p in sorted(meta_dir.glob("PADCHEST*.csv*")):
            if p.is_file() and "sample" not in p.name.lower():
                return p, False
    return None, False
    return None, False


def _validate_metadata_file(path: Path, allow_mock: bool = False) -> None:
    """Ensure sample CSVs are rejected and official metadata exists."""
    name_lower = path.name.lower()
    if "sample" in name_lower and not allow_mock:
        raise ValueError(
            f"Sample PadChest metadata file '{path.name}' is explicitly rejected for the official workflow. "
            "Please provide the official full metadata file: 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz' "
            "or 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv' in 'data/raw/PadChest/metadata/'."
        )


def _validate_columns(df: pd.DataFrame, path: Path, allow_mock: bool = False) -> None:
    req_cols = ["ImageID", "Labels"] if allow_mock else OFFICIAL_REQUIRED_COLUMNS
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"PadChest metadata at {path} is missing required columns: {missing}.\n"
            f"Official PadChest schema requires: {OFFICIAL_REQUIRED_COLUMNS}"
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
        n_pos = int((df["target"] == 1).sum()) if "target" in df.columns else 0
        n_neg = int((df["target"] == 0).sum()) if "target" in df.columns else 0
        n_pts = df["patient_id"].nunique() if "patient_id" in df.columns else 0

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
        n_pts = df["patient_id"].nunique() if "patient_id" in df.columns else 0
        n_std = df["study_id"].nunique() if "study_id" in df.columns else "N/A"
        n_pos = int((df["target"] == 1).sum()) if "target" in df.columns else 0
        n_neg = int((df["target"] == 0).sum()) if "target" in df.columns else 0
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


def verify_existing_splits(output_dir: Path, verbose: bool = True) -> dict[str, Any]:
    """Audit and verify existing processed PadChest split files without modifying them.

    Guards against stale or sample-derived splits.
    """
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file() and (output_dir / "cohort_summary.json").is_file():
        summary_path = output_dir / "cohort_summary.json"

    for p, name in [
        (train_path, "train.csv"),
        (val_path, "val.csv"),
        (test_path, "test.csv"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Existing split file not found: {p}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Verify column schema
    for s_name, s_df in splits.items():
        missing_cols = [c for c in STANDARD_COLUMNS if c not in s_df.columns]
        if missing_cols:
            raise ValueError(f"Split '{s_name}' is missing standard columns: {missing_cols}")

    # Verify patient leakage
    assert_no_leakage(splits)
    assert_no_study_leakage(splits)

    train_pts = set(train_df["patient_id"].astype(str).unique())
    val_pts = set(val_df["patient_id"].astype(str).unique())
    test_pts = set(test_df["patient_id"].astype(str).unique())
    total_pts = len(train_pts | val_pts | test_pts)
    total_imgs = len(train_df) + len(val_df) + len(test_df)

    summary_data = {}
    if summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Safeguard: detect and reject sample-derived split files
    params = summary_data.get("parameters", {})
    if summary_data.get("is_sample_metadata") is True or params.get("is_sample_metadata") is True:
        raise ValueError(
            f"Existing split manifests in '{output_dir}' were generated from sample metadata! "
            "Sample-derived artifacts are invalidated for official research. "
            "Please run 'python scripts/prepare_padchest.py --config configs/padchest.yaml' to regenerate from official metadata."
        )

    if total_imgs < 1000:
        raise ValueError(
            f"Existing PadChest cohort in '{output_dir}' contains only {total_imgs} images, indicating a sample artifact. "
            "The official PadChest cohort requires >= 60,000 images. "
            "Please run 'python scripts/prepare_padchest.py --config configs/padchest.yaml' to regenerate."
        )

    if verbose:
        print("\n=== Existing PadChest Split Verification ===")
        print(f"Directory: {output_dir}")
        print(f"Total Cohort: {total_imgs:,} images across {total_pts:,} unique patients")
        _print_split_table(splits, title="Verified Split Manifests:")
        print("  Patient Leakage: PASSED (Zero patient overlap across Train, Val, Test)")
        print("  Study Leakage:   PASSED (Zero study overlap across Train, Val, Test)")
        if "metadata_sha256" in params:
            print(f"  Source Metadata: {params.get('metadata_file')} (SHA256: {params.get('metadata_sha256')[:16]}...)")
        print("============================================\n")

    return {
        "status": "VERIFIED_VALID",
        "output_dir": str(output_dir),
        "total_images": total_imgs,
        "total_patients": total_pts,
        "splits": {
            k: {
                "images": len(v),
                "patients": v["patient_id"].nunique(),
                "positive": int((v["target"] == 1).sum()),
                "negative": int((v["target"] == 0).sum()),
            }
            for k, v in splits.items()
        },
        "summary": summary_data,
    }


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
    allow_small_mock: bool = False,
) -> dict[str, Any]:
    """Execute the full PadChest preprocessing pipeline."""
    set_reproducibility(seed)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ------------------------------------------------------------------
    # 1. Load and validate metadata
    # ------------------------------------------------------------------
    log(f"\n[1/6] Loading PadChest metadata from: {metadata_path}")
    _validate_metadata_file(metadata_path, allow_mock=allow_small_mock)

    df_raw = pd.read_csv(metadata_path, low_memory=False)
    _validate_columns(df_raw, metadata_path, allow_mock=allow_small_mock)

    if len(df_raw) < MIN_OFFICIAL_RECORDS and not allow_small_mock:
        raise ValueError(
            f"Loaded metadata from '{metadata_path.name}' contains only {len(df_raw):,} records, "
            f"which is suspiciously small (expected >= {MIN_OFFICIAL_RECORDS:,} for the full official PadChest dataset). "
            "Please ensure you are pointing to the official full metadata file: PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
        )

    # Compute source metadata checksum and attributes
    meta_sha256 = compute_file_sha256(metadata_path)
    meta_size_bytes = metadata_path.stat().st_size
    is_sample = "sample" in metadata_path.name.lower()

    total_raw_rows = len(df_raw)
    n_unique_patients = df_raw["PatientID"].nunique() if "PatientID" in df_raw.columns else 0
    n_unique_studies = df_raw["StudyID"].nunique() if "StudyID" in df_raw.columns else 0
    n_unique_images = df_raw["ImageID"].nunique()
    duplicate_images = total_raw_rows - n_unique_images

    proj_dist = df_raw["Projection"].value_counts().to_dict() if "Projection" in df_raw.columns else {}
    missing_labels = int(df_raw["Labels"].isna().sum()) if "Labels" in df_raw.columns else 0

    log(f"      Rows: {total_raw_rows:,}  |  Columns: {len(df_raw.columns)}")
    log(f"      File Size: {meta_size_bytes / (1024 * 1024):.2f} MB  |  SHA256: {meta_sha256[:16]}...")
    log(f"      Unique patients: {n_unique_patients:,}")
    log(f"      Unique studies:  {n_unique_studies:,}")
    log(f"      Unique images:   {n_unique_images:,} (duplicates: {duplicate_images})")
    log(f"      Missing/empty labels: {missing_labels:,}")
    if proj_dist:
        proj_str = ", ".join(f"{k}: {v:,}" for k, v in sorted(proj_dist.items(), key=lambda x: -x[1])[:5])
        log(f"      Top Projections: {proj_str}")

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
            "SAMPLE_METADATA: Preprocessing ran against sample CSV, NOT the full "
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
            "is_official_full_metadata": not is_sample,
            "metadata_file": metadata_path.name,
            "metadata_sha256": meta_sha256,
            "metadata_size_bytes": meta_size_bytes,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_metadata_rows": total_raw_rows,
            "unique_patient_count": n_unique_patients,
            "unique_study_count": n_unique_studies,
            "unique_image_count": n_unique_images,
            "duplicate_image_ids": duplicate_images,
            "missing_labels_count": missing_labels,
            "projection_distribution": proj_dist,
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

        # Write matching cohort manifests
        all_cohort = pd.concat(splits.values(), ignore_index=True)
        all_cohort.to_csv(output_dir / "padchest_cohort.csv", index=False)
        splits["train"].to_csv(output_dir / "train_cohort.csv", index=False)
        splits["val"].to_csv(output_dir / "val_cohort.csv", index=False)
        splits["test"].to_csv(output_dir / "test_cohort.csv", index=False)
        (output_dir / "cohort_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        saved = sorted([p.name for p in output_dir.glob("*") if p.is_file()])
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
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML or JSON configuration file.",
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
        "--verify_only",
        action="store_true",
        help="Verify existing processed split CSVs without regenerating them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration file if specified
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if k == "metadata" and args.metadata is None and v is not None:
                args.metadata = v
            elif k == "processed_dir" and args.output_dir == "data/processed/PadChest" and v is not None:
                args.output_dir = v
            elif k == "seed" and args.seed == RANDOM_SEED and v is not None:
                args.seed = int(v)
            elif k == "train_size" and args.train_size == 0.8 and v is not None:
                args.train_size = float(v)
            elif k == "val_size" and args.val_size == 0.1 and v is not None:
                args.val_size = float(v)

    # Resolve output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    # If verify_only is requested, verify existing splits and exit
    if args.verify_only:
        try:
            verify_existing_splits(output_dir, verbose=not args.quiet)
            sys.exit(0)
        except Exception as exc:
            print(f"[ERROR] Split verification failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Resolve metadata path
    if args.metadata:
        metadata_path = Path(args.metadata).expanduser()
        if not metadata_path.is_absolute():
            metadata_path = (REPO_ROOT / metadata_path).resolve()
        if not metadata_path.is_file():
            print(f"[ERROR] Metadata file not found: {metadata_path}", file=sys.stderr)
            sys.exit(1)
        is_auto = False
    else:
        metadata_path, _ = _locate_metadata(REPO_ROOT)
        is_auto = True
        if metadata_path is None:
            print(
                "[ERROR] No official PadChest metadata CSV found. Searched:\n"
                f"  {REPO_ROOT / 'data' / 'raw' / 'PadChest' / 'metadata' / 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz'}\n"
                f"  {REPO_ROOT / 'data' / 'raw' / 'PadChest' / 'metadata' / 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'}\n"
                "Provide --metadata <path> or place the official metadata CSV.GZ at the expected location.",
                file=sys.stderr,
            )
            sys.exit(1)

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
