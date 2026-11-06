#!/usr/bin/env python
"""Preflight verification guard for the PadChest federated client (Client 03).

Usage:
  # Standard full-dataset preflight:
  python scripts/preflight_padchest.py --config configs/padchest.yaml

  # Partial archive mode (e.g. 0.zip and 1.zip downloaded):
  python scripts/preflight_padchest.py --config configs/padchest_partial.yaml --allow_partial

  # Pilot / smoke-test mode:
  python scripts/preflight_padchest.py --config configs/padchest_pilot.yaml --smoke_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import load_config
from src.config import STANDARD_COLUMNS
from src.data.padchest_dataset import PadChestPathResolver, filter_padchest_frontal_views
from src.federated.preflight import check_dependencies, audit_client_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PadChest Federated Client Preflight Verification Guard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/padchest.yaml",
        help="Path to YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to raw metadata CSV or CSV.GZ (overrides config).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to raw images directory (overrides config).",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Path to processed splits directory (overrides config).",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Permit partial archive downloads (e.g., archives 0.zip and 1.zip) without failing preflight.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Allow small sample mode (relaxes minimum image count thresholds for quick smoke testing).",
    )
    parser.add_argument(
        "--patient_limit",
        type=int,
        default=None,
        help="Optional patient limit to test sampling logic.",
    )
    return parser.parse_args()


def run_padchest_preflight(
    config_path: str | Path | None = "configs/padchest.yaml",
    metadata_override: str | Path | None = None,
    image_dir_override: str | Path | None = None,
    processed_dir_override: str | Path | None = None,
    allow_partial: bool = False,
    smoke_test: bool = False,
    patient_limit: int | None = None,
    verbose: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Execute complete preflight audit for the PadChest client.

    Returns:
      (is_ready, blockers_list, audit_details_dict)
    """
    blockers: list[str] = []
    warnings_list: list[str] = []
    details: dict[str, Any] = {}

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # 1. Load Configuration
    cfg: dict[str, Any] = {}
    if config_path and Path(config_path).is_file():
        cfg = load_config(config_path)

    metadata_path_str = metadata_override or cfg.get(
        "metadata", "data/raw/PadChest/metadata/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    )
    image_dir_str = image_dir_override or cfg.get("image_dir", "data/raw/PadChest/images")
    proc_dir_str = processed_dir_override or cfg.get("processed_dir", "data/processed/PadChest")
    is_partial = allow_partial or bool(cfg.get("allow_partial", False))
    is_smoke = smoke_test or bool(cfg.get("pilot", False)) or is_partial

    metadata_path = (REPO_ROOT / metadata_path_str).resolve() if not Path(metadata_path_str).is_absolute() else Path(metadata_path_str)
    image_dir = (REPO_ROOT / image_dir_str).resolve() if not Path(image_dir_str).is_absolute() else Path(image_dir_str)
    proc_dir = (REPO_ROOT / proc_dir_str).resolve() if not Path(proc_dir_str).is_absolute() else Path(proc_dir_str)

    # Auto-resolve .csv vs .csv.gz if needed
    if not metadata_path.is_file():
        if metadata_path.suffix == ".csv" and metadata_path.with_suffix(".csv.gz").is_file():
            metadata_path = metadata_path.with_suffix(".csv.gz")
        elif metadata_path.suffix == ".gz" and metadata_path.with_suffix("").is_file():
            metadata_path = metadata_path.with_suffix("")
        else:
            meta_dir = metadata_path.parent
            if meta_dir.is_dir():
                matches = list(meta_dir.glob("PADCHEST*.csv*"))
                if matches:
                    metadata_path = matches[0]

    mode_str = "PARTIAL ARCHIVE SUBSET (0.zip & 1.zip)" if is_partial else ("SMOKE TEST / PILOT" if is_smoke else "OFFICIAL FULL DATASET")

    log("=" * 78)
    log("  PADCHEST FEDERATED CLIENT PREFLIGHT VERIFICATION GUARD")
    log("=" * 78)
    log(f"Execution Mode:       {mode_str}")
    log(f"Config File:          {config_path}")
    log(f"Metadata Path:        {metadata_path}")
    log(f"Image Directory:      {image_dir}")
    log(f"Processed Splits:     {proc_dir}")
    log("-" * 78)

    # 2. Check System & Runtime Dependencies
    deps_ok, dep_details = check_dependencies()
    details["dependencies"] = dep_details
    log("[Check 1/6] Runtime Dependencies:")
    for dep, ok in dep_details.items():
        log(f"  - {dep:16s}: {'INSTALLED' if ok else 'MISSING'}")
    if not deps_ok:
        missing = [d for d, ok in dep_details.items() if not ok]
        blockers.append(f"Missing required Python dependencies: {', '.join(missing)}")

    # 3. Check Metadata File
    log("\n[Check 2/6] Metadata File Integrity:")
    if not metadata_path.is_file():
        blockers.append(
            f"Metadata file not found at '{metadata_path}'. "
            "Please place 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv' or "
            "'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz' in 'data/raw/PadChest/metadata/'."
        )
        log(f"  [BLOCKED] Metadata file missing: {metadata_path}")
    else:
        file_size_mb = metadata_path.stat().st_size / (1024 * 1024)
        log(f"  [PASSED] Metadata file exists: {metadata_path.name} ({file_size_mb:.1f} MB)")
        is_sample_csv = "sample" in metadata_path.name.lower()
        if is_sample_csv:
            blockers.append(
                f"Sample metadata file '{metadata_path.name}' is explicitly rejected for PadChest workflows. "
                "Official training and partial archive workflows require the official full metadata CSV.GZ."
            )

    # 4. Check Processed Manifests & Leakage
    log("\n[Check 3/6] Processed Manifests & Patient Isolation:")
    train_csv = proc_dir / "train.csv"
    val_csv = proc_dir / "val.csv"
    test_csv = proc_dir / "test.csv"
    summary_json = proc_dir / "summary.json"
    if not summary_json.is_file() and (proc_dir / "cohort_summary.json").is_file():
        summary_json = proc_dir / "cohort_summary.json"

    # Support nested directory structure as fallback (train/train.csv)
    if not train_csv.is_file() and (proc_dir / "train" / "train.csv").is_file():
        train_csv = proc_dir / "train" / "train.csv"
        val_csv = proc_dir / "val" / "val.csv"
        test_csv = proc_dir / "test" / "test.csv"

    splits_exist = train_csv.is_file() and val_csv.is_file() and test_csv.is_file()
    if not splits_exist:
        blockers.append(
            f"Processed split CSVs (train.csv, val.csv, test.csv) missing in '{proc_dir}'. "
            "Run 'python scripts/prepare_padchest.py --config configs/padchest.yaml' first."
        )
        log("  [BLOCKED] Processed CSV files missing on disk.")
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()
    else:
        # Check if existing processed splits are stale or sample-derived
        if summary_json.is_file():
            import json as _json
            try:
                s_data = _json.loads(summary_json.read_text(encoding="utf-8"))
                s_params = s_data.get("parameters", {})
                if s_data.get("is_sample_metadata") is True or s_params.get("is_sample_metadata") is True:
                    blockers.append(
                        f"Existing processed splits in '{proc_dir}' were generated from sample metadata! "
                        "Run 'python scripts/prepare_padchest.py --config configs/padchest.yaml' to regenerate from official metadata."
                    )
            except Exception:
                pass

        train_df = pd.read_csv(train_csv, low_memory=False)
        val_df = pd.read_csv(val_csv, low_memory=False)
        test_df = pd.read_csv(test_csv, low_memory=False)

        # Check column schema
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            missing_cols = [c for c in STANDARD_COLUMNS if c not in df.columns]
            if missing_cols:
                blockers.append(f"Split '{name}' is missing standard columns: {missing_cols}")

        # Check empty splits
        if len(train_df) == 0:
            blockers.append("Training manifest 'train.csv' is empty (0 records).")
        if len(val_df) == 0:
            blockers.append("Validation manifest 'val.csv' is empty (0 records).")
        if len(test_df) == 0:
            blockers.append("Test manifest 'test.csv' is empty (0 records).")

        # Patient leakage check
        train_pts = set(train_df["patient_id"].astype(str).unique()) if "patient_id" in train_df.columns else set()
        val_pts = set(val_df["patient_id"].astype(str).unique()) if "patient_id" in val_df.columns else set()
        test_pts = set(test_df["patient_id"].astype(str).unique()) if "patient_id" in test_df.columns else set()

        leak_train_val = train_pts & val_pts
        leak_train_test = train_pts & test_pts
        leak_val_test = val_pts & test_pts

        if leak_train_val:
            blockers.append(f"CRITICAL: {len(leak_train_val)} overlapping patient(s) between Train and Validation!")
        if leak_train_test:
            blockers.append(f"CRITICAL: {len(leak_train_test)} overlapping patient(s) between Train and Test!")
        if leak_val_test:
            blockers.append(f"CRITICAL: {len(leak_val_test)} overlapping patient(s) between Validation and Test!")

        if not (leak_train_val or leak_train_test or leak_val_test):
            log(f"  [PASSED] Patient isolation verified (Zero leakage across {len(train_pts | val_pts | test_pts):,} patients)")

        # Target class presence check
        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "target" in df.columns:
                n_pos = int((df["target"] == 1).sum())
                n_neg = int((df["target"] == 0).sum())
                if n_pos == 0 and not is_smoke:
                    blockers.append(f"Split '{name}' has 0 positive Pleural Effusion samples (target=1).")
                if n_neg == 0 and not is_smoke:
                    blockers.append(f"Split '{name}' has 0 negative Pleural Effusion samples (target=0).")
                log(f"  - {name:<6s}: {len(df):>7,d} images | {df['patient_id'].nunique():>6,d} pts | Pos: {n_pos:>5,d} | Neg: {n_neg:>5,d}")

    # 5. Image Directory & Archive Resolution Audit
    log("\n[Check 4/6] Image Path Resolution & Archive Indexing:")
    search_roots = []
    if image_dir.exists():
        search_roots.append(image_dir)
    search_roots.extend([
        REPO_ROOT / "data" / "raw" / "PadChest" / "images",
        REPO_ROOT / "data" / "raw" / "PadChest",
    ])

    resolver = PadChestPathResolver(search_roots=search_roots)
    num_indexed = resolver.build_index()
    log(f"  Indexed on disk: {num_indexed:,} image files (macOS ._* sidecars excluded)")

    if splits_exist and len(train_df) > 0:
        # Apply frontal filtering for realistic audit
        train_frontal, _ = filter_padchest_frontal_views(train_df, "train")
        val_frontal, _ = filter_padchest_frontal_views(val_df, "val")
        test_frontal, _ = filter_padchest_frontal_views(test_df, "test")

        train_audit = resolver.audit_split_availability(train_frontal, "train")
        val_audit = resolver.audit_split_availability(val_frontal, "val")
        test_audit = resolver.audit_split_availability(test_frontal, "test")

        log(f"  Frontal Train: Accessible: {train_audit['accessible_images']:,} / {train_audit['total_official_images']:,} ({train_audit['accessible_fraction']:.1%})")
        log(f"  Frontal Val:   Accessible: {val_audit['accessible_images']:,} / {val_audit['total_official_images']:,} ({val_audit['accessible_fraction']:.1%})")
        log(f"  Frontal Test:  Accessible: {test_audit['accessible_images']:,} / {test_audit['total_official_images']:,} ({test_audit['accessible_fraction']:.1%})")

        details["audit"] = {
            "train": train_audit,
            "val": val_audit,
            "test": test_audit,
        }

        if train_audit["accessible_images"] == 0:
            blockers.append(
                f"Zero accessible PadChest training images found on disk in searched roots: {train_audit['searched_roots']}. "
                "Download and extract the PadChest image archives to 'data/raw/PadChest/images/'."
            )
        elif not is_smoke and not is_partial and train_audit["accessible_images"] < 10000:
            blockers.append(
                f"Only {train_audit['accessible_images']:,} training images resolved on disk ({train_audit['accessible_fraction']:.1%} coverage). "
                "Official full benchmark requires all PadChest archives. "
                "If you are training on initial archives (e.g. 0.zip and 1.zip), pass '--allow_partial' or use 'configs/padchest_partial.yaml'."
            )

        if val_audit["accessible_images"] == 0 and not is_smoke and not is_partial:
            blockers.append(
                "Zero accessible PadChest validation images resolved on disk. "
                "Validation split must contain accessible frontal images for model selection."
            )

    # 6. Federated Client Preflight Audit
    log("\n[Check 5/6] Federated Client Adapter Readiness:")
    try:
        fed_status = audit_client_readiness("client_03_PadChest", REPO_ROOT, patient_limit=patient_limit)
        details["federated_status"] = fed_status
        if not fed_status.is_ready and not is_smoke and not is_partial:
            if not any("Zero accessible PadChest" in b for b in blockers):
                blockers.append(f"Federated adapter blocked: {fed_status.blocker_reason}")
            log(f"  [BLOCKED] {fed_status.blocker_reason}")
        else:
            log(f"  [PASSED] Federated adapter compatible ({fed_status.usable_train_images} train images)")
    except Exception as exc:
        blockers.append(f"Federated adapter audit raised exception: {exc}")
        log(f"  [ERROR] {exc}")

    # 7. Summary & Result
    is_ready = (len(blockers) == 0)
    log("\n" + "=" * 78)
    if is_ready:
        if is_partial:
            log("PREFLIGHT RESULT: PASSED (PARTIAL ARCHIVE MODE)")
            log("  Notice: Evaluated on partial archive download (e.g. 0.zip and 1.zip).")
            log("  Training will run on physically available frontal radiographs.")
            log("  Evaluation status will be marked: PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL.")
        else:
            log("PREFLIGHT RESULT: PASSED - PadChest client is fully ready for training.")
    else:
        log("PREFLIGHT RESULT: FAILED - Training cannot proceed.")
        log("Blocking Issues:")
        for b in blockers:
            log(f"  * {b}")
    log("=" * 78 + "\n")

    return is_ready, blockers, details


def main() -> None:
    args = parse_args()
    is_ready, blockers, _ = run_padchest_preflight(
        config_path=args.config,
        metadata_override=args.metadata,
        image_dir_override=args.image_dir,
        processed_dir_override=args.processed_dir,
        allow_partial=args.allow_partial,
        smoke_test=args.smoke_test,
        patient_limit=args.patient_limit,
        verbose=True,
    )

    if not is_ready:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
