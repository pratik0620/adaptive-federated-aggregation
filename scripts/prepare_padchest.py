#!/usr/bin/env python
"""PadChest dataset preparation and split verification runner.

Usage:
  # Verify existing processed splits (non-destructive):
  python scripts/prepare_padchest.py --config configs/padchest.yaml --verify_only

  # Dry-run against official 160K metadata:
  python scripts/prepare_padchest.py --config configs/padchest.yaml --dry_run

  # Generate fresh splits from metadata:
  python scripts/prepare_padchest.py --config configs/padchest.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config_loader import load_config
from src.preprocess_padchest import run_padchest_preprocessing, verify_existing_splits, _locate_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PadChest Dataset Preparation and Verification Runner",
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
        help="Path to raw PadChest metadata CSV (overrides config).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save/verify processed splits (overrides config).",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Verify existing processed split manifests without regenerating.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute split statistics without writing files to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic patient splitting.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Fraction of patients allocated to train split.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of patients allocated to validation split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg: dict = {}
    if args.config and Path(args.config).is_file():
        cfg = load_config(args.config)

    # Merge config defaults
    metadata_path_str = args.metadata or cfg.get("metadata")
    output_dir_str = args.output_dir or cfg.get("processed_dir", "data/processed/PadChest")
    seed = args.seed if args.seed != 42 or "seed" not in cfg else int(cfg["seed"])
    train_size = args.train_size if args.train_size != 0.8 or "train_size" not in cfg else float(cfg["train_size"])
    val_size = args.val_size if args.val_size != 0.1 or "val_size" not in cfg else float(cfg["val_size"])

    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    print("=" * 74)
    print("  PadChest Dataset Ingestion & Cohort Preparation")
    print("=" * 74)
    print(f"Config:       {args.config}")
    print(f"Output Dir:   {output_dir}")
    print(f"Verify Only:  {args.verify_only}")
    print(f"Dry Run:      {args.dry_run}")
    print("=" * 74 + "\n")

    if args.verify_only:
        try:
            report = verify_existing_splits(output_dir, verbose=True)
            print("[SUCCESS] All existing PadChest cohort splits verified successfully.")
            sys.exit(0)
        except Exception as exc:
            print(f"[ERROR] Split verification failed: {exc}", file=sys.stderr)
            sys.exit(1)

    # Resolve metadata CSV
    if metadata_path_str:
        metadata_path = Path(metadata_path_str)
        if not metadata_path.is_absolute():
            metadata_path = (REPO_ROOT / metadata_path).resolve()
    else:
        metadata_path, is_sample = _locate_metadata(REPO_ROOT)

    if not metadata_path or not metadata_path.is_file():
        print(
            f"[ERROR] Metadata CSV not found at '{metadata_path}'.\n"
            "Please check the path or download the official PadChest metadata to data/raw/PadChest/metadata/",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        summary = run_padchest_preprocessing(
            metadata_path=metadata_path,
            output_dir=output_dir,
            seed=seed,
            train_size=train_size,
            val_size=val_size,
            dry_run=args.dry_run,
            verbose=True,
        )
        print("\n[SUCCESS] PadChest preparation completed.")
    except Exception as exc:
        print(f"\n[ERROR] PadChest preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
