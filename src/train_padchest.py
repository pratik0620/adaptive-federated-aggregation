"""CLI entry point for reproducible PadChest ResNet-50 binary training.

Binary classification of Pleural Effusion on PadChest (Client 03).
Preserves patient-level 80/10/10 split definitions, enforces frontal-view filtering
(PA, AP, and AP_horizontal, excluding Lateral and Costal views), and guarantees zero patient leakage.

Usage examples:
  # Small pilot run (100 patients, 2 epochs, batch size 16):
  python src/train_padchest.py --pilot

  # Sliced patient range (e.g. patients 0 to 500, 5 epochs):
  python src/train_padchest.py --patient_start 0 --patient_end 500 --epochs 5

  # Full dataset run with custom image root and batch size:
  python src/train_padchest.py --image_dir data/raw/PadChest/images --batch_size 32 --epochs 10

  # Dry-run verification (validates paths, patient splits, and manifests without starting training):
  python src/train_padchest.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Ensure repository root is in sys.path when executed directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data.padchest_dataset import (
        PadChestPathResolver,
        create_padchest_dataloaders,
        filter_padchest_frontal_views,
    )
    from src.data.patient_sampler import format_patient_selection_report, select_training_patients
    from src.models.resnet import build_resnet50_binary
    from src.trainer import ClientTrainer
except ImportError:
    from data.padchest_dataset import (
        PadChestPathResolver,
        create_padchest_dataloaders,
        filter_padchest_frontal_views,
    )
    from data.patient_sampler import format_patient_selection_report, select_training_patients
    from models.resnet import build_resnet50_binary
    from trainer import ClientTrainer


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random, numpy, and torch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PadChest training pipeline."""
    parser = argparse.ArgumentParser(
        description="Reproducible PadChest Training Pipeline (Pleural Effusion)"
    )

    # Paths
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Repository root directory. Defaults to current working directory.",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed/PadChest",
        help="Directory containing processed PadChest train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing raw PadChest images or archives (e.g. data/raw/PadChest/images).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/padchest",
        help="Base directory for saving run artifacts, checkpoints, and logs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this training run. If omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a JSON configuration file to load arguments from.",
    )

    # Frontal view policy
    parser.add_argument(
        "--frontal_only",
        dest="frontal_only",
        action="store_true",
        default=True,
        help="Filter for frontal radiographs (PA, AP), excluding Lateral and Costal views (default: True).",
    )
    parser.add_argument(
        "--no-frontal-only",
        dest="frontal_only",
        action="store_false",
        help="Include all view positions (do not filter out lateral or costal views).",
    )
    parser.add_argument(
        "--include_ap_horizontal",
        dest="include_ap_horizontal",
        action="store_true",
        default=True,
        help="Include AP_horizontal (bedside supine AP) as frontal radiographs (default: True).",
    )
    parser.add_argument(
        "--no_ap_horizontal",
        dest="include_ap_horizontal",
        action="store_false",
        help="Exclude AP_horizontal views, restricting strictly to upright PA and AP.",
    )

    # Pilot and patient selection
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a quick pilot subset (default: 100 patients, 2 epochs, batch size 16).",
    )
    parser.add_argument(
        "--patient_limit",
        type=int,
        default=None,
        help="Maximum number of unique training patients to select (None = all).",
    )
    parser.add_argument(
        "--patient_start",
        type=int,
        default=None,
        help="Starting index in the reproducible seeded sequence of eligible training patients.",
    )
    parser.add_argument(
        "--patient_end",
        type=int,
        default=None,
        help="Ending index in the reproducible seeded sequence of eligible training patients.",
    )
    parser.add_argument(
        "--sample_from_accessible",
        action="store_true",
        help="Restrict training patient sampling to patients with accessible images in local archives.",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Optimizer weight decay.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer algorithm.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--metric_to_optimize",
        type=str,
        default="val_auroc",
        choices=["val_auroc", "val_loss"],
        help="Validation metric used for best checkpoint selection.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help="Optional positive class loss weighting factor for BCEWithLogitsLoss.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Hardware & performance
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu'). Defaults to auto-detecting CUDA if available.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (0 is safest on Windows).",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision (AMP fp16) on CUDA.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image resize square resolution (default: 224).",
    )

    # Model architecture
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet-pretrained weights for ResNet-50 backbone (default: True).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate in classification head.",
    )

    # Dataset path resolution handling
    parser.add_argument(
        "--strict_paths",
        action="store_true",
        help="Raise FileNotFoundError if any split image cannot be found on disk.",
    )
    parser.add_argument(
        "--filter_missing",
        action="store_true",
        default=True,
        help="Filter splits to images actually present on disk (reports dropped counts explicitly).",
    )
    parser.add_argument(
        "--no-filter-missing",
        dest="filter_missing",
        action="store_false",
        help="Do not filter missing images.",
    )

    # Dry-run flag
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths, patient splits, and write manifests without executing model training.",
    )

    return parser.parse_args()


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from a JSON file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    # Load external config file if provided
    if args.config:
        file_config = load_config_file(args.config)
        for key, val in file_config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, val)

    # Pilot run defaults
    if args.pilot:
        if args.patient_limit is None:
            args.patient_limit = 100
        if args.epochs == 10:
            args.epochs = 2
        if args.batch_size == 32:
            args.batch_size = 16

    # Determine project root and directories
    repo_root = Path(args.data_root).resolve() if args.data_root else Path.cwd().resolve()
    processed_dir = (
        (repo_root / args.processed_dir).resolve()
        if not Path(args.processed_dir).is_absolute()
        else Path(args.processed_dir)
    )

    # Validate processed CSVs exist
    train_csv_path = processed_dir / "train.csv"
    val_csv_path = processed_dir / "val.csv"
    test_csv_path = processed_dir / "test.csv"

    for path, name in [
        (train_csv_path, "Train split CSV"),
        (val_csv_path, "Validation split CSV"),
        (test_csv_path, "Test split CSV"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} not found at {path}. "
                "Ensure PadChest preprocessing has been executed or verify --processed_dir."
            )

    # Setup run name and output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    elif args.pilot:
        run_name = f"padchest_pilot_resnet50_{timestamp}"
    else:
        run_name = f"padchest_resnet50_{timestamp}"

    output_dir = (
        (repo_root / args.output_dir / run_name).resolve()
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir) / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    set_seed(args.seed)

    # Device selection
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("======================================================================")
    print("  PadChest Binary Training Pipeline (Client 03 — Pleural Effusion)")
    print("======================================================================")
    print(f"Run Name:               {run_name}")
    print(f"Device:                 {device} ({'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Processed Dir:          {processed_dir}")
    print(f"Output Dir:             {output_dir}")
    print(f"Random Seed:            {args.seed}")
    print(f"Frontal Only:           {args.frontal_only}")
    print(f"Include AP_horizontal:  {args.include_ap_horizontal}")
    print(f"Pilot Mode:             {args.pilot}")
    print(f"Epochs:                 {args.epochs}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Learning Rate:          {args.lr}")
    print(f"Optimizer:              {args.optimizer.upper()}")
    print("======================================================================\n")

    # 1. Load Processed CSVs
    print("[1/5] Loading processed PadChest metadata CSVs...")
    train_df = pd.read_csv(train_csv_path, low_memory=False)
    val_df = pd.read_csv(val_csv_path, low_memory=False)
    test_df = pd.read_csv(test_csv_path, low_memory=False)

    print(f"  Official Train Split:       {len(train_df):,} images ({train_df['patient_id'].nunique():,} unique patients)")
    print(f"  Official Validation Split:  {len(val_df):,} images ({val_df['patient_id'].nunique():,} unique patients)")
    print(f"  Official Held-Out Test:     {len(test_df):,} images ({test_df['patient_id'].nunique():,} unique patients)")

    # Apply frontal-view filtering if enabled
    frontal_reports = {}
    if args.frontal_only:
        print(f"\n  Filtering for frontal radiographs (PA, AP{' + AP_horizontal' if args.include_ap_horizontal else ''})...")
        train_df, frontal_reports["train"] = filter_padchest_frontal_views(
            train_df, split_name="train", include_ap_horizontal=args.include_ap_horizontal
        )
        val_df, frontal_reports["val"] = filter_padchest_frontal_views(
            val_df, split_name="val", include_ap_horizontal=args.include_ap_horizontal
        )
        test_df, frontal_reports["test"] = filter_padchest_frontal_views(
            test_df, split_name="test", include_ap_horizontal=args.include_ap_horizontal
        )

        print(
            f"  Train:      Retained {frontal_reports['train']['frontal_images_retained']:,} frontal "
            f"(excluded {frontal_reports['train']['lateral_images_excluded']:,} lateral, {frontal_reports['train']['costal_images_excluded']} costal)"
        )
        print(
            f"  Validation: Retained {frontal_reports['val']['frontal_images_retained']:,} frontal "
            f"(excluded {frontal_reports['val']['lateral_images_excluded']:,} lateral, {frontal_reports['val']['costal_images_excluded']} costal)"
        )
        print(
            f"  Test:       Retained {frontal_reports['test']['frontal_images_retained']:,} frontal "
            f"(excluded {frontal_reports['test']['lateral_images_excluded']:,} lateral, {frontal_reports['test']['costal_images_excluded']} costal)"
        )

    # 2. Path Resolution & Archive Availability Audit
    print("\n[2/5] Auditing image archive availability across PadChest splits...")
    search_roots = []
    if args.image_dir:
        search_roots.append(args.image_dir)
    search_roots.extend([
        repo_root / "data" / "raw" / "PadChest" / "images",
        repo_root / "data" / "raw" / "PadChest",
        repo_root / "data" / "raw",
    ])

    resolver = PadChestPathResolver(search_roots=search_roots)
    num_indexed = resolver.build_index()
    print(f"  Indexed {num_indexed:,} valid radiographs on disk (macOS ._* sidecars strictly excluded).")

    # Audit official splits before sampling
    train_audit = resolver.audit_split_availability(train_df, split_name="train")
    val_audit = resolver.audit_split_availability(val_df, split_name="val")
    test_audit = resolver.audit_split_availability(test_df, split_name="test")

    print("\n  === PadChest Archive Availability Audit ===")
    print(
        f"  Train:      Accessible: {train_audit['accessible_images']:,} / {train_audit['total_official_images']:,} "
        f"({train_audit['accessible_fraction']:.1%}) | Missing: {train_audit['missing_images']:,} "
        f"| Unique Patients: {train_audit['accessible_patients']:,} / {train_audit['total_official_patients']:,}"
    )
    print(
        f"  Validation: Accessible: {val_audit['accessible_images']:,} / {val_audit['total_official_images']:,} "
        f"({val_audit['accessible_fraction']:.1%}) | Missing: {val_audit['missing_images']:,} "
        f"| Unique Patients: {val_audit['accessible_patients']:,} / {val_audit['total_official_patients']:,}"
    )
    print(
        f"  Test:       Accessible: {test_audit['accessible_images']:,} / {test_audit['total_official_images']:,} "
        f"({test_audit['accessible_fraction']:.1%}) | Missing: {test_audit['missing_images']:,} "
        f"| Unique Patients: {test_audit['accessible_patients']:,} / {test_audit['total_official_patients']:,}"
    )

    if not test_audit["is_complete"]:
        print(
            f"\n  [AUDIT ALERT] Incomplete PadChest test archive detected ({test_audit['accessible_images']:,} / "
            f"{test_audit['total_official_images']:,} images accessible). "
            "\n  Evaluation metrics on this subset MUST NOT be described as official full-test results."
        )

    # 3. Patient Selection on Training Split
    print("\n[3/5] Applying reproducible seeded patient sampling on PadChest training split...")
    accessible_train_pts = (
        resolver.get_accessible_patient_ids(train_df) if args.sample_from_accessible else None
    )

    selected_train_df, patient_manifest = select_training_patients(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        patient_limit=args.patient_limit,
        patient_start=args.patient_start,
        patient_end=args.patient_end,
        seed=args.seed,
        accessible_patient_ids=accessible_train_pts,
    )
    print(format_patient_selection_report(patient_manifest, title="=== PadChest Training Patient Selection Report ==="))

    # Save patient manifest immediately
    (output_dir / "patient_manifest.json").write_text(
        json.dumps(patient_manifest, indent=2, default=str), encoding="utf-8"
    )

    # Resolve paths for selected training set, validation set, and test set
    resolved_train_df, train_path_report = resolver.resolve_dataframe(
        selected_train_df,
        split_name="train",
        strict=args.strict_paths,
        filter_missing=args.filter_missing,
    )
    resolved_val_df, val_path_report = resolver.resolve_dataframe(
        val_df,
        split_name="val",
        strict=args.strict_paths,
        filter_missing=args.filter_missing,
    )
    resolved_test_df, test_path_report = resolver.resolve_dataframe(
        test_df,
        split_name="test",
        strict=args.strict_paths,
        filter_missing=args.filter_missing,
    )

    path_reports = {
        "client": "PadChest",
        "frontal_view_filter": frontal_reports,
        "audit_official_splits": {
            "train": train_audit,
            "val": val_audit,
            "test": test_audit,
        },
        "resolved_training_subset": train_path_report,
        "resolved_validation_set": val_path_report,
        "resolved_test_set": test_path_report,
    }
    (output_dir / "path_resolution_report.json").write_text(
        json.dumps(path_reports, indent=2, default=str), encoding="utf-8"
    )

    if len(resolved_train_df) == 0 and not args.dry_run:
        raise RuntimeError(
            "No resolved PadChest training images available on disk! "
            "Please check that PadChest images have been downloaded/extracted, "
            "provide --image_dir, or use --sample_from_accessible to sample among locally available images."
        )

    # Consolidate full configuration
    full_config = {
        "run_name": run_name,
        "task": "binary_classification",
        "target_disease": "Pleural Effusion",
        "dataset_name": "PadChest",
        "dataset": "PadChest",
        "client_node": "client_03_PadChest",
        "model_architecture": "resnet50",
        "classification_head": "single_logit_binary",
        "num_classes": 1,
        "pretrained_backbone": args.pretrained,
        "dropout_rate": args.dropout,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "metric_to_optimize": args.metric_to_optimize,
        "pos_weight": args.pos_weight,
        "seed": args.seed,
        "device": str(device),
        "use_amp": args.use_amp,
        "pilot": args.pilot,
        "frontal_only": args.frontal_only,
        "include_ap_horizontal": args.include_ap_horizontal,
        "patient_limit": args.patient_limit,
        "patient_start": args.patient_start,
        "patient_end": args.patient_end,
        "sample_from_accessible": args.sample_from_accessible,
        "total_official_train_images": len(train_df),
        "total_official_val_images": len(val_df),
        "total_official_test_images": len(test_df),
        "resolved_train_images": len(resolved_train_df),
        "resolved_val_images": len(resolved_val_df),
        "resolved_test_images": len(resolved_test_df),
        "test_is_complete": test_audit["is_complete"],
        "test_coverage_percentage": test_audit["accessible_fraction"] * 100.0,
    }

    (output_dir / "config.json").write_text(
        json.dumps(full_config, indent=2, default=str), encoding="utf-8"
    )

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] --dry_run flag detected.")
        print(f"  Configuration, patient manifest, and path resolution reports saved to: {output_dir}")
        print("  Model training and dataloading were NOT executed.")
        return

    # 4. Construct DataLoaders and Model Architecture
    print("\n[4/5] Building DataLoaders and ResNet-50 binary architecture...")
    loaders = create_padchest_dataloaders(
        train_df=resolved_train_df,
        val_df=resolved_val_df,
        test_df=resolved_test_df,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_resnet50_binary(
        pretrained=args.pretrained,
        dropout_rate=args.dropout,
        device=device,
    )

    # 5. Initialize Trainer and Execute Training Loop
    print("\n[5/5] Initializing ClientTrainer for PadChest...")
    trainer = ClientTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders.get("test"),
        config=full_config,
        output_dir=output_dir,
        device=device,
    )

    results = trainer.train()

    print("\n======================================================================")
    print("  PadChest Training Pipeline Execution Finished Successfully")
    print("======================================================================")
    print(f"Artifacts and Checkpoints Directory: {output_dir}")
    print(f"Best Validation Epoch:               {results.get('best_epoch')}")
    print(f"Total Training Time:                 {results.get('total_training_time', 0):.1f}s")
    print("Saved files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
