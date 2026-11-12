#!/usr/bin/env python
"""PadChest training pipeline runner with automatic preflight validation.

Usage:
  # Official training from YAML configuration:
  python scripts/train_padchest.py --config configs/padchest.yaml

  # Partial archive training (0.zip and 1.zip):
  python scripts/train_padchest.py --config configs/padchest_partial.yaml

  # Pilot / smoke run (100 patients, 2 epochs):
  python scripts/train_padchest.py --config configs/padchest_pilot.yaml

  # Dry-run check without launching training:
  python scripts/train_padchest.py --config configs/padchest_partial.yaml --dry_run
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
from scripts.preflight_padchest import run_padchest_preflight
from src.train_padchest import main as train_padchest_main


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="PadChest Training Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/padchest.yaml",
        help="Path to YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths, manifests, and configurations without training.",
    )
    parser.add_argument(
        "--skip_preflight",
        action="store_true",
        help="Skip the pre-training dataset readiness audit.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Permit partial archive downloads (e.g. 0.zip and 1.zip).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run in smoke-test mode with relaxed image count thresholds.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, unknown_args = parse_args()

    cfg: dict = {}
    if args.config and Path(args.config).is_file():
        cfg = load_config(args.config)

    is_partial = args.allow_partial or bool(cfg.get("allow_partial", False))
    is_smoke = args.smoke_test or bool(cfg.get("pilot", False))

    # 1. Run Preflight Guard
    if not args.skip_preflight and not args.dry_run:
        print("[Runner] Executing PadChest preflight readiness audit before training...")
        is_ready, blockers, _ = run_padchest_preflight(
            config_path=args.config,
            allow_partial=is_partial,
            smoke_test=is_smoke,
            verbose=True,
        )
        if not is_ready:
            print(
                "\n[PREFLIGHT BLOCKED] Training aborted because preflight checks failed.",
                file=sys.stderr,
            )
            print("Please resolve the blocking issues listed above before starting training.", file=sys.stderr)
            sys.exit(1)
        print("[Runner] Preflight passed successfully. Proceeding to model training.\n")

    # 2. Forward to src.train_padchest
    forward_argv = [sys.argv[0], "--config", args.config]
    if args.dry_run:
        forward_argv.append("--dry_run")
    forward_argv.extend(unknown_args)
    sys.argv = forward_argv

    train_padchest_main()


if __name__ == "__main__":
    main()
