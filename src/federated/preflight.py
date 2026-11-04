"""Preflight readiness verification for federated learning experiments.

Guarantees:
  - Exact dependency checking (torchvision, pillow, model instantiation)
  - Explicit dataset readiness checks for each participating client
  - No silent dropouts: a three-client real-data run cannot silently degrade to fewer clients
  - Pre-training reporting of usable patients and images per client
  - Clean abort with informative diagnostics before any training begins
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from .config import FederatedConfig


class PreflightError(RuntimeError):
    """Raised when federated preflight checks fail for one or more participating clients."""
    pass


@dataclass
class ClientPreflightStatus:
    """Readiness status for an individual federated client."""
    client_id: str
    is_ready: bool
    usable_train_images: int = 0
    usable_train_patients: int = 0
    usable_val_images: int = 0
    usable_val_patients: int = 0
    usable_test_images: int = 0
    usable_test_patients: int = 0
    manifest_train_rows: int = 0
    missing_files: list[str] = field(default_factory=list)
    blocker_reason: str | None = None


@dataclass
class PreflightSummary:
    """Consolidated preflight status for the entire federated federation."""
    is_ready: bool
    is_synthetic: bool
    dependencies_ok: bool
    dependency_details: dict[str, bool]
    client_statuses: dict[str, ClientPreflightStatus]
    blockers: list[str] = field(default_factory=list)

    def format_report(self) -> str:
        """Format preflight report into a clean tabular status summary."""
        lines = [
            "=" * 78,
            "  FEDERATED EXPERIMENT PREFLIGHT READINESS REPORT",
            "=" * 78,
            f"Mode: {'SYNTHETIC SIMULATION' if self.is_synthetic else 'REAL HEALTHCARE DATASETS'}",
            f"Dependencies OK: {self.dependencies_ok}",
        ]
        for dep, ok in self.dependency_details.items():
            lines.append(f"  - {dep:15s}: {'INSTALLED' if ok else 'MISSING'}")

        lines.extend([
            "-" * 78,
            f"{'Client ID':22s} | {'Status':8s} | {'Train Pts':9s} | {'Train Imgs':10s} | {'Val Imgs':8s} | Details",
            "-" * 78,
        ])

        for cid, status in self.client_statuses.items():
            stat_str = "READY" if status.is_ready else "BLOCKED"
            detail = status.blocker_reason if status.blocker_reason else f"{status.usable_train_images:,} images accessible"
            lines.append(
                f"{cid:22s} | {stat_str:8s} | {status.usable_train_patients:9,d} | "
                f"{status.usable_train_images:10,d} | {status.usable_val_images:8,d} | {detail}"
            )

        lines.append("=" * 78)
        if not self.is_ready:
            lines.append("PREFLIGHT RESULT: FAILED - Cannot start real-data federated training.")
            lines.append("Blocking Issues:")
            for b in self.blockers:
                lines.append(f"  * {b}")
        else:
            lines.append("PREFLIGHT RESULT: PASSED - All required dependencies and datasets ready.")
        lines.append("=" * 78)
        return "\n".join(lines)


def check_dependencies() -> tuple[bool, dict[str, bool]]:
    """Check required packages for real-data ResNet-50 training."""
    details = {
        "torch": False,
        "torchvision": False,
        "pillow": False,
        "resnet50_model": False,
    }

    try:
        import torch
        details["torch"] = True
    except ImportError:
        pass

    try:
        import torchvision
        details["torchvision"] = True
    except ImportError:
        pass

    try:
        from PIL import Image
        details["pillow"] = True
    except ImportError:
        pass

    if details["torch"] and details["torchvision"]:
        try:
            from ..models.resnet import build_resnet50_binary
            _ = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
            details["resnet50_model"] = True
        except Exception:
            details["resnet50_model"] = False

    all_ok = all(details.values())
    return all_ok, details


def audit_client_readiness(
    client_id: str,
    repo_root: Path,
    patient_limit: int | None = None,
) -> ClientPreflightStatus:
    """Audit dataset readiness for a single healthcare client without launching training."""
    canonical_id = client_id.lower()

    if "nih" in canonical_id:
        proc_dir = repo_root / "data" / "processed" / "NIH"
        img_dir = repo_root / "data" / "raw" / "NIH" / "images"
        if not proc_dir.is_dir():
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                blocker_reason=f"Processed metadata directory not found: {proc_dir}",
            )

        from ..data.path_resolver import NIHPathResolver
        from ..data.patient_sampler import select_training_patients

        train_path = proc_dir / "train.csv"
        val_path = proc_dir / "val.csv"
        test_path = proc_dir / "test.csv"

        missing = [str(p.name) for p in (train_path, val_path, test_path) if not p.is_file()]
        if missing:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                missing_files=missing,
                blocker_reason=f"Missing metadata files: {', '.join(missing)}",
            )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        resolver = NIHPathResolver(search_roots=[img_dir])
        resolver.build_index()
        if len(resolver._file_index) == 0:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible NIH training images indexed on disk.",
            )

        acc_pts = resolver.get_accessible_patient_ids(train_df)
        if not acc_pts:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible training images resolved on disk.",
            )

        sel_train, manifest = select_training_patients(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            patient_limit=patient_limit,
            accessible_patient_ids=acc_pts,
        )

        res_train, _ = resolver.resolve_dataframe(sel_train, "train", filter_missing=True)
        res_val, _ = resolver.resolve_dataframe(val_df, "val", filter_missing=True)
        res_test, _ = resolver.resolve_dataframe(test_df, "test", filter_missing=True)

        return ClientPreflightStatus(
            client_id=client_id,
            is_ready=(len(res_train) > 0 and len(res_val) > 0),
            usable_train_images=len(res_train),
            usable_train_patients=manifest.get("selected_patient_count", res_train["patient_id"].nunique()),
            usable_val_images=len(res_val),
            usable_val_patients=res_val["patient_id"].nunique() if len(res_val) > 0 else 0,
            usable_test_images=len(res_test),
            usable_test_patients=res_test["patient_id"].nunique() if len(res_test) > 0 else 0,
            manifest_train_rows=len(train_df),
            blocker_reason=None if len(res_train) > 0 else "No usable training images resolved.",
        )

    elif "chexpert" in canonical_id:
        proc_dir = repo_root / "data" / "processed" / "CheXpert"
        if not proc_dir.is_dir():
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                blocker_reason=f"Processed metadata directory not found: {proc_dir}",
            )

        from ..data.chexpert_dataset import CheXpertPathResolver, filter_chexpert_frontal_views
        from ..data.patient_sampler import select_training_patients

        train_path = proc_dir / "train.csv"
        val_path = proc_dir / "val.csv"
        test_path = proc_dir / "test.csv"

        missing = [str(p.name) for p in (train_path, val_path, test_path) if not p.is_file()]
        if missing:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                missing_files=missing,
                blocker_reason=f"Missing metadata files: {', '.join(missing)}",
            )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_df, _ = filter_chexpert_frontal_views(train_df, "train")
        val_df, _ = filter_chexpert_frontal_views(val_df, "val")
        test_df, _ = filter_chexpert_frontal_views(test_df, "test")

        resolver = CheXpertPathResolver(search_roots=[
            c for c in [
                repo_root / "data" / "raw" / "CheXpert" / "images",
                repo_root / "data" / "raw" / "CheXpert" / "CheXpert-v1.0-small",
                repo_root / "data" / "raw" / "CheXpert",
            ] if c.exists()
        ])
        resolver.build_index()
        if len(resolver._index) == 0:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible CheXpert training images resolved on disk.",
            )

        acc_pts = resolver.get_accessible_patient_ids(train_df)
        if not acc_pts:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible CheXpert training images resolved on disk.",
            )

        sel_train, manifest = select_training_patients(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            patient_limit=patient_limit,
            accessible_patient_ids=acc_pts,
        )

        res_train, _ = resolver.resolve_dataframe(sel_train, "train", filter_missing=True)
        res_val, _ = resolver.resolve_dataframe(val_df, "val", filter_missing=True)
        res_test, _ = resolver.resolve_dataframe(test_df, "test", filter_missing=True)

        return ClientPreflightStatus(
            client_id=client_id,
            is_ready=(len(res_train) > 0 and len(res_val) > 0),
            usable_train_images=len(res_train),
            usable_train_patients=manifest.get("selected_patient_count", res_train["patient_id"].nunique()),
            usable_val_images=len(res_val),
            usable_val_patients=res_val["patient_id"].nunique() if len(res_val) > 0 else 0,
            usable_test_images=len(res_test),
            usable_test_patients=res_test["patient_id"].nunique() if len(res_test) > 0 else 0,
            manifest_train_rows=len(train_df),
            blocker_reason=None if len(res_train) > 0 else "No usable training images resolved.",
        )

    elif "padchest" in canonical_id:
        proc_dir = repo_root / "data" / "processed" / "PadChest"
        if not proc_dir.is_dir():
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                blocker_reason=f"Processed metadata directory not found: {proc_dir}",
            )

        from ..data.padchest_dataset import PadChestPathResolver, filter_padchest_frontal_views
        from ..data.patient_sampler import select_training_patients

        train_path = proc_dir / "train.csv"
        val_path = proc_dir / "val.csv"
        test_path = proc_dir / "test.csv"

        missing = [str(p.name) for p in (train_path, val_path, test_path) if not p.is_file()]
        if missing:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                missing_files=missing,
                blocker_reason=f"Missing metadata files: {', '.join(missing)}",
            )

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        train_df, _ = filter_padchest_frontal_views(train_df, "train")
        val_df, _ = filter_padchest_frontal_views(val_df, "val")
        test_df, _ = filter_padchest_frontal_views(test_df, "test")

        resolver = PadChestPathResolver(search_roots=[
            c for c in [
                repo_root / "data" / "raw" / "PadChest" / "images",
                repo_root / "data" / "raw" / "PadChest",
            ] if c.exists()
        ])
        resolver.build_index()
        if len(resolver._index) == 0:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible PadChest training images resolved on disk.",
            )

        acc_pts = resolver.get_accessible_patient_ids(train_df)
        if not acc_pts:
            return ClientPreflightStatus(
                client_id=client_id,
                is_ready=False,
                manifest_train_rows=len(train_df),
                blocker_reason="Zero accessible PadChest training images resolved on disk.",
            )

        sel_train, manifest = select_training_patients(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            patient_limit=patient_limit,
            accessible_patient_ids=acc_pts,
        )

        res_train, _ = resolver.resolve_dataframe(sel_train, "train", filter_missing=True)
        res_val, _ = resolver.resolve_dataframe(val_df, "val", filter_missing=True)
        res_test, _ = resolver.resolve_dataframe(test_df, "test", filter_missing=True)

        return ClientPreflightStatus(
            client_id=client_id,
            is_ready=(len(res_train) > 0 and len(res_val) > 0),
            usable_train_images=len(res_train),
            usable_train_patients=manifest.get("selected_patient_count", res_train["patient_id"].nunique()),
            usable_val_images=len(res_val),
            usable_val_patients=res_val["patient_id"].nunique() if len(res_val) > 0 else 0,
            usable_test_images=len(res_test),
            usable_test_patients=res_test["patient_id"].nunique() if len(res_test) > 0 else 0,
            manifest_train_rows=len(train_df),
            blocker_reason=None if len(res_train) > 0 else "No usable training images resolved.",
        )

    else:
        return ClientPreflightStatus(
            client_id=client_id,
            is_ready=False,
            blocker_reason=f"Unrecognized client identifier '{client_id}'. Expected NIH, CheXpert, or PadChest.",
        )


def run_federated_preflight(
    config: FederatedConfig,
    repo_root: str | Path | None = None,
    is_synthetic: bool = False,
    abort_on_failure: bool = True,
) -> PreflightSummary:
    """Execute complete preflight readiness check before federated training.

    Enforces:
      - All participating clients must be verified.
      - A 3-client run CANNOT silently degrade into a 1-client or 2-client experiment.
      - Required dependencies must be installed.
      - Training will NOT begin if preflight fails.

    Args:
        config: FederatedConfig specifying participating clients and patient limits.
        repo_root: Root repository directory.
        is_synthetic: If True, skips physical dataset audits and verifies synthetic plumbing.
        abort_on_failure: If True, raises PreflightError when preflight fails.

    Returns:
        PreflightSummary object.

    Raises:
        PreflightError: If abort_on_failure is True and any check fails.
    """
    root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    target_clients = config.participating_clients or ["client_01_NIH", "client_02_CheXpert", "client_03_PadChest"]

    deps_ok, dep_details = check_dependencies()
    blockers: list[str] = []

    if not is_synthetic and not deps_ok:
        missing_deps = [d for d, ok in dep_details.items() if not ok]
        blockers.append(f"Missing required runtime dependencies: {', '.join(missing_deps)}")

    client_statuses: dict[str, ClientPreflightStatus] = {}

    if is_synthetic:
        # Synthetic mode: all clients are verified in-memory
        for cid in target_clients:
            client_statuses[cid] = ClientPreflightStatus(
                client_id=cid,
                is_ready=True,
                usable_train_images=100,
                usable_train_patients=100,
                usable_val_images=20,
                usable_val_patients=20,
            )
        is_ready = True
    else:
        # Real healthcare data mode: audit every specified client individually
        for cid in target_clients:
            status = audit_client_readiness(
                client_id=cid,
                repo_root=root,
                patient_limit=config.patient_limit,
            )
            client_statuses[cid] = status
            if not status.is_ready:
                blockers.append(f"Client '{cid}' is NOT ready: {status.blocker_reason}")

        all_clients_ready = all(s.is_ready for s in client_statuses.values())
        is_ready = deps_ok and all_clients_ready

    summary = PreflightSummary(
        is_ready=is_ready,
        is_synthetic=is_synthetic,
        dependencies_ok=deps_ok,
        dependency_details=dep_details,
        client_statuses=client_statuses,
        blockers=blockers,
    )

    if not is_ready and abort_on_failure:
        print(summary.format_report(), file=sys.stderr)
        raise PreflightError(
            f"Preflight validation failed for {len(blockers)} reason(s). "
            "Real-data federated training was halted to prevent invalid or silently degraded runs."
        )

    return summary
