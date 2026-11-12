"""Regression tests for federated learning correctness fixes."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.federated.adapters import build_padchest_client
from src.federated.client import FederatedClient, SyntheticFederatedClient
from src.federated.config import (
    FederatedConfig,
    create_federated_arg_parser,
    resolve_federated_config,
)
from src.federated.preflight import (
    audit_client_readiness,
    check_dependencies,
    check_resnet50_pretrained_cached,
    run_federated_preflight,
)
from src.federated.server import FederatedServer
from src.models.resnet import build_resnet50_binary


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)


# ---------------------------------------------------------------------------
# Fix 1: CLI configuration precedence
# ---------------------------------------------------------------------------
def test_cli_preserves_json_config_when_flags_omitted(tmp_path):
    """JSON configuration values must remain intact when CLI flags are not supplied."""
    cfg_path = tmp_path / "exp_config.json"
    cfg_payload = {
        "experiment_name": "json_capped_run",
        "initialization_id": "custom_init_01",
        "aggregation_method": "capped_fedavg",
        "cap_value": 0.55,
        "num_rounds": 12,
        "local_epochs": 3,
        "batch_size": 16,
        "lr": 5e-4,
        "optimizer": "sgd",
        "seed": 777,
        "device": "cpu",
        "output_dir": str(tmp_path / "custom_outputs"),
        "patient_limit": 25,
        "participating_clients": ["client_01_NIH", "client_03_PadChest"],
    }
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")

    parser = create_federated_arg_parser()
    args = parser.parse_args(["--config", str(cfg_path)])
    resolved = resolve_federated_config(args)

    assert resolved.experiment_name == "json_capped_run"
    assert resolved.initialization_id == "custom_init_01"
    assert resolved.aggregation_method == "capped_fedavg"
    assert resolved.cap_value == pytest.approx(0.55)
    assert resolved.num_rounds == 12
    assert resolved.local_epochs == 3
    assert resolved.batch_size == 16
    assert resolved.lr == pytest.approx(5e-4)
    assert resolved.optimizer == "sgd"
    assert resolved.seed == 777
    assert resolved.output_dir == str(tmp_path / "custom_outputs")
    assert resolved.patient_limit == 25
    assert resolved.participating_clients == ["client_01_NIH", "client_03_PadChest"]


def test_cli_explicit_flags_override_json_config(tmp_path):
    """Explicit CLI flags must override only the corresponding JSON configuration keys."""
    cfg_path = tmp_path / "exp_config.json"
    cfg_payload = {
        "experiment_name": "json_capped_run",
        "aggregation_method": "capped_fedavg",
        "cap_value": 0.55,
        "num_rounds": 12,
        "local_epochs": 3,
        "batch_size": 16,
        "lr": 5e-4,
        "optimizer": "sgd",
        "seed": 777,
        "patient_limit": 25,
        "participating_clients": ["client_01_NIH", "client_03_PadChest"],
    }
    cfg_path.write_text(json.dumps(cfg_payload), encoding="utf-8")

    parser = create_federated_arg_parser()
    args = parser.parse_args([
        "--config", str(cfg_path),
        "--rounds", "2",
        "--lr", "0.01",
        "--seed", "123",
        "--aggregation_method", "fedavg",
        "--clients", "client_02_CheXpert",
    ])
    resolved = resolve_federated_config(args)

    # Overridden by CLI
    assert resolved.num_rounds == 2
    assert resolved.lr == pytest.approx(0.01)
    assert resolved.seed == 123
    assert resolved.aggregation_method == "fedavg"
    assert resolved.participating_clients == ["client_02_CheXpert"]

    # Preserved from JSON
    assert resolved.experiment_name == "json_capped_run"
    assert resolved.cap_value == pytest.approx(0.55)
    assert resolved.local_epochs == 3
    assert resolved.batch_size == 16
    assert resolved.optimizer == "sgd"
    assert resolved.patient_limit == 25


def test_cli_falls_back_to_dataclass_defaults_when_unspecified():
    """Calling parser without --config or overrides must retain FederatedConfig defaults."""
    parser = create_federated_arg_parser()
    args = parser.parse_args([])
    resolved = resolve_federated_config(args)
    defaults = FederatedConfig()

    assert resolved == defaults


# ---------------------------------------------------------------------------
# Fix 2: PadChest validation coverage from actual filtered val DataFrame
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("available_frontal_val", [3, 6])
def test_padchest_validation_coverage_derived_from_filtered_val_df(tmp_path, available_frontal_val):
    """PadChest adapter must compute coverage against actual frontal val count, not 10000."""
    proc_dir = tmp_path / "processed"
    img_dir = tmp_path / "images"
    proc_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    train_rows = [
        {
            "image_id": f"train_{i}.png",
            "patient_id": f"p_train_{i}",
            "target": i % 2,
            "projection": "PA",
            "view_position": "",
            "image_path": f"images/train_{i}.png",
        }
        for i in range(4)
    ]
    # Add non-frontal train row that filter_padchest_frontal_views should drop
    train_rows.append(
        {
            "image_id": "train_lat.png",
            "patient_id": "p_train_lat",
            "target": 1,
            "projection": "L",
            "view_position": "LATERAL",
            "image_path": "images/train_lat.png",
        }
    )

    # Validation split has 6 frontal rows + 3 lateral rows -> official frontal val total = 6
    val_rows = [
        {
            "image_id": f"val_frontal_{i}.png",
            "patient_id": f"p_val_{i}",
            "target": i % 2,
            "projection": "PA" if i % 2 == 0 else "AP",
            "view_position": "",
            "image_path": f"images/val_frontal_{i}.png",
        }
        for i in range(6)
    ]
    val_rows.extend(
        [
            {
                "image_id": f"val_lat_{i}.png",
                "patient_id": f"p_val_lat_{i}",
                "target": 0,
                "projection": "L",
                "view_position": "",
                "image_path": f"images/val_lat_{i}.png",
            }
            for i in range(3)
        ]
    )

    test_rows = [
        {
            "image_id": f"test_{i}.png",
            "patient_id": f"p_test_{i}",
            "target": i % 2,
            "projection": "PA",
            "view_position": "",
            "image_path": f"images/test_{i}.png",
        }
        for i in range(2)
    ]

    pd.DataFrame(train_rows).to_csv(proc_dir / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(proc_dir / "val.csv", index=False)
    pd.DataFrame(test_rows).to_csv(proc_dir / "test.csv", index=False)

    for row in train_rows[:4] + test_rows:
        _write_dummy_png(img_dir / row["image_id"])
    for i in range(available_frontal_val):
        _write_dummy_png(img_dir / f"val_frontal_{i}.png")

    model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
    client = build_padchest_client(
        repo_root=tmp_path,
        processed_dir=proc_dir,
        image_dir=img_dir,
        batch_size=2,
        img_size=32,
        seed=42,
        device="cpu",
        model=model,
    )

    expected_missing = 6 - available_frontal_val
    expected_cov = available_frontal_val / 6 * 100.0
    expected_full = available_frontal_val == 6

    assert client.metadata["missing_samples_count"] == expected_missing
    assert client.metadata["coverage_percentage"] == pytest.approx(expected_cov)
    assert client.metadata["is_official_full_val"] is expected_full
    expected_status = (
        "OFFICIAL_FULL_VALIDATION_BENCHMARK"
        if expected_full
        else "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL"
    )
    assert client.metadata["evaluation_status"] == expected_status


# ---------------------------------------------------------------------------
# Fix 3: Client selection validation
# ---------------------------------------------------------------------------
def test_run_round_rejects_unknown_participating_clients(tmp_path):
    """Explicit participating_client_ids with zero registered matches must raise ValueError."""
    model = nn.Linear(4, 1)
    c1 = SyntheticFederatedClient("c1", input_dim=4, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("c2", input_dim=4, model=copy.deepcopy(model))

    server = FederatedServer(
        global_model=model,
        clients=[c1, c2],
        config=FederatedConfig(num_rounds=1, local_epochs=1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="None of the requested participating clients"):
        server.run_round(round_idx=1, participating_client_ids=["does_not_exist"])

    with pytest.raises(ValueError, match="None of the requested participating clients"):
        server.run_round(round_idx=1, participating_client_ids=[])

    assert len(server.round_history) == 0


def test_run_round_preserves_valid_partial_participation(tmp_path):
    """Valid partial participating_client_ids subset must run only those clients."""
    model = nn.Linear(4, 1)
    c1 = SyntheticFederatedClient("c1", num_train_samples=30, input_dim=4, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("c2", num_train_samples=70, input_dim=4, model=copy.deepcopy(model))
    c3 = SyntheticFederatedClient("c3", num_train_samples=100, input_dim=4, model=copy.deepcopy(model))

    server = FederatedServer(
        global_model=model,
        clients=[c1, c2, c3],
        config=FederatedConfig(num_rounds=1, local_epochs=1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )

    summary = server.run_round(round_idx=1, participating_client_ids=["c2"])
    assert summary["participating_clients"] == ["c2"]
    assert summary["aggregation_weights"] == {"c2": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# Fix 4: Sample count correctness with drop_last=True
# ---------------------------------------------------------------------------
def test_client_fit_round_sample_count_excludes_dropped_samples(tmp_path):
    """sample_count must equal actual processed samples when drop_last=True drops a partial batch."""
    torch.manual_seed(0)
    # Dataset size 10 is NOT divisible by batch_size=4 -> drop_last=True yields 2 batches = 8 samples
    x1 = torch.randn(10, 4)
    y1 = torch.tensor([0.0, 1.0] * 5)
    loader1 = DataLoader(TensorDataset(x1, y1), batch_size=4, shuffle=False, drop_last=True)

    # Dataset size 8 divisible by batch_size=4 -> 8 processed samples
    x2 = torch.randn(8, 4)
    y2 = torch.tensor([0.0, 1.0] * 4)
    loader2 = DataLoader(TensorDataset(x2, y2), batch_size=4, shuffle=False, drop_last=True)

    base_model = nn.Linear(4, 1)
    c1 = FederatedClient("c1", model=copy.deepcopy(base_model), train_loader=loader1, val_loader=None)
    c2 = FederatedClient("c2", model=copy.deepcopy(base_model), train_loader=loader2, val_loader=None)

    assert c1.get_sample_count() == 10
    res1 = c1.fit_round(local_epochs=2, lr=1e-2)
    assert res1["sample_count"] == 8

    server = FederatedServer(
        global_model=copy.deepcopy(base_model),
        clients=[c1, c2],
        config=FederatedConfig(num_rounds=1, local_epochs=1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )
    summary = server.run_round(round_idx=1)
    assert summary["sample_counts"] == {"c1": 8, "c2": 8}
    assert summary["aggregation_weights"]["c1"] == pytest.approx(0.5)
    assert summary["aggregation_weights"]["c2"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fix 5: Best checkpoint selection when validation AUROC is unavailable
# ---------------------------------------------------------------------------
def test_server_skips_best_checkpoint_when_val_auroc_missing(tmp_path):
    """Unavailable validation AUROC must not trigger -1.0 best checkpoint selection."""
    model = nn.Linear(4, 1)
    train_x = torch.randn(8, 4)
    train_y = torch.tensor([0.0, 1.0] * 4)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4, drop_last=False)

    # Single-class validation targets yield auroc=None from compute_binary_metrics
    val_x = torch.randn(4, 4)
    val_y = torch.zeros(4)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=4, drop_last=False)

    c1 = FederatedClient("c1", model=copy.deepcopy(model), train_loader=train_loader, val_loader=val_loader)
    server = FederatedServer(
        global_model=model,
        clients=[c1],
        config=FederatedConfig(num_rounds=2, local_epochs=1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )

    history = server.run_federated()
    assert len(history) == 2
    assert all(r["mean_val_auroc"] is None for r in history)
    assert not (tmp_path / "best_global_checkpoint.pt").exists()
    assert (tmp_path / "final_global_checkpoint.pt").is_file()
    assert server.best_round == -1

    meta = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["best_round"] is None
    assert meta["best_metric_value"] is None
    assert meta["checkpoint_paths"]["best_checkpoint"] is None

    final_ckpt = torch.load(tmp_path / "final_global_checkpoint.pt", map_location="cpu")
    assert final_ckpt["best_val_metric"] is None


# ---------------------------------------------------------------------------
# Fix 7: Logging correction for aggregation method
# ---------------------------------------------------------------------------
def test_server_logs_selected_aggregation_method(tmp_path, capsys):
    """Server logs must accurately state Capped FedAvg vs Standard FedAvg."""
    model = nn.Linear(4, 1)
    c1 = SyntheticFederatedClient("c1", num_train_samples=60, input_dim=4, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("c2", num_train_samples=40, input_dim=4, model=copy.deepcopy(model))

    capped_cfg = FederatedConfig(
        aggregation_method="capped_fedavg",
        cap_value=0.45,
        num_rounds=1,
        local_epochs=1,
        output_dir=str(tmp_path / "capped"),
    )
    capped_server = FederatedServer(
        global_model=copy.deepcopy(model),
        clients=[c1, c2],
        config=capped_cfg,
        output_dir=tmp_path / "capped",
    )
    capped_server.run_federated()
    capped_out = capsys.readouterr().out
    assert "Capped FedAvg (cap=0.45)" in capped_out
    assert "standard FedAvg" not in capped_out

    std_cfg = FederatedConfig(
        aggregation_method="fedavg",
        num_rounds=1,
        local_epochs=1,
        output_dir=str(tmp_path / "fedavg"),
    )
    std_server = FederatedServer(
        global_model=copy.deepcopy(model),
        clients=[copy.deepcopy(c1), copy.deepcopy(c2)],
        config=std_cfg,
        output_dir=tmp_path / "fedavg",
    )
    std_server.run_federated()
    std_out = capsys.readouterr().out
    assert "Standard FedAvg" in std_out


# ---------------------------------------------------------------------------
# Fix 8: Preflight consistency checks
# ---------------------------------------------------------------------------
def test_preflight_nih_resolves_flat_raw_directory(tmp_path):
    """Preflight NIH auditing must match adapters by checking raw/NIH directly in addition to raw/NIH/images."""
    proc_dir = tmp_path / "data" / "processed" / "NIH"
    raw_flat_dir = tmp_path / "data" / "raw" / "NIH"
    proc_dir.mkdir(parents=True)
    raw_flat_dir.mkdir(parents=True)

    train_df = pd.DataFrame([
        {"image_id": "nih_tr_1.png", "patient_id": 1, "target": 0, "image_path": "nih_tr_1.png"},
        {"image_id": "nih_tr_2.png", "patient_id": 2, "target": 1, "image_path": "nih_tr_2.png"},
    ])
    val_df = pd.DataFrame([
        {"image_id": "nih_va_1.png", "patient_id": 3, "target": 0, "image_path": "nih_va_1.png"},
    ])
    test_df = pd.DataFrame([
        {"image_id": "nih_te_1.png", "patient_id": 4, "target": 1, "image_path": "nih_te_1.png"},
    ])

    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    for fname in ["nih_tr_1.png", "nih_tr_2.png", "nih_va_1.png", "nih_te_1.png"]:
        _write_dummy_png(raw_flat_dir / fname)

    status = audit_client_readiness("client_01_NIH", repo_root=tmp_path)
    assert status.is_ready is True
    assert status.usable_train_images == 2
    assert status.usable_val_images == 1
    assert status.blocker_reason is None


def test_preflight_passes_config_seed_and_checks_pretrained_cache(tmp_path, monkeypatch):
    """Preflight must forward config.seed to patient sampling and detect missing/present ResNet50 cache."""
    captured_seeds: list[int] = []
    import src.data.patient_sampler as ps_mod

    orig_select = ps_mod.select_training_patients

    def spy_select(*args, **kwargs):
        captured_seeds.append(kwargs.get("seed"))
        return orig_select(*args, **kwargs)

    monkeypatch.setattr(ps_mod, "select_training_patients", spy_select)

    # Point torch hub cache to an empty temporary directory
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

    # Minimal valid NIH directory
    proc_dir = tmp_path / "data" / "processed" / "NIH"
    img_dir = tmp_path / "data" / "raw" / "NIH" / "images"
    proc_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)

    pd.DataFrame([
        {"image_id": "tr1.png", "patient_id": 10, "target": 0, "image_path": "tr1.png"},
        {"image_id": "tr2.png", "patient_id": 11, "target": 1, "image_path": "tr2.png"},
        {"image_id": "tr3.png", "patient_id": 12, "target": 0, "image_path": "tr3.png"},
    ]).to_csv(proc_dir / "train.csv", index=False)
    pd.DataFrame([
        {"image_id": "va1.png", "patient_id": 20, "target": 1, "image_path": "va1.png"},
    ]).to_csv(proc_dir / "val.csv", index=False)
    pd.DataFrame([
        {"image_id": "te1.png", "patient_id": 30, "target": 0, "image_path": "te1.png"},
    ]).to_csv(proc_dir / "test.csv", index=False)

    for fname in ["tr1.png", "tr2.png", "tr3.png", "va1.png", "te1.png"]:
        _write_dummy_png(img_dir / fname)

    cfg = FederatedConfig(
        participating_clients=["client_01_NIH"],
        patient_limit=2,
        seed=31415,
    )

    cached_before, expected_cache_path = check_resnet50_pretrained_cached()
    assert cached_before is False
    assert expected_cache_path is not None

    deps_strict_ok, deps_strict_details = check_dependencies(check_pretrained=True)
    assert deps_strict_ok is False
    assert deps_strict_details["resnet50_pretrained_cached"] is False

    summary_warn = run_federated_preflight(
        config=cfg,
        repo_root=tmp_path,
        is_synthetic=False,
        abort_on_failure=False,
        require_pretrained_cache=False,
    )
    assert captured_seeds == [31415]
    assert summary_warn.is_ready is True
    assert any("not cached locally" in w for w in summary_warn.warnings)

    summary_strict = run_federated_preflight(
        config=cfg,
        repo_root=tmp_path,
        is_synthetic=False,
        abort_on_failure=False,
        require_pretrained_cache=True,
    )
    assert summary_strict.is_ready is False
    assert any("resnet50_pretrained_cached" in b for b in summary_strict.blockers)

    # Populate mock cache file and re-verify
    expected_cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_cache_path.write_bytes(b"cached")
    cached_after, _ = check_resnet50_pretrained_cached()
    assert cached_after is True


def test_cli_dry_run_and_failed_preflight_do_not_create_output_dir(tmp_path):
    """run_federated.py must not create output directories during --dry_run or failed preflight."""
    repo_root = Path(__file__).resolve().parent.parent
    dry_out = tmp_path / "dry_out"
    fail_out = tmp_path / "fail_out"

    dry_res = subprocess.run(
        [
            sys.executable,
            "src/federated/run_federated.py",
            "--synthetic",
            "--dry_run",
            "--output_dir",
            str(dry_out),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_res.returncode == 0, dry_res.stderr
    assert not dry_out.exists()

    fail_res = subprocess.run(
        [
            sys.executable,
            "src/federated/run_federated.py",
            "--clients",
            "unrecognized_client",
            "--output_dir",
            str(fail_out),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail_res.returncode != 0
    assert not fail_out.exists()
