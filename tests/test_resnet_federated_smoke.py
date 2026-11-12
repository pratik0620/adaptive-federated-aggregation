"""CPU-compatible ResNet50Binary federated smoke test and demo config parity checks."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_synthetic_demonstration import (
    LinearClassifier,
    create_synthetic_datasets,
    run_experiment,
)
from src.federated.client import FederatedClient
from src.federated.config import FederatedConfig
from src.federated.server import FederatedServer
from src.models.resnet import build_resnet50_binary, load_checkpoint


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _create_synthetic_rgb_client_datasets(
    seed: int = 42,
) -> dict[str, tuple[TensorDataset, TensorDataset]]:
    """Generate small 224x224 RGB datasets with distinct sizes and distributions."""
    specs = [
        ("client_01_nih_sim", 12, 4, 0.8, 1.0, 9, 3, seed + 1),
        ("client_02_chexpert_sim", 8, 4, 0.0, 0.8, 4, 2, seed + 2),
        ("client_03_padchest_sim", 4, 4, -0.8, 0.6, 1, 1, seed + 3),
    ]
    datasets: dict[str, tuple[TensorDataset, TensorDataset]] = {}
    for cid, n_train, n_val, shift, scale, n_pos_train, n_pos_val, c_seed in specs:
        rng = torch.Generator().manual_seed(c_seed)
        tx = shift + scale * torch.randn(n_train, 3, 224, 224, generator=rng)
        ty = torch.cat([torch.ones(n_pos_train), torch.zeros(n_train - n_pos_train)])
        # Offset positive examples slightly so signal is learnable within 1 epoch
        tx[:n_pos_train] += 0.25

        vx = shift + scale * torch.randn(n_val, 3, 224, 224, generator=rng)
        vy = torch.cat([torch.ones(n_pos_val), torch.zeros(n_val - n_pos_val)])
        vx[:n_pos_val] += 0.25

        datasets[cid] = (TensorDataset(tx, ty), TensorDataset(vx, vy))
    return datasets


def _run_resnet50_federated_round(
    *,
    aggregation_method: str,
    cap_value: float | None,
    datasets: dict[str, tuple[TensorDataset, TensorDataset]],
    initial_model: torch.nn.Module,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 4,
    lr: float = 1e-3,
) -> tuple[FederatedServer, list[dict[str, Any]], dict[str, dict[str, torch.Tensor]]]:
    _set_all_seeds(seed)
    broadcasts: dict[str, dict[str, torch.Tensor]] = {}
    clients: list[FederatedClient] = []

    for cid, (train_ds, val_ds) in datasets.items():
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        client = FederatedClient(
            client_id=cid,
            model=copy.deepcopy(initial_model),
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
            metadata={"unique_patient_count": len(train_ds), "is_synthetic": True},
        )
        orig_set = client.set_parameters

        def _make_spy(c_id: str, fn: Any) -> Any:
            def _spy_set(sd: dict[str, torch.Tensor]) -> None:
                if c_id not in broadcasts:
                    broadcasts[c_id] = {k: v.detach().cpu().clone() for k, v in sd.items()}
                fn(sd)

            return _spy_set

        client.set_parameters = _make_spy(cid, orig_set)  # type: ignore[method-assign]
        clients.append(client)

    config = FederatedConfig(
        experiment_name=f"resnet50_smoke_{aggregation_method}",
        initialization_id="resnet50_unpretrained_seed42",
        aggregation_method=aggregation_method,
        cap_value=cap_value,
        num_rounds=1,
        local_epochs=1,
        batch_size=batch_size,
        lr=lr,
        optimizer="adamw",
        weight_decay=1e-4,
        seed=seed,
        device="cpu",
        participating_clients=list(datasets.keys()),
        output_dir=str(output_dir),
    )

    server = FederatedServer(
        global_model=copy.deepcopy(initial_model),
        clients=clients,
        config=config,
        device="cpu",
        output_dir=output_dir,
    )
    history = server.run_federated()
    return server, history, broadcasts


def test_resnet50_federated_smoke_fedavg_and_capped_fedavg(tmp_path):
    """Execute 1 complete federated round with ResNet50Binary(pretrained=False) for both methods."""
    seed = 42
    _set_all_seeds(seed)
    initial_model = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
    initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
    assert len(initial_state) == 320

    datasets = _create_synthetic_rgb_client_datasets(seed=seed)
    client_ids = list(datasets.keys())

    fedavg_dir = tmp_path / "resnet50_fedavg"
    capped_dir = tmp_path / "resnet50_capped"

    srv_fedavg, hist_fedavg, bcast_fedavg = _run_resnet50_federated_round(
        aggregation_method="fedavg",
        cap_value=None,
        datasets=datasets,
        initial_model=initial_model,
        output_dir=fedavg_dir,
        seed=seed,
    )
    srv_capped, hist_capped, bcast_capped = _run_resnet50_federated_round(
        aggregation_method="capped_fedavg",
        cap_value=0.40,
        datasets=datasets,
        initial_model=initial_model,
        output_dir=capped_dir,
        seed=seed,
    )

    # 1. Broadcast parity: all 3 clients received identical initial ResNet-50 state dicts
    for cid in client_ids:
        for key in initial_state:
            assert torch.equal(bcast_fedavg[cid][key], initial_state[key])
            assert torch.equal(bcast_capped[cid][key], initial_state[key])

    # 2. Check sample counts and aggregation weights
    rf = hist_fedavg[0]
    rc = hist_capped[0]
    expected_counts = {
        "client_01_nih_sim": 12,
        "client_02_chexpert_sim": 8,
        "client_03_padchest_sim": 4,
    }
    assert rf["sample_counts"] == expected_counts
    assert rc["sample_counts"] == expected_counts

    assert sum(rf["aggregation_weights"].values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(rc["aggregation_weights"].values()) == pytest.approx(1.0, abs=1e-12)

    assert rf["aggregation_weights"]["client_01_nih_sim"] == pytest.approx(12 / 24)
    assert rf["aggregation_weights"]["client_02_chexpert_sim"] == pytest.approx(8 / 24)
    assert rf["aggregation_weights"]["client_03_padchest_sim"] == pytest.approx(4 / 24)

    # Pre-normalization cap=0.40 on [0.5, 1/3, 1/6]: clipped=[0.4, 1/3, 1/6], sum=0.9
    assert rc["aggregation_weights"]["client_01_nih_sim"] == pytest.approx(0.40 / 0.90)
    assert rc["aggregation_weights"]["client_02_chexpert_sim"] == pytest.approx((1 / 3) / 0.90)
    assert rc["aggregation_weights"]["client_03_padchest_sim"] == pytest.approx((1 / 6) / 0.90)

    # 3. Validate complete ResNet-50 state_dict aggregation and BatchNorm buffers
    final_fedavg = srv_fedavg.get_global_parameters()
    final_capped = srv_capped.get_global_parameters()
    assert set(final_fedavg.keys()) == set(initial_state.keys())
    assert set(final_capped.keys()) == set(initial_state.keys())

    bn_mean_keys = [k for k in initial_state if k.endswith(".running_mean")]
    bn_var_keys = [k for k in initial_state if k.endswith(".running_var")]
    bn_nbt_keys = [k for k in initial_state if k.endswith(".num_batches_tracked")]
    assert len(bn_mean_keys) == len(bn_var_keys) == len(bn_nbt_keys) == 53

    for state in (final_fedavg, final_capped):
        for k in initial_state:
            assert state[k].dtype == initial_state[k].dtype
            if state[k].is_floating_point():
                assert torch.isfinite(state[k]).all(), f"Non-finite values in {k}"
        for k in bn_var_keys:
            assert (state[k] > 0).all(), f"Non-positive running_var in {k}"
        for k in bn_nbt_keys:
            assert state[k].dtype == torch.int64
            assert int(state[k].item()) > 0

    assert any(not torch.equal(initial_state[k], final_fedavg[k]) for k in initial_state)
    assert any(not torch.equal(initial_state[k], final_capped[k]) for k in initial_state)
    assert any(
        not torch.allclose(final_fedavg[k].float(), final_capped[k].float())
        for k in initial_state
    )

    # 4. Validation metrics and checkpoint save/reload equivalence
    for run_dir, srv, rec in [
        (fedavg_dir, srv_fedavg, rf),
        (capped_dir, srv_capped, rc),
    ]:
        assert rec["mean_val_auroc"] is not None and math.isfinite(rec["mean_val_auroc"])
        for cid in client_ids:
            vm = rec["val_metrics"][cid]
            for metric_key in ("auroc", "pr_auc", "sensitivity", "specificity", "f1", "balanced_accuracy", "loss"):
                assert vm[metric_key] is not None and math.isfinite(vm[metric_key])

        for fname in (
            "federated_config.json",
            "run_metadata.json",
            "round_metrics.json",
            "history.csv",
            "global_checkpoint_round_01.pt",
            "best_global_checkpoint.pt",
            "final_global_checkpoint.pt",
        ):
            assert (run_dir / fname).is_file(), f"Missing {fname} in {run_dir}"

        reloaded = build_resnet50_binary(pretrained=False, dropout_rate=0.0, device="cpu")
        ckpt = load_checkpoint(run_dir / "final_global_checkpoint.pt", model=reloaded)
        assert ckpt["epoch"] == 1
        assert ckpt["model_architecture"] == "resnet50"

        probe = torch.randn(2, 3, 224, 224)
        srv.global_model.eval()
        reloaded.eval()
        with torch.no_grad():
            assert torch.allclose(srv.global_model(probe), reloaded(probe))


def test_synthetic_demonstration_records_actual_batch_size(tmp_path):
    """run_synthetic_demonstration.run_experiment must align DataLoader and FederatedConfig batch_size."""
    _set_all_seeds(42)
    datasets, _ = create_synthetic_datasets(dim=16, seed=42)
    model = LinearClassifier(in_features=16)

    out_dir = tmp_path / "demo_bs8"
    srv, _ = run_experiment(
        aggregation_method="fedavg",
        cap_value=None,
        datasets=datasets,
        initial_model=model,
        num_rounds=1,
        local_epochs=1,
        batch_size=8,
        lr=0.02,
        seed=42,
        output_dir=out_dir,
    )

    saved_cfg = json.loads((out_dir / "federated_config.json").read_text(encoding="utf-8"))
    saved_meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))

    for client in srv.clients.values():
        assert client.train_loader.batch_size == 8
    assert saved_cfg["batch_size"] == 8
    assert saved_meta["config"]["batch_size"] == 8
