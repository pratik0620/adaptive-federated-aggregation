"""End-to-end synthetic federated integration tests comparing FedAvg and Capped FedAvg.

Uses three synthetic clients with imbalanced sample counts and heterogeneous
covariate/label distributions, identical initial weights, identical seeds, and
identical local training settings, storing all outputs exclusively in temporary
directories.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import FederatedClient
from src.federated.config import FederatedConfig
from src.federated.server import FederatedServer
from src.models.resnet import load_checkpoint


class SyntheticBatchNormNet(nn.Module):
    """Lightweight binary classifier with BatchNorm1d buffers for synthetic FL integration."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.fc1(x)))
        return self.fc2(x)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_both_classes(y: torch.Tensor) -> torch.Tensor:
    y = y.clone()
    if int((y == 1).sum().item()) == 0:
        y[0] = 1.0
    if int((y == 0).sum().item()) == 0:
        y[-1] = 0.0
    return y


def _create_heterogeneous_three_client_datasets(
    dim: int = 16,
    seed: int = 42,
) -> dict[str, tuple[TensorDataset, TensorDataset]]:
    """Build three synthetic client datasets with sample-size and distribution skew."""
    rng = torch.Generator().manual_seed(seed)
    base_w = torch.randn(dim, 1, generator=rng)

    # Heterogeneous client regimes (different sizes, covariate shifts, prevalence offsets, and feature perturbations)
    specs = {
        "client_01_dominant": {
            "n_train": 240,
            "n_val": 48,
            "mean_shift": 0.9,
            "scale": 1.2,
            "logit_bias": 3.5,
            "w_noise_std": 0.15,
            "client_seed": seed + 101,
        },
        "client_02_medium": {
            "n_train": 48,
            "n_val": 48,
            "mean_shift": 0.0,
            "scale": 1.0,
            "logit_bias": 0.0,
            "w_noise_std": 0.35,
            "client_seed": seed + 202,
        },
        "client_03_minority": {
            "n_train": 32,
            "n_val": 48,
            "mean_shift": -0.9,
            "scale": 0.85,
            "logit_bias": -3.5,
            "w_noise_std": 0.50,
            "client_seed": seed + 303,
        },
    }

    datasets: dict[str, tuple[TensorDataset, TensorDataset]] = {}
    for cid, spec in specs.items():
        c_rng = torch.Generator().manual_seed(spec["client_seed"])
        local_w = base_w + spec["w_noise_std"] * torch.randn(dim, 1, generator=c_rng)

        tx_base = spec["scale"] * torch.randn(spec["n_train"], dim, generator=c_rng)
        tx = spec["mean_shift"] + tx_base
        t_logits = (tx_base @ local_w).squeeze(-1) + spec["logit_bias"]
        ty = _ensure_both_classes((t_logits > 0.0).float())

        vx_base = spec["scale"] * torch.randn(spec["n_val"], dim, generator=c_rng)
        vx = spec["mean_shift"] + vx_base
        v_logits = (vx_base @ local_w).squeeze(-1) + spec["logit_bias"]
        vy = _ensure_both_classes((v_logits > 0.0).float())

        datasets[cid] = (TensorDataset(tx, ty), TensorDataset(vx, vy))

    return datasets


def _run_synthetic_federation(
    *,
    aggregation_method: str,
    cap_value: float | None,
    datasets: dict[str, tuple[TensorDataset, TensorDataset]],
    initial_model: nn.Module,
    output_dir: Path,
    seed: int = 42,
    num_rounds: int = 4,
    local_epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.02,
    optimizer: str = "adamw",
    weight_decay: float = 1e-4,
) -> tuple[FederatedServer, list[dict[str, Any]], dict[int, dict[str, dict[str, torch.Tensor]]]]:
    """Execute one deterministic synthetic FL run and record per-round broadcast states."""
    _set_all_seeds(seed)
    broadcast_records: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    round_state = {"current_round": 0}

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
            metadata={
                "unique_patient_count": len(train_ds),
                "is_synthetic": True,
                "dataset_name": f"Synthetic_{cid}",
            },
        )

        # Spy on set_parameters (first call per round = Step 1 broadcast) and fit_round entry
        orig_set_params = client.set_parameters
        orig_fit_round = client.fit_round

        def _make_spies(c_id: str, c_obj: FederatedClient, orig_set: Any, orig_fit: Any) -> tuple[Any, Any]:
            def _spy_set_params(state_dict: dict[str, torch.Tensor]) -> None:
                r_idx = len(server.round_history) + 1
                round_map = broadcast_records.setdefault(r_idx, {})
                if c_id not in round_map:
                    round_map[c_id] = {
                        k: v.detach().cpu().clone() for k, v in state_dict.items()
                    }
                orig_set(state_dict)

            def _spy_fit_round(*args: Any, **kwargs: Any) -> dict[str, Any]:
                r_idx = len(server.round_history) + 1
                _assert_state_dicts_equal(c_obj.get_parameters(), broadcast_records[r_idx][c_id])
                return orig_fit(*args, **kwargs)

            return _spy_set_params, _spy_fit_round

        spy_set, spy_fit = _make_spies(cid, client, orig_set_params, orig_fit_round)
        client.set_parameters = spy_set  # type: ignore[method-assign]
        client.fit_round = spy_fit  # type: ignore[method-assign]
        clients.append(client)

    participating_clients = list(datasets.keys())
    config = FederatedConfig(
        experiment_name=f"synthetic_e2e_{aggregation_method}",
        initialization_id="synthetic_shared_seed42",
        aggregation_method=aggregation_method,
        cap_value=cap_value,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        seed=seed,
        device="cpu",
        participating_clients=participating_clients,
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
    return server, history, broadcast_records


def _assert_state_dicts_equal(sd_a: dict[str, torch.Tensor], sd_b: dict[str, torch.Tensor]) -> None:
    assert set(sd_a.keys()) == set(sd_b.keys())
    for key in sd_a:
        assert sd_a[key].dtype == sd_b[key].dtype, f"Dtype mismatch for {key}"
        assert torch.equal(sd_a[key], sd_b[key]), f"Tensor mismatch for {key}"


def test_three_client_heterogeneous_synthetic_fedavg_and_capped_fedavg_end_to_end(tmp_path):
    """Verify full end-to-end 3-client synthetic FedAvg vs. Capped FedAvg integration."""
    seed = 42
    _set_all_seeds(seed)
    initial_model = SyntheticBatchNormNet(in_features=16, hidden_dim=16)
    initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
    datasets = _create_heterogeneous_three_client_datasets(dim=16, seed=seed)

    # Confirm non-IID heterogeneity across the three client datasets before running FL
    client_names = list(datasets.keys())
    means = [datasets[c][0].tensors[0].mean().item() for c in client_names]
    prevs = [datasets[c][0].tensors[1].mean().item() for c in client_names]
    sizes = [len(datasets[c][0]) for c in client_names]
    assert sizes == [240, 48, 32]
    assert means[0] > means[1] > means[2]
    assert prevs[0] > prevs[1] > prevs[2]

    common_kwargs = dict(
        datasets=datasets,
        initial_model=initial_model,
        seed=seed,
        num_rounds=4,
        local_epochs=2,
        batch_size=16,
        lr=0.02,
        optimizer="adamw",
        weight_decay=1e-4,
    )

    fedavg_dir = tmp_path / "fedavg_run"
    capped_dir = tmp_path / "capped_fedavg_run"

    server_fedavg, hist_fedavg, bcast_fedavg = _run_synthetic_federation(
        aggregation_method="fedavg",
        cap_value=None,
        output_dir=fedavg_dir,
        **common_kwargs,
    )
    server_capped, hist_capped, bcast_capped = _run_synthetic_federation(
        aggregation_method="capped_fedavg",
        cap_value=0.40,
        output_dir=capped_dir,
        **common_kwargs,
    )

    # 1. Verify both methods started from identical initial parameters and config settings
    assert server_fedavg.config.seed == server_capped.config.seed == seed
    assert server_fedavg.config.num_rounds == server_capped.config.num_rounds == 4
    assert server_fedavg.config.local_epochs == server_capped.config.local_epochs == 2
    assert server_fedavg.config.batch_size == server_capped.config.batch_size == 16
    assert server_fedavg.config.lr == server_capped.config.lr == pytest.approx(0.02)
    assert server_fedavg.config.optimizer == server_capped.config.optimizer == "adamw"
    assert server_fedavg.config.participating_clients == server_capped.config.participating_clients == client_names

    for cid in client_names:
        _assert_state_dicts_equal(bcast_fedavg[1][cid], initial_state)
        _assert_state_dicts_equal(bcast_capped[1][cid], initial_state)

    # 2. Verify all 4 rounds completed for both algorithms
    assert len(hist_fedavg) == 4
    assert len(hist_capped) == 4

    expected_fedavg_weights = {
        "client_01_dominant": 240 / 320,
        "client_02_medium": 48 / 320,
        "client_03_minority": 32 / 320,
    }
    # Pre-normalization clipping at c=0.40: [0.40, 0.15, 0.10] / 0.65
    expected_capped_weights = {
        "client_01_dominant": 0.40 / 0.65,
        "client_02_medium": 0.15 / 0.65,
        "client_03_minority": 0.10 / 0.65,
    }

    for r_idx in range(1, 5):
        # All clients receive identical global parameters at the start of every round
        for bcast_map in (bcast_fedavg, bcast_capped):
            ref_bcast = bcast_map[r_idx][client_names[0]]
            for cid in client_names[1:]:
                _assert_state_dicts_equal(ref_bcast, bcast_map[r_idx][cid])

        rf = hist_fedavg[r_idx - 1]
        rc = hist_capped[r_idx - 1]

        assert rf["round"] == r_idx
        assert rc["round"] == r_idx
        assert rf["sample_counts"] == {"client_01_dominant": 240, "client_02_medium": 48, "client_03_minority": 32}
        assert rc["sample_counts"] == {"client_01_dominant": 240, "client_02_medium": 48, "client_03_minority": 32}

        # Aggregation weights sum to 1 and match exact mathematical formulations
        assert sum(rf["aggregation_weights"].values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(rc["aggregation_weights"].values()) == pytest.approx(1.0, abs=1e-12)
        for cid in client_names:
            assert rf["aggregation_weights"][cid] == pytest.approx(expected_fedavg_weights[cid], rel=1e-9)
            assert rc["aggregation_weights"][cid] == pytest.approx(expected_capped_weights[cid], rel=1e-9)

        # Per-client train and validation metrics recorded with finite values
        assert rf["mean_val_auroc"] is not None and math.isfinite(rf["mean_val_auroc"])
        assert rc["mean_val_auroc"] is not None and math.isfinite(rc["mean_val_auroc"])
        for cid in client_names:
            for rec in (rf, rc):
                assert math.isfinite(rec["train_metrics"][cid]["loss"])
                val_m = rec["val_metrics"][cid]
                for metric_key in ("auroc", "pr_auc", "sensitivity", "specificity", "f1", "balanced_accuracy", "loss"):
                    assert val_m[metric_key] is not None
                    assert math.isfinite(val_m[metric_key])

    # 3. Verify valid updated parameters, BatchNorm buffer dtypes, and distinct FedAvg vs Capped solutions
    final_fedavg = server_fedavg.get_global_parameters()
    final_capped = server_capped.get_global_parameters()

    assert final_fedavg["bn1.num_batches_tracked"].dtype == torch.int64
    assert final_capped["bn1.num_batches_tracked"].dtype == torch.int64
    assert int(final_fedavg["bn1.num_batches_tracked"].item()) > 0
    assert int(final_capped["bn1.num_batches_tracked"].item()) > 0

    for key in initial_state:
        for state in (final_fedavg, final_capped):
            if state[key].is_floating_point():
                assert torch.isfinite(state[key]).all(), f"Non-finite parameter in {key}"

    assert any(not torch.equal(initial_state[k], final_fedavg[k]) for k in initial_state)
    assert any(not torch.equal(initial_state[k], final_capped[k]) for k in initial_state)
    assert any(not torch.allclose(final_fedavg[k].float(), final_capped[k].float()) for k in initial_state)

    # 4. Verify checkpoints, metadata, CSV, and JSON exports inside tmp_path
    for run_dir, expected_method, expected_cap in [
        (fedavg_dir, "fedavg", None),
        (capped_dir, "capped_fedavg", 0.40),
    ]:
        for fname in (
            "federated_config.json",
            "run_metadata.json",
            "round_metrics.json",
            "history.csv",
            "best_global_checkpoint.pt",
            "final_global_checkpoint.pt",
            "global_checkpoint_round_01.pt",
            "global_checkpoint_round_02.pt",
            "global_checkpoint_round_03.pt",
            "global_checkpoint_round_04.pt",
        ):
            assert (run_dir / fname).is_file(), f"Missing output file {fname} in {run_dir}"

        meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        assert meta["aggregation_method"] == expected_method
        assert meta["cap_value"] == expected_cap
        assert meta["total_rounds_completed"] == 4
        assert meta["best_round"] in (1, 2, 3, 4)
        assert meta["best_metric_value"] is not None

        hist_df = pd.read_csv(run_dir / "history.csv")
        assert len(hist_df) == 4
        assert set(["round", "mean_val_auroc"]).issubset(hist_df.columns)

        reloaded = SyntheticBatchNormNet(in_features=16, hidden_dim=16)
        ckpt = load_checkpoint(run_dir / "final_global_checkpoint.pt", model=reloaded)
        assert ckpt["epoch"] == 4
        probe_x = torch.randn(8, 16)
        reloaded.eval()
        with torch.no_grad():
            expected_server = server_fedavg if expected_method == "fedavg" else server_capped
            expected_server.global_model.eval()
            assert torch.allclose(expected_server.global_model(probe_x), reloaded(probe_x))

    # 5. Verify exact reproducibility across repeated runs for both algorithms
    server_fedavg_repeat, hist_fedavg_repeat, _ = _run_synthetic_federation(
        aggregation_method="fedavg",
        cap_value=None,
        output_dir=tmp_path / "fedavg_repeat",
        **common_kwargs,
    )
    server_capped_repeat, hist_capped_repeat, _ = _run_synthetic_federation(
        aggregation_method="capped_fedavg",
        cap_value=0.40,
        output_dir=tmp_path / "capped_fedavg_repeat",
        **common_kwargs,
    )

    _assert_state_dicts_equal(final_fedavg, server_fedavg_repeat.get_global_parameters())
    _assert_state_dicts_equal(final_capped, server_capped_repeat.get_global_parameters())
    for r_idx in range(4):
        assert hist_fedavg[r_idx]["mean_val_auroc"] == pytest.approx(
            hist_fedavg_repeat[r_idx]["mean_val_auroc"], abs=1e-12
        )
        assert hist_capped[r_idx]["mean_val_auroc"] == pytest.approx(
            hist_capped_repeat[r_idx]["mean_val_auroc"], abs=1e-12
        )
