"""Comprehensive test suite for the three-client federated learning baseline (FedAvg).

Automated tests covering:
  1. Correct sample-weighted FedAvg using small known tensors.
  2. Aggregation weights summing to 1.0 and nonnegative validation.
  3. Rejection of mismatched parameter names, shapes, or NaNs.
  4. Explicit handling of floating-point parameters and integer buffers.
  5. Distribution of identical global parameters to all clients.
  6. Client updates aggregated and broadcast correctly.
  7. Optimizer state reset between rounds (zero momentum drift).
  8. Deterministic reproducibility with a fixed seed.
  9. Handling of non-participating clients in a communication round.
  10. Strict privacy: zero raw image or patient-record exchange.
  11. Synthetic end-to-end integration smoke test (2 clients, 2 rounds, save & reload).
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.aggregation import (
    aggregate_fedavg,
    compute_fedavg_weights,
    validate_client_state_dicts,
)
from src.federated.client import FederatedClient, SyntheticFederatedClient
from src.federated.config import FederatedConfig
from src.federated.server import FederatedServer
from src.models.resnet import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Test 1: Sample-weighted FedAvg using small known tensors
# ---------------------------------------------------------------------------
def test_sample_weighted_fedavg_known_tensors():
    """Verify FedAvg computes exact weighted averages with small known numerical tensors."""
    # Client 1: n_1 = 100, theta_1 = [1.0, 2.0]
    # Client 2: n_2 = 300, theta_2 = [3.0, 6.0]
    # Expected: p_1 = 0.25, p_2 = 0.75
    # theta_global = 0.25 * [1.0, 2.0] + 0.75 * [3.0, 6.0] = [2.5, 5.0]
    c1_state = {"fc.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32)}
    c2_state = {"fc.weight": torch.tensor([[3.0, 6.0]], dtype=torch.float32)}

    states = [c1_state, c2_state]
    sample_counts = [100, 300]
    cids = ["c1", "c2"]

    agg_state, weights = aggregate_fedavg(states, sample_counts, cids)

    assert weights["c1"] == pytest.approx(0.25)
    assert weights["c2"] == pytest.approx(0.75)

    expected_tensor = torch.tensor([[2.5, 5.0]], dtype=torch.float32)
    assert torch.allclose(agg_state["fc.weight"], expected_tensor)


# ---------------------------------------------------------------------------
# Test 2: Aggregation weights summing to one
# ---------------------------------------------------------------------------
def test_aggregation_weights_sum_to_one():
    """Verify aggregation weights are strictly positive and sum to 1.0 across diverse cohorts."""
    sample_counts = [137, 289, 412, 1024]
    cids = [f"node_{i}" for i in range(len(sample_counts))]

    weights = compute_fedavg_weights(sample_counts, cids)

    total_weight = sum(weights.values())
    assert total_weight == pytest.approx(1.0, abs=1e-6)

    total_samples = sum(sample_counts)
    for cid, n_k in zip(cids, sample_counts):
        assert weights[cid] > 0.0
        assert weights[cid] == pytest.approx(n_k / total_samples)

    # Test rejection of non-positive or empty sample counts
    with pytest.raises(ValueError, match="strictly positive integers"):
        compute_fedavg_weights([100, 0, 200], ["c1", "c2", "c3"])

    with pytest.raises(ValueError, match="strictly positive integers"):
        compute_fedavg_weights([100, -50], ["c1", "c2"])

    with pytest.raises(ValueError, match="empty list"):
        compute_fedavg_weights([], [])


# ---------------------------------------------------------------------------
# Test 3: Rejection of mismatched parameter names, shapes, or NaNs
# ---------------------------------------------------------------------------
def test_rejection_of_mismatched_parameters_and_nans():
    """Verify validation rejects missing keys, shape mismatches, dtype mismatches, and NaNs."""
    base_state = {
        "weight": torch.ones(4, 4),
        "bias": torch.zeros(4),
    }

    # 1. Missing parameter key
    missing_key_state = {"weight": torch.ones(4, 4)}
    with pytest.raises(KeyError, match="Missing keys"):
        aggregate_fedavg([base_state, missing_key_state], [100, 100], ["c1", "c2"])

    # 2. Extra parameter key
    extra_key_state = {
        "weight": torch.ones(4, 4),
        "bias": torch.zeros(4),
        "extra_layer": torch.ones(2),
    }
    with pytest.raises(KeyError, match="Unexpected keys"):
        aggregate_fedavg([base_state, extra_key_state], [100, 100], ["c1", "c2"])

    # 3. Shape mismatch
    mismatched_shape_state = {
        "weight": torch.ones(4, 8),
        "bias": torch.zeros(4),
    }
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_fedavg([base_state, mismatched_shape_state], [100, 100], ["c1", "c2"])

    # 4. NaN value in weights
    nan_state = {
        "weight": torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
        "bias": torch.zeros(2),
    }
    clean_state = {
        "weight": torch.ones(2, 2),
        "bias": torch.zeros(2),
    }
    with pytest.raises(ValueError, match="NaN value detected"):
        aggregate_fedavg([clean_state, nan_state], [100, 100], ["c1", "c2"])

    # 5. Non-tensor parameter (e.g., passing optimizer state dict by mistake)
    non_tensor_state = {
        "weight": [1.0, 2.0],  # list instead of torch.Tensor
        "bias": torch.zeros(2),
    }
    with pytest.raises(TypeError, match="not torch.Tensor"):
        aggregate_fedavg([clean_state, non_tensor_state], [100, 100], ["c1", "c2"])


# ---------------------------------------------------------------------------
# Test 4: Handling of floating-point parameters and integer buffers
# ---------------------------------------------------------------------------
def test_handling_floating_point_parameters_and_integer_buffers():
    """Verify FedAvg explicitly averages float tensors and rounds integer buffers correctly."""
    # Simulates ResNet BatchNorm buffers (float running_mean/var, int64 num_batches_tracked)
    c1_state = {
        "weight": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "bn.running_mean": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(10, dtype=torch.int64),
    }
    c2_state = {
        "weight": torch.tensor([3.0, 4.0], dtype=torch.float32),
        "bn.running_mean": torch.tensor([0.6, 0.8], dtype=torch.float32),
        "bn.num_batches_tracked": torch.tensor(30, dtype=torch.int64),
    }

    # Equal weighting (50% each)
    agg_state, _ = aggregate_fedavg([c1_state, c2_state], [100, 100], ["c1", "c2"])

    # Floats averaged cleanly
    assert torch.allclose(agg_state["weight"], torch.tensor([2.0, 3.0], dtype=torch.float32))
    assert torch.allclose(agg_state["bn.running_mean"], torch.tensor([0.4, 0.6], dtype=torch.float32))

    # Integer buffer explicitly rounded and kept as torch.int64
    assert agg_state["bn.num_batches_tracked"].dtype == torch.int64
    assert agg_state["bn.num_batches_tracked"].item() == 20

    # Unequal weighting test for integer buffer:
    # c1: n=100 (p=0.25, val=10), c2: n=300 (p=0.75, val=30)
    # 0.25 * 10 + 0.75 * 30 = 2.5 + 22.5 = 25
    agg_state2, _ = aggregate_fedavg([c1_state, c2_state], [100, 300], ["c1", "c2"])
    assert agg_state2["bn.num_batches_tracked"].dtype == torch.int64
    assert agg_state2["bn.num_batches_tracked"].item() == 25


# ---------------------------------------------------------------------------
# Test 5: Identical global parameters distributed to all clients
# ---------------------------------------------------------------------------
def test_identical_global_parameters_distributed_to_clients(tmp_path):
    """Verify that the server broadcasts identical global parameter values to all clients."""
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 1))

    c1 = SyntheticFederatedClient("c1", input_dim=8, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("c2", input_dim=8, model=copy.deepcopy(model))
    c3 = SyntheticFederatedClient("c3", input_dim=8, model=copy.deepcopy(model))

    server = FederatedServer(
        global_model=model,
        clients=[c1, c2, c3],
        output_dir=tmp_path,
    )

    # Scramble client parameters before broadcasting
    with torch.no_grad():
        for p in c1.model.parameters():
            p.add_(5.0)
        for p in c2.model.parameters():
            p.add_(-10.0)

    server.broadcast_parameters()

    global_params = server.get_global_parameters()
    for cid, client in server.clients.items():
        client_params = client.get_parameters()
        for k in global_params.keys():
            assert torch.equal(global_params[k], client_params[k]), (
                f"Client {cid} parameter '{k}' does not match global server parameter!"
            )


# ---------------------------------------------------------------------------
# Test 6: Client updates aggregated and broadcast correctly
# ---------------------------------------------------------------------------
def test_client_updates_aggregated_and_broadcast(tmp_path):
    """Verify that client parameter changes are aggregated into the global model."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1, bias=False)

        def forward(self, x):
            return self.linear(x)

    server_model = SimpleNet()
    with torch.no_grad():
        server_model.linear.weight.fill_(0.0)

    c1_model = SimpleNet()
    c2_model = SimpleNet()

    c1 = SyntheticFederatedClient("client_A", num_train_samples=20, input_dim=4, model=c1_model)
    c2 = SyntheticFederatedClient("client_B", num_train_samples=60, input_dim=4, model=c2_model)

    server = FederatedServer(
        global_model=server_model,
        clients=[c1, c2],
        config=FederatedConfig(num_rounds=1, local_epochs=1, lr=0.1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )

    # Initial weights are zero
    assert torch.all(server.get_global_parameters()["linear.weight"] == 0.0)

    # Run round 1
    summary = server.run_round(round_idx=1)

    updated_global_weight = server.get_global_parameters()["linear.weight"]
    # Model parameters must have updated from initial zero
    assert not torch.all(updated_global_weight == 0.0)

    # Verify clients now receive updated weights upon next broadcast
    server.broadcast_parameters()
    assert torch.allclose(c1.get_parameters()["linear.weight"], updated_global_weight)
    assert torch.allclose(c2.get_parameters()["linear.weight"], updated_global_weight)


# ---------------------------------------------------------------------------
# Test 7: Optimizer reset behavior between rounds
# ---------------------------------------------------------------------------
def test_optimizer_reset_between_rounds():
    """Verify that local optimizer state (AdamW momentum) is completely reset each round."""
    model = nn.Linear(4, 1)
    client = SyntheticFederatedClient(
        "client_reset_test",
        num_train_samples=30,
        input_dim=4,
        model=model,
        config={"optimizer": "adamw", "lr": 1e-3},
    )

    # Before training, fresh optimizer has empty state
    opt_before = client._build_fresh_optimizer()
    assert len(opt_before.state) == 0

    # Execute round 1: performs AdamW steps which populate momentum state
    res1 = client.fit_round(local_epochs=2)
    assert res1["sample_count"] > 0

    # Inside fit_round, _build_fresh_optimizer was called.
    # Verify that calling _build_fresh_optimizer again yields a clean optimizer with NO stale state
    fresh_opt = client._build_fresh_optimizer()
    assert len(fresh_opt.state) == 0

    # Ensure optimizer state does NOT attach to model state_dict
    state_dict = client.get_parameters()
    assert "optimizer_state_dict" not in state_dict
    assert "exp_avg" not in state_dict


# ---------------------------------------------------------------------------
# Test 8: Reproducibility with a fixed seed
# ---------------------------------------------------------------------------
def test_reproducibility_with_fixed_seed(tmp_path):
    """Verify identical federated training trajectories when initialized with identical seed."""
    def run_experiment(seed: int, run_dir: Path) -> dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = nn.Sequential(nn.Linear(6, 1))
        c1 = SyntheticFederatedClient("c1", num_train_samples=40, input_dim=6, model=copy.deepcopy(model), seed=seed)
        c2 = SyntheticFederatedClient("c2", num_train_samples=40, input_dim=6, model=copy.deepcopy(model), seed=seed + 1)

        server = FederatedServer(
            global_model=model,
            clients=[c1, c2],
            config=FederatedConfig(num_rounds=2, local_epochs=1, seed=seed, output_dir=str(run_dir)),
            output_dir=run_dir,
        )
        server.run_federated()
        return server.get_global_parameters()

    run1_weights = run_experiment(seed=999, run_dir=tmp_path / "run1")
    run2_weights = run_experiment(seed=999, run_dir=tmp_path / "run2")

    for k in run1_weights.keys():
        assert torch.equal(run1_weights[k], run2_weights[k]), (
            f"Non-deterministic parameter divergence detected for '{k}' across repeated seeded runs."
        )


# ---------------------------------------------------------------------------
# Test 9: Handling of client that does not participate in a round
# ---------------------------------------------------------------------------
def test_non_participating_client_handling(tmp_path):
    """Verify that a non-participating client does not contribute to FedAvg weights."""
    model = nn.Linear(4, 1)
    c1 = SyntheticFederatedClient("client_active_1", num_train_samples=100, input_dim=4, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("client_active_2", num_train_samples=200, input_dim=4, model=copy.deepcopy(model))
    c3 = SyntheticFederatedClient("client_dormant", num_train_samples=500, input_dim=4, model=copy.deepcopy(model))
    server = FederatedServer(
        global_model=model,
        clients=[c1, c2, c3],
        config=FederatedConfig(num_rounds=1, local_epochs=1, output_dir=str(tmp_path)),
        output_dir=tmp_path,
    )

    # Only client_active_1 and client_active_2 participate
    summary = server.run_round(round_idx=1, participating_client_ids=["client_active_1", "client_active_2"])

    weights = summary["aggregation_weights"]
    assert "client_active_1" in weights
    assert "client_active_2" in weights
    assert "client_dormant" not in weights

    # Total samples should be 100 + 200 = 300 (dormant client's 500 excluded)
    assert weights["client_active_1"] == pytest.approx(100 / 300)
    assert weights["client_active_2"] == pytest.approx(200 / 300)

    # Validation can still evaluate across all 3 registered clients
    assert "client_dormant" in summary["val_metrics"]


# ---------------------------------------------------------------------------
# Test 10: Privacy check - zero raw image or patient record exchange
# ---------------------------------------------------------------------------
def test_privacy_no_raw_data_exchange():
    """Verify that client interface emits only state dictionaries, sample counts, and metrics."""
    client = SyntheticFederatedClient("privacy_node", num_train_samples=25, input_dim=8)
    round_result = client.fit_round(local_epochs=1)

    # Allowed keys in client transmission
    allowed_keys = {"client_id", "parameters", "sample_count", "train_metrics"}
    assert set(round_result.keys()) == allowed_keys

    # Parameters must be pure torch.Tensor objects
    for param_name, tensor in round_result["parameters"].items():
        assert isinstance(param_name, str)
        assert isinstance(tensor, torch.Tensor)

    # Ensure no patient records, images, or metadata tables leaked
    assert "images" not in round_result
    assert "patient_id" not in round_result
    assert "dataframe" not in round_result
    assert "records" not in round_result

    # Evaluate interface also preserves privacy
    val_metrics = client.evaluate(split="val")
    assert "images" not in val_metrics
    assert "patient_records" not in val_metrics


# ---------------------------------------------------------------------------
# Test 11: Synthetic end-to-end integration smoke test
# ---------------------------------------------------------------------------
def test_synthetic_end_to_end_smoke_test():
    """Run an end-to-end synthetic federated experiment (2 clients, 2 rounds, save & reload)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "smoke_test_run"

        model = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

        initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

        c1 = SyntheticFederatedClient("client_synthetic_01", num_train_samples=60, input_dim=12, model=copy.deepcopy(model))
        c2 = SyntheticFederatedClient("client_synthetic_02", num_train_samples=40, input_dim=12, model=copy.deepcopy(model))

        config = FederatedConfig(
            num_rounds=2,
            local_epochs=2,
            batch_size=8,
            lr=1e-2,
            optimizer="sgd",
            seed=42,
            output_dir=str(output_dir),
            participating_clients=["client_synthetic_01", "client_synthetic_02"],
        )

        server = FederatedServer(
            global_model=model,
            clients=[c1, c2],
            config=config,
            output_dir=output_dir,
        )

        history = server.run_federated(num_rounds=2)

        # 1. Verify two communication rounds executed
        assert len(history) == 2
        assert history[0]["round"] == 1
        assert history[1]["round"] == 2

        # 2. Verify model parameters changed from initial state
        final_weights = server.get_global_parameters()
        has_changed = False
        for k in initial_weights.keys():
            if not torch.allclose(initial_weights[k], final_weights[k]):
                has_changed = True
                break
        assert has_changed, "Global model parameters did not change after 2 rounds of training!"

        # 3. Verify checkpoints were saved
        final_ckpt_path = output_dir / "final_global_checkpoint.pt"
        best_ckpt_path = output_dir / "best_global_checkpoint.pt"
        history_csv_path = output_dir / "history.csv"
        metrics_json_path = output_dir / "round_metrics.json"

        assert final_ckpt_path.is_file(), "Final global checkpoint not found!"
        assert best_ckpt_path.is_file(), "Best global checkpoint not found!"
        assert history_csv_path.is_file(), "history.csv not found!"
        assert metrics_json_path.is_file(), "round_metrics.json not found!"

        # 4. Reload checkpoint and verify equivalence
        reloaded_model = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )
        loaded_data = load_checkpoint(final_ckpt_path, model=reloaded_model)
        assert loaded_data["epoch"] == 2

        dummy_x = torch.randn(4, 12)
        with torch.no_grad():
            out_server = server.global_model(dummy_x)
            out_reloaded = reloaded_model(dummy_x)
            assert torch.allclose(out_server, out_reloaded), "Reloaded model outputs do not match server model!"


# ---------------------------------------------------------------------------
# Test 12: Run metadata logging and designated test evaluation stage
# ---------------------------------------------------------------------------
def test_run_metadata_and_test_evaluation_stage():
    """Verify that experiment metadata is logged and test evaluation runs only at designated stage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "logging_test_run"

        model = nn.Linear(8, 1)
        c1 = SyntheticFederatedClient("c1", input_dim=8, model=copy.deepcopy(model))

        config = FederatedConfig(
            experiment_name="test_experiment_exp01",
            initialization_id="init_seed_42_resnet",
            aggregation_method="fedavg",
            cap_value=None,
            num_rounds=1,
            local_epochs=1,
            seed=42,
            output_dir=str(output_dir),
        )

        server = FederatedServer(
            global_model=model,
            clients=[c1],
            config=config,
            output_dir=output_dir,
        )

        # Run federated rounds
        server.run_federated()

        # 1. Verify run_metadata.json exists and has all experiment tracking fields
        metadata_path = output_dir / "run_metadata.json"
        assert metadata_path.is_file(), "run_metadata.json was not created!"
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert meta["experiment_name"] == "test_experiment_exp01"
        assert meta["initialization_id"] == "init_seed_42_resnet"
        assert meta["aggregation_method"] == "fedavg"
        assert meta["seed"] == 42
        assert "best_checkpoint" in meta["checkpoint_paths"]

        # 2. Verify test split metrics are NOT in history.csv
        history_df = pd.read_csv(output_dir / "history.csv")
        for col in history_df.columns:
            assert "test" not in col.lower(), f"Test split leaked into round history column: {col}"

        # 3. Execute held-out test evaluation stage
        test_payload = server.evaluate_test_benchmark()
        assert (output_dir / "test_metrics.json").is_file()
        assert test_payload["experiment_name"] == "test_experiment_exp01"
        assert "c1" in test_payload["clients"]
