"""Unit and integration tests for Capped Federated Averaging (Capped FedAvg).

Verifies:
  1. Standard FedAvg remains completely unchanged and reproducible.
  2. Capped weights are computed correctly per the mathematical definition:
       p_k = n_k / sum(n_j)
       capped_weight_k = min(p_k, c)
       q_k = capped_weight_k / sum(capped_weight_j)
  3. Capped weights strictly sum to 1.0 and are non-negative.
  4. Dominant client is capped and smaller clients receive relatively more weight.
  5. Equal-sized clients behave sensibly (equal weights preserved).
  6. Rejection of invalid cap values (None, <=0, >1.0, NaN, Inf, non-numeric).
  7. Rejection of malformed state dictionaries (mismatched keys, shapes, dtypes, NaNs).
  8. Aggregated tensors match an exact hand-calculated numerical example.
  9. Integer model buffers (BatchNorm num_batches_tracked) are aggregated correctly without dtype errors.
  10. The FederatedServer actually selects and uses Capped FedAvg when configured.
"""

from __future__ import annotations

import copy
import math
import pytest
import torch
import torch.nn as nn

from src.federated.aggregation import (
    aggregate_capped_fedavg,
    aggregate_fedavg,
    aggregate_parameters,
    compute_capped_fedavg_weights,
    compute_fedavg_weights,
    validate_cap_value,
)
from src.federated.client import SyntheticFederatedClient
from src.federated.config import FederatedConfig
from src.federated.server import FederatedServer


# ---------------------------------------------------------------------------
# Test 1: Standard FedAvg remains unchanged
# ---------------------------------------------------------------------------
def test_standard_fedavg_remains_unchanged():
    """Verify standard FedAvg produces identical results and when cap >= max(p_k)."""
    sample_counts = [100, 200, 300]
    std_weights = compute_fedavg_weights(sample_counts)

    assert std_weights["client_0"] == pytest.approx(100 / 600)
    assert std_weights["client_1"] == pytest.approx(200 / 600)
    assert std_weights["client_2"] == pytest.approx(300 / 600)

    # When cap=1.0 or cap >= max(p_k), Capped FedAvg must equal standard FedAvg
    capped_weights = compute_capped_fedavg_weights(sample_counts, cap_value=1.0)
    for cid in std_weights:
        assert capped_weights[cid] == pytest.approx(std_weights[cid], abs=1e-7)


# ---------------------------------------------------------------------------
# Test 2: Capped weights calculation & mathematical correctness
# ---------------------------------------------------------------------------
def test_capped_weights_calculation():
    """Verify hand-derived formula with n = [100, 20, 20] and c = 0.40."""
    # N = 140
    # p = [100/140, 20/140, 20/140] = [5/7, 1/7, 1/7] ~ [0.7142857, 0.1428571, 0.1428571]
    # min(p, 0.40) = [0.40, 1/7, 1/7]
    # sum = 0.40 + 2/7 = 2/5 + 2/7 = 24/35 ~ 0.6857143
    # q_0 = (2/5) / (24/35) = 14/24 = 7/12 ~ 0.5833333
    # q_1 = (1/7) / (24/35) = 5/24 ~ 0.2083333
    # q_2 = (1/7) / (24/35) = 5/24 ~ 0.2083333
    sample_counts = [100, 20, 20]
    weights = compute_capped_fedavg_weights(sample_counts, cap_value=0.40)

    assert weights["client_0"] == pytest.approx(7 / 12, rel=1e-5)
    assert weights["client_1"] == pytest.approx(5 / 24, rel=1e-5)
    assert weights["client_2"] == pytest.approx(5 / 24, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Capped weights sum to 1.0 and are strictly positive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cap", [0.10, 0.25, 0.40, 0.50, 0.75, 1.0])
def test_capped_weights_sum_to_one(cap: float):
    """Verify unity constraint across diverse cap values and sample distributions."""
    sample_counts = [1500, 400, 100]
    weights = compute_capped_fedavg_weights(sample_counts, cap_value=cap)

    total = sum(weights.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    for cid, w in weights.items():
        assert w >= 0.0, f"Weight for {cid} was negative: {w}"


# ---------------------------------------------------------------------------
# Test 4: Dominant client capping gives smaller clients more weight
# ---------------------------------------------------------------------------
def test_dominant_client_capped_redistributes_to_minority():
    """Verify that capping client 0 strictly increases the weights of client 1 and 2."""
    sample_counts = [800, 100, 100]
    p_weights = compute_fedavg_weights(sample_counts)
    q_weights = compute_capped_fedavg_weights(sample_counts, cap_value=0.50)

    # Dominant client loses weight
    assert q_weights["client_0"] < p_weights["client_0"]
    # Smaller clients gain proportional weight
    assert q_weights["client_1"] > p_weights["client_1"]
    assert q_weights["client_2"] > p_weights["client_2"]
    # Symmetric smaller clients remain equal
    assert q_weights["client_1"] == pytest.approx(q_weights["client_2"])


# ---------------------------------------------------------------------------
# Test 5: Equal-sized clients behave sensibly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cap", [0.20, 0.3333, 0.40, 0.80])
def test_equal_sized_clients_behave_sensibly(cap: float):
    """Verify equal-sized clients always receive identical 1/K weight regardless of cap."""
    sample_counts = [100, 100, 100]
    weights = compute_capped_fedavg_weights(sample_counts, cap_value=cap)

    assert weights["client_0"] == pytest.approx(1 / 3, abs=1e-6)
    assert weights["client_1"] == pytest.approx(1 / 3, abs=1e-6)
    assert weights["client_2"] == pytest.approx(1 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Invalid cap values are strictly rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_cap", [None, 0.0, -0.2, 1.5, float("nan"), float("inf"), "0.4"])
def test_invalid_cap_values_rejected(invalid_cap):
    """Verify validate_cap_value and compute_capped_fedavg_weights reject invalid caps."""
    sample_counts = [50, 50]
    with pytest.raises((ValueError, TypeError)):
        validate_cap_value(invalid_cap)

    with pytest.raises((ValueError, TypeError)):
        compute_capped_fedavg_weights(sample_counts, cap_value=invalid_cap)


# ---------------------------------------------------------------------------
# Test 7: Malformed state dictionaries are rejected
# ---------------------------------------------------------------------------
def test_malformed_state_dicts_rejected():
    """Verify state dict validation catches mismatched keys, shapes, dtypes, and NaNs."""
    s1 = {"w": torch.tensor([1.0, 2.0])}
    s_missing = {"other": torch.tensor([1.0, 2.0])}
    s_shape = {"w": torch.tensor([[1.0], [2.0]])}
    s_dtype = {"w": torch.tensor([1, 2], dtype=torch.int32)}
    s_nan = {"w": torch.tensor([1.0, float("nan")])}

    # Key mismatch
    with pytest.raises(KeyError):
        aggregate_capped_fedavg([s1, s_missing], [10, 10], cap_value=0.5)

    # Shape mismatch
    with pytest.raises(ValueError):
        aggregate_capped_fedavg([s1, s_shape], [10, 10], cap_value=0.5)

    # Dtype mismatch
    with pytest.raises(TypeError):
        aggregate_capped_fedavg([s1, s_dtype], [10, 10], cap_value=0.5)

    # NaN in tensor
    with pytest.raises(ValueError):
        aggregate_capped_fedavg([s1, s_nan], [10, 10], cap_value=0.5)


# ---------------------------------------------------------------------------
# Test 8: Aggregated tensors match hand-calculated example
# ---------------------------------------------------------------------------
def test_aggregated_tensors_match_hand_calculation():
    """Verify tensor math against hand-derived values.

    n = [100, 20, 20], cap = 0.40 => q = [7/12, 5/24, 5/24] ~ [0.5833333, 0.2083333, 0.2083333]
    Client 0: W = [12.0, 24.0]
    Client 1: W = [0.0, 0.0]
    Client 2: W = [0.0, 0.0]

    Expected aggregated tensor:
      W_agg = (7/12) * [12.0, 24.0] + 0 + 0 = [7.0, 14.0]
    """
    s0 = {"weight": torch.tensor([12.0, 24.0], dtype=torch.float32)}
    s1 = {"weight": torch.tensor([0.0, 0.0], dtype=torch.float32)}
    s2 = {"weight": torch.tensor([0.0, 0.0], dtype=torch.float32)}

    agg_state, weights = aggregate_capped_fedavg([s0, s1, s2], [100, 20, 20], cap_value=0.40)

    expected = torch.tensor([7.0, 14.0], dtype=torch.float32)
    assert torch.allclose(agg_state["weight"], expected, atol=1e-5)
    assert weights["client_0"] == pytest.approx(7 / 12, rel=1e-5)


# ---------------------------------------------------------------------------
# Test 9: Integer buffers handled correctly without casting errors
# ---------------------------------------------------------------------------
def test_integer_buffers_aggregated_correctly():
    """Verify BatchNorm num_batches_tracked integer buffers round properly."""
    s0 = {
        "weight": torch.tensor([1.0]),
        "num_batches_tracked": torch.tensor(100, dtype=torch.int64),
    }
    s1 = {
        "weight": torch.tensor([2.0]),
        "num_batches_tracked": torch.tensor(20, dtype=torch.int64),
    }

    # Equal sample counts => equal weight 0.5
    agg_state, _ = aggregate_capped_fedavg([s0, s1], [50, 50], cap_value=0.5)

    assert agg_state["num_batches_tracked"].dtype == torch.int64
    # (100 + 20) * 0.5 = 60
    assert agg_state["num_batches_tracked"].item() == 60


# ---------------------------------------------------------------------------
# Test 10: FederatedServer uses selected aggregation method
# ---------------------------------------------------------------------------
def test_federated_server_uses_capped_fedavg(tmp_path):
    """Verify FederatedServer selects Capped FedAvg when configured."""
    model = nn.Linear(4, 1)
    c1 = SyntheticFederatedClient("c1", num_train_samples=100, input_dim=4, model=copy.deepcopy(model))
    c2 = SyntheticFederatedClient("c2", num_train_samples=20, input_dim=4, model=copy.deepcopy(model))
    c3 = SyntheticFederatedClient("c3", num_train_samples=20, input_dim=4, model=copy.deepcopy(model))

    config = FederatedConfig(
        aggregation_method="capped_fedavg",
        cap_value=0.40,
        num_rounds=1,
        local_epochs=1,
        output_dir=str(tmp_path),
    )
    server = FederatedServer(model, clients=[c1, c2, c3], config=config, output_dir=tmp_path)

    summary = server.run_round(1)
    weights = summary["aggregation_weights"]

    # In round 1, weights should be Capped FedAvg weights [7/12, 5/24, 5/24]
    assert weights["c1"] == pytest.approx(7 / 12, rel=1e-4)
    assert weights["c2"] == pytest.approx(5 / 24, rel=1e-4)
    assert weights["c3"] == pytest.approx(5 / 24, rel=1e-4)
    assert summary["aggregation_method"] == "capped_fedavg"
    assert summary["cap_value"] == 0.40


def test_aggregate_parameters_dispatcher():
    """Verify aggregate_parameters dispatches properly between fedavg and capped_fedavg."""
    s = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0])}]
    counts = [10, 10]

    # Standard
    _, w_std = aggregate_parameters(s, counts, method="fedavg")
    assert w_std["client_0"] == 0.5

    # Capped
    _, w_cap = aggregate_parameters(s, counts, method="capped_fedavg", cap_value=0.40)
    assert w_cap["client_0"] == 0.5

    # Missing cap_value raises
    with pytest.raises(ValueError):
        aggregate_parameters(s, counts, method="capped_fedavg")

    # Unknown method raises
    with pytest.raises(ValueError):
        aggregate_parameters(s, counts, method="unknown_method")
