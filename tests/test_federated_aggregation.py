from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated import AdaptiveFederatedAggregator, ClientUpdate, FedAvgAggregator
from src.trainer import ClientTrainer


def _client(client_id: str, values: list[float], count: int) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        state_dict={
            "weight": torch.tensor(values, dtype=torch.float32),
            "num_batches_tracked": torch.tensor(count, dtype=torch.int64),
        },
        num_examples=count,
    )


def test_aggregation_uses_sqrt_sample_weights_and_preserves_integer_buffers() -> None:
    global_state = {
        "weight": torch.tensor([0.0, 0.0]),
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    result = AdaptiveFederatedAggregator().aggregate(
        global_state,
        [_client("small", [1.0, 1.0], 1), _client("large", [3.0, 3.0], 9)],
    )

    assert result.client_weights == pytest.approx({"small": 0.25, "large": 0.75})
    assert torch.allclose(result.state_dict["weight"], torch.tensor([2.5, 2.5]))
    assert result.state_dict["num_batches_tracked"].item() == 9
    assert result.projected_conflicts == 0


def test_conflicting_client_updates_are_projected() -> None:
    global_state = {
        "weight": torch.zeros(2),
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    result = AdaptiveFederatedAggregator().aggregate(
        global_state,
        [_client("positive", [1.0, 0.0], 1), _client("negative", [-1.0, 0.0], 1)],
    )

    assert result.projected_conflicts == 2
    assert result.pairwise_cosines["positive|negative"] == pytest.approx(-1.0)
    assert result.pairwise_cosines["negative|positive"] == pytest.approx(-1.0)
    assert torch.allclose(result.state_dict["weight"], torch.zeros(2))


def test_aligned_updates_are_averaged_without_projection() -> None:
    global_state = {
        "weight": torch.zeros(2),
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    result = AdaptiveFederatedAggregator().aggregate(
        global_state,
        [_client("one", [1.0, 2.0], 1), _client("two", [3.0, 6.0], 1)],
    )

    assert result.projected_conflicts == 0
    assert result.pairwise_cosines["one|two"] == pytest.approx(1.0)
    assert torch.allclose(result.state_dict["weight"], torch.tensor([2.0, 4.0]))


def test_fedavg_baseline_uses_sample_proportions_without_projection() -> None:
    global_state = {
        "weight": torch.zeros(1),
        "num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }
    result = FedAvgAggregator().aggregate(
        global_state,
        [_client("small", [1.0], 1), _client("large", [-1.0], 9)],
    )

    assert result.client_weights == pytest.approx({"small": 0.1, "large": 0.9})
    assert result.projected_conflicts == 0
    assert result.pairwise_cosines["small|large"] == pytest.approx(-1.0)
    assert torch.allclose(result.state_dict["weight"], torch.tensor([-0.8]))


def test_invalid_client_updates_fail_explicitly() -> None:
    aggregator = AdaptiveFederatedAggregator()
    global_state = {"weight": torch.zeros(1)}

    with pytest.raises(ValueError, match="At least one"):
        aggregator.aggregate(global_state, [])
    with pytest.raises(ValueError, match="at least one"):
        aggregator.aggregate(global_state, [_client("empty", [1.0], 0)])
    with pytest.raises(ValueError, match="non-finite"):
        aggregator.aggregate(
            global_state,
            [ClientUpdate("bad", {"weight": torch.tensor([float("nan")])}, 1)],
        )
    with pytest.raises(ValueError, match="state keys"):
        aggregator.aggregate(
            global_state,
            [ClientUpdate("bad", {"other": torch.ones(1)}, 1)],
        )


def test_test_metrics_are_marked_as_partial_when_client_data_is_incomplete() -> None:
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 1), torch.zeros(2)),
        batch_size=1,
    )
    trainer = ClientTrainer(
        model=nn.Linear(1, 1),
        train_loader=loader,
        val_loader=loader,
        config={"total_official_test_images": 10},
        device="cpu",
    )

    metrics = trainer.annotate_test_coverage({"total_samples": 2, "warnings": []})

    assert metrics["evaluation_status"] == "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL"
    assert metrics["coverage_percentage"] == pytest.approx(20.0)
    assert metrics["missing_samples_count"] == 8
    assert any("MUST NOT be reported" in warning for warning in metrics["warnings"])
