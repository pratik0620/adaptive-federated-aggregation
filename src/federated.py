"""Adaptive conflict-aware aggregation for federated binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClientUpdate:
    """A client's locally trained model state and training-set size."""

    client_id: str
    state_dict: Mapping[str, Tensor]
    num_examples: int


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated global parameters and round diagnostics."""

    state_dict: dict[str, Tensor]
    client_weights: dict[str, float]
    pairwise_cosines: dict[str, float]
    projected_conflicts: int


class AdaptiveFederatedAggregator:
    """Project conflicting client updates, then aggregate with sqrt-size weights.

    Pairwise gradient conflict is approximated by the cosine similarity of each
    client's model delta from the current global model. Conflicting components
    are removed with a deterministic PCGrad-style projection. The remaining
    deltas are weighted by sqrt(client sample count), reducing the dominance of
    larger clients compared with sample-proportional FedAvg.
    """

    def __init__(
        self,
        sample_weight_power: float = 0.5,
        project_conflicts: bool = True,
    ) -> None:
        if not isfinite(sample_weight_power) or sample_weight_power < 0:
            raise ValueError("sample_weight_power must be finite and non-negative.")
        self.sample_weight_power = sample_weight_power
        self.project_conflicts = project_conflicts

    def aggregate(
        self,
        global_state: Mapping[str, Tensor],
        client_updates: list[ClientUpdate],
    ) -> AggregationResult:
        if not client_updates:
            raise ValueError("At least one client update is required.")

        client_ids = [update.client_id for update in client_updates]
        if any(not client_id for client_id in client_ids):
            raise ValueError("Client IDs must be non-empty.")
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("Client IDs must be unique within a round.")
        if any(update.num_examples <= 0 for update in client_updates):
            raise ValueError("Each client must contribute at least one training example.")

        keys = set(global_state)
        if not keys:
            raise ValueError("The global model state must not be empty.")
        for update in client_updates:
            if set(update.state_dict) != keys:
                raise ValueError(
                    f"Client {update.client_id!r} state keys do not match the global model."
                )
            for key, global_tensor in global_state.items():
                local_tensor = update.state_dict[key]
                if local_tensor.shape != global_tensor.shape:
                    raise ValueError(
                        f"Client {update.client_id!r} tensor {key!r} has shape "
                        f"{tuple(local_tensor.shape)}, expected {tuple(global_tensor.shape)}."
                    )
                if local_tensor.dtype != global_tensor.dtype:
                    raise ValueError(
                        f"Client {update.client_id!r} tensor {key!r} has dtype "
                        f"{local_tensor.dtype}, expected {global_tensor.dtype}."
                    )

        float_keys = [key for key, value in global_state.items() if value.is_floating_point()]
        if not float_keys:
            raise ValueError("The global model has no floating-point parameters to aggregate.")

        reference = {
            key: global_state[key].detach().to(device="cpu", dtype=torch.float32)
            for key in float_keys
        }
        deltas: list[dict[str, Tensor]] = []
        for update in client_updates:
            delta = {
                key: (
                    update.state_dict[key].detach().to(device="cpu", dtype=torch.float32)
                    - reference[key]
                )
                for key in float_keys
            }
            if any(not torch.isfinite(value).all() for value in delta.values()):
                raise ValueError(f"Client {update.client_id!r} update contains non-finite values.")
            deltas.append(delta)

        norms = [self._dot(delta, delta) for delta in deltas]
        pairwise_cosines: dict[str, float] = {}
        conflict_count = 0
        projected = [{key: value.clone() for key, value in delta.items()} for delta in deltas]

        for index, update in enumerate(client_updates):
            for other_index, other_update in enumerate(client_updates):
                if index == other_index:
                    continue
                dot = self._dot(deltas[index], deltas[other_index])
                denominator = sqrt(norms[index] * norms[other_index])
                cosine = dot / denominator if denominator > 0 else 0.0
                pairwise_cosines[f"{update.client_id}|{other_update.client_id}"] = cosine
                if self.project_conflicts and dot < 0 and norms[other_index] > 0:
                    scale = dot / norms[other_index]
                    for key in float_keys:
                        projected[index][key].sub_(deltas[other_index][key], alpha=scale)
                    conflict_count += 1

        raw_weights = [
            float(update.num_examples) ** self.sample_weight_power
            for update in client_updates
        ]
        total_weight = sum(raw_weights)
        weights = [weight / total_weight for weight in raw_weights]
        weighted_updates = {
            key: sum(
                (projected[index][key] * weights[index] for index in range(len(client_updates))),
                torch.zeros_like(reference[key]),
            )
            for key in float_keys
        }

        result: dict[str, Tensor] = {}
        for key, global_tensor in global_state.items():
            if key in weighted_updates:
                update = weighted_updates[key].to(
                    device=global_tensor.device, dtype=global_tensor.dtype
                )
                result[key] = global_tensor.detach() + update
            elif global_tensor.is_floating_point():
                result[key] = global_tensor.detach().clone()
            elif global_tensor.dtype == torch.bool or global_tensor.dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                result[key] = torch.stack(
                    [
                        client.state_dict[key].detach().to(device=global_tensor.device)
                        for client in client_updates
                    ]
                ).amax(dim=0)
            else:
                result[key] = global_tensor.detach().clone()

        return AggregationResult(
            state_dict=result,
            client_weights={
                update.client_id: weights[index]
                for index, update in enumerate(client_updates)
            },
            pairwise_cosines=pairwise_cosines,
            projected_conflicts=conflict_count,
        )

    @staticmethod
    def _dot(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> float:
        return sum(
            torch.sum(left[key] * right[key], dtype=torch.float64).item()
            for key in left
        )


class FedAvgAggregator(AdaptiveFederatedAggregator):
    """Sample-proportional FedAvg baseline without conflict projection."""

    def __init__(self) -> None:
        super().__init__(sample_weight_power=1.0, project_conflicts=False)
