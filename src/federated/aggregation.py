"""Federated parameter aggregation algorithms and state validation.

Implements standard Federated Averaging (FedAvg) with strict validation:
  - Exact parameter name, shape, and dtype matching across clients
  - Strict absence of NaN or Inf values
  - Positive sample-count weighting summing to 1.0
  - Explicit handling of integer model buffers (such as BatchNorm num_batches_tracked)
  - Decoupled functional interface for easy future extension (e.g. Capped FedAvg)
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import torch


def compute_fedavg_weights(
    sample_counts: Sequence[int],
    client_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute normalized FedAvg aggregation weights p_k = n_k / N.

    Args:
        sample_counts: List of integer sample counts for each participating client.
        client_ids: Optional list of client identifiers. Defaults to client_0, client_1, ...

    Returns:
        Dictionary mapping client_id to float weight p_k.

    Raises:
        ValueError: If sample_counts is empty, contains non-positive numbers, or total N <= 0.
    """
    if not sample_counts:
        raise ValueError("Cannot compute FedAvg weights for an empty list of clients.")

    if client_ids is None:
        client_ids = [f"client_{i}" for i in range(len(sample_counts))]
    elif len(client_ids) != len(sample_counts):
        raise ValueError(
            f"Length mismatch: {len(client_ids)} client_ids provided for {len(sample_counts)} sample counts."
        )

    for cid, n_k in zip(client_ids, sample_counts):
        if not isinstance(n_k, (int, torch.integer_types if hasattr(torch, "integer_types") else int)) or n_k <= 0:
            raise ValueError(
                f"Client '{cid}' reported invalid sample count: {n_k}. "
                "Sample counts must be strictly positive integers."
            )

    total_samples = sum(sample_counts)
    if total_samples <= 0:
        raise ValueError(f"Total sample count across clients must be > 0, got {total_samples}.")

    weights: dict[str, float] = {}
    for cid, n_k in zip(client_ids, sample_counts):
        p_k = float(n_k / total_samples)
        if p_k < 0.0:
            raise ValueError(f"Negative weight calculated for client '{cid}': {p_k}")
        weights[cid] = p_k

    # Validate numerical stability: sum must equal 1.0 within tolerance
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"FedAvg aggregation weights sum to {weight_sum:.8f}, which violates unity constraint (1.0)."
        )

    return weights


def validate_client_state_dicts(
    client_states: Sequence[dict[str, torch.Tensor]],
    client_ids: Sequence[str] | None = None,
) -> None:
    """Validate that all client state dictionaries are structurally identical and clean.

    Checks:
      1. Non-empty sequence of client state dictionaries.
      2. All values are PyTorch Tensors (rejects optimizer states or raw checkpoints).
      3. Exactly matching parameter keys across all clients (no missing/extra keys).
      4. Identical tensor shapes for each parameter across all clients.
      5. Identical tensor data types for each parameter across all clients.
      6. No NaN or Inf values in any parameter tensor.

    Raises:
        ValueError or TypeError if any validation rule is violated.
    """
    if not client_states:
        raise ValueError("Cannot validate an empty sequence of client state dictionaries.")

    if client_ids is None:
        client_ids = [f"client_{i}" for i in range(len(client_states))]
    elif len(client_ids) != len(client_states):
        raise ValueError(
            f"Length mismatch: {len(client_ids)} client_ids provided for {len(client_states)} states."
        )

    # Reference client
    ref_id = client_ids[0]
    ref_state = client_states[0]

    if not isinstance(ref_state, dict):
        raise TypeError(f"Client '{ref_id}' state must be a dict, got {type(ref_state).__name__}.")

    ref_keys = set(ref_state.keys())
    if not ref_keys:
        raise ValueError(f"Client '{ref_id}' state dictionary is empty.")

    # Validate each client against reference
    for cid, state in zip(client_ids, client_states):
        if not isinstance(state, dict):
            raise TypeError(
                f"Client '{cid}' state must be a dict of tensors, got {type(state).__name__}."
            )

        keys = set(state.keys())
        missing = ref_keys - keys
        extra = keys - ref_keys
        if missing or extra:
            err_msg = [f"Parameter key mismatch for client '{cid}':"]
            if missing:
                err_msg.append(f"  Missing keys: {sorted(list(missing))[:5]} (total {len(missing)})")
            if extra:
                err_msg.append(f"  Unexpected keys: {sorted(list(extra))[:5]} (total {len(extra)})")
            raise KeyError("\n".join(err_msg))

        # Check tensor type, shapes, dtypes, and NaNs
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Parameter '{key}' in client '{cid}' is of type {type(tensor).__name__}, not torch.Tensor. "
                    "Make sure only model state dictionaries are aggregated, not optimizer states or checkpoints."
                )

            ref_tensor = ref_state[key]
            if tensor.shape != ref_tensor.shape:
                raise ValueError(
                    f"Parameter shape mismatch for '{key}': client '{cid}' has {tuple(tensor.shape)}, "
                    f"while reference client '{ref_id}' has {tuple(ref_tensor.shape)}."
                )

            if tensor.dtype != ref_tensor.dtype:
                raise TypeError(
                    f"Parameter dtype mismatch for '{key}': client '{cid}' has {tensor.dtype}, "
                    f"while reference client '{ref_id}' has {ref_tensor.dtype}."
                )

            if tensor.is_floating_point():
                if torch.isnan(tensor).any():
                    raise ValueError(f"NaN value detected in parameter '{key}' of client '{cid}'.")
                if torch.isinf(tensor).any():
                    raise ValueError(f"Inf value detected in parameter '{key}' of client '{cid}'.")


def aggregate_fedavg(
    client_states: Sequence[dict[str, torch.Tensor]],
    sample_counts: Sequence[int],
    client_ids: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Aggregate client model parameters using standard Federated Averaging (FedAvg).

    Mathematical Formulation:
      p_k = n_k / N,  where N = sum_{k=1}^K n_k
      theta_{t+1} = sum_{k=1}^K p_k * theta_{t+1}^{(k)}

    Integer Buffer Handling:
      Model buffers with integer dtypes (e.g., BatchNorm num_batches_tracked: torch.int64)
      are aggregated explicitly via weighted average in float64 and rounded to the nearest
      integer, preserving their original integer dtype rather than throwing type errors
      or performing truncated integer division.

    Args:
        client_states: Sequence of client state dictionaries.
        sample_counts: Number of training examples n_k actually used by each client.
        client_ids: Optional list of client identifiers.

    Returns:
        Tuple of (aggregated_state_dict, client_weights_dict).

    Raises:
        ValueError, TypeError, or KeyError if inputs fail validation.
    """
    if len(client_states) != len(sample_counts):
        raise ValueError(
            f"Mismatched counts: {len(client_states)} client states vs {len(sample_counts)} sample counts."
        )

    if client_ids is None:
        client_ids = [f"client_{i}" for i in range(len(client_states))]

    # 1. Compute and validate aggregation weights p_k
    weights = compute_fedavg_weights(sample_counts, client_ids)

    # 2. Validate state dictionaries
    validate_client_state_dicts(client_states, client_ids)

    ref_state = client_states[0]
    aggregated_state: dict[str, torch.Tensor] = {}

    # 3. Perform sample-weighted aggregation parameter-by-parameter
    for key, ref_tensor in ref_state.items():
        orig_dtype = ref_tensor.dtype

        if ref_tensor.is_floating_point():
            # Accumulate in float64 on CPU for numerical stability
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                p_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += p_k * client_tensor

            aggregated_state[key] = accum.to(dtype=orig_dtype)

        elif orig_dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            # Integer-valued buffer (e.g. BatchNorm num_batches_tracked)
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                p_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += p_k * client_tensor

            # Round to nearest integer and cast back to original integer dtype
            aggregated_state[key] = torch.round(accum).to(dtype=orig_dtype)

        elif orig_dtype == torch.bool:
            # Boolean buffers: majority vote or ref value
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                p_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += p_k * client_tensor
            aggregated_state[key] = (accum >= 0.5).to(dtype=torch.bool)

        else:
            raise TypeError(
                f"Unsupported parameter dtype '{orig_dtype}' for parameter '{key}'."
            )

    return aggregated_state, weights
