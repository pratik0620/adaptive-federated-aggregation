"""Federated parameter aggregation algorithms and state validation.

Implements:
  1. Standard Federated Averaging (FedAvg): sample-weighted aggregation p_k = n_k / N
  2. Capped Federated Averaging (Capped FedAvg): capped sample weights q_k = min(p_k, c) / sum(min(p_j, c))
  3. Strict parameter validation, state dictionary integrity, and integer buffer preservation
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Sequence

import torch


def validate_cap_value(cap_value: Any) -> float:
    """Validate cap parameter c for Capped FedAvg.

    Enforces:
      - Not None
      - Real numeric type (float or int)
      - Finite (not NaN or Inf)
      - Strictly positive and at most 1.0: c in (0.0, 1.0]

    Args:
        cap_value: Candidate cap value.

    Returns:
        Validated float cap value.

    Raises:
        TypeError: If cap_value is not float or int.
        ValueError: If cap_value is None, non-finite, <= 0.0, or > 1.0.
    """
    if cap_value is None:
        raise ValueError(
            "Cap value cannot be None for Capped FedAvg. Please provide a float in (0.0, 1.0]."
        )
    if not isinstance(cap_value, (int, float)):
        raise TypeError(
            f"Cap value must be a numeric float or int, got {type(cap_value).__name__}."
        )
    c = float(cap_value)
    if math.isnan(c) or math.isinf(c):
        raise ValueError(f"Cap value must be a finite real number, got {c}.")
    if c <= 0.0:
        raise ValueError(f"Cap value must be strictly positive (> 0.0), got {c}.")
    if c > 1.0:
        raise ValueError(f"Cap value must be <= 1.0, got {c}.")
    return c


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


def compute_capped_fedavg_weights(
    sample_counts: Sequence[int],
    cap_value: float,
    client_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute normalized Capped FedAvg aggregation weights q_k.

    Mathematical Formulation:
      1. p_k = n_k / sum_{j=1}^K n_j
      2. capped_weight_k = min(p_k, c)
      3. q_k = capped_weight_k / sum_{j=1}^K capped_weight_j

    Properties:
      - For c >= max_k(p_k), q_k == p_k (recovers standard FedAvg).
      - For c < max_k(p_k), the dominant client's weight is capped at c before
        renormalization, distributing more proportional influence to minority silos.
      - Non-negative: q_k >= 0 for all k.
      - Unity constraint: sum_{k=1}^K q_k == 1.0 within numerical tolerance.

    Args:
        sample_counts: Sequence of integer sample counts for each participating client.
        cap_value: Float upper bound c in (0.0, 1.0].
        client_ids: Optional sequence of client identifiers.

    Returns:
        Dictionary mapping client_id to float weight q_k.

    Raises:
        ValueError or TypeError: If cap_value or sample_counts fail validation.
    """
    c = validate_cap_value(cap_value)

    if not sample_counts:
        raise ValueError("Cannot compute Capped FedAvg weights for an empty list of clients.")

    if client_ids is None:
        client_ids = [f"client_{i}" for i in range(len(sample_counts))]
    elif len(client_ids) != len(sample_counts):
        raise ValueError(
            f"Length mismatch: {len(client_ids)} client_ids provided for {len(sample_counts)} sample counts."
        )

    # Step 1: Compute standard proportional weights p_k
    p_weights = compute_fedavg_weights(sample_counts=sample_counts, client_ids=client_ids)

    # Step 2: Apply cap min(p_k, c)
    capped_unnormalized: dict[str, float] = {}
    for cid in client_ids:
        p_k = p_weights[cid]
        capped_unnormalized[cid] = min(p_k, c)

    # Step 3: Renormalize to sum to 1.0
    sum_capped = sum(capped_unnormalized.values())
    if sum_capped <= 0.0:
        raise ValueError(
            f"Sum of capped weights must be > 0, got {sum_capped}. Check cap_value={c}."
        )

    q_weights: dict[str, float] = {}
    for cid in client_ids:
        q_k = float(capped_unnormalized[cid] / sum_capped)
        if q_k < 0.0:
            raise ValueError(f"Negative capped weight calculated for client '{cid}': {q_k}")
        q_weights[cid] = q_k

    # Validate unity constraint
    weight_sum = sum(q_weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Capped FedAvg weights sum to {weight_sum:.8f}, which violates unity constraint (1.0)."
        )

    return q_weights


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


def _aggregate_tensors_with_weights(
    client_states: Sequence[dict[str, torch.Tensor]],
    weights: dict[str, float],
    client_ids: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Internal helper to aggregate model state dictionaries using a pre-computed weight map.

    Handles:
      - Floating-point tensors (accumulated in float64 on CPU for numerical stability)
      - Integer buffers (e.g. BatchNorm num_batches_tracked: float64 weighted sum rounded to nearest integer)
      - Boolean buffers (thresholded majority at 0.5)
    """
    ref_state = client_states[0]
    aggregated_state: dict[str, torch.Tensor] = {}

    for key, ref_tensor in ref_state.items():
        orig_dtype = ref_tensor.dtype

        if ref_tensor.is_floating_point():
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                w_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += w_k * client_tensor

            aggregated_state[key] = accum.to(dtype=orig_dtype)

        elif orig_dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                w_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += w_k * client_tensor

            aggregated_state[key] = torch.round(accum).to(dtype=orig_dtype)

        elif orig_dtype == torch.bool:
            accum = torch.zeros(ref_tensor.shape, dtype=torch.float64, device="cpu")
            for cid, state in zip(client_ids, client_states):
                w_k = weights[cid]
                client_tensor = state[key].detach().to(device="cpu", dtype=torch.float64)
                accum += w_k * client_tensor
            aggregated_state[key] = (accum >= 0.5).to(dtype=torch.bool)

        else:
            raise TypeError(
                f"Unsupported parameter dtype '{orig_dtype}' for parameter '{key}'."
            )

    return aggregated_state


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

    # 1. Compute and validate standard weights p_k
    weights = compute_fedavg_weights(sample_counts, client_ids)

    # 2. Validate state dictionaries
    validate_client_state_dicts(client_states, client_ids)

    # 3. Perform sample-weighted aggregation
    aggregated_state = _aggregate_tensors_with_weights(client_states, weights, client_ids)
    return aggregated_state, weights


def aggregate_capped_fedavg(
    client_states: Sequence[dict[str, torch.Tensor]],
    sample_counts: Sequence[int],
    cap_value: float,
    client_ids: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Aggregate client model parameters using Capped Federated Averaging (Capped FedAvg).

    Mathematical Formulation:
      p_k = n_k / sum_{j=1}^K n_j
      capped_weight_k = min(p_k, c)
      q_k = capped_weight_k / sum_{j=1}^K capped_weight_j
      theta_{t+1} = sum_{k=1}^K q_k * theta_{t+1}^{(k)}

    Args:
        client_states: Sequence of client state dictionaries.
        sample_counts: Number of training examples n_k actually used by each client.
        cap_value: Upper bound on client aggregation weight, c in (0.0, 1.0].
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

    # 1. Compute and validate capped weights q_k
    weights = compute_capped_fedavg_weights(sample_counts, cap_value, client_ids)

    # 2. Validate state dictionaries
    validate_client_state_dicts(client_states, client_ids)

    # 3. Perform sample-weighted aggregation
    aggregated_state = _aggregate_tensors_with_weights(client_states, weights, client_ids)
    return aggregated_state, weights


def aggregate_parameters(
    client_states: Sequence[dict[str, torch.Tensor]],
    sample_counts: Sequence[int],
    method: str = "fedavg",
    cap_value: float | None = None,
    client_ids: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Unified entrypoint dispatching between standard FedAvg and Capped FedAvg.

    Args:
        client_states: Sequence of client state dictionaries.
        sample_counts: Sequence of sample counts.
        method: Aggregation algorithm ('fedavg' or 'capped_fedavg').
        cap_value: Cap value c, required if method is 'capped_fedavg'.
        client_ids: Optional sequence of client identifiers.

    Returns:
        Tuple of (aggregated_state_dict, client_weights_dict).
    """
    canonical = method.lower().strip()
    if canonical == "fedavg":
        return aggregate_fedavg(client_states, sample_counts, client_ids)
    elif canonical in ("capped_fedavg", "capped-fedavg", "capped"):
        if cap_value is None:
            raise ValueError(
                "cap_value must be provided when aggregation method is 'capped_fedavg'."
            )
        return aggregate_capped_fedavg(client_states, sample_counts, cap_value, client_ids)
    else:
        raise ValueError(
            f"Unknown aggregation method: '{method}'. Supported methods: 'fedavg', 'capped_fedavg'."
        )
