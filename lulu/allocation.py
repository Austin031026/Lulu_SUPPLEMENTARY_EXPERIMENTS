"""Deterministic reasoning-weight allocation controls for ReN experiments.

These transforms deliberately keep the bounded ReN score separate from the
weight actually applied by the training arm.  This makes the RQ2 controls
interpretable:

* ``ren``: use each state-specific ReN weight as computed;
* ``uniform``: preserve each rollout's total ReN mass, spread uniformly;
* ``shuffled``: preserve the exact within-rollout multiset, permute positions;
* ``causal_matched``: preserve the exact multiset, rank positions only by D_C;
* ``vanilla``: weight every reasoning position by one;
* ``none``: no reasoning Teacher loss.

The randomized arm is deterministic from seed/round/source identity so that a
frozen experiment plan completely determines the ablation.
"""
from __future__ import annotations

import hashlib
import torch


ARMS = ("ren", "uniform", "shuffled", "causal_matched", "vanilla", "none", "opsd")


def _seed(seed: int, round_index: int, source_id: object, record_index: int) -> int:
    text = f"{int(seed)}:{int(round_index)}:{source_id}:{int(record_index)}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def _stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Stable descending argsort with deterministic position tie-breaking."""
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:  # Older torch fallback; tiny deterministic tie-breaker.
        order = torch.arange(values.numel(), device=values.device, dtype=torch.float64)
        adjusted = values.to(torch.float64) - order * torch.finfo(torch.float64).eps
        return torch.argsort(adjusted, descending=True)


def allocate_reasoning_weights(
    base_weight: torch.Tensor,
    causal_kl: torch.Tensor,
    reasoning_mask: torch.Tensor,
    *,
    arm: str,
    seed: int,
    round_index: int,
    source_id: object,
    record_index: int,
) -> torch.Tensor:
    """Return the reasoning weights actually applied by an experimental arm.

    Inputs are full-response vectors.  Control positions always receive zero
    here; they are supervised by the separate control objective.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown reasoning allocation arm {arm!r}; expected one of {ARMS}")
    if base_weight.ndim != 1 or causal_kl.shape != base_weight.shape or reasoning_mask.shape != base_weight.shape:
        raise ValueError("base_weight, causal_kl and reasoning_mask must be matching one-dimensional tensors")
    if reasoning_mask.dtype != torch.bool:
        raise ValueError("reasoning_mask must be boolean")
    if not torch.isfinite(base_weight).all() or not torch.isfinite(causal_kl).all():
        raise FloatingPointError("non-finite allocation inputs")
    if bool((base_weight < 0).any()):
        raise ValueError("bounded ReN weights must be nonnegative")

    result = torch.zeros_like(base_weight, dtype=torch.float32)
    positions = reasoning_mask.nonzero(as_tuple=False).flatten()
    if not positions.numel() or arm == "none":
        return result
    if arm in ("vanilla", "opsd"):
        result[positions] = 1.0
        return result

    values = base_weight[positions].to(torch.float32)
    if arm == "ren":
        result[positions] = values
    elif arm == "uniform":
        result[positions] = values.mean()
    elif arm == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(seed, round_index, source_id, record_index))
        permutation = torch.randperm(values.numel(), generator=generator)
        result[positions] = values.cpu()[permutation].to(result.device)
    elif arm == "causal_matched":
        ranked_positions = _stable_descending(causal_kl[positions].to(torch.float64))
        ranked_weights = values[_stable_descending(values.to(torch.float64))]
        assigned = torch.empty_like(values)
        assigned[ranked_positions] = ranked_weights
        result[positions] = assigned
    else:  # pragma: no cover - guarded above.
        raise AssertionError(arm)
    return result


def allocation_invariants(base_weight: torch.Tensor, applied_weight: torch.Tensor, reasoning_mask: torch.Tensor) -> dict:
    """Small audit payload shared by tests and training diagnostics."""
    selected = reasoning_mask.bool()
    base = base_weight[selected].to(torch.float64)
    applied = applied_weight[selected].to(torch.float64)
    return {
        "reasoning_positions": int(selected.sum()),
        "base_sum": float(base.sum()),
        "applied_sum": float(applied.sum()),
        "base_mean": float(base.mean()) if base.numel() else 0.0,
        "applied_mean": float(applied.mean()) if applied.numel() else 0.0,
        "same_multiset": bool(
            base.numel() == applied.numel()
            and torch.allclose(torch.sort(base).values, torch.sort(applied).values, rtol=0, atol=1e-8)
        ),
    }
