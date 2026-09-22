from __future__ import annotations

import torch
from torch import Tensor


def require_finite(name: str, tensor: Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")


def require_unit_interval(
    name: str,
    tensor: Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> None:
    require_finite(name, tensor)
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be floating-point")
    if bool(
        (
            (tensor < -float(tolerance))
            | (tensor > 1.0 + float(tolerance))
        ).any()
    ):
        raise ValueError(f"{name} must stay inside [0,1]")


def require_probability_simplex(
    name: str,
    tensor: Tensor,
    *,
    dim: int = -1,
    tolerance: float = 1.0e-5,
) -> None:
    require_unit_interval(name, tensor, tolerance=tolerance)
    total = tensor.sum(dim=dim)
    if not torch.allclose(
        total,
        torch.ones_like(total),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError(f"{name} must sum to one on axis {dim}")


def exact_masked_softmax(
    logits: Tensor,
    mask: Tensor,
    *,
    dim: int = -1,
    allow_empty: bool = False,
) -> Tensor:
    """Softmax over exactly the valid set with zero mass outside it.

    A finite sentinel such as -1e4 is not a structural mask: a learned valid
    logit can become smaller and allow an invalid item to win. Invalid entries
    therefore receive the dtype floor before softmax and are explicitly zeroed
    afterward. Empty slices are either rejected or returned as exact zeros.
    """
    if not logits.is_floating_point():
        raise ValueError("masked-softmax logits must be floating-point")
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("masked-softmax mask must be bool with logits shape")

    axis = dim if dim >= 0 else logits.ndim + dim
    if not 0 <= axis < logits.ndim:
        raise ValueError("masked-softmax dimension outside tensor rank")

    valid_values = logits.masked_select(mask)
    if valid_values.numel() and not bool(torch.isfinite(valid_values).all()):
        raise ValueError("valid masked-softmax logits must be finite")

    active = mask.any(dim=axis)
    if not allow_empty and bool((~active).any()):
        raise ValueError("masked-softmax slice has no valid entries")

    floor = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~mask, floor)
    probability = torch.softmax(masked_logits, dim=axis)
    probability = probability * mask.to(probability.dtype)
    denominator = probability.sum(dim=axis, keepdim=True)

    if bool(active.any()):
        active_denominator = denominator.squeeze(axis).masked_select(active)
        if not bool(torch.isfinite(active_denominator).all()):
            raise ValueError("masked-softmax normalization became non-finite")
        if bool((active_denominator <= 0).any()):
            raise ValueError("masked-softmax valid slice lost all probability mass")

    probability = probability / denominator.clamp_min(
        torch.finfo(probability.dtype).tiny
    )
    if allow_empty and bool((~active).any()):
        expanded_active = active.unsqueeze(axis)
        probability = torch.where(
            expanded_active,
            probability,
            torch.zeros_like(probability),
        )
    return probability


def exact_masked_logits(
    logits: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return finite logits with invalid positions reserved at the dtype floor."""
    if not logits.is_floating_point():
        raise ValueError("masked logits must be floating-point")
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("masked-logit mask must be bool with logits shape")
    valid_values = logits.masked_select(mask)
    if valid_values.numel() and not bool(torch.isfinite(valid_values).all()):
        raise ValueError("valid masked logits must be finite")
    floor = torch.finfo(logits.dtype).min
    if valid_values.numel() and bool((valid_values <= floor).any()):
        raise ValueError("valid logit reached reserved invalid floor")
    return logits.masked_fill(~mask, floor)
