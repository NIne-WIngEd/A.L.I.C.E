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
