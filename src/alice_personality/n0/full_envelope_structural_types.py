from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


ROLE_SOURCE = 0
ROLE_TARGET = 1
ROLE_SYMMETRIC = 2
ROLE_NONE = 3

TRAVERSAL_LOCAL = 0
TRAVERSAL_PATH = 1
TRAVERSAL_AGGREGATE = 2

DIRECTION_FORWARD = 0
DIRECTION_REVERSE = 1
DIRECTION_BIDIRECTIONAL = 2

MOD_RELIABILITY = 0
MOD_RECENCY = 1
MOD_TEMPORAL_CONSTRAINT = 2
MOD_PROVENANCE_CONSTRAINT = 3

CONTROL_FALLBACK = 0
CONTROL_RELATIONAL = 1
CONTROL_DEFER = 2


@dataclass(frozen=True)
class FullEnvelopeOperatorState:
    relation_distribution: Tensor
    relation_step_mass: Tensor
    stop_probability: Tensor
    unknown_probability: Tensor
    truncation_probability: Tensor
    role_distribution: Tensor
    traversal_distribution: Tensor
    direction_distribution: Tensor
    step_direction_distribution: Tensor
    modifier_weight: Tensor
    step_modifier_weight: Tensor
    applicability: Tensor
    control_distribution: Tensor
    continuous_state: Tensor
    uncertainty: Tensor

    def validate(
        self,
        *,
        relation_count: int,
        model_dim: int,
    ) -> dict[str, int]:
        if self.relation_distribution.ndim != 3:
            raise ValueError("relation_distribution must be [B,S,R]")
        batch, steps, relations = self.relation_distribution.shape
        if relations != relation_count:
            raise ValueError("operator/schema relation cardinality drift")
        if self.relation_step_mass.shape != (batch, steps):
            raise ValueError("relation_step_mass must be [B,S]")
        if self.stop_probability.shape != (batch, steps):
            raise ValueError("stop_probability must be [B,S]")
        if self.unknown_probability.shape != (batch, steps):
            raise ValueError("unknown_probability must be [B,S]")
        if self.truncation_probability.shape != (batch,):
            raise ValueError("truncation_probability must be [B]")
        if (
            self.role_distribution.ndim != 2
            or self.role_distribution.size(0) != batch
            or self.role_distribution.size(1) != 4
        ):
            raise ValueError("role_distribution must be [B,4]")
        if (
            self.traversal_distribution.ndim != 2
            or self.traversal_distribution.size(0) != batch
            or self.traversal_distribution.size(1) != 3
        ):
            raise ValueError("traversal_distribution must be [B,3]")
        if (
            self.direction_distribution.ndim != 2
            or self.direction_distribution.size(0) != batch
            or self.direction_distribution.size(1) != 3
        ):
            raise ValueError("direction_distribution must be [B,3]")
        if self.step_direction_distribution.shape != (batch, steps, 3):
            raise ValueError("step_direction_distribution must be [B,S,3]")
        if (
            self.modifier_weight.ndim != 2
            or self.modifier_weight.shape != (batch, 4)
        ):
            raise ValueError("modifier_weight must be [B,4]")
        if self.step_modifier_weight.shape != (batch, steps, 4):
            raise ValueError("step_modifier_weight must be [B,S,4]")
        if self.applicability.shape != (batch,):
            raise ValueError("applicability must be [B]")
        if (
            self.control_distribution.ndim != 2
            or self.control_distribution.shape != (batch, 3)
        ):
            raise ValueError("control_distribution must be [B,3]")
        if self.continuous_state.shape != (batch, model_dim):
            raise ValueError("continuous_state shape drift")
        if self.uncertainty.shape != (batch,):
            raise ValueError("uncertainty must be [B]")
        probability_tensors = (
            ("relation_step_mass", self.relation_step_mass),
            ("stop_probability", self.stop_probability),
            ("unknown_probability", self.unknown_probability),
            ("truncation_probability", self.truncation_probability),
            ("modifier_weight", self.modifier_weight),
            ("step_modifier_weight", self.step_modifier_weight),
            ("applicability", self.applicability),
            ("uncertainty", self.uncertainty),
        )
        for name, tensor in probability_tensors:
            if bool(((tensor < -1.0e-6) | (tensor > 1.0 + 1.0e-6)).any()):
                raise ValueError(f"{name} must stay inside [0,1]")

        normalized_distributions = (
            ("relation_distribution", self.relation_distribution),
            ("role_distribution", self.role_distribution),
            ("traversal_distribution", self.traversal_distribution),
            ("direction_distribution", self.direction_distribution),
            ("step_direction_distribution", self.step_direction_distribution),
            ("control_distribution", self.control_distribution),
        )
        for name, tensor in normalized_distributions:
            if bool(((tensor < -1.0e-6) | (tensor > 1.0 + 1.0e-6)).any()):
                raise ValueError(f"{name} must stay inside [0,1]")
            total = tensor.sum(dim=-1)
            if not torch.allclose(
                total,
                torch.ones_like(total),
                atol=1.0e-5,
                rtol=1.0e-5,
            ):
                raise ValueError(f"{name} must sum to one on its candidate axis")

        event_mass = (
            self.relation_step_mass
            + self.stop_probability
            + self.unknown_probability
        )
        expected_mass = torch.cat(
            [
                torch.ones(
                    batch,
                    1,
                    device=event_mass.device,
                    dtype=event_mass.dtype,
                ),
                self.relation_step_mass[:, :-1],
            ],
            dim=1,
        )
        if not torch.allclose(
            event_mass,
            expected_mass,
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError(
                "relation/stop/unknown masses violate recurrent survival conservation"
            )
        if not torch.allclose(
            self.truncation_probability,
            self.relation_step_mass[:, -1],
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise ValueError(
                "truncation_probability must equal residual survival after final slot"
            )

        for name, tensor in (
            ("relation_distribution", self.relation_distribution),
            ("relation_step_mass", self.relation_step_mass),
            ("stop_probability", self.stop_probability),
            ("unknown_probability", self.unknown_probability),
            ("truncation_probability", self.truncation_probability),
            ("role_distribution", self.role_distribution),
            ("traversal_distribution", self.traversal_distribution),
            ("direction_distribution", self.direction_distribution),
            ("step_direction_distribution", self.step_direction_distribution),
            ("modifier_weight", self.modifier_weight),
            ("step_modifier_weight", self.step_modifier_weight),
            ("applicability", self.applicability),
            ("control_distribution", self.control_distribution),
            ("continuous_state", self.continuous_state),
            ("uncertainty", self.uncertainty),
        ):
            if tensor.is_floating_point() and not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{name} contains non-finite values")
        return {
            "batch": batch,
            "steps": steps,
            "relations": relations,
        }


def masked_sparsemax(
    logits: Tensor,
    mask: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    """Exact-zero adaptive projection with fail-closed empty rows."""
    if logits.shape != mask.shape:
        raise ValueError("logits/mask shape mismatch")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")

    dim = dim if dim >= 0 else logits.ndim + dim
    if not (0 <= dim < logits.ndim):
        raise ValueError("invalid sparsemax dimension")

    moved_logits = logits.movedim(dim, -1)
    moved_mask = mask.movedim(dim, -1)
    width = moved_logits.size(-1)
    flat_logits = moved_logits.reshape(-1, width)
    flat_mask = moved_mask.reshape(-1, width)
    flat_out = torch.zeros_like(flat_logits)

    for row in range(flat_logits.size(0)):
        valid = flat_mask[row]
        if not bool(valid.any()):
            continue
        values = flat_logits[row][valid]
        sorted_values, _ = torch.sort(values, descending=True)
        cumsum = torch.cumsum(sorted_values, dim=0)
        k = torch.arange(
            1,
            sorted_values.numel() + 1,
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        support = 1 + k * sorted_values > cumsum
        k_star = int(support.sum().item())
        if k_star <= 0:
            raise RuntimeError("sparsemax support search failed")
        tau = (cumsum[k_star - 1] - 1) / k_star
        projected = torch.clamp(values - tau, min=0)
        total = projected.sum()
        if not bool(total > 0):
            raise RuntimeError("sparsemax produced zero valid mass")
        flat_out[row][valid] = projected / total

    return flat_out.reshape(moved_logits.shape).movedim(-1, dim)
