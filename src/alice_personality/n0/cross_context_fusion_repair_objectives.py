from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from alice_personality.n0.cross_context_fusion_objectives import (
    disagreement_preservation_loss,
    soft_distribution_cross_entropy,
)


@dataclass(frozen=True)
class FusionRepairObjectiveWeights:
    """Objective for the full-scale source-anchored fusion repair.

    Exact source preservation is now an architectural channel, not something
    the optimizer has to approximate.  The learned contextualized stream is
    therefore free to transform while a soft alignment term keeps it grounded.
    """

    fused_semantic: float = 0.35
    view_routing: float = 0.40
    contextualized_source_alignment: float = 0.10
    disagreement_geometry: float = 0.15

    def validate(self) -> None:
        values = (
            self.fused_semantic,
            self.view_routing,
            self.contextualized_source_alignment,
            self.disagreement_geometry,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("fusion repair objective weights must be non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one fusion repair objective weight must be positive")


def contextualized_source_alignment_loss(
    contextualized_view_summaries: torch.Tensor,
    source_view_summaries: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if contextualized_view_summaries.shape != source_view_summaries.shape:
        raise ValueError("contextualized and source summaries must have matching shapes")
    if contextualized_view_summaries.ndim != 3:
        raise ValueError("view summaries must have shape [batch, views, semantic_size]")
    if available.shape != contextualized_view_summaries.shape[:2]:
        raise ValueError("available view mask shape mismatch")
    cosine = F.cosine_similarity(
        contextualized_view_summaries,
        source_view_summaries,
        dim=-1,
    )
    active = available.to(cosine.dtype)
    return ((1.0 - cosine) * active).sum() / active.sum().clamp_min(1.0)


def cross_context_fusion_repair_objective(
    *,
    fused_state: torch.Tensor,
    semantic_target: torch.Tensor,
    view_weights: torch.Tensor,
    target_view_distribution: torch.Tensor,
    contextualized_view_summaries: torch.Tensor,
    source_view_summaries: torch.Tensor,
    view_available: torch.Tensor,
    weights: FusionRepairObjectiveWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or FusionRepairObjectiveWeights()
    weights.validate()

    if fused_state.shape != semantic_target.shape or fused_state.ndim != 2:
        raise ValueError("fused semantic tensors must have shape [batch, semantic_size]")
    if view_weights.shape != target_view_distribution.shape:
        raise ValueError("view routing tensors must match")

    semantic = (1.0 - F.cosine_similarity(fused_state, semantic_target, dim=-1)).mean()
    routing = soft_distribution_cross_entropy(
        view_weights,
        target_view_distribution,
        available=view_available,
    )
    alignment = contextualized_source_alignment_loss(
        contextualized_view_summaries,
        source_view_summaries,
        view_available,
    )
    geometry = disagreement_preservation_loss(
        contextualized_view_summaries,
        source_view_summaries,
        view_available,
    )
    total = (
        weights.fused_semantic * semantic
        + weights.view_routing * routing
        + weights.contextualized_source_alignment * alignment
        + weights.disagreement_geometry * geometry
    )
    return {
        "loss": total,
        "fused_semantic_loss": semantic,
        "view_routing_loss": routing,
        "contextualized_source_alignment_loss": alignment,
        "disagreement_geometry_loss": geometry,
    }
