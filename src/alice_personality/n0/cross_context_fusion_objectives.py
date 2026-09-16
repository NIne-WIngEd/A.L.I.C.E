from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FusionObjectiveWeights:
    fused_semantic: float = 0.40
    view_routing: float = 0.25
    view_preservation: float = 0.20
    disagreement_preservation: float = 0.15

    def validate(self) -> None:
        values = (
            self.fused_semantic,
            self.view_routing,
            self.view_preservation,
            self.disagreement_preservation,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("fusion objective weights must be non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one fusion objective weight must be positive")


def soft_distribution_cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    available: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != available.shape:
        raise ValueError("prediction, target, and available must have matching shapes")
    if available.dtype != torch.bool:
        raise ValueError("available must be bool")
    if (target < 0).any() or not torch.isfinite(target).all():
        raise ValueError("target view distribution must be finite and non-negative")
    unavailable_target_mass = target.masked_select(~available)
    if unavailable_target_mass.numel() and torch.any(unavailable_target_mass > 1e-8):
        raise ValueError("fusion target may assign mass only to an available view")
    masked_target = target * available.to(target.dtype)
    denominator = masked_target.sum(dim=-1, keepdim=True)
    if not torch.all(denominator > 0):
        raise ValueError("every fusion target must assign mass to an available view")
    normalized_target = masked_target / denominator
    return -(normalized_target * prediction.clamp_min(1e-8).log()).sum(dim=-1).mean()


def view_preservation_loss(
    contextualized_view_summaries: torch.Tensor,
    source_view_targets: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if contextualized_view_summaries.shape != source_view_targets.shape:
        raise ValueError("view summary preservation tensors must match")
    if contextualized_view_summaries.ndim != 3:
        raise ValueError("view summaries must have shape [batch, views, semantic_size]")
    if available.shape != contextualized_view_summaries.shape[:2]:
        raise ValueError("available shape mismatch")
    cosine = F.cosine_similarity(contextualized_view_summaries, source_view_targets, dim=-1)
    active = available.to(cosine.dtype)
    return ((1.0 - cosine) * active).sum() / active.sum().clamp_min(1.0)


def disagreement_preservation_loss(
    contextualized_view_summaries: torch.Tensor,
    source_view_targets: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if contextualized_view_summaries.shape != source_view_targets.shape:
        raise ValueError("disagreement preservation tensors must match")
    output = F.normalize(contextualized_view_summaries, dim=-1)
    source = F.normalize(source_view_targets, dim=-1)
    output_similarity = torch.einsum("bvd,bwd->bvw", output, output)
    source_similarity = torch.einsum("bvd,bwd->bvw", source, source)
    pair_mask = available.unsqueeze(2) & available.unsqueeze(1)
    eye = torch.eye(available.size(1), device=available.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~eye
    active = pair_mask.to(output_similarity.dtype)
    squared = (output_similarity - source_similarity).pow(2)
    return (squared * active).sum() / active.sum().clamp_min(1.0)


def cross_context_fusion_objective(
    *,
    fused_state: torch.Tensor,
    semantic_target: torch.Tensor,
    view_weights: torch.Tensor,
    target_view_distribution: torch.Tensor,
    contextualized_view_summaries: torch.Tensor,
    source_view_targets: torch.Tensor,
    view_available: torch.Tensor,
    weights: FusionObjectiveWeights | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or FusionObjectiveWeights()
    weights.validate()
    if fused_state.shape != semantic_target.shape:
        raise ValueError("fused_state and semantic_target must match")
    if fused_state.ndim != 2:
        raise ValueError("fused_state must have shape [batch, semantic_size]")
    if view_weights.shape != target_view_distribution.shape:
        raise ValueError("view routing tensors must match")

    semantic = (1.0 - F.cosine_similarity(fused_state, semantic_target, dim=-1)).mean()
    routing = soft_distribution_cross_entropy(
        view_weights,
        target_view_distribution,
        available=view_available,
    )
    preservation = view_preservation_loss(
        contextualized_view_summaries,
        source_view_targets,
        view_available,
    )
    disagreement = disagreement_preservation_loss(
        contextualized_view_summaries,
        source_view_targets,
        view_available,
    )
    total = (
        weights.fused_semantic * semantic
        + weights.view_routing * routing
        + weights.view_preservation * preservation
        + weights.disagreement_preservation * disagreement
    )
    return {
        "loss": total,
        "fused_semantic_loss": semantic,
        "view_routing_loss": routing,
        "view_preservation_loss": preservation,
        "disagreement_preservation_loss": disagreement,
    }
