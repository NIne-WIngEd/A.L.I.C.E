from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentPoolObjectiveWeights:
    semantic_alignment: float = 0.45
    pooled_alignment: float = 0.20
    slot_diversity: float = 0.12
    view_coverage: float = 0.10
    channel_coverage: float = 0.05
    counterfactual_margin: float = 0.08


def _require_unit_interval(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def multi_slot_semantic_alignment_loss(
    latent_slots: torch.Tensor,
    target_semantic: torch.Tensor,
    *,
    temperature: float = 0.12,
) -> torch.Tensor:
    """Require the slot set to contain target-relevant information.

    No specific slot is assigned a fixed semantic role. A smooth maximum lets
    whichever slot(s) are useful align to the target without forcing all slots
    to become copies of one another.
    """
    if latent_slots.ndim != 3 or target_semantic.ndim != 2:
        raise ValueError("latent_slots must be [batch, slots, dim] and target_semantic [batch, dim]")
    if latent_slots.shape[0] != target_semantic.shape[0] or latent_slots.shape[2] != target_semantic.shape[1]:
        raise ValueError("latent/target shape mismatch")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    slots = F.normalize(latent_slots, dim=-1)
    target = F.normalize(target_semantic, dim=-1).unsqueeze(1)
    cosine = (slots * target).sum(dim=-1)
    smooth_best = temperature * torch.logsumexp(cosine / temperature, dim=-1)
    return (1.0 - smooth_best).mean()


def pooled_semantic_alignment_loss(
    pooled_state: torch.Tensor,
    target_semantic: torch.Tensor,
) -> torch.Tensor:
    if pooled_state.shape != target_semantic.shape:
        raise ValueError("pooled_state and target_semantic must have the same shape")
    return (1.0 - F.cosine_similarity(pooled_state, target_semantic, dim=-1)).mean()


def slot_diversity_loss(
    latent_slots: torch.Tensor,
    *,
    similarity_margin: float = 0.82,
) -> torch.Tensor:
    """Discourage representational collapse without naming or orthogonalizing traits."""
    _require_unit_interval("similarity_margin", similarity_margin)
    if latent_slots.ndim != 3 or latent_slots.shape[1] < 2:
        raise ValueError("latent_slots must contain at least two slots")
    normalized = F.normalize(latent_slots, dim=-1)
    cosine = torch.einsum("bsd,btd->bst", normalized, normalized)
    count = cosine.shape[1]
    diagonal = torch.eye(count, device=cosine.device, dtype=torch.bool).unsqueeze(0)
    off_diagonal = cosine.masked_select(~diagonal.expand_as(cosine))
    return F.relu(off_diagonal - similarity_margin).pow(2).mean()


def view_coverage_loss(
    view_attention_mass: torch.Tensor,
    view_available: torch.Tensor,
    *,
    minimum_best_slot_mass: float = 0.18,
) -> torch.Tensor:
    """Make available views representable by at least one slot.

    This is coverage, not decision authority. A stale or contradictory view may
    still deserve representation so later uncertainty/historical reasoning can
    see it. No exact internal view percentage is supervised.
    """
    _require_unit_interval("minimum_best_slot_mass", minimum_best_slot_mass)
    if view_attention_mass.ndim != 3:
        raise ValueError("view_attention_mass must be [batch, slots, views]")
    if view_available.shape != (view_attention_mass.shape[0], view_attention_mass.shape[2]):
        raise ValueError("view_available shape mismatch")
    best = view_attention_mass.max(dim=1).values
    deficits = F.relu(minimum_best_slot_mass - best)
    selected = deficits.masked_select(view_available)
    if selected.numel() == 0:
        return deficits.sum() * 0.0
    return selected.pow(2).mean()


def channel_coverage_loss(
    channel_attention_mass: torch.Tensor,
    *,
    minimum_best_slot_mass: float = 0.12,
) -> torch.Tensor:
    """Keep both exact-source and contextualized channels causally reachable."""
    _require_unit_interval("minimum_best_slot_mass", minimum_best_slot_mass)
    if channel_attention_mass.ndim != 3 or channel_attention_mass.shape[-1] != 2:
        raise ValueError("channel_attention_mass must be [batch, slots, 2]")
    best = channel_attention_mass.max(dim=1).values
    return F.relu(minimum_best_slot_mass - best).pow(2).mean()


def counterfactual_target_margin_loss(
    correct_slots: torch.Tensor,
    counterfactual_slots: torch.Tensor,
    target_semantic: torch.Tensor,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
    """Require the correct evidence state to fit the target better than a corrupted one."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if correct_slots.shape != counterfactual_slots.shape:
        raise ValueError("correct/counterfactual slot shape mismatch")
    if correct_slots.ndim != 3 or target_semantic.shape != correct_slots[:, 0].shape:
        raise ValueError("target shape mismatch")
    target = F.normalize(target_semantic, dim=-1).unsqueeze(1)
    correct = (F.normalize(correct_slots, dim=-1) * target).sum(dim=-1).max(dim=1).values
    corrupted = (F.normalize(counterfactual_slots, dim=-1) * target).sum(dim=-1).max(dim=1).values
    return F.relu(margin - (correct - corrupted)).mean()


def adaptive_latent_pool_objective(
    *,
    latent_slots: torch.Tensor,
    pooled_state: torch.Tensor,
    target_semantic: torch.Tensor,
    view_attention_mass: torch.Tensor,
    view_available: torch.Tensor,
    channel_attention_mass: torch.Tensor,
    counterfactual_slots: torch.Tensor | None = None,
    weights: LatentPoolObjectiveWeights | None = None,
) -> dict[str, torch.Tensor]:
    w = weights or LatentPoolObjectiveWeights()
    semantic = multi_slot_semantic_alignment_loss(latent_slots, target_semantic)
    pooled = pooled_semantic_alignment_loss(pooled_state, target_semantic)
    diversity = slot_diversity_loss(latent_slots)
    view_coverage = view_coverage_loss(view_attention_mass, view_available)
    channel_coverage = channel_coverage_loss(channel_attention_mass)
    if counterfactual_slots is None:
        counterfactual = latent_slots.sum() * 0.0
    else:
        counterfactual = counterfactual_target_margin_loss(
            latent_slots,
            counterfactual_slots,
            target_semantic,
        )
    total = (
        w.semantic_alignment * semantic
        + w.pooled_alignment * pooled
        + w.slot_diversity * diversity
        + w.view_coverage * view_coverage
        + w.channel_coverage * channel_coverage
        + w.counterfactual_margin * counterfactual
    )
    return {
        "total": total,
        "semantic_alignment": semantic,
        "pooled_alignment": pooled,
        "slot_diversity": diversity,
        "view_coverage": view_coverage,
        "channel_coverage": channel_coverage,
        "counterfactual_margin": counterfactual,
    }
