from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentPoolV02ObjectiveWeights:
    best_slot_semantic: float = 0.30
    pooled_alignment: float = 0.18
    slot_redundancy: float = 0.20
    view_coverage: float = 0.08
    view_specialization: float = 0.10
    channel_coverage: float = 0.04
    counterfactual_margin: float = 0.10


def _require_unit_interval(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def best_slot_semantic_alignment_loss(
    latent_slots: torch.Tensor,
    target_semantic: torch.Tensor,
) -> torch.Tensor:
    """Require at least one useful target-aligned slot without rewarding copies.

    The failed v0.1 objective used an unnormalized log-sum-exp smooth maximum.
    With S slots that term increases by temperature*log(S) when all slots copy
    the same target, so duplicate slots were explicitly rewarded. A hard set
    maximum expresses the actual requirement: the set must contain a strong
    target-relevant slot, while additional slots remain free to preserve other
    judgment-relevant evidence.
    """
    if latent_slots.ndim != 3 or target_semantic.ndim != 2:
        raise ValueError("latent_slots must be [batch, slots, dim] and target_semantic [batch, dim]")
    if latent_slots.shape[0] != target_semantic.shape[0] or latent_slots.shape[2] != target_semantic.shape[1]:
        raise ValueError("latent/target shape mismatch")
    target = F.normalize(target_semantic, dim=-1).unsqueeze(1)
    cosine = (F.normalize(latent_slots, dim=-1) * target).sum(dim=-1)
    return (1.0 - cosine.max(dim=1).values).mean()


def pooled_semantic_alignment_loss(
    pooled_state: torch.Tensor,
    target_semantic: torch.Tensor,
) -> torch.Tensor:
    if pooled_state.shape != target_semantic.shape:
        raise ValueError("pooled_state and target_semantic must have the same shape")
    return (1.0 - F.cosine_similarity(pooled_state, target_semantic, dim=-1)).mean()


def slot_redundancy_loss(
    latent_slots: torch.Tensor,
    *,
    similarity_margin: float = 0.90,
) -> torch.Tensor:
    """Penalize near-duplicate slots while allowing substantial shared semantics."""
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
    """Every available evidence view must be reachable by at least one slot."""
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


def normalized_view_specialization(
    view_attention_mass: torch.Tensor,
    view_available: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-example mutual-information style specialization score in [0, 1].

    It is permutation-free and does not name slot roles. A high score means
    different slots make different use of available views while the slot set as
    a whole still spans those views. Examples with only one available view are
    neutral and receive score 1 because specialization is not identifiable.
    """
    if view_attention_mass.ndim != 3:
        raise ValueError("view_attention_mass must be [batch, slots, views]")
    if view_available.shape != (view_attention_mass.shape[0], view_attention_mass.shape[2]):
        raise ValueError("view_available shape mismatch")
    mass = torch.where(
        view_available.unsqueeze(1),
        view_attention_mass.clamp_min(0.0),
        torch.zeros_like(view_attention_mass),
    )
    per_slot = mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    slot_entropy = -(per_slot * per_slot.clamp_min(eps).log()).sum(dim=-1).mean(dim=1)
    marginal = per_slot.mean(dim=1)
    marginal_entropy = -(marginal * marginal.clamp_min(eps).log()).sum(dim=-1)
    mutual_information = (marginal_entropy - slot_entropy).clamp_min(0.0)
    available_count = view_available.sum(dim=-1)
    normalizer = available_count.clamp_min(2).to(mass.dtype).log()
    score = mutual_information / normalizer.clamp_min(eps)
    return torch.where(available_count > 1, score.clamp(0.0, 1.0), torch.ones_like(score))


def view_specialization_loss(
    view_attention_mass: torch.Tensor,
    view_available: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - normalized_view_specialization(view_attention_mass, view_available)).mean()


def channel_coverage_loss(
    channel_attention_mass: torch.Tensor,
    *,
    minimum_best_slot_mass: float = 0.12,
) -> torch.Tensor:
    if channel_attention_mass.ndim != 3 or channel_attention_mass.shape[-1] != 2:
        raise ValueError("channel_attention_mass must be [batch, slots, 2]")
    _require_unit_interval("minimum_best_slot_mass", minimum_best_slot_mass)
    best = channel_attention_mass.max(dim=1).values
    return F.relu(minimum_best_slot_mass - best).pow(2).mean()


def counterfactual_target_margin_loss(
    correct_slots: torch.Tensor,
    counterfactual_slots: torch.Tensor,
    target_semantic: torch.Tensor,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
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


def centered_slot_effective_rank(
    latent_slots: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalized effective rank of slot-specific residual directions.

    Common semantic content is removed by centering over slots first. Thus this
    metric does not demand that slots forget shared context merely to look
    different. A collapsed slot set has rank 0; diverse residual structure
    approaches 1 relative to the maximum S-1 centered rank.
    """
    if latent_slots.ndim != 3 or latent_slots.shape[1] < 2:
        raise ValueError("latent_slots must be [batch, slots, dim] with at least two slots")
    centered = latent_slots.float() - latent_slots.float().mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum(dim=-1, keepdim=True)
    probability = energy / total.clamp_min(eps)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=-1)
    effective = entropy.exp()
    maximum = float(min(latent_slots.shape[1] - 1, latent_slots.shape[2]))
    normalized = effective / max(maximum, 1.0)
    return torch.where(total.squeeze(-1) > eps, normalized.clamp(0.0, 1.0), torch.zeros_like(normalized))


def adaptive_latent_pool_v0_2_objective(
    *,
    latent_slots: torch.Tensor,
    pooled_state: torch.Tensor,
    target_semantic: torch.Tensor,
    view_attention_mass: torch.Tensor,
    view_available: torch.Tensor,
    channel_attention_mass: torch.Tensor,
    counterfactual_slots: torch.Tensor | None = None,
    weights: LatentPoolV02ObjectiveWeights | None = None,
) -> dict[str, torch.Tensor]:
    w = weights or LatentPoolV02ObjectiveWeights()
    semantic = best_slot_semantic_alignment_loss(latent_slots, target_semantic)
    pooled = pooled_semantic_alignment_loss(pooled_state, target_semantic)
    redundancy = slot_redundancy_loss(latent_slots)
    coverage = view_coverage_loss(view_attention_mass, view_available)
    specialization = view_specialization_loss(view_attention_mass, view_available)
    channel = channel_coverage_loss(channel_attention_mass)
    if counterfactual_slots is None:
        counterfactual = latent_slots.sum() * 0.0
    else:
        counterfactual = counterfactual_target_margin_loss(
            latent_slots,
            counterfactual_slots,
            target_semantic,
        )
    total = (
        w.best_slot_semantic * semantic
        + w.pooled_alignment * pooled
        + w.slot_redundancy * redundancy
        + w.view_coverage * coverage
        + w.view_specialization * specialization
        + w.channel_coverage * channel
        + w.counterfactual_margin * counterfactual
    )
    return {
        "total": total,
        "best_slot_semantic": semantic,
        "pooled_alignment": pooled,
        "slot_redundancy": redundancy,
        "view_coverage": coverage,
        "view_specialization": specialization,
        "channel_coverage": channel,
        "counterfactual_margin": counterfactual,
    }
