from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FullEnvelopeBehavioralWeights:
    public_judgment: float = 0.30
    support_selection: float = 0.16
    endpoint_roles: float = 0.12
    decisive_view_causality: float = 0.12
    irrelevant_view_invariance: float = 0.08
    latent_noncollapse: float = 0.10
    source_view_recoverability: float = 0.06
    permutation_consistency: float = 0.06

    def validate(self) -> None:
        values = tuple(vars(self).values())
        if any(v < 0.0 for v in values):
            raise ValueError("behavioral weights must be non-negative")
        if abs(sum(values) - 1.0) > 1.0e-9:
            raise ValueError("behavioral weights must sum to 1")


def public_judgment_loss(
    candidate_logits: Tensor,
    target_index: Tensor,
    *,
    hard_negative_margin: float = 0.20,
) -> Tensor:
    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must be [B,C]")
    if target_index.shape != (candidate_logits.size(0),):
        raise ValueError("target_index shape drift")
    ce = F.cross_entropy(candidate_logits, target_index.long())
    if candidate_logits.size(1) <= 1:
        return ce
    correct = candidate_logits.gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    mask = torch.ones_like(candidate_logits, dtype=torch.bool)
    mask.scatter_(1, target_index.long()[:, None], False)
    hardest = candidate_logits.masked_fill(~mask, -1.0e4).max(dim=-1).values
    margin = F.relu(float(hard_negative_margin) - (correct - hardest)).mean()
    return ce + margin


def support_selection_loss(
    support_logits: Tensor,
    support_target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if support_logits.shape != support_target.shape:
        raise ValueError("support target shape drift")
    if valid_mask.shape != support_logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("support valid mask shape drift")
    if not bool(valid_mask.any()):
        return support_logits.sum() * 0.0
    target = support_target.float()
    selected = target[valid_mask]
    positives = selected.sum()
    negatives = selected.numel() - positives
    pos_weight = (
        torch.as_tensor(
            float(negatives) / max(float(positives), 1.0),
            device=support_logits.device,
            dtype=support_logits.dtype,
        )
        if float(positives) > 0.0
        else None
    )
    return F.binary_cross_entropy_with_logits(
        support_logits[valid_mask],
        selected,
        pos_weight=pos_weight,
    )


def endpoint_role_loss(
    source_weight: Tensor,
    target_weight: Tensor,
    source_target: Tensor,
    target_target: Tensor,
    active_mask: Tensor,
) -> Tensor:
    if source_weight.shape != target_weight.shape:
        raise ValueError("source/target endpoint weight shape drift")
    batch, fields = source_weight.shape
    if source_target.shape != (batch,) or target_target.shape != (batch,):
        raise ValueError("endpoint target index shape drift")
    if active_mask.shape != (batch,) or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be bool [B]")
    if not bool(active_mask.any()):
        return source_weight.sum() * 0.0
    if int(source_target[active_mask].min()) < 0 or int(source_target[active_mask].max()) >= fields:
        raise ValueError("source endpoint target outside field range")
    if int(target_target[active_mask].min()) < 0 or int(target_target[active_mask].max()) >= fields:
        raise ValueError("target endpoint target outside field range")
    source_prob = source_weight.gather(
        1, source_target.long()[:, None]
    ).squeeze(1).clamp_min(1.0e-8)
    target_prob = target_weight.gather(
        1, target_target.long()[:, None]
    ).squeeze(1).clamp_min(1.0e-8)
    return -0.5 * (
        source_prob[active_mask].log().mean()
        + target_prob[active_mask].log().mean()
    )


def decisive_view_causal_margin_loss(
    normal_candidate_logits: Tensor,
    ablated_candidate_logits: Tensor,
    target_index: Tensor,
    *,
    margin: float = 0.15,
) -> Tensor:
    if normal_candidate_logits.shape != ablated_candidate_logits.shape:
        raise ValueError("normal/ablated candidate-logit shape drift")
    if target_index.shape != (normal_candidate_logits.size(0),):
        raise ValueError("causal target_index shape drift")
    normal = torch.softmax(normal_candidate_logits, dim=-1).gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    ablated = torch.softmax(ablated_candidate_logits, dim=-1).gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    return F.relu(float(margin) - (normal - ablated)).mean()


def irrelevant_view_invariance_loss(
    normal_candidate_logits: Tensor,
    irrelevant_removed_logits: Tensor,
) -> Tensor:
    if normal_candidate_logits.shape != irrelevant_removed_logits.shape:
        raise ValueError("invariance candidate-logit shape drift")
    p = torch.log_softmax(normal_candidate_logits, dim=-1)
    q = torch.log_softmax(irrelevant_removed_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    midpoint = 0.5 * (p_prob + q_prob)
    log_mid = midpoint.clamp_min(1.0e-8).log()
    return 0.5 * (
        F.kl_div(log_mid, p_prob, reduction="batchmean")
        + F.kl_div(log_mid, q_prob, reduction="batchmean")
    )


def latent_noncollapse_loss(
    latent_slots: Tensor,
    *,
    similarity_margin: float = 0.90,
) -> Tensor:
    if latent_slots.ndim != 3 or latent_slots.size(1) < 2:
        raise ValueError("latent_slots must be [B,S,D] with S>=2")
    normalized = F.normalize(latent_slots.float(), dim=-1)
    cosine = torch.einsum("bsd,btd->bst", normalized, normalized)
    slots = latent_slots.size(1)
    offdiag = ~torch.eye(
        slots,
        device=latent_slots.device,
        dtype=torch.bool,
    )[None, :, :]
    selected = cosine.masked_select(offdiag.expand_as(cosine))
    redundancy = F.relu(selected - float(similarity_margin)).square().mean()

    centered = latent_slots.float() - latent_slots.float().mean(
        dim=1, keepdim=True
    )
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum(dim=-1)
    effective_rank = entropy.exp()
    maximum = float(min(slots - 1, latent_slots.size(-1)))
    rank_fraction = effective_rank / max(maximum, 1.0)
    rank_penalty = F.relu(0.50 - rank_fraction).square().mean()
    return redundancy + rank_penalty


def source_view_recoverability_loss(
    latent_slots: Tensor,
    source_views: Tensor,
    view_available: Tensor,
) -> Tensor:
    if latent_slots.ndim != 3 or source_views.ndim != 3:
        raise ValueError("latent/source views must be rank 3")
    if latent_slots.size(0) != source_views.size(0):
        raise ValueError("latent/source batch drift")
    if latent_slots.size(-1) != source_views.size(-1):
        raise ValueError("latent/source width drift")
    if view_available.shape != source_views.shape[:2]:
        raise ValueError("view_available shape drift")
    slots = F.normalize(latent_slots.float(), dim=-1)
    views = F.normalize(source_views.float(), dim=-1)
    cosine = torch.einsum("bsd,bvd->bsv", slots, views)
    best = cosine.max(dim=1).values
    selected = (1.0 - best).masked_select(view_available)
    if selected.numel() == 0:
        return cosine.sum() * 0.0
    return selected.mean()


def permutation_consistency_loss(
    original: Tensor,
    permuted: Tensor,
) -> Tensor:
    if original.shape != permuted.shape:
        raise ValueError("permutation-consistency shape drift")
    return (1.0 - F.cosine_similarity(
        original.float(),
        permuted.float(),
        dim=-1,
    )).mean()


def full_envelope_behavioral_objective(
    *,
    candidate_logits: Tensor,
    target_index: Tensor,
    support_logits: Tensor,
    support_target: Tensor,
    support_valid_mask: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    source_target_index: Tensor,
    target_target_index: Tensor,
    endpoint_active_mask: Tensor,
    decisive_ablated_candidate_logits: Tensor,
    irrelevant_removed_candidate_logits: Tensor,
    latent_slots: Tensor,
    source_views: Tensor,
    view_available: Tensor,
    pooled_state_permuted: Tensor,
    pooled_state_original: Tensor,
    weights: FullEnvelopeBehavioralWeights | None = None,
) -> dict[str, Tensor]:
    w = weights or FullEnvelopeBehavioralWeights()
    w.validate()
    judgment = public_judgment_loss(candidate_logits, target_index)
    support = support_selection_loss(
        support_logits,
        support_target,
        support_valid_mask,
    )
    endpoints = endpoint_role_loss(
        source_weight,
        target_weight,
        source_target_index,
        target_target_index,
        endpoint_active_mask,
    )
    decisive = decisive_view_causal_margin_loss(
        candidate_logits,
        decisive_ablated_candidate_logits,
        target_index,
    )
    irrelevant = irrelevant_view_invariance_loss(
        candidate_logits,
        irrelevant_removed_candidate_logits,
    )
    noncollapse = latent_noncollapse_loss(latent_slots)
    recoverability = source_view_recoverability_loss(
        latent_slots,
        source_views,
        view_available,
    )
    permutation = permutation_consistency_loss(
        pooled_state_original,
        pooled_state_permuted,
    )

    total = (
        w.public_judgment * judgment
        + w.support_selection * support
        + w.endpoint_roles * endpoints
        + w.decisive_view_causality * decisive
        + w.irrelevant_view_invariance * irrelevant
        + w.latent_noncollapse * noncollapse
        + w.source_view_recoverability * recoverability
        + w.permutation_consistency * permutation
    )
    return {
        "loss": total,
        "public_judgment": judgment,
        "support_selection": support,
        "endpoint_roles": endpoints,
        "decisive_view_causality": decisive,
        "irrelevant_view_invariance": irrelevant,
        "latent_noncollapse": noncollapse,
        "source_view_recoverability": recoverability,
        "permutation_consistency": permutation,
    }
