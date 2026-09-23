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
    candidate_valid_mask: Tensor | None = None,
    hard_negative_margin: float = 0.20,
) -> Tensor:
    if candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must be [B,C]")
    batch, candidates = candidate_logits.shape
    if target_index.shape != (batch,):
        raise ValueError("target_index shape drift")
    if candidate_valid_mask is None:
        candidate_valid_mask = torch.ones_like(
            candidate_logits,
            dtype=torch.bool,
        )
    if (
        candidate_valid_mask.shape != candidate_logits.shape
        or candidate_valid_mask.dtype != torch.bool
    ):
        raise ValueError("candidate_valid_mask must be bool [B,C]")
    if bool((candidate_valid_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every example requires a valid judgment candidate")
    if bool((target_index < 0).any()) or bool((target_index >= candidates).any()):
        raise ValueError("target_index outside candidate axis")
    target_valid = candidate_valid_mask.gather(
        1,
        target_index.long()[:, None],
    ).squeeze(1)
    if not bool(target_valid.all()):
        raise ValueError("judgment target points to a padded candidate")

    valid_logits = candidate_logits.masked_select(candidate_valid_mask)
    if not bool(torch.isfinite(valid_logits).all()):
        raise ValueError("valid judgment candidate logits must be finite")
    invalid_floor = torch.finfo(candidate_logits.dtype).min
    if bool((valid_logits <= invalid_floor).any()):
        raise ValueError(
            "valid judgment candidate logit reached reserved invalid floor"
        )
    masked_logits = candidate_logits.masked_fill(
        ~candidate_valid_mask,
        invalid_floor,
    )
    ce = F.cross_entropy(masked_logits, target_index.long())
    correct = masked_logits.gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    negative_mask = candidate_valid_mask.clone()
    negative_mask.scatter_(1, target_index.long()[:, None], False)
    has_negative = negative_mask.any(dim=-1)
    if not bool(has_negative.any()):
        return ce
    hardest = masked_logits.masked_fill(
        ~negative_mask,
        invalid_floor,
    ).max(dim=-1).values
    margin = F.relu(
        float(hard_negative_margin) - (correct - hardest)
    )
    margin = margin.masked_select(has_negative).mean()
    return ce + margin


def support_selection_loss(
    support_logits: Tensor,
    support_target: Tensor,
    valid_mask: Tensor,
    *,
    null_support_logit: Tensor | None = None,
) -> Tensor:
    if support_logits.shape != support_target.shape:
        raise ValueError("support target shape drift")
    if valid_mask.shape != support_logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("support valid mask shape drift")
    batch = support_logits.size(0)
    target = support_target.float()

    if bool(valid_mask.any()):
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
        edge_loss = F.binary_cross_entropy_with_logits(
            support_logits[valid_mask],
            selected,
            pos_weight=pos_weight,
        )
    else:
        edge_loss = support_logits.sum() * 0.0

    if null_support_logit is None:
        return edge_loss
    if null_support_logit.shape != (batch,):
        raise ValueError("null_support_logit must be [B]")
    positive_by_row = (
        target * valid_mask.to(target.dtype)
    ).sum(dim=-1) > 0
    null_target = (~positive_by_row).to(null_support_logit.dtype)
    null_loss = F.binary_cross_entropy_with_logits(
        null_support_logit,
        null_target,
    )
    return 0.5 * (edge_loss + null_loss)


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
    # Inactive rows may use -1 as an explicit no-endpoint sentinel. Clamp only
    # for the mechanical gather; inactive rows are excluded from the loss.
    safe_source_target = source_target.long().clamp(min=0, max=fields - 1)
    safe_target_target = target_target.long().clamp(min=0, max=fields - 1)
    source_prob = source_weight.gather(
        1, safe_source_target[:, None]
    ).squeeze(1).clamp_min(1.0e-8)
    target_prob = target_weight.gather(
        1, safe_target_target[:, None]
    ).squeeze(1).clamp_min(1.0e-8)
    return -0.5 * (
        source_prob[active_mask].log().mean()
        + target_prob[active_mask].log().mean()
    )


def _mask_candidate_logits(
    logits: Tensor,
    candidate_valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2:
        raise ValueError("candidate logits must be [B,C]")
    if candidate_valid_mask is None:
        candidate_valid_mask = torch.ones_like(logits,dtype=torch.bool)
    if (
        candidate_valid_mask.shape != logits.shape
        or candidate_valid_mask.dtype != torch.bool
    ):
        raise ValueError("candidate_valid_mask must be bool [B,C]")
    if bool((candidate_valid_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every example requires at least one valid candidate")
    valid_logits = logits.masked_select(candidate_valid_mask)
    if not bool(torch.isfinite(valid_logits).all()):
        raise ValueError("valid candidate logits must be finite")
    invalid_floor = torch.finfo(logits.dtype).min
    if bool((valid_logits <= invalid_floor).any()):
        raise ValueError("valid candidate logit reached reserved invalid floor")
    return (
        logits.masked_fill(~candidate_valid_mask, invalid_floor),
        candidate_valid_mask,
    )


def _normalized_candidate_entropy(
    logits: Tensor,
    candidate_valid_mask: Tensor | None,
) -> Tensor:
    masked,valid=_mask_candidate_logits(logits,candidate_valid_mask)
    log_probability=torch.log_softmax(masked.float(),dim=-1)
    probability=log_probability.exp()
    contribution=-(probability*log_probability).masked_fill(~valid,0.0)
    raw=contribution.sum(dim=-1)
    count=valid.sum(dim=-1)
    denom=count.clamp_min(2).float().log()
    return torch.where(
        count>1,
        raw/denom,
        torch.zeros_like(raw),
    )


def evidence_removal_uncertainty_margin_loss(
    normal_candidate_logits: Tensor,
    ablated_candidate_logits: Tensor,
    *,
    candidate_valid_mask: Tensor | None = None,
    active_mask: Tensor | None = None,
    minimum_entropy_increase: float = 0.05,
) -> Tensor:
    """Require decisive-evidence removal to increase judgment uncertainty.

    A target-probability drop alone can be satisfied by switching to a different
    answer with high confidence. N0's uncertainty contract instead requires the
    governed public judgment distribution to become less certain when decisive
    evidence is removed.
    """
    if normal_candidate_logits.shape!=ablated_candidate_logits.shape:
        raise ValueError("evidence-removal uncertainty logit geometry drift")
    if float(minimum_entropy_increase)<0.0:
        raise ValueError("minimum_entropy_increase must be non-negative")
    normal_entropy=_normalized_candidate_entropy(
        normal_candidate_logits,candidate_valid_mask
    )
    ablated_entropy=_normalized_candidate_entropy(
        ablated_candidate_logits,candidate_valid_mask
    )
    batch=normal_candidate_logits.size(0)
    if active_mask is None:
        active_mask=torch.ones(
            batch,device=normal_candidate_logits.device,dtype=torch.bool
        )
    if active_mask.shape!=(batch,) or active_mask.dtype!=torch.bool:
        raise ValueError("evidence-removal uncertainty active_mask must be bool [B]")
    raw=F.relu(
        float(minimum_entropy_increase)
        -(ablated_entropy-normal_entropy)
    )
    selected=raw.masked_select(active_mask)
    if selected.numel()==0:
        valid=(
            torch.ones_like(normal_candidate_logits,dtype=torch.bool)
            if candidate_valid_mask is None
            else candidate_valid_mask
        )
        return (
            normal_candidate_logits.masked_select(valid).sum()
            + ablated_candidate_logits.masked_select(valid).sum()
        )*0.0
    loss=selected.mean()
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("evidence-removal uncertainty loss became non-finite")
    return loss


def decisive_view_causal_margin_loss(
    normal_candidate_logits: Tensor,
    ablated_candidate_logits: Tensor,
    target_index: Tensor,
    *,
    candidate_valid_mask: Tensor | None = None,
    active_mask: Tensor | None = None,
    margin: float = 0.15,
) -> Tensor:
    if normal_candidate_logits.shape != ablated_candidate_logits.shape:
        raise ValueError("normal/ablated candidate-logit shape drift")
    if target_index.shape != (normal_candidate_logits.size(0),):
        raise ValueError("causal target_index shape drift")
    normal_logits, valid = _mask_candidate_logits(
        normal_candidate_logits,
        candidate_valid_mask,
    )
    ablated_logits, _ = _mask_candidate_logits(
        ablated_candidate_logits,
        valid,
    )
    target_valid = valid.gather(
        1,target_index.long()[:,None]
    ).squeeze(1)
    if not bool(target_valid.all()):
        raise ValueError("causal target points to padded candidate")
    normal = torch.softmax(normal_logits, dim=-1).gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    ablated = torch.softmax(ablated_logits, dim=-1).gather(
        1, target_index.long()[:, None]
    ).squeeze(1)
    raw = F.relu(float(margin) - (normal - ablated))
    if active_mask is None:
        active_mask = torch.ones(
            normal_logits.size(0),
            device=normal_logits.device,
            dtype=torch.bool,
        )
    if active_mask.shape != (normal_logits.size(0),) or active_mask.dtype != torch.bool:
        raise ValueError("decisive causal active_mask must be bool [B]")
    selected = raw.masked_select(active_mask)
    if selected.numel() == 0:
        return (
            normal_candidate_logits.masked_select(valid).sum()
            + ablated_candidate_logits.masked_select(valid).sum()
        ) * 0.0
    return selected.mean()


def irrelevant_view_invariance_loss(
    normal_candidate_logits: Tensor,
    irrelevant_removed_logits: Tensor,
    *,
    candidate_valid_mask: Tensor | None = None,
    active_mask: Tensor | None = None,
) -> Tensor:
    """Jensen-Shannon invariance with finite gradients under masked candidates.

    The previous implementation passed model-derived probabilities as the
    differentiable target argument of F.kl_div. Masked logits can underflow
    those probabilities to exact zero. KL's target-side derivative contains
    log(target), so the forward value stayed finite while backward produced
    NaNs across the shared backbone.

    Compute JSD directly in log-space instead. Invalid candidates are excluded
    explicitly, and no log is ever taken on a probability tensor.
    """
    if normal_candidate_logits.shape != irrelevant_removed_logits.shape:
        raise ValueError("invariance candidate-logit shape drift")
    normal_logits, valid = _mask_candidate_logits(
        normal_candidate_logits,
        candidate_valid_mask,
    )
    removed_logits, _ = _mask_candidate_logits(
        irrelevant_removed_logits,
        valid,
    )
    log_p = torch.log_softmax(normal_logits, dim=-1)
    log_q = torch.log_softmax(removed_logits, dim=-1)
    p_prob = log_p.exp()
    q_prob = log_q.exp()
    log_mid = torch.logaddexp(log_p, log_q) - torch.log(
        torch.tensor(
            2.0,
            device=log_p.device,
            dtype=log_p.dtype,
        )
    )
    per_candidate = 0.5 * (
        p_prob * (log_p - log_mid)
        + q_prob * (log_q - log_mid)
    )
    per_candidate = per_candidate.masked_fill(~valid, 0.0)
    per_row = per_candidate.sum(dim=-1)
    if active_mask is None:
        active_mask = torch.ones(
            normal_logits.size(0),
            device=normal_logits.device,
            dtype=torch.bool,
        )
    if active_mask.shape != (normal_logits.size(0),) or active_mask.dtype != torch.bool:
        raise ValueError("irrelevant invariance active_mask must be bool [B]")
    selected = per_row.masked_select(active_mask)
    if selected.numel() == 0:
        return (
            normal_candidate_logits.masked_select(valid).sum()
            + irrelevant_removed_logits.masked_select(valid).sum()
        ) * 0.0
    loss = selected.mean()
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("irrelevant-view invariance loss became non-finite")
    return loss


def latent_noncollapse_loss(
    latent_slots: Tensor,
    *,
    similarity_margin: float = 0.90,
    minimum_directional_spread: float = 0.20,
) -> Tensor:
    """Numerically stable anti-collapse objective for latent slots.

    The earlier implementation differentiated through SVD singular values.
    That gradient is undefined when singular values repeat, which is common
    early in training and produced finite forward losses with NaN backbone
    gradients. This version uses only smooth first/second-order geometry.

    Pairwise absolute cosine catches both duplicate and anti-correlated
    rank-one collapse. Directional spread catches the all-equal/all-zero case
    without an eigendecomposition or SVD.
    """
    if latent_slots.ndim != 3 or latent_slots.size(1) < 1:
        raise ValueError("latent_slots must be [B,S,D] with S>=1")
    if not 0.0 <= float(similarity_margin) < 1.0:
        raise ValueError("similarity_margin must be in [0,1)")
    if not 0.0 <= float(minimum_directional_spread) <= 1.0:
        raise ValueError("minimum_directional_spread must be in [0,1]")
    if latent_slots.size(1) == 1:
        return latent_slots.sum() * 0.0

    # A larger epsilon avoids extreme derivatives for nearly-zero slots while
    # preserving direction information at ordinary activation magnitudes.
    normalized = F.normalize(
        latent_slots.float(),
        dim=-1,
        eps=1.0e-4,
    )
    cosine = torch.einsum("bsd,btd->bst", normalized, normalized)
    slots = latent_slots.size(1)
    offdiag = ~torch.eye(
        slots,
        device=latent_slots.device,
        dtype=torch.bool,
    )[None, :, :]
    selected = cosine.masked_select(offdiag.expand_as(cosine))
    redundancy = F.relu(
        selected.abs() - float(similarity_margin)
    ).square().mean()

    # For unit vectors this equals 1 - ||mean direction||^2. It is zero for
    # duplicate directions (and for all-zero slots) and grows as slots occupy
    # distinct directions. No spectral decomposition is required.
    mean_direction = normalized.mean(dim=1, keepdim=True)
    centered_direction = normalized - mean_direction
    directional_spread = centered_direction.square().sum(
        dim=-1
    ).mean(dim=-1)
    spread_penalty = F.relu(
        float(minimum_directional_spread) - directional_spread
    ).square().mean()

    loss = redundancy + spread_penalty
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("latent noncollapse loss became non-finite")
    return loss


def source_view_recoverability_loss(
    latent_slots: Tensor,
    source_views: Tensor,
    view_available: Tensor,
    recoverable_view_mask: Tensor | None = None,
) -> Tensor:
    if latent_slots.ndim != 3 or source_views.ndim != 3:
        raise ValueError("latent/source views must be rank 3")
    if latent_slots.size(0) != source_views.size(0):
        raise ValueError("latent/source batch drift")
    if latent_slots.size(-1) != source_views.size(-1):
        raise ValueError("latent/source width drift")
    if view_available.shape != source_views.shape[:2]:
        raise ValueError("view_available shape drift")
    if view_available.dtype != torch.bool:
        raise ValueError("view_available must be bool")
    if recoverable_view_mask is None:
        recoverable_view_mask = view_available
    if (
        recoverable_view_mask.shape != view_available.shape
        or recoverable_view_mask.dtype != torch.bool
    ):
        raise ValueError("recoverable_view_mask must be bool [B,V]")
    recoverable_view_mask = recoverable_view_mask & view_available
    slots = F.normalize(latent_slots.float(), dim=-1)
    views = F.normalize(source_views.float(), dim=-1)
    cosine = torch.einsum("bsd,bvd->bsv", slots, views)
    best = cosine.max(dim=1).values
    selected = (1.0 - best).masked_select(recoverable_view_mask)
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
    candidate_valid_mask: Tensor | None = None,
    support_logits: Tensor,
    support_target: Tensor,
    support_valid_mask: Tensor,
    null_support_logit: Tensor | None = None,
    source_weight: Tensor,
    target_weight: Tensor,
    source_target_index: Tensor,
    target_target_index: Tensor,
    endpoint_active_mask: Tensor,
    decisive_ablated_candidate_logits: Tensor,
    decisive_view_active_mask: Tensor | None = None,
    irrelevant_removed_candidate_logits: Tensor,
    irrelevant_view_active_mask: Tensor | None = None,
    latent_slots: Tensor,
    source_views: Tensor,
    view_available: Tensor,
    recoverable_view_mask: Tensor | None = None,
    pooled_state_permuted: Tensor,
    pooled_state_original: Tensor,
    weights: FullEnvelopeBehavioralWeights | None = None,
) -> dict[str, Tensor]:
    w = weights or FullEnvelopeBehavioralWeights()
    w.validate()
    judgment = public_judgment_loss(
        candidate_logits,
        target_index,
        candidate_valid_mask=candidate_valid_mask,
    )
    support = support_selection_loss(
        support_logits,
        support_target,
        support_valid_mask,
        null_support_logit=null_support_logit,
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
        candidate_valid_mask=candidate_valid_mask,
        active_mask=decisive_view_active_mask,
    )
    irrelevant = irrelevant_view_invariance_loss(
        candidate_logits,
        irrelevant_removed_candidate_logits,
        candidate_valid_mask=candidate_valid_mask,
        active_mask=irrelevant_view_active_mask,
    )
    noncollapse = latent_noncollapse_loss(latent_slots)
    recoverability = source_view_recoverability_loss(
        latent_slots,
        source_views,
        view_available,
        recoverable_view_mask,
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
