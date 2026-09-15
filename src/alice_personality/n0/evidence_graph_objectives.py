from __future__ import annotations

import torch
import torch.nn.functional as F

from .structured_state_objectives import masked_field_preservation_loss


def soft_evidence_selection_loss(
    field_weights: torch.Tensor,
    target_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against a possibly multi-source evidence distribution.

    A soft target lets unresolved conflicts or jointly supporting evidence remain
    represented instead of forcing every graph into a single winning node.
    """
    if field_weights.ndim != 2 or field_weights.shape != target_distribution.shape:
        raise ValueError("field_weights and target_distribution must match [batch, fields]")
    if valid_mask.shape != field_weights.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, fields]")
    if not torch.isfinite(field_weights).all() or not torch.isfinite(target_distribution).all():
        raise ValueError("evidence distributions must be finite")
    if (field_weights < 0).any() or (target_distribution < 0).any():
        raise ValueError("evidence distributions must be non-negative")
    if (target_distribution.masked_fill(~valid_mask, 0.0).sum(dim=-1) <= 0).any():
        raise ValueError("each example needs positive target mass on a valid field")
    if (target_distribution.masked_select(~valid_mask) > 0).any():
        raise ValueError("target evidence mass may not be assigned to padding fields")

    target = target_distribution / target_distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    log_weights = field_weights.clamp_min(1e-8).log()
    return -(target * log_weights).sum(dim=-1).mean()


def pooled_semantic_alignment_loss(
    pooled_state: torch.Tensor,
    semantic_target: torch.Tensor,
) -> torch.Tensor:
    """Align the graph read with the semantic conclusion implied by the evidence."""
    if pooled_state.ndim != 2 or pooled_state.shape != semantic_target.shape:
        raise ValueError("pooled_state and semantic_target must match [batch, hidden]")
    return (1.0 - F.cosine_similarity(pooled_state, semantic_target, dim=-1)).mean()


def relation_counterfactual_margin_loss(
    correct_pooled_state: torch.Tensor,
    corrupted_pooled_state: torch.Tensor,
    semantic_target: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
    margin: float = 0.10,
) -> torch.Tensor:
    """Require correct relations to explain the target better than corrupted ones.

    This prevents the sidecar from learning to ignore edges while succeeding only
    from node semantics or query wording.
    """
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if (
        correct_pooled_state.ndim != 2
        or correct_pooled_state.shape != corrupted_pooled_state.shape
        or correct_pooled_state.shape != semantic_target.shape
    ):
        raise ValueError("correct, corrupted, and target states must match [batch, hidden]")

    correct = F.cosine_similarity(correct_pooled_state, semantic_target, dim=-1)
    corrupted = F.cosine_similarity(corrupted_pooled_state, semantic_target, dim=-1)
    per_example = F.relu(margin - correct + corrupted)
    if active_mask is not None:
        if active_mask.ndim != 1 or active_mask.numel() != per_example.numel():
            raise ValueError("active_mask must have shape [batch]")
        if active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be bool")
        if not active_mask.any():
            return per_example.new_zeros(())
        per_example = per_example[active_mask]
    return per_example.mean()


def pooled_permutation_consistency_loss(
    original: torch.Tensor,
    permuted: torch.Tensor,
) -> torch.Tensor:
    if original.ndim != 2 or original.shape != permuted.shape:
        raise ValueError("original and permuted pooled states must match [batch, hidden]")
    return (1.0 - F.cosine_similarity(original, permuted, dim=-1)).mean()


def evidence_graph_objective(
    *,
    field_weights: torch.Tensor,
    target_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    pooled_state: torch.Tensor,
    semantic_target: torch.Tensor,
    corrupted_pooled_state: torch.Tensor,
    counterfactual_active_mask: torch.Tensor,
    field_states: torch.Tensor,
    field_semantic: torch.Tensor,
    permuted_pooled_state: torch.Tensor,
    selection_weight: float = 0.35,
    semantic_weight: float = 0.30,
    counterfactual_weight: float = 0.20,
    field_preservation_weight: float = 0.10,
    permutation_weight: float = 0.05,
    counterfactual_margin: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Capability-first public relation objective for the N0 evidence graph."""
    weights = {
        "selection": selection_weight,
        "semantic": semantic_weight,
        "counterfactual": counterfactual_weight,
        "field_preservation": field_preservation_weight,
        "permutation": permutation_weight,
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("evidence graph objective weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("evidence graph objective weights must sum to 1.0")

    selection = soft_evidence_selection_loss(field_weights, target_distribution, valid_mask)
    semantic = pooled_semantic_alignment_loss(pooled_state, semantic_target)
    counterfactual = relation_counterfactual_margin_loss(
        pooled_state,
        corrupted_pooled_state,
        semantic_target,
        active_mask=counterfactual_active_mask,
        margin=counterfactual_margin,
    )
    field_preservation = masked_field_preservation_loss(field_states, field_semantic, valid_mask)
    permutation = pooled_permutation_consistency_loss(pooled_state, permuted_pooled_state)
    total = (
        selection_weight * selection
        + semantic_weight * semantic
        + counterfactual_weight * counterfactual
        + field_preservation_weight * field_preservation
        + permutation_weight * permutation
    )
    return {
        "loss": total,
        "selection": selection,
        "semantic": semantic,
        "counterfactual": counterfactual,
        "field_preservation": field_preservation,
        "permutation": permutation,
    }
