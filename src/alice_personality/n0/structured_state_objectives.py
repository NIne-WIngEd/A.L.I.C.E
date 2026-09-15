from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_semantic_alignment_loss(
    structured: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch symmetric contrastive alignment for pooled structured state."""
    if structured.ndim != 2 or target.ndim != 2 or structured.shape != target.shape:
        raise ValueError("structured and target must be aligned [batch, hidden] tensors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if structured.size(0) < 2:
        raise ValueError("contrastive alignment requires batch size >= 2")

    structured = F.normalize(structured, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = structured @ target.transpose(0, 1) / temperature
    labels = torch.arange(structured.size(0), device=structured.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def masked_field_preservation_loss(
    field_states: torch.Tensor,
    semantic_values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep typed field states anchored to the semantic content they refine."""
    if field_states.shape != semantic_values.shape or field_states.ndim != 3:
        raise ValueError("field_states and semantic_values must match [batch, fields, hidden]")
    if valid_mask.shape != field_states.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, fields]")
    if not valid_mask.any():
        raise ValueError("field preservation requires at least one valid field")

    field_states = F.normalize(field_states, dim=-1)
    semantic_values = F.normalize(semantic_values, dim=-1)
    per_field = 1.0 - (field_states * semantic_values).sum(dim=-1)
    return per_field[valid_mask].mean()


def pooled_permutation_consistency_loss(
    original: torch.Tensor,
    permuted: torch.Tensor,
) -> torch.Tensor:
    """Penalize order-sensitive pooled state when field order is non-semantic."""
    if original.shape != permuted.shape or original.ndim != 2:
        raise ValueError("original and permuted must match [batch, hidden]")
    return (1.0 - F.cosine_similarity(original, permuted, dim=-1)).mean()


def binary_rationale_compatibility_loss(
    structured: torch.Tensor,
    rationale: torch.Tensor,
    labels: torch.Tensor,
    *,
    logit_scale: torch.Tensor | float = 5.0,
    positive_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Positive/negative compatibility against governed rationale supervision."""
    if structured.shape != rationale.shape or structured.ndim != 2:
        raise ValueError("structured and rationale must match [batch, hidden]")
    if labels.ndim != 1 or labels.numel() != structured.size(0):
        raise ValueError("labels must have shape [batch]")
    structured = F.normalize(structured, dim=-1)
    rationale = F.normalize(rationale, dim=-1)
    logits = (structured * rationale).sum(dim=-1) * logit_scale
    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
        if pos_weight.numel() != 1 or not torch.isfinite(pos_weight) or pos_weight.item() <= 0.0:
            raise ValueError("positive_weight must be one finite positive scalar")
    return F.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.dtype),
        pos_weight=pos_weight,
    )


def structured_state_objective(
    *,
    pooled_state: torch.Tensor,
    semantic_target: torch.Tensor,
    rationale_target: torch.Tensor,
    field_states: torch.Tensor,
    field_semantic: torch.Tensor,
    valid_mask: torch.Tensor,
    permuted_pooled_state: torch.Tensor,
    compatibility_labels: torch.Tensor,
    compatibility_positive_weight: torch.Tensor | float | None = None,
    alignment_weight: float = 0.45,
    compatibility_weight: float = 0.30,
    field_preservation_weight: float = 0.15,
    permutation_weight: float = 0.10,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Joint identity-neutral structured-state objective.

    `semantic_target` represents the meaning of the structured fields themselves.
    `rationale_target` is separate governed supervision used only for compatibility.
    Keeping these targets distinct prevents negative candidates from being pulled
    toward the same rationale that they are simultaneously trained to reject.
    """
    weights = {
        "alignment": alignment_weight,
        "compatibility": compatibility_weight,
        "field_preservation": field_preservation_weight,
        "permutation": permutation_weight,
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("structured-state objective weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("structured-state objective weights must sum to 1.0")

    alignment = symmetric_semantic_alignment_loss(
        pooled_state,
        semantic_target,
        temperature=temperature,
    )
    compatibility = binary_rationale_compatibility_loss(
        pooled_state,
        rationale_target,
        compatibility_labels,
        positive_weight=compatibility_positive_weight,
    )
    field_preservation = masked_field_preservation_loss(
        field_states,
        field_semantic,
        valid_mask,
    )
    permutation = pooled_permutation_consistency_loss(
        pooled_state,
        permuted_pooled_state,
    )
    total = (
        alignment_weight * alignment
        + compatibility_weight * compatibility
        + field_preservation_weight * field_preservation
        + permutation_weight * permutation
    )
    return {
        "loss": total,
        "alignment": alignment,
        "compatibility": compatibility,
        "field_preservation": field_preservation,
        "permutation": permutation,
    }
