from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def validate_teacher_row_for_v02(row: Mapping[str, Any]) -> None:
    """Validate the extra supervision N0 v0.2 requires from a teacher row.

    v0.1 could train from preferred candidate indices alone. v0.2 requires an
    explicit rationale because the reason for the preference is itself a
    training target. This remains public semantic supervision; it is not an
    authorization path for private identity gradients.
    """
    candidates = row.get("candidates")
    preferred = row.get("preferred_indices")
    rationale = str(row.get("rationale", "")).strip()

    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("v0.2 teacher row requires at least two candidates")
    if not isinstance(preferred, list) or not preferred:
        raise ValueError("v0.2 teacher row requires preferred_indices")
    if not rationale:
        raise ValueError("v0.2 teacher row requires a non-empty rationale")

    preferred_set = {int(index) for index in preferred}
    if min(preferred_set) < 0 or max(preferred_set) >= len(candidates):
        raise ValueError("v0.2 teacher row has preferred index out of range")


def principle_alignment_examples(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn one preference row into rationale-alignment supervision.

    Every candidate is paired with the teacher rationale. Preferred candidates
    are positive rationale matches; unsupported candidates are negatives. This
    makes the rationale gradient-bearing rather than inert JSON metadata.
    """
    validate_teacher_row_for_v02(row)
    preferred = {int(index) for index in row["preferred_indices"]}
    prompt = str(row["prompt"])
    rationale = str(row["rationale"])
    competency = str(row.get("competency", "unknown"))
    row_id = str(row.get("id", "unknown"))

    examples: list[dict[str, Any]] = []
    for index, candidate in enumerate(row["candidates"]):
        examples.append(
            {
                "row_id": row_id,
                "competency": competency,
                "prompt": prompt,
                "candidate": str(candidate),
                "rationale": rationale,
                "label": 1.0 if index in preferred else 0.0,
            }
        )
    return examples


def principle_alignment_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary compatibility loss between a candidate interpretation and rationale."""
    if logits.shape != labels.shape:
        raise ValueError("principle alignment logits and labels must have the same shape")
    return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


def symmetric_contrastive_loss(
    semantic_embeddings: torch.Tensor,
    rationale_embeddings: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Symmetric InfoNCE between semantic examples and their governing rationales.

    Each row's semantic representation should be closest to its own rationale
    representation, while other rationales in the batch act as negatives.
    """
    if semantic_embeddings.ndim != 2 or rationale_embeddings.ndim != 2:
        raise ValueError("contrastive embeddings must be rank-2 tensors")
    if semantic_embeddings.shape != rationale_embeddings.shape:
        raise ValueError("semantic and rationale embeddings must have identical shape")
    if semantic_embeddings.size(0) < 2:
        raise ValueError("contrastive batches need at least two examples")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    semantic = F.normalize(semantic_embeddings, dim=-1)
    rationale = F.normalize(rationale_embeddings, dim=-1)
    logits = semantic @ rationale.transpose(0, 1)
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.transpose(0, 1), labels)
    )


def weighted_objective_sum(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> torch.Tensor:
    """Combine only explicitly present objectives with normalized positive weights."""
    active = [(name, loss) for name, loss in losses.items() if name in weights and weights[name] > 0]
    if not active:
        raise ValueError("no active N0 v0.2 objectives were supplied")

    total_weight = sum(float(weights[name]) for name, _ in active)
    if total_weight <= 0:
        raise ValueError("objective weight sum must be positive")

    result = active[0][1] * (float(weights[active[0][0]]) / total_weight)
    for name, loss in active[1:]:
        result = result + loss * (float(weights[name]) / total_weight)
    return result
