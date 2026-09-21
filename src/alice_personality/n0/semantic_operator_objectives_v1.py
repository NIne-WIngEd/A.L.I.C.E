from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SemanticOperatorObjectiveWeights:
    relation_sequence: float = 0.24
    factor_semantics: float = 0.18
    event_control: float = 0.12
    applicability: float = 0.08
    query_evidence: float = 0.10
    causal_relation_margin: float = 0.10
    causal_factor_margin: float = 0.08
    uncertainty: float = 0.10

    def validate(self) -> None:
        values = (
            self.relation_sequence,
            self.factor_semantics,
            self.event_control,
            self.applicability,
            self.query_evidence,
            self.causal_relation_margin,
            self.causal_factor_margin,
            self.uncertainty,
        )
        if any(x < 0.0 for x in values):
            raise ValueError("objective weights must be non-negative")
        if abs(sum(values) - 1.0) > 1.0e-9:
            raise ValueError("objective weights must sum to 1")


def masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    if logits.ndim != target.ndim + 1:
        raise ValueError("logits/target rank mismatch")
    if target.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("target/mask geometry drift")
    if not bool(mask.any()):
        return logits.sum() * 0.0
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask])


def factor_semantic_loss(
    factor_logits: Mapping[str, Tensor],
    factor_targets: Mapping[str, Tensor],
) -> Tensor:
    if set(factor_logits) != set(factor_targets):
        raise ValueError("factor logits/targets names must match")
    losses = []
    for name in sorted(factor_logits):
        logits = factor_logits[name]
        target = factor_targets[name]
        if logits.ndim != 2 or target.shape != (logits.size(0),):
            raise ValueError(f"factor target shape drift for {name!r}")
        losses.append(F.cross_entropy(logits, target.long()))
    if not losses:
        raise ValueError("at least one factor bank is required")
    return torch.stack(losses).mean()


def binary_token_evidence_loss(
    predicted: Tensor,
    target: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted/target token evidence shape drift")
    if valid_mask.shape != predicted.shape or valid_mask.dtype != torch.bool:
        raise ValueError("token evidence valid mask shape drift")
    if not bool(valid_mask.any()):
        return predicted.sum() * 0.0
    probability = predicted.clamp(1.0e-5, 1.0 - 1.0e-5)
    return F.binary_cross_entropy(
        probability[valid_mask],
        target.float()[valid_mask],
    )


def pairwise_margin_loss(
    correct_score: Tensor,
    counterfactual_score: Tensor,
    *,
    margin: float = 0.20,
) -> Tensor:
    if correct_score.shape != counterfactual_score.shape:
        raise ValueError("causal margin tensors must have matching shapes")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    return F.relu(float(margin) - (correct_score - counterfactual_score)).mean()


def uncertainty_supervision_loss(
    uncertainty: Tensor,
    target: Tensor,
) -> Tensor:
    if uncertainty.shape != target.shape:
        raise ValueError("uncertainty target shape drift")
    probability = uncertainty.clamp(1.0e-5, 1.0 - 1.0e-5)
    return F.binary_cross_entropy(probability, target.float())


def semantic_operator_objective(
    *,
    relation_logits: Tensor,
    relation_targets: Tensor,
    relation_step_mask: Tensor,
    factor_logits: Mapping[str, Tensor],
    factor_targets: Mapping[str, Tensor],
    event_distribution: Tensor,
    event_targets: Tensor,
    event_mask: Tensor,
    applicability: Tensor,
    applicability_target: Tensor,
    relation_query_evidence: Tensor,
    query_evidence_target: Tensor,
    query_evidence_valid_mask: Tensor,
    uncertainty: Tensor,
    uncertainty_target: Tensor,
    correct_relation_score: Tensor,
    counterfactual_relation_score: Tensor,
    correct_factor_score: Tensor,
    counterfactual_factor_score: Tensor,
    weights: SemanticOperatorObjectiveWeights | None = None,
) -> dict[str, Tensor]:
    w = weights or SemanticOperatorObjectiveWeights()
    w.validate()

    if relation_logits.ndim != 3:
        raise ValueError("relation_logits must be [B,S,R]")
    batch, steps, _ = relation_logits.shape
    if relation_targets.shape != (batch, steps):
        raise ValueError("relation_targets shape drift")
    if relation_step_mask.shape != (batch, steps):
        raise ValueError("relation_step_mask shape drift")

    relation = masked_cross_entropy(
        relation_logits,
        relation_targets.long(),
        relation_step_mask.bool(),
    )
    factor = factor_semantic_loss(factor_logits, factor_targets)

    if event_distribution.shape != (batch, steps, 3):
        raise ValueError("event_distribution shape drift")
    event_logits = torch.log(event_distribution.clamp_min(1.0e-8))
    event = masked_cross_entropy(
        event_logits,
        event_targets.long(),
        event_mask.bool(),
    )

    if applicability.shape != applicability_target.shape:
        raise ValueError("applicability target shape drift")
    applicability_loss = F.binary_cross_entropy(
        applicability.clamp(1.0e-5, 1.0 - 1.0e-5),
        applicability_target.float(),
    )

    evidence = binary_token_evidence_loss(
        relation_query_evidence,
        query_evidence_target,
        query_evidence_valid_mask.bool(),
    )
    relation_margin = pairwise_margin_loss(
        correct_relation_score,
        counterfactual_relation_score,
    )
    factor_margin = pairwise_margin_loss(
        correct_factor_score,
        counterfactual_factor_score,
    )
    uncertainty_loss = uncertainty_supervision_loss(
        uncertainty,
        uncertainty_target,
    )

    total = (
        w.relation_sequence * relation
        + w.factor_semantics * factor
        + w.event_control * event
        + w.applicability * applicability_loss
        + w.query_evidence * evidence
        + w.causal_relation_margin * relation_margin
        + w.causal_factor_margin * factor_margin
        + w.uncertainty * uncertainty_loss
    )
    return {
        "loss": total,
        "relation_sequence": relation,
        "factor_semantics": factor,
        "event_control": event,
        "applicability": applicability_loss,
        "query_evidence": evidence,
        "causal_relation_margin": relation_margin,
        "causal_factor_margin": factor_margin,
        "uncertainty": uncertainty_loss,
    }
