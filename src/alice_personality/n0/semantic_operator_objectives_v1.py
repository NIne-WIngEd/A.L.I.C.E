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


def relation_semantic_supervision_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    relation_plurality_target_distribution: Tensor | None = None,
    relation_plurality_mask: Tensor | None = None,
) -> Tensor:
    if logits.ndim != target.ndim + 1:
        raise ValueError("relation logits/target rank mismatch")
    if target.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("relation target/mask geometry drift")
    if relation_plurality_target_distribution is None:
        if relation_plurality_mask is not None:
            raise ValueError("plurality mask supplied without target distribution")
        return masked_cross_entropy(logits,target,mask)
    if relation_plurality_mask is None:
        raise ValueError("plurality target distribution supplied without mask")
    if relation_plurality_target_distribution.shape != logits.shape:
        raise ValueError("plurality target distribution geometry drift")
    if relation_plurality_mask.shape != target.shape or relation_plurality_mask.dtype != torch.bool:
        raise ValueError("plurality mask geometry drift")
    if bool((relation_plurality_mask & ~mask).any()):
        raise ValueError("plurality supervision must be an active relation step")

    if bool(relation_plurality_mask.any()):
        selected=relation_plurality_target_distribution[
            relation_plurality_mask
        ]
        if bool((selected < 0.0).any()):
            raise ValueError("plurality target probability must be non-negative")
        total=selected.sum(dim=-1)
        if not torch.allclose(
            total,torch.ones_like(total),atol=1.0e-6,rtol=1.0e-6
        ):
            raise ValueError("plurality target distribution must sum to one")
        if bool((selected.gt(0.0).sum(dim=-1) < 2).any()):
            raise ValueError(
                "plurality target needs at least two positive hypotheses"
            )

    if not bool(mask.any()):
        return logits.sum()*0.0
    log_probability=F.log_softmax(logits.float(),dim=-1)
    hard=-log_probability.gather(
        -1,target.long().unsqueeze(-1)
    ).squeeze(-1)
    soft=-(
        relation_plurality_target_distribution.to(log_probability.dtype)
        * log_probability
    ).sum(dim=-1)
    per_step=torch.where(relation_plurality_mask,soft,hard)
    return per_step.masked_select(mask).mean()


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


def step_factor_semantic_loss(
    step_factor_logits: Mapping[str, Tensor],
    step_factor_targets: Mapping[str, Tensor],
    step_mask: Tensor,
) -> Tensor:
    if set(step_factor_logits) != set(step_factor_targets):
        raise ValueError("step factor logits/targets names must match")
    if step_mask.ndim != 2 or step_mask.dtype != torch.bool:
        raise ValueError("step factor mask must be bool [B,S]")
    losses = []
    for name in sorted(step_factor_logits):
        logits = step_factor_logits[name]
        target = step_factor_targets[name]
        if logits.ndim != 3:
            raise ValueError(f"step factor logits {name!r} must be [B,S,C]")
        if target.shape != logits.shape[:2] or step_mask.shape != logits.shape[:2]:
            raise ValueError(f"step factor target geometry drift for {name!r}")
        losses.append(
            masked_cross_entropy(
                logits,
                target.long(),
                step_mask,
            )
        )
    if not losses:
        raise ValueError("at least one step-conditioned factor bank is required")
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


def mapping_token_evidence_loss(
    predicted: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    valid_mask: Mapping[str, Tensor],
    *,
    label: str,
) -> Tensor:
    if set(predicted) != set(target) or set(predicted) != set(valid_mask):
        raise ValueError(f"{label} evidence bank names must match")
    losses = [
        binary_token_evidence_loss(
            predicted[name],
            target[name],
            valid_mask[name].bool(),
        )
        for name in sorted(predicted)
    ]
    if not losses:
        raise ValueError(f"{label} evidence requires at least one bank")
    return torch.stack(losses).mean()


def pairwise_margin_loss(
    correct_score: Tensor,
    counterfactual_score: Tensor,
    *,
    margin: float = 0.20,
    valid_mask: Tensor | None = None,
) -> Tensor:
    if correct_score.shape != counterfactual_score.shape:
        raise ValueError("causal margin tensors must have matching shapes")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    raw = F.relu(float(margin) - (correct_score - counterfactual_score))
    if valid_mask is None:
        if raw.numel() == 0:
            return (correct_score.sum() + counterfactual_score.sum()) * 0.0
        return raw.mean()
    if valid_mask.shape != raw.shape or valid_mask.dtype != torch.bool:
        raise ValueError("causal margin valid_mask shape drift")
    selected = raw.masked_select(valid_mask)
    if selected.numel() == 0:
        return (correct_score.sum() + counterfactual_score.sum()) * 0.0
    return selected.mean()


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
    relation_plurality_target_distribution: Tensor | None = None,
    relation_plurality_mask: Tensor | None = None,
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
    relation_margin_valid_mask: Tensor | None = None,
    factor_margin_valid_mask: Tensor | None = None,
    step_factor_logits: Mapping[str, Tensor] | None = None,
    step_factor_targets: Mapping[str, Tensor] | None = None,
    step_factor_mask: Tensor | None = None,
    relation_schema_evidence: Tensor | None = None,
    relation_schema_evidence_target: Tensor | None = None,
    relation_schema_evidence_valid_mask: Tensor | None = None,
    factor_schema_evidence: Mapping[str, Tensor] | None = None,
    factor_schema_evidence_target: Mapping[str, Tensor] | None = None,
    factor_schema_evidence_valid_mask: Mapping[str, Tensor] | None = None,
    step_factor_schema_evidence: Mapping[str, Tensor] | None = None,
    step_factor_schema_evidence_target: Mapping[str, Tensor] | None = None,
    step_factor_schema_evidence_valid_mask: Mapping[str, Tensor] | None = None,
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

    relation = relation_semantic_supervision_loss(
        relation_logits,
        relation_targets.long(),
        relation_step_mask.bool(),
        relation_plurality_target_distribution=(
            relation_plurality_target_distribution
        ),
        relation_plurality_mask=relation_plurality_mask,
    )
    global_factor = factor_semantic_loss(factor_logits, factor_targets)
    supplied_step = (
        step_factor_logits is not None
        or step_factor_targets is not None
        or step_factor_mask is not None
    )
    if supplied_step:
        if (
            step_factor_logits is None
            or step_factor_targets is None
            or step_factor_mask is None
        ):
            raise ValueError(
                "step factor logits, targets and mask must be supplied together"
            )
        step_factor = step_factor_semantic_loss(
            step_factor_logits,
            step_factor_targets,
            step_factor_mask,
        )
        factor = 0.5 * (global_factor + step_factor)
    else:
        step_factor = global_factor * 0.0
        factor = global_factor

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

    query_evidence_loss = binary_token_evidence_loss(
        relation_query_evidence,
        query_evidence_target,
        query_evidence_valid_mask.bool(),
    )
    evidence_channels = [query_evidence_loss]

    relation_schema_supplied = (
        relation_schema_evidence is not None
        or relation_schema_evidence_target is not None
        or relation_schema_evidence_valid_mask is not None
    )
    if relation_schema_supplied:
        if (
            relation_schema_evidence is None
            or relation_schema_evidence_target is None
            or relation_schema_evidence_valid_mask is None
        ):
            raise ValueError(
                "relation schema evidence, target and valid mask must be supplied together"
            )
        relation_schema_evidence_loss = binary_token_evidence_loss(
            relation_schema_evidence,
            relation_schema_evidence_target,
            relation_schema_evidence_valid_mask.bool(),
        )
        evidence_channels.append(relation_schema_evidence_loss)
    else:
        relation_schema_evidence_loss = query_evidence_loss * 0.0

    factor_schema_supplied = (
        factor_schema_evidence is not None
        or factor_schema_evidence_target is not None
        or factor_schema_evidence_valid_mask is not None
    )
    if factor_schema_supplied:
        if (
            factor_schema_evidence is None
            or factor_schema_evidence_target is None
            or factor_schema_evidence_valid_mask is None
        ):
            raise ValueError(
                "factor schema evidence mappings must be supplied together"
            )
        factor_schema_evidence_loss = mapping_token_evidence_loss(
            factor_schema_evidence,
            factor_schema_evidence_target,
            factor_schema_evidence_valid_mask,
            label="factor schema",
        )
        evidence_channels.append(factor_schema_evidence_loss)
    else:
        factor_schema_evidence_loss = query_evidence_loss * 0.0

    step_factor_schema_supplied = (
        step_factor_schema_evidence is not None
        or step_factor_schema_evidence_target is not None
        or step_factor_schema_evidence_valid_mask is not None
    )
    if step_factor_schema_supplied:
        if (
            step_factor_schema_evidence is None
            or step_factor_schema_evidence_target is None
            or step_factor_schema_evidence_valid_mask is None
        ):
            raise ValueError(
                "step factor schema evidence mappings must be supplied together"
            )
        step_factor_schema_evidence_loss = mapping_token_evidence_loss(
            step_factor_schema_evidence,
            step_factor_schema_evidence_target,
            step_factor_schema_evidence_valid_mask,
            label="step factor schema",
        )
        evidence_channels.append(step_factor_schema_evidence_loss)
    else:
        step_factor_schema_evidence_loss = query_evidence_loss * 0.0

    # Keep the precommitted 0.10 evidence-family macro weight unchanged.
    # Multiple supervised evidence surfaces are macro-averaged inside the
    # family so adding factor banks or token surfaces cannot silently increase
    # their optimization weight.
    evidence = torch.stack(evidence_channels).mean()

    relation_margin = pairwise_margin_loss(
        correct_relation_score,
        counterfactual_relation_score,
        valid_mask=relation_margin_valid_mask,
    )
    factor_margin = pairwise_margin_loss(
        correct_factor_score,
        counterfactual_factor_score,
        valid_mask=factor_margin_valid_mask,
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
        "global_factor_semantics": global_factor,
        "step_factor_semantics": step_factor,
        "event_control": event,
        "applicability": applicability_loss,
        "token_evidence": evidence,
        "query_evidence": query_evidence_loss,
        "relation_schema_evidence": relation_schema_evidence_loss,
        "factor_schema_evidence": factor_schema_evidence_loss,
        "step_factor_schema_evidence": step_factor_schema_evidence_loss,
        "causal_relation_margin": relation_margin,
        "causal_factor_margin": factor_margin,
        "uncertainty": uncertainty_loss,
    }
