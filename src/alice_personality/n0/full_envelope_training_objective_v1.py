from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
    full_envelope_behavioral_objective,
)
from alice_personality.n0.full_envelope_loss_balancer_v1 import (
    MacroFamilyLossBalancer,
)
from alice_personality.n0.semantic_operator_objectives_v1 import (
    semantic_operator_objective,
)


DEFAULT_FAMILY_WEIGHTS: dict[str, float] = {
    "broad_semantic_replay": 1.0,
    "governed_judgment_replay": 1.0,
    "relation_program_semantics": 1.0,
    "dynamic_factor_semantics": 1.0,
    "uncertainty_and_control": 1.0,
    "token_evidence_grounding": 1.0,
    "structural_support_and_roles": 1.0,
    "multi_view_causal_preservation": 1.0,
    "latent_judgment_and_noncollapse": 1.0,
    "natural_relation_semantics": 1.0,
}


def _require_public_judgment(
    outputs: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Tensor]:
    value = outputs.get("public_judgment")
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} outputs require public_judgment")
    logits = value.get("candidate_logits")
    if not isinstance(logits, Tensor):
        raise ValueError(f"{label} public_judgment missing candidate_logits")
    return value


def _gather_class_score(
    logits: Tensor,
    target: Tensor,
    *,
    name: str,
) -> Tensor:
    if logits.ndim < 2:
        raise ValueError(f"{name} logits need a candidate axis")
    if target.shape != logits.shape[:-1]:
        raise ValueError(f"{name} target geometry drift")
    if bool((target < 0).any()) or bool((target >= logits.size(-1)).any()):
        raise ValueError(f"{name} target outside candidate axis")
    return logits.gather(
        -1,
        target.long().unsqueeze(-1),
    ).squeeze(-1)


def _factor_margin_scores(
    factor_logits: Mapping[str, Tensor],
    correct_target: Mapping[str, Tensor],
    counterfactual_target: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    if set(factor_logits) != set(correct_target):
        raise ValueError("factor logits/correct target bank mismatch")
    if set(factor_logits) != set(counterfactual_target):
        raise ValueError("factor logits/counterfactual target bank mismatch")
    correct_values = []
    counter_values = []
    valid_values = []
    for name in sorted(factor_logits):
        logits = factor_logits[name]
        correct = correct_target[name]
        counter = counterfactual_target[name]
        if logits.ndim != 2:
            raise ValueError(f"factor logits {name!r} must be [B,C]")
        if correct.shape != (logits.size(0),) or counter.shape != correct.shape:
            raise ValueError(f"factor target geometry drift for {name!r}")
        if bool((correct == counter).any()):
            raise ValueError(
                f"factor counterfactual must differ from correct target for {name!r}"
            )
        correct_values.append(
            _gather_class_score(
                logits,
                correct,
                name=f"factor/{name}/correct",
            )
        )
        counter_values.append(
            _gather_class_score(
                logits,
                counter,
                name=f"factor/{name}/counterfactual",
            )
        )
        valid_values.append(torch.ones_like(correct, dtype=torch.bool))
    return (
        torch.stack(correct_values, dim=-1),
        torch.stack(counter_values, dim=-1),
        torch.stack(valid_values, dim=-1),
    )


def semantic_operator_supervision(
    *,
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Tensor]:
    semantic = outputs.get("semantic_operator")
    operator = outputs.get("operator")
    if not isinstance(semantic, Mapping) or operator is None:
        raise ValueError("full-envelope outputs missing semantic operator state")

    relation_logits = semantic["relation_logits"]
    relation_targets = targets["relation_targets"]
    relation_step_mask = targets["relation_step_mask"].bool()
    counterfactual_relation_targets = targets[
        "counterfactual_relation_targets"
    ]
    correct_relation_score = _gather_class_score(
        relation_logits,
        relation_targets,
        name="relation/correct",
    )
    counterfactual_relation_score = _gather_class_score(
        relation_logits,
        counterfactual_relation_targets,
        name="relation/counterfactual",
    )
    if bool(
        (
            relation_step_mask
            & relation_targets.eq(counterfactual_relation_targets)
        ).any()
    ):
        raise ValueError(
            "active relation counterfactual target must differ from correct target"
        )

    (
        correct_factor_score,
        counterfactual_factor_score,
        factor_margin_mask,
    ) = _factor_margin_scores(
        semantic["factor_logits"],
        targets["factor_targets"],
        targets["counterfactual_factor_targets"],
    )

    event_distribution = torch.stack(
        [
            operator.relation_step_mass,
            operator.stop_probability,
            operator.unknown_probability,
        ],
        dim=-1,
    )

    return semantic_operator_objective(
        relation_logits=relation_logits,
        relation_targets=relation_targets,
        relation_step_mask=relation_step_mask,
        factor_logits=semantic["factor_logits"],
        factor_targets=targets["factor_targets"],
        step_factor_logits=semantic["step_factor_logits"],
        step_factor_targets=targets["step_factor_targets"],
        step_factor_mask=targets["step_factor_mask"].bool(),
        event_distribution=event_distribution,
        event_targets=targets["event_targets"],
        event_mask=targets["event_mask"].bool(),
        applicability=operator.applicability,
        applicability_target=targets["applicability_target"],
        relation_query_evidence=semantic["relation_query_evidence"],
        query_evidence_target=targets["query_evidence_target"],
        query_evidence_valid_mask=targets[
            "query_evidence_valid_mask"
        ].bool(),
        relation_schema_evidence=semantic["relation_schema_evidence"],
        relation_schema_evidence_target=targets[
            "relation_schema_evidence_target"
        ],
        relation_schema_evidence_valid_mask=targets[
            "relation_schema_evidence_valid_mask"
        ].bool(),
        factor_schema_evidence=semantic["factor_schema_evidence"],
        factor_schema_evidence_target=targets[
            "factor_schema_evidence_target"
        ],
        factor_schema_evidence_valid_mask={
            name: value.bool()
            for name, value in targets[
                "factor_schema_evidence_valid_mask"
            ].items()
        },
        step_factor_schema_evidence=semantic[
            "step_factor_schema_evidence"
        ],
        step_factor_schema_evidence_target=targets[
            "step_factor_schema_evidence_target"
        ],
        step_factor_schema_evidence_valid_mask={
            name: value.bool()
            for name, value in targets[
                "step_factor_schema_evidence_valid_mask"
            ].items()
        },
        uncertainty=operator.uncertainty,
        uncertainty_target=targets["uncertainty_target"],
        correct_relation_score=correct_relation_score,
        counterfactual_relation_score=counterfactual_relation_score,
        relation_margin_valid_mask=relation_step_mask,
        correct_factor_score=correct_factor_score,
        counterfactual_factor_score=counterfactual_factor_score,
        factor_margin_valid_mask=factor_margin_mask,
    )


def behavioral_supervision(
    *,
    primary_outputs: Mapping[str, Any],
    decisive_ablated_outputs: Mapping[str, Any],
    irrelevant_removed_outputs: Mapping[str, Any],
    permuted_outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Tensor]:
    primary_judgment = _require_public_judgment(
        primary_outputs,
        label="primary",
    )
    decisive_judgment = _require_public_judgment(
        decisive_ablated_outputs,
        label="decisive_ablated",
    )
    irrelevant_judgment = _require_public_judgment(
        irrelevant_removed_outputs,
        label="irrelevant_removed",
    )

    binder = primary_outputs["binder"]
    executor = primary_outputs["executor"]
    latent = primary_outputs["latent"]
    permuted_latent = permuted_outputs["latent"]

    support_target = targets["support_target"]
    support_valid_mask = targets["support_valid_mask"].bool()
    if support_target.shape != binder["support_logits"].shape:
        raise ValueError("support target geometry drift")
    if support_valid_mask.shape != support_target.shape:
        raise ValueError("support valid-mask geometry drift")
    if bool(
        (
            support_target.bool()
            & ~support_valid_mask
        ).any()
    ):
        raise ValueError("positive support target lies outside support-valid mask")

    return full_envelope_behavioral_objective(
        candidate_logits=primary_judgment["candidate_logits"],
        target_index=targets["public_target_index"],
        candidate_valid_mask=primary_judgment[
            "candidate_valid_mask"
        ],
        support_logits=binder["support_logits"],
        support_target=support_target,
        support_valid_mask=support_valid_mask,
        null_support_logit=binder["null_support_logit"],
        source_weight=executor["source_support_weight"],
        target_weight=executor["target_support_weight"],
        source_target_index=targets["source_target_index"],
        target_target_index=targets["target_target_index"],
        endpoint_active_mask=targets["endpoint_active_mask"].bool(),
        decisive_ablated_candidate_logits=decisive_judgment[
            "candidate_logits"
        ],
        irrelevant_removed_candidate_logits=irrelevant_judgment[
            "candidate_logits"
        ],
        latent_slots=latent["latent_slots"],
        source_views=primary_outputs["source_views"],
        view_available=primary_outputs["view_available"],
        recoverable_view_mask=targets[
            "recoverable_view_mask"
        ].bool(),
        pooled_state_permuted=permuted_latent["pooled_state"],
        pooled_state_original=latent["pooled_state"],
    )


def full_envelope_family_losses(
    *,
    semantic_losses: Mapping[str, Tensor],
    behavioral_losses: Mapping[str, Tensor],
    broad_semantic_replay_loss: Tensor,
    governed_judgment_replay_loss: Tensor,
    natural_relation_loss: Tensor,
) -> dict[str, dict[str, Tensor]]:
    return {
        "broad_semantic_replay": {
            "replay": broad_semantic_replay_loss,
        },
        "governed_judgment_replay": {
            "replay": governed_judgment_replay_loss,
        },
        "relation_program_semantics": {
            "relation_sequence": semantic_losses["relation_sequence"],
            "causal_relation_margin": semantic_losses[
                "causal_relation_margin"
            ],
        },
        "dynamic_factor_semantics": {
            "factor_semantics": semantic_losses["factor_semantics"],
            "causal_factor_margin": semantic_losses[
                "causal_factor_margin"
            ],
        },
        "uncertainty_and_control": {
            "event_control": semantic_losses["event_control"],
            "applicability": semantic_losses["applicability"],
            "uncertainty": semantic_losses["uncertainty"],
        },
        "token_evidence_grounding": {
            "token_evidence": semantic_losses["token_evidence"],
        },
        "structural_support_and_roles": {
            "support_selection": behavioral_losses["support_selection"],
            "endpoint_roles": behavioral_losses["endpoint_roles"],
        },
        "multi_view_causal_preservation": {
            "decisive_view_causality": behavioral_losses[
                "decisive_view_causality"
            ],
            "irrelevant_view_invariance": behavioral_losses[
                "irrelevant_view_invariance"
            ],
            "source_view_recoverability": behavioral_losses[
                "source_view_recoverability"
            ],
            "permutation_consistency": behavioral_losses[
                "permutation_consistency"
            ],
        },
        "latent_judgment_and_noncollapse": {
            "public_judgment": behavioral_losses["public_judgment"],
            "latent_noncollapse": behavioral_losses["latent_noncollapse"],
        },
        "natural_relation_semantics": {
            "natural_relation": natural_relation_loss,
        },
    }


class FullEnvelopeJointTrainingObjectiveV1(nn.Module):
    """One pre-result loss contract for the full public N0 successor.

    Raw semantic/operator and behavioral components are assigned to exactly one
    macro family. Replay and natural-relation losses enter as independent lanes.
    No family weight is learned and no TEST/final result can adapt weighting.
    """

    def __init__(
        self,
        family_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        weights = (
            dict(DEFAULT_FAMILY_WEIGHTS)
            if family_weights is None
            else {str(k): float(v) for k, v in family_weights.items()}
        )
        if set(weights) != set(DEFAULT_FAMILY_WEIGHTS):
            raise ValueError(
                "family weights must exactly match the precommitted family set"
            )
        self.balancer = MacroFamilyLossBalancer(weights)

    def forward(
        self,
        *,
        primary_outputs: Mapping[str, Any],
        decisive_ablated_outputs: Mapping[str, Any],
        irrelevant_removed_outputs: Mapping[str, Any],
        permuted_outputs: Mapping[str, Any],
        operator_targets: Mapping[str, Any],
        behavioral_targets: Mapping[str, Any],
        broad_semantic_replay_loss: Tensor,
        governed_judgment_replay_loss: Tensor,
        natural_relation_loss: Tensor,
        update_ema: bool,
    ) -> dict[str, Any]:
        semantic_losses = semantic_operator_supervision(
            outputs=primary_outputs,
            targets=operator_targets,
        )
        behavioral_losses = behavioral_supervision(
            primary_outputs=primary_outputs,
            decisive_ablated_outputs=decisive_ablated_outputs,
            irrelevant_removed_outputs=irrelevant_removed_outputs,
            permuted_outputs=permuted_outputs,
            targets=behavioral_targets,
        )
        families = full_envelope_family_losses(
            semantic_losses=semantic_losses,
            behavioral_losses=behavioral_losses,
            broad_semantic_replay_loss=broad_semantic_replay_loss,
            governed_judgment_replay_loss=governed_judgment_replay_loss,
            natural_relation_loss=natural_relation_loss,
        )
        balanced = self.balancer(
            families,
            update_ema=update_ema,
        )
        return {
            "loss": balanced["loss"],
            "balanced": balanced,
            "semantic_operator": semantic_losses,
            "behavioral": behavioral_losses,
            "families": families,
        }

    def parameter_report(self) -> dict[str, Any]:
        report = self.balancer.parameter_report()
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "macro_family_count": len(DEFAULT_FAMILY_WEIGHTS),
            "family_names": tuple(DEFAULT_FAMILY_WEIGHTS),
            "learned_family_weight_parameters": report[
                "learned_family_weight_parameters"
            ],
            "test_adaptive_weights": report["test_adaptive_weights"],
            "component_double_counting": False,
            "replay_lanes_separate_from_full_envelope_behavior": True,
            "natural_relation_lane_separate": True,
        }
