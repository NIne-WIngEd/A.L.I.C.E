from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
    decisive_view_causal_margin_loss,
    endpoint_role_loss,
    irrelevant_view_invariance_loss,
    latent_noncollapse_loss,
    permutation_consistency_loss,
    public_judgment_loss,
    source_view_recoverability_loss,
    support_selection_loss,
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
        if bool((correct < 0).any()) or bool((correct >= logits.size(-1)).any()):
            raise ValueError(f"factor correct target outside bank: {name!r}")
        if bool((counter < -1).any()) or bool((counter >= logits.size(-1)).any()):
            raise ValueError(f"factor counterfactual target outside bank: {name!r}")
        valid = counter.ge(0) & counter.ne(correct)
        safe_counter = torch.where(valid, counter, correct)
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
                safe_counter,
                name=f"factor/{name}/counterfactual",
            )
        )
        valid_values.append(valid)
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
    if counterfactual_relation_targets.shape != relation_targets.shape:
        raise ValueError("counterfactual relation target geometry drift")
    if bool((counterfactual_relation_targets < -1).any()) or bool(
        (counterfactual_relation_targets >= relation_logits.size(-1)).any()
    ):
        raise ValueError("counterfactual relation target outside runtime bank")
    relation_margin_mask = (
        relation_step_mask
        & counterfactual_relation_targets.ge(0)
        & counterfactual_relation_targets.ne(relation_targets)
    )
    safe_counterfactual_relation_targets = torch.where(
        relation_margin_mask,
        counterfactual_relation_targets,
        relation_targets,
    )
    correct_relation_score = _gather_class_score(
        relation_logits,
        relation_targets,
        name="relation/correct",
    )
    counterfactual_relation_score = _gather_class_score(
        relation_logits,
        safe_counterfactual_relation_targets,
        name="relation/counterfactual",
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
        relation_plurality_target_distribution=targets.get(
            "relation_plurality_target_distribution"
        ),
        relation_plurality_mask=targets.get("relation_plurality_mask"),
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
        relation_margin_valid_mask=relation_margin_mask,
        correct_factor_score=correct_factor_score,
        counterfactual_factor_score=counterfactual_factor_score,
        factor_margin_valid_mask=factor_margin_mask,
    )


def behavioral_supervision(
    *,
    primary_outputs: Mapping[str, Any],
    decisive_ablated_outputs: Mapping[str, Any] | None,
    irrelevant_removed_outputs: Mapping[str, Any] | None,
    permuted_outputs: Mapping[str, Any] | None,
    targets: Mapping[str, Any],
    active_families: tuple[str, ...] | None = None,
) -> dict[str, Tensor]:
    behavioral_families=(
        "structural_support_and_roles",
        "multi_view_causal_preservation",
        "latent_judgment_and_noncollapse",
    )
    active=(
        set(behavioral_families)
        if active_families is None
        else set(map(str,active_families)) & set(behavioral_families)
    )
    unknown=(
        set()
        if active_families is None
        else set(map(str,active_families))-set(DEFAULT_FAMILY_WEIGHTS)
    )
    if unknown:
        raise ValueError(
            "unknown active macro families: "+repr(sorted(unknown))
        )
    if not active:
        return {}

    losses: dict[str, Tensor]={}

    if "structural_support_and_roles" in active:
        binder=primary_outputs.get("binder")
        executor=primary_outputs.get("executor")
        if not isinstance(binder,Mapping) or not isinstance(executor,Mapping):
            raise ValueError(
                "structural supervision requires Binder and Executor outputs"
            )
        support_target=targets["support_target"]
        support_valid_mask=targets["support_valid_mask"].bool()
        if support_target.shape != binder["support_logits"].shape:
            raise ValueError("support target geometry drift")
        if support_valid_mask.shape != support_target.shape:
            raise ValueError("support valid-mask geometry drift")
        if bool((support_target.bool() & ~support_valid_mask).any()):
            raise ValueError(
                "positive support target lies outside support-valid mask"
            )
        losses["support_selection"]=support_selection_loss(
            binder["support_logits"],
            support_target,
            support_valid_mask,
            null_support_logit=binder["null_support_logit"],
        )
        losses["endpoint_roles"]=endpoint_role_loss(
            executor["source_support_weight"],
            executor["target_support_weight"],
            targets["source_target_index"],
            targets["target_target_index"],
            targets["endpoint_active_mask"].bool(),
        )

    primary_judgment=None
    if (
        "multi_view_causal_preservation" in active
        or "latent_judgment_and_noncollapse" in active
    ):
        primary_judgment=_require_public_judgment(
            primary_outputs,
            label="primary",
        )

    if "latent_judgment_and_noncollapse" in active:
        latent=primary_outputs.get("latent")
        if not isinstance(latent,Mapping):
            raise ValueError(
                "latent/judgment supervision requires latent outputs"
            )
        assert primary_judgment is not None
        losses["public_judgment"]=public_judgment_loss(
            primary_judgment["candidate_logits"],
            targets["public_target_index"],
            candidate_valid_mask=primary_judgment[
                "candidate_valid_mask"
            ],
        )
        losses["latent_noncollapse"]=latent_noncollapse_loss(
            latent["latent_slots"]
        )

    if "multi_view_causal_preservation" in active:
        if (
            decisive_ablated_outputs is None
            or irrelevant_removed_outputs is None
            or permuted_outputs is None
        ):
            raise ValueError(
                "multi-view causal supervision requires decisive, irrelevant, "
                "and permutation counterfactual outputs"
            )
        assert primary_judgment is not None
        decisive_judgment=_require_public_judgment(
            decisive_ablated_outputs,
            label="decisive_ablated",
        )
        irrelevant_judgment=_require_public_judgment(
            irrelevant_removed_outputs,
            label="irrelevant_removed",
        )
        latent=primary_outputs.get("latent")
        permuted_latent=permuted_outputs.get("latent")
        if not isinstance(latent,Mapping) or not isinstance(
            permuted_latent,Mapping
        ):
            raise ValueError(
                "multi-view causal supervision requires latent outputs"
            )
        losses["decisive_view_causality"]=decisive_view_causal_margin_loss(
            primary_judgment["candidate_logits"],
            decisive_judgment["candidate_logits"],
            targets["public_target_index"],
            candidate_valid_mask=primary_judgment[
                "candidate_valid_mask"
            ],
            active_mask=targets["decisive_view_active_mask"].bool(),
        )
        losses["irrelevant_view_invariance"]=irrelevant_view_invariance_loss(
            primary_judgment["candidate_logits"],
            irrelevant_judgment["candidate_logits"],
            candidate_valid_mask=primary_judgment[
                "candidate_valid_mask"
            ],
            active_mask=targets["irrelevant_view_active_mask"].bool(),
        )
        losses["source_view_recoverability"]=source_view_recoverability_loss(
            latent["latent_slots"],
            primary_outputs["source_views"],
            primary_outputs["view_available"],
            targets["recoverable_view_mask"].bool(),
        )
        losses["permutation_consistency"]=permutation_consistency_loss(
            latent["pooled_state"],
            permuted_latent["pooled_state"],
        )

    return losses


def full_envelope_family_losses(
    *,
    semantic_losses: Mapping[str, Tensor],
    behavioral_losses: Mapping[str, Tensor],
    broad_semantic_replay_loss: Tensor,
    governed_judgment_replay_loss: Tensor,
    natural_relation_loss: Tensor,
    active_families: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Tensor]]:
    active=(
        tuple(DEFAULT_FAMILY_WEIGHTS)
        if active_families is None
        else tuple(str(x) for x in active_families)
    )
    if not active or len(active)!=len(set(active)):
        raise ValueError("active family set must be non-empty and unique")
    unknown=set(active)-set(DEFAULT_FAMILY_WEIGHTS)
    if unknown:
        raise ValueError(
            "unknown active macro families: "+repr(sorted(unknown))
        )

    families: dict[str, dict[str, Tensor]]={}
    if "broad_semantic_replay" in active:
        families["broad_semantic_replay"]={
            "replay":broad_semantic_replay_loss,
        }
    if "governed_judgment_replay" in active:
        families["governed_judgment_replay"]={
            "replay":governed_judgment_replay_loss,
        }
    if "relation_program_semantics" in active:
        families["relation_program_semantics"]={
            "relation_sequence":semantic_losses["relation_sequence"],
            "causal_relation_margin":semantic_losses[
                "causal_relation_margin"
            ],
        }
    if "dynamic_factor_semantics" in active:
        families["dynamic_factor_semantics"]={
            "factor_semantics":semantic_losses["factor_semantics"],
            "causal_factor_margin":semantic_losses[
                "causal_factor_margin"
            ],
        }
    if "uncertainty_and_control" in active:
        families["uncertainty_and_control"]={
            "event_control":semantic_losses["event_control"],
            "applicability":semantic_losses["applicability"],
            "uncertainty":semantic_losses["uncertainty"],
        }
    if "token_evidence_grounding" in active:
        families["token_evidence_grounding"]={
            "token_evidence":semantic_losses["token_evidence"],
        }
    if "structural_support_and_roles" in active:
        families["structural_support_and_roles"]={
            "support_selection":behavioral_losses["support_selection"],
            "endpoint_roles":behavioral_losses["endpoint_roles"],
        }
    if "multi_view_causal_preservation" in active:
        families["multi_view_causal_preservation"]={
            "decisive_view_causality":behavioral_losses[
                "decisive_view_causality"
            ],
            "irrelevant_view_invariance":behavioral_losses[
                "irrelevant_view_invariance"
            ],
            "source_view_recoverability":behavioral_losses[
                "source_view_recoverability"
            ],
            "permutation_consistency":behavioral_losses[
                "permutation_consistency"
            ],
        }
    if "latent_judgment_and_noncollapse" in active:
        families["latent_judgment_and_noncollapse"]={
            "public_judgment":behavioral_losses["public_judgment"],
            "latent_noncollapse":behavioral_losses["latent_noncollapse"],
        }
    if "natural_relation_semantics" in active:
        families["natural_relation_semantics"]={
            "natural_relation":natural_relation_loss,
        }
    return families



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
        decisive_ablated_outputs: Mapping[str, Any] | None,
        irrelevant_removed_outputs: Mapping[str, Any] | None,
        permuted_outputs: Mapping[str, Any] | None,
        operator_targets: Mapping[str, Any],
        behavioral_targets: Mapping[str, Any],
        broad_semantic_replay_loss: Tensor,
        governed_judgment_replay_loss: Tensor,
        natural_relation_loss: Tensor,
        update_ema: bool,
        semantic_operator_outputs: Mapping[str, Any] | None = None,
        active_families: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        active=(
            tuple(DEFAULT_FAMILY_WEIGHTS)
            if active_families is None
            else tuple(str(x) for x in active_families)
        )
        if not active or len(active)!=len(set(active)):
            raise ValueError("active family set must be non-empty and unique")
        unknown=set(active)-set(DEFAULT_FAMILY_WEIGHTS)
        if unknown:
            raise ValueError(
                "unknown active macro families: "+repr(sorted(unknown))
            )

        semantic_family_names={
            "relation_program_semantics",
            "dynamic_factor_semantics",
            "uncertainty_and_control",
            "token_evidence_grounding",
        }
        semantic_losses={}
        if set(active) & semantic_family_names:
            semantic_source=(
                primary_outputs
                if semantic_operator_outputs is None
                else semantic_operator_outputs
            )
            semantic_losses=semantic_operator_supervision(
                outputs=semantic_source,
                targets=operator_targets,
            )

        behavioral_losses=behavioral_supervision(
            primary_outputs=primary_outputs,
            decisive_ablated_outputs=decisive_ablated_outputs,
            irrelevant_removed_outputs=irrelevant_removed_outputs,
            permuted_outputs=permuted_outputs,
            targets=behavioral_targets,
            active_families=active,
        )
        families=full_envelope_family_losses(
            semantic_losses=semantic_losses,
            behavioral_losses=behavioral_losses,
            broad_semantic_replay_loss=broad_semantic_replay_loss,
            governed_judgment_replay_loss=governed_judgment_replay_loss,
            natural_relation_loss=natural_relation_loss,
            active_families=active,
        )
        balanced=self.balancer(
            families,
            update_ema=update_ema,
            active_families=active,
        )
        return {
            "loss": balanced["loss"],
            "balanced": balanced,
            "semantic_operator": semantic_losses,
            "semantic_operator_source": (
                "full_fabric_primary_lane"
                if semantic_operator_outputs is None
                else "dedicated_semantic_operator_lane"
            ),
            "behavioral": behavioral_losses,
            "families":families,
            "active_families":active,
            "inactive_family_losses_computed":False,
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
            "semantic_operator_lane_may_be_separate_from_full_fabric": True,
            "semantic_operator_lane_requires_fake_downstream_labels": False,
            "natural_relation_lane_separate": True,
        }
