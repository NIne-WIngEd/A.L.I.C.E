from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from torch import nn


J1="J1_joint_semantic_operator"
J2="J2_reopened_representation_interfaces"
J3="J3_full_public_n0_coadaptation"

ALL_FAMILIES=(
    "broad_semantic_replay",
    "governed_judgment_replay",
    "relation_program_semantics",
    "dynamic_factor_semantics",
    "uncertainty_and_control",
    "token_evidence_grounding",
    "structural_support_and_roles",
    "multi_view_causal_preservation",
    "latent_judgment_and_noncollapse",
    "natural_relation_semantics",
)

STAGE_ACTIVE_FAMILIES={
    J1:(
        "broad_semantic_replay",
        "governed_judgment_replay",
        "relation_program_semantics",
        "dynamic_factor_semantics",
        "uncertainty_and_control",
        "token_evidence_grounding",
        "natural_relation_semantics",
    ),
    J2:(
        "broad_semantic_replay",
        "governed_judgment_replay",
        "relation_program_semantics",
        "dynamic_factor_semantics",
        "uncertainty_and_control",
        "token_evidence_grounding",
        "natural_relation_semantics",
        "structural_support_and_roles",
    ),
    J3:ALL_FAMILIES,
}

STAGE_TRAINABLE_COMPONENTS={
    J1:(
        "semantic_backbone_all_layers",
        "semantic_existing_heads_for_replay",
        "schema_conditioned_semantic_operator",
        "long_context_segment_bridge",
    ),
    J2:(
        "semantic_backbone_all_layers",
        "schema_conditioned_semantic_operator",
        "structured_successor",
        "dynamic_evidence_graph",
        "binder_successor",
        "production_executor_successor",
        "long_context_segment_bridge",
    ),
    J3:(
        "semantic_backbone_all_layers",
        "schema_conditioned_semantic_operator",
        "structured_successor",
        "evidence_successor",
        "dynamic_evidence_graph",
        "production_executor_successor",
        "binder_successor",
        "public_judgment_probe",
        "dynamic_fusion_successor",
        "dynamic_latent_successor",
        "long_context_segment_bridge",
        "raw_semantic_view_layer_gate",
        "semantic_input_summary_layer_gate",
    ),
}


@dataclass(frozen=True)
class StagePolicy:
    name: str
    active_macro_families: tuple[str,...]
    trainable_components: tuple[str,...]

    @property
    def architecture_reduced(self) -> bool:
        return False


def resolve_stage_policy(stage: str) -> StagePolicy:
    name=str(stage)
    if name not in STAGE_ACTIVE_FAMILIES:
        raise ValueError(f"unknown N0 joint-training stage: {name!r}")
    return StagePolicy(
        name=name,
        active_macro_families=tuple(STAGE_ACTIVE_FAMILIES[name]),
        trainable_components=tuple(STAGE_TRAINABLE_COMPONENTS[name]),
    )


def _named_component_parameters(system: nn.Module) -> dict[str,list[nn.Parameter]]:
    semantic_model=system.semantic_model
    backbone=system.backbone
    backbone_ids={id(p) for p in backbone.parameters()}
    replay_heads=[
        p for p in semantic_model.parameters()
        if id(p) not in backbone_ids
    ]
    return {
        "semantic_backbone_all_layers":list(backbone.parameters()),
        "semantic_existing_heads_for_replay":replay_heads,
        "schema_conditioned_semantic_operator":list(
            system.stack.semantic_operator.parameters()
        ),
        "long_context_segment_bridge":list(
            system.semantic_input.segment_bridge.parameters()
        ),
        "semantic_input_summary_layer_gate":list(
            system.semantic_input.summary_layer_gate.parameters()
        ),
        "structured_successor":list(system.stack.structured.parameters()),
        "dynamic_evidence_graph":list(system.stack.evidence_graph.parameters()),
        "binder_successor":list(system.stack.binder.parameters()),
        "production_executor_successor":list(system.stack.executor.parameters()),
        "evidence_successor":list(system.stack.evidence_view.parameters()),
        "raw_semantic_view_layer_gate":list(
            system.stack.raw_semantic_layer_gate.parameters()
        ),
        "dynamic_fusion_successor":list(system.stack.fusion.parameters()),
        "dynamic_latent_successor":list(system.stack.latent.parameters()),
        "public_judgment_probe":list(
            system.stack.public_judgment_probe.parameters()
        ),
    }


def _unique_parameters(groups: Iterable[Iterable[nn.Parameter]]) -> list[nn.Parameter]:
    out=[]
    seen=set()
    for group in groups:
        for parameter in group:
            key=id(parameter)
            if key in seen:
                continue
            seen.add(key)
            out.append(parameter)
    return out


def apply_stage_trainability(
    system: nn.Module,
    *,
    stage: str,
) -> dict[str,Any]:
    """Apply the precommitted stage policy without changing module topology."""
    policy=resolve_stage_policy(stage)
    components=_named_component_parameters(system)

    unknown=set(policy.trainable_components)-set(components)
    if unknown:
        raise RuntimeError(
            "stage policy references unknown trainable components: "
            +repr(sorted(unknown))
        )

    all_component_parameters=_unique_parameters(components.values())
    all_system_parameters=list(system.parameters())
    component_ids={id(p) for p in all_component_parameters}
    uncovered=[
        name for name,p in system.named_parameters()
        if id(p) not in component_ids
    ]
    if uncovered:
        raise RuntimeError(
            "stage policy does not own every system parameter: "
            +repr(uncovered)
        )

    active=_unique_parameters(
        components[name] for name in policy.trainable_components
    )
    active_ids={id(p) for p in active}
    for parameter in all_system_parameters:
        parameter.requires_grad_(id(parameter) in active_ids)

    trainable_names=[
        name for name,p in system.named_parameters()
        if p.requires_grad
    ]
    frozen_names=[
        name for name,p in system.named_parameters()
        if not p.requires_grad
    ]
    return {
        "stage":policy.name,
        "active_macro_families":policy.active_macro_families,
        "trainable_components":policy.trainable_components,
        "architecture_reduced":False,
        "topology_parameter_count":sum(p.numel() for p in all_system_parameters),
        "trainable_parameter_count":sum(
            p.numel() for p in all_system_parameters if p.requires_grad
        ),
        "trainable_parameter_names":tuple(trainable_names),
        "frozen_parameter_names":tuple(frozen_names),
        "all_parameters_owned":True,
        "inactive_modules_removed_from_topology":False,
    }
