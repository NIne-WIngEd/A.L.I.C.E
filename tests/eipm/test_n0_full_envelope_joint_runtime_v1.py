from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_full_public_mixture_manifest_is_executable_before_gradient() -> None:
    contract_path=ROOT/"configs/eipm/n0/n0_v02_full_public_mixture_contract_v1.json"
    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_full_public_mixture_manifest_v1.py"
    audit_path=ROOT/"scripts/eipm/n0/audit_n0_v02_full_public_mixture_manifest_v1.py"

    assert contract_path.is_file(), "full public-mixture contract missing"
    assert builder_path.is_file(), "full public-mixture manifest builder missing"
    assert audit_path.is_file(), "full public-mixture manifest auditor missing"

    contract=json.loads(contract_path.read_text())
    assert contract["schema"]=="alice.eipm.n0.full-public-mixture-contract.v1"
    assert contract["authority"]["private_identity_data"] is False
    assert contract["authority"]["final_rows_training_allowed"] is False
    assert contract["authority"]["final_results_may_be_observed"] is False
    assert contract["required_training_lanes"] == [
        "broad_semantic_replay",
        "governed_judgment_replay",
        "semantic_operator_intervention",
        "full_envelope_behavioral",
        "runtime_view_supplement",
        "long_context_supplement",
        "natural_relation",
    ]
    assert set(contract["required_macro_families"]) == {
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
    }
    assert contract["mixture"]["row_count_is_capability_ceiling"] is False
    assert contract["mixture"]["fixed_lane_sampling_ratio_as_capability_definition"] is False
    assert contract["validation_before_gradient"]["materialized_manifest_required"] is True
    assert contract["validation_before_gradient"]["all_lane_hashes_bound_required"] is True
    assert contract["validation_before_gradient"]["full_macro_family_coverage_required"] is True
    assert contract["validation_before_gradient"]["final_freeze_hash_bound_but_final_rows_excluded"] is True


def test_semantic_operator_intervention_rows_own_explicit_uncertainty_targets() -> None:
    import importlib.util

    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_semantic_operator_intervention_curriculum_v1.py"
    spec=importlib.util.spec_from_file_location("semantic_intervention_builder_uncertainty",builder_path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows=[]
    for example in range(12):
        relation=module.TRAIN_RELATIONS[example % len(module.TRAIN_RELATIONS)]
        second=module.TRAIN_RELATIONS[(example + 1) % len(module.TRAIN_RELATIONS)]
        import random
        rows.append(
            module.make_row(
                split="train",
                relation=relation,
                second=second,
                example=example,
                candidates=4,
                rng=random.Random(20260922 + example),
            )
        )

    assert all("uncertainty_target" in row for row in rows)
    by_intervention={row["intervention"]:row for row in rows}
    assert by_intervention["unknown_defer"]["uncertainty_target"] == 1.0
    for row in rows:
        assert 0.0 <= float(row["uncertainty_target"]) <= 1.0
    assert by_intervention["source_target_role"]["uncertainty_target"] == 0.0


def test_semantic_operator_batch_compiler_is_registered() -> None:
    path=ROOT/"src/alice_personality/n0/semantic_operator_batch_v1.py"
    assert path.is_file(), "semantic-operator optimizer-facing batch compiler missing"
    source=path.read_text()
    assert "compile_semantic_operator_batch" in source
    assert "compile_operator_evidence_targets" in source
    assert "counterfactual_factor_targets" in source
    assert "step_factor_schema_evidence_valid_mask" in source


def test_natural_relation_optimizer_compiler_is_registered() -> None:
    path=ROOT/"src/alice_personality/n0/natural_relation_batch_v1.py"
    assert path.is_file(), "natural-relation optimizer-facing compiler missing"
    source=path.read_text()
    assert "compile_natural_relation_batch" in source
    assert "natural_relation_semantic_loss" in source
    assert "relation_keys_model_visible" in source
    assert "factor_labels_fabricated" in source
    assert "downstream_fabric_labels_fabricated" in source


def test_registered_joint_step_executes_all_public_training_lanes_without_placeholder_losses() -> None:
    path=ROOT/"src/alice_personality/n0/full_envelope_joint_step_v1.py"
    assert path.is_file(), "registered all-lane joint-step runtime missing"
    source=path.read_text()
    assert "execute_full_envelope_joint_step" in source
    assert "broad_semantic_replay_loss" in source
    assert "governed_judgment_replay_loss" in source
    assert "natural_relation_semantic_loss" in source
    assert 'task="semantic_operator"' in source
    assert 'task="full_envelope"' in source
    assert 'task="natural_relation"' in source
    assert 'task="mlm"' in source
    assert 'task="teacher"' in source
    assert "\"placeholder_losses_used\":False" in source.replace(" ","")


def test_joint_training_stages_have_explicit_causal_loss_family_activation() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    stages={
        row["name"]:row
        for row in plan["optimization_strategy"]["stages"]
    }
    all_families=set(plan["loss_balancing"]["families"])
    j1={
        "broad_semantic_replay",
        "governed_judgment_replay",
        "relation_program_semantics",
        "dynamic_factor_semantics",
        "uncertainty_and_control",
        "token_evidence_grounding",
        "natural_relation_semantics",
    }
    j2=j1|{"structural_support_and_roles"}
    j3=set(all_families)
    assert set(stages["J1_joint_semantic_operator"]["active_macro_families"])==j1
    assert set(stages["J2_reopened_representation_interfaces"]["active_macro_families"])==j2
    assert set(stages["J3_full_public_n0_coadaptation"]["active_macro_families"])==j3

    j2_trainable=set(stages["J2_reopened_representation_interfaces"]["trainable"])
    assert "binder_successor" in j2_trainable
    assert "production_executor_successor" in j2_trainable
    assert "dynamic_fusion_successor" not in j2_trainable
    assert "dynamic_latent_successor" not in j2_trainable
    assert "public_judgment_probe" not in j2_trainable
    assert "evidence_successor_interface" not in j2_trainable
    j3_trainable=set(stages["J3_full_public_n0_coadaptation"]["trainable"])
    assert "raw_semantic_view_layer_gate" in j3_trainable
    assert "semantic_input_summary_layer_gate" in j3_trainable
    assert plan["optimization_strategy"]["module_bindings"]["raw_semantic_view_layer_gate"]=="stack.raw_semantic_layer_gate"
    assert plan["optimization_strategy"]["module_bindings"]["semantic_input_summary_layer_gate"]=="semantic_input.summary_layer_gate"

    j2_gates=set(plan["stage_gates"]["J2"])
    j3_gates=set(plan["stage_gates"]["J3"])
    assert "long_internal_and_additional_view_descriptor_semantics" not in j2_gates
    assert "long_additional_runtime_view_source_and_descriptor_semantics" not in j2_gates
    assert "long_internal_and_additional_view_descriptor_semantics" in j3_gates
    assert "long_additional_runtime_view_source_and_descriptor_semantics" in j3_gates

    assert stages["J1_joint_semantic_operator"]["architecture_reduced"] is False
    assert stages["J2_reopened_representation_interfaces"]["architecture_reduced"] is False
    assert stages["J3_full_public_n0_coadaptation"]["architecture_reduced"] is False
    assert plan["optimization_strategy"]["stage_family_policy"]["inactive_family_loss_computed"] is False
    assert plan["optimization_strategy"]["stage_family_policy"]["inactive_modules_removed_from_topology"] is False
    assert plan["optimization_strategy"]["stage_family_policy"]["newly_activated_family_owners_train_together"] is True


def test_stage_policy_owns_every_parameter_without_reducing_topology() -> None:
    from test_n0_full_envelope_trainable_system_v1 import _system
    from alice_personality.n0.full_envelope_stage_policy_v1 import (
        J1,J2,J3,apply_stage_trainability,
    )

    system=_system()
    total=sum(p.numel() for p in system.parameters())
    reports=[]
    for stage in (J1,J2,J3):
        report=apply_stage_trainability(system,stage=stage)
        reports.append(report)
        assert report["topology_parameter_count"]==total
        assert report["all_parameters_owned"] is True
        assert report["architecture_reduced"] is False
        assert report["inactive_modules_removed_from_topology"] is False
        assert report["trainable_parameter_count"]>0
        assert report["trainable_parameter_count"]<=total

    j1,j2,j3=reports
    assert j1["trainable_parameter_count"] < j3["trainable_parameter_count"]
    assert j2["trainable_parameter_count"] < j3["trainable_parameter_count"]
    assert set(j1["active_macro_families"]) < set(j2["active_macro_families"])
    assert set(j2["active_macro_families"]) < set(j3["active_macro_families"])


def test_stage_policy_keeps_random_downstream_owners_inactive_until_their_losses_activate() -> None:
    from test_n0_full_envelope_trainable_system_v1 import _system
    from alice_personality.n0.full_envelope_stage_policy_v1 import (
        J1,J2,J3,apply_stage_trainability,
    )

    system=_system()
    apply_stage_trainability(system,stage=J1)
    assert any(p.requires_grad for p in system.stack.semantic_operator.parameters())
    assert not any(p.requires_grad for p in system.stack.binder.parameters())
    assert not any(p.requires_grad for p in system.stack.public_judgment_probe.parameters())

    apply_stage_trainability(system,stage=J2)
    assert any(p.requires_grad for p in system.stack.binder.parameters())
    assert any(p.requires_grad for p in system.stack.executor.parameters())
    assert not any(p.requires_grad for p in system.stack.fusion.parameters())
    assert not any(p.requires_grad for p in system.stack.latent.parameters())
    assert not any(p.requires_grad for p in system.stack.public_judgment_probe.parameters())
    assert not any(p.requires_grad for p in system.stack.evidence_view.parameters())

    apply_stage_trainability(system,stage=J3)
    assert any(p.requires_grad for p in system.stack.evidence_view.parameters())
    assert any(p.requires_grad for p in system.stack.fusion.parameters())
    assert any(p.requires_grad for p in system.stack.latent.parameters())
    assert any(p.requires_grad for p in system.stack.public_judgment_probe.parameters())
    assert any(p.requires_grad for p in system.stack.raw_semantic_layer_gate.parameters())
    assert any(p.requires_grad for p in system.semantic_input.summary_layer_gate.parameters())


def test_joint_step_stage_policy_prevents_inactive_downstream_loss_execution() -> None:
    source=(
        ROOT/"src/alice_personality/n0/full_envelope_joint_step_v1.py"
    ).read_text()
    objective=(
        ROOT/"src/alice_personality/n0/full_envelope_training_objective_v1.py"
    ).read_text()
    assert "stage:" in source
    assert "resolve_stage_policy" in source
    # Lock the causal ownership source, not one formatting of the ternary.
    # The executable J1 test below proves that inactive full-fabric paths are
    # actually skipped; this static assertion only verifies policy wiring.
    assert "policy.active_macro_families" in source
    assert "requires_full_fabric_primary" in source
    assert "requires_full_fabric_counterfactuals" in source
    assert "active_families:" in objective
    assert "active_families=active" in objective.replace(" ","")
    assert "behavioral_supervision(" in objective
    assert "active_families=active" in objective.replace(" ","")


def test_j1_joint_step_does_not_require_or_execute_full_fabric() -> None:
    import torch
    from torch import nn

    from alice_personality.n0.full_envelope_joint_step_v1 import (
        execute_full_envelope_joint_step,
    )
    from alice_personality.n0.full_envelope_stage_policy_v1 import J1

    class _Objective(nn.Module):
        def forward(self, **kwargs):
            active=tuple(kwargs["active_families"])
            assert "structural_support_and_roles" not in active
            assert "multi_view_causal_preservation" not in active
            assert "latent_judgment_and_noncollapse" not in active
            assert kwargs["primary_outputs"] == {}
            assert kwargs["decisive_ablated_outputs"] is None
            assert kwargs["irrelevant_removed_outputs"] is None
            assert kwargs["permuted_outputs"] is None
            return {"loss":kwargs["broad_semantic_replay_loss"]}

    class _System(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor=nn.Parameter(torch.tensor(1.0))
            self.tasks=[]
        def forward(self, *, task, batch):
            self.tasks.append(task)
            if task=="mlm":
                return {"loss":self.anchor.square()}
            if task=="teacher":
                # The governed replay lane includes an actual cross-example
                # contrastive objective, so the fixture must represent at least
                # two teacher groups instead of weakening production semantics.
                score=self.anchor.repeat(4)
                semantic=self.anchor.repeat(4,3)
                rationale=self.anchor.repeat(2,3)
                return {
                    "scores":score,
                    "semantic":semantic,
                    "rationale":rationale,
                    "alignment_logits":self.anchor.repeat(4),
                }
            if task=="semantic_operator":
                return {"semantic_operator":{},"operator":object()}
            if task=="natural_relation":
                logits=self.anchor.repeat(1,2)
                return {
                    "semantic_operator":{
                        "relation_logits":logits[:,None,:],
                    },
                }
            if task=="full_envelope":
                raise AssertionError("J1 must not execute full-envelope fabric")
            raise AssertionError(task)

    system=_System()
    teacher={
        "group_sizes":[2,2],
        "preferred_masks":[
            torch.tensor([True,False]),
            torch.tensor([False,True]),
        ],
        "principle_tags":["p","q"],
    }
    semantic={"batch":{},"operator_targets":{}}
    natural={
        "batch":{},
        "target_relation_index":torch.tensor([0]),
    }
    result=execute_full_envelope_joint_step(
        system=system,
        objective=_Objective(),
        mlm_batch={},
        teacher_batch=teacher,
        semantic_operator_compiled=semantic,
        full_fabric_compiled=None,
        natural_relation_compiled=natural,
        update_ema=False,
        stage=J1,
    )
    assert "full_envelope" not in system.tasks
    assert result["requires_full_fabric_primary"] is False
    assert result["requires_full_fabric_counterfactuals"] is False


def test_registered_production_topology_has_one_source_of_truth_and_no_hidden_ceiling() -> None:
    topology_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json"
    factory_path=ROOT/"src/alice_personality/n0/full_envelope_runtime_factory_v1.py"
    assert topology_path.is_file()
    assert factory_path.is_file()
    topology=json.loads(topology_path.read_text())
    assert topology["schema"]=="alice.eipm.n0.full-envelope-registered-topology.v1"
    system=topology["registered_system"]
    invariants=topology["topology_invariants"]
    assert system["semantic_dim"]==system["model_dim"]==640
    assert system["num_hidden_states"]==17
    assert system["num_attention_heads"]==10
    assert system["native_window_tokens"]==4096
    assert system["overlap_tokens"]==256
    assert system["segment_bridge_layers"]==2
    assert topology["topology_invariants"]["full_architecture"] is True
    assert topology["topology_invariants"]["reduced_pilot"] is False
    for key in (
        "runtime_relation_ceiling","runtime_factor_ceiling",
        "runtime_field_ceiling","runtime_edge_ceiling",
        "runtime_view_ceiling","runtime_slot_ceiling",
        "runtime_reasoning_step_ceiling","product_context_token_ceiling",
    ):
        assert invariants[key] is None
    source=factory_path.read_text()
    assert "load_registered_full_envelope_system" in source
    assert "semantic initialization checkpoint hash drift" in source
    assert "runtime profile attempted to change learned topology" in source


def test_gpu_memory_dry_run_is_exact_topology_no_gradient_and_not_training_authority() -> None:
    cfg_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json"
    script_path=ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py"
    sbatch_path=ROOT/"scripts/eipm/n0/magnolia_p100x2_n0_v02_full_envelope_gpu_memory_v1.sbatch"
    assert cfg_path.is_file()
    assert script_path.is_file()
    assert sbatch_path.is_file()
    cfg=json.loads(cfg_path.read_text())
    assert cfg["schema"]=="alice.eipm.n0.full-envelope-gpu-memory-dry-run.v1"
    assert cfg["registered_topology"]=="configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json"
    assert cfg["stage"]=="J3_full_public_n0_coadaptation"
    assert cfg["qualification"]["exact_registered_production_topology_required"] is True
    assert cfg["qualification"]["real_optimizer_facing_public_lanes_required"] is True
    assert cfg["qualification"]["full_j3_counterfactual_path_required"] is True
    assert cfg["qualification"]["no_gradient"] is True
    assert cfg["qualification"]["no_backward"] is True
    assert cfg["qualification"]["no_optimizer_object"] is True
    assert cfg["authorization"]["optimizer"] is False
    assert cfg["authorization"]["gradient"] is False
    assert cfg["authorization"]["gpu_training"] is False
    assert cfg["route"]["microbatch_size_is_capability_ceiling"] is False
    source=script_path.read_text()
    assert "torch.inference_mode()" in source
    assert "apply_stage_trainability(system,stage=J3)" in source.replace(" ","")
    assert "execute_full_envelope_joint_step" in source
    assert "optimizer_object_created" in source
    assert "torch.optim" not in source
