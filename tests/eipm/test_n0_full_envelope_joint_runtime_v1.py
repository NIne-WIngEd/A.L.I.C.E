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
        "semantic_operator_long_context",
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


def test_stage_training_scheduler_routes_long_semantics_into_j1_without_downstream_fabric() -> None:
    path=ROOT/"src/alice_personality/n0/full_envelope_training_batch_scheduler_v1.py"
    assert path.is_file(), "stage-aware full-envelope training batch scheduler missing"
    source=path.read_text()
    assert "FullEnvelopeTrainingBatchSchedulerV1" in source
    assert "semantic_operator_intervention" in source
    assert "long_context_semantic" in source
    assert "full_envelope_behavioral" in source
    assert "runtime_view_supplement" in source
    assert "long_context_fabric" in source
    assert "natural_relation" in source

    from alice_personality.n0.full_envelope_stage_policy_v1 import J1,J2,J3
    from alice_personality.n0.full_envelope_training_batch_scheduler_v1 import (
        FullEnvelopeTrainingBatchSchedulerV1,
    )

    def public(row):
        return {
            **row,
            "private_identity_data":False,
            "final_validation_only":False,
        }
    lanes={
        "semantic_operator_intervention":[public({"id":"semantic-a"})],
        "long_context_semantic":[
            public({"id":"long-query","lane":"semantic_operator_long_context","long_context_surface":"query"}),
            public({"id":"long-relation","lane":"semantic_operator_long_context","long_context_surface":"relation_schema"}),
            public({"id":"long-factor","lane":"semantic_operator_long_context","long_context_surface":"factor_schema"}),
        ],
        "full_envelope_behavioral":[public({"id":"behavior-a"})],
        "runtime_view_supplement":[public({"id":"runtime-a"})],
        "long_context_fabric":[
            public({"id":"long-field","lane":"full_envelope_long_context_supplement","long_context_surface":"field_text"})
        ],
        "natural_relation":[public({"id":"natural-a"})],
    }
    scheduler=FullEnvelopeTrainingBatchSchedulerV1(lanes=lanes,seed=20260922)

    seen_j1={scheduler.next_lane(stage=J1,kind="semantic") for _ in range(8)}
    assert seen_j1=={"semantic_operator_intervention","long_context_semantic"}
    assert scheduler.active_full_fabric_lanes(stage=J1)==()
    assert scheduler.active_full_fabric_lanes(stage=J2)==(
        "full_envelope_behavioral","long_context_fabric",
    )
    assert scheduler.active_full_fabric_lanes(stage=J3)==(
        "full_envelope_behavioral","runtime_view_supplement","long_context_fabric",
    )
    assert scheduler.active_natural_lanes(stage=J1)==("natural_relation",)


def test_stage_training_scheduler_filters_long_surfaces_by_owner_stage() -> None:
    from alice_personality.n0.full_envelope_stage_policy_v1 import J1,J2,J3
    from alice_personality.n0.full_envelope_training_batch_scheduler_v1 import (
        FullEnvelopeTrainingBatchSchedulerV1,
        LONG_J2_FABRIC_SURFACES,
        LONG_J3_FABRIC_SURFACES,
    )
    surfaces=[
        "query","relation_schema","factor_schema","type_schema","field_text",
        "field_descriptor","candidate_text","internal_view_descriptor",
        "additional_view_descriptor","additional_view_source",
    ]
    long_rows=[
        {
            "id":f"long-{surface}",
            "long_context_surface":surface,
            "private_identity_data":False,
            "final_validation_only":False,
        }
        for surface in surfaces
    ]
    semantic=[
        {
            "id":f"semantic-long-{surface}",
            "lane":"semantic_operator_long_context",
            "long_context_surface":surface,
            "private_identity_data":False,
            "final_validation_only":False,
        }
        for surface in ("query","relation_schema","factor_schema")
    ]
    fabric=[
        {**row,"lane":"full_envelope_long_context_supplement"}
        for row in long_rows
    ]
    semantic,fabric=FullEnvelopeTrainingBatchSchedulerV1.validate_long_context_lanes(
        semantic_rows=semantic,
        fabric_rows=fabric,
    )
    scheduler=FullEnvelopeTrainingBatchSchedulerV1(
        lanes={
            "semantic_operator_intervention":[{
                "id":"semantic","private_identity_data":False,
                "final_validation_only":False,
            }],
            "long_context_semantic":semantic,
            "full_envelope_behavioral":[{
                "id":"behavior","private_identity_data":False,
                "final_validation_only":False,
            }],
            "runtime_view_supplement":[{
                "id":"runtime","private_identity_data":False,
                "final_validation_only":False,
            }],
            "long_context_fabric":fabric,
            "natural_relation":[{
                "id":"natural","private_identity_data":False,
                "final_validation_only":False,
            }],
        },
        seed=1,
    )
    assert scheduler.active_full_fabric_lanes(stage=J1)==()
    j2=scheduler._eligible_rows(stage=J2,lane="long_context_fabric")
    j3=scheduler._eligible_rows(stage=J3,lane="long_context_fabric")
    assert {x["long_context_surface"] for x in j2}==set(LONG_J2_FABRIC_SURFACES)
    assert {x["long_context_surface"] for x in j3}==set(LONG_J3_FABRIC_SURFACES)


def test_j1_long_semantic_rows_compile_through_registered_semantic_operator_task() -> None:
    import importlib.util
    import sys

    contract_path=ROOT/"configs/eipm/n0/n0_v02_semantic_operator_long_context_contract_v1.json"
    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_semantic_operator_long_context_curriculum_v1.py"
    audit_path=ROOT/"scripts/eipm/n0/audit_n0_v02_semantic_operator_long_context_curriculum_v1.py"
    assert contract_path.is_file(), "J1 semantic long-context contract missing"
    assert builder_path.is_file(), "J1 semantic long-context builder missing"
    assert audit_path.is_file(), "J1 semantic long-context audit missing"

    contract=json.loads(contract_path.read_text())
    assert contract["schema"]=="alice.eipm.n0.semantic-operator-long-context-contract.v1"
    assert contract["required_surfaces"]==["query","relation_schema","factor_schema"]
    assert contract["authority"]["private_identity_data"] is False
    assert contract["authority"]["final_rows_training_allowed"] is False
    assert contract["operating_point"]["native_window_tokens"]==4096
    assert contract["operating_point"]["long_word_target"]>4096
    assert contract["operating_point"]["long_word_target_is_product_ceiling"] is False

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location("semantic_long_builder",builder_path)
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from alice_personality.n0.semantic_operator_batch_v1 import (
            compile_semantic_operator_batch,
        )
        from test_n0_semantic_operator_foundation_v1 import _OffsetTokenizer

        class _SemanticLongTokenizer(_OffsetTokenizer):
            def __call__(
                self,
                texts,
                *,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                max_length=None,
            ):
                # The registered semantic compiler first requests ordinary
                # token tensors and then exact evidence-offset tensors. One
                # test tokenizer must faithfully support both interfaces.
                return super().__call__(
                    texts,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                    max_length=max_length,
                )

        for surface in ("query","relation_schema","factor_schema"):
            row=module.materialize(
                split="train",
                surface=surface,
                target_words=256,
                placement_variant="tail",
            )
            assert row["lane"]=="semantic_operator_long_context"
            assert row["training_authorized"] is True
            assert row["final_validation_only"] is False
            assert "factor_targets" in row
            assert "step_factor_targets" in row
            assert "runtime_operator_slots" in row
            assert "query_relation_evidence_char_spans" in row
            assert "relation_schema_evidence_char_spans" in row
            assert "factor_schema_evidence_char_spans" in row
            compiled=compile_semantic_operator_batch(
                rows=[row],
                tokenizer=_SemanticLongTokenizer(),
            )
            assert compiled["metadata"]["batch_size"]==1
            assert compiled["metadata"]["fabricated_downstream_labels"] is False
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_runtime_qualification_token_aligns_dedicated_J1_long_semantic_evidence() -> None:
    runner=(
        ROOT/"scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh"
    ).read_text()
    assert "build_n0_v02_semantic_operator_long_context_curriculum_v1.py" in runner
    assert "audit_n0_v02_semantic_operator_long_context_curriculum_v1.py" in runner
    assert "SEMANTIC_LONG_TOKEN_AUDIT" in runner
    assert "audit_n0_v02_semantic_operator_long_token_alignment_v1.py" in runner
    long_audit=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_semantic_operator_long_token_alignment_v1.py"
    ).read_text()
    assert "max_length=None" in long_audit
    assert "first_positive_evidence_token" in long_audit
    assert "native_window_tokens" in long_audit
    assert "PASS_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1" in runner


def test_stage_scheduler_rejects_behavioral_long_rows_as_J1_semantic_authority() -> None:
    import pytest
    from alice_personality.n0.full_envelope_training_batch_scheduler_v1 import (
        FullEnvelopeTrainingBatchSchedulerV1,
    )

    def row(identifier, **extra):
        return {
            "id":identifier,
            "private_identity_data":False,
            "final_validation_only":False,
            **extra,
        }

    with pytest.raises(ValueError,match="semantic long-context row authority drift"):
        FullEnvelopeTrainingBatchSchedulerV1(
            lanes={
                "semantic_operator_intervention":[row("semantic")],
                "long_context_semantic":[
                    row("wrong-query",lane="full_envelope_long_context_supplement",long_context_surface="query"),
                    row("wrong-relation",lane="full_envelope_long_context_supplement",long_context_surface="relation_schema"),
                    row("wrong-factor",lane="full_envelope_long_context_supplement",long_context_surface="factor_schema"),
                ],
                "full_envelope_behavioral":[row("behavior")],
                "runtime_view_supplement":[row("runtime")],
                "long_context_fabric":[
                    row("fabric",lane="full_envelope_long_context_supplement",long_context_surface="field_text")
                ],
                "natural_relation":[row("natural")],
            },
            seed=1,
        )


def test_gpu_memory_dry_run_covers_both_semantic_cardinality_and_long_semantic_lanes() -> None:
    qualifier=(
        ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py"
    ).read_text()
    sbatch=(
        ROOT/"scripts/eipm/n0/magnolia_p100x2_n0_v02_full_envelope_gpu_memory_v1.sbatch"
    ).read_text()
    assert "--semantic-long-rows" in qualifier
    assert "semantic_long_rows=train_rows(args.semantic_long_rows)" in qualifier.replace(" ","")
    assert '"max_runtime_axes"' in qualifier
    assert '"long_context_semantic"' in qualifier
    assert "semantic_case_receipts" in qualifier
    assert "--semantic-long-rows '$MIXTURE_ROOT/semantic-long/rows.jsonl'" in sbatch


def test_successor_trainer_dev_evaluator_and_checkpoint_contract_exist_before_gradient() -> None:
    trainer=ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    evaluator=ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    checkpoint_contract=ROOT/"configs/eipm/n0/n0_v02_full_envelope_checkpoint_contract_v1.json"
    dev_contract=ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_validation_contract_v1.json"

    assert trainer.is_file(), "successor staged full-envelope trainer missing"
    assert evaluator.is_file(), "successor full-envelope DEV evaluator missing"
    assert checkpoint_contract.is_file(), "successor checkpoint provenance contract missing"
    assert dev_contract.is_file(), "successor DEV gate contract missing"

    checkpoint=json.loads(checkpoint_contract.read_text())
    assert checkpoint["schema"]=="alice.eipm.n0.full-envelope-checkpoint-contract.v1"
    assert checkpoint["registered_system"]=="N0FullEnvelopeTrainableSystemV1"
    assert checkpoint["required_lineage"]==[
        "source_revision",
        "registered_topology_sha256",
        "semantic_initialization_sha256",
        "full_public_mixture_manifest_sha256",
        "full_public_mixture_audit_sha256",
        "stage",
        "stage_policy",
        "optimizer_contract",
        "scheduler_contract",
        "objective_contract",
        "tokenizer_sha256",
        "public_corpus_receipt_sha256",
        "teacher_audit_sha256",
        "static_proof_receipt_sha256",
        "operator_evidence_token_receipt_sha256",
        "training_authorization_sha256",
        "optimizer_step",
        "checkpoint_evaluation_cadence_steps",
        "stage_checkpoint_parent_receipt_sha256",
        "stage_transition_predecessor_checkpoint_receipt_sha256",
        "stage_transition_predecessor_dev_receipt_sha256",
        "stage_transition_predecessor_selection_receipt_sha256",
        "accelerator_state_tree_sha256",
        "full_system_sha256",
        "objective_state_sha256",
    ]
    assert checkpoint["authority"]["final_results_observed"] is False
    assert checkpoint["authority"]["final_checkpoint_selection_allowed"] is False
    assert checkpoint["authority"]["private_identity_data"] is False

    dev=json.loads(dev_contract.read_text())
    assert dev["schema"]=="alice.eipm.n0.full-envelope-dev-validation-contract.v1"
    assert dev["registered_system"]=="N0FullEnvelopeTrainableSystemV1"
    assert dev["checkpoint_selection_surface"]=="DEV_ONLY"
    assert dev["final_rows_allowed"] is False
    assert dev["final_results_allowed"] is False
    assert dev["shortcut_preflight_must_remain_passed"] is True
    assert set(dev["stage_sections"])=={
        "J1_joint_semantic_operator",
        "J2_reopened_representation_interfaces",
        "J3_full_public_n0_coadaptation",
    }


def test_successor_trainer_reuses_registered_stage_and_joint_step_without_historical_model_authority() -> None:
    trainer=(
        ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    ).read_text()
    required=(
        "load_registered_full_envelope_system",
        "apply_stage_trainability",
        "FullEnvelopeTrainingBatchSchedulerV1",
        "execute_full_envelope_joint_step",
        "FullEnvelopeJointTrainingObjectiveV1",
        "compile_semantic_operator_batch",
        "compile_behavioral_batch",
        "compile_natural_relation_batch",
        "verify_public_corpus_v021",
        "verify_teacher_registry",
        "PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1",
    )
    for symbol in required:
        assert symbol in trainer
    assert "AliceN0V02Model(" not in trainer
    assert "SourceAnchoredCrossContextFusion(" not in trainer
    assert "AdaptiveMultiViewLatentPool(" not in trainer
    assert "final_validation" not in trainer.lower() or "forbid" in trainer.lower()


def test_dev_evaluator_is_dev_only_and_cannot_open_final() -> None:
    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert "split" in evaluator
    assert '"dev"' in evaluator or "'dev'" in evaluator
    assert "final_validation_only" in evaluator
    assert "final_results_observed" in evaluator
    assert "final_opening_authorized" in evaluator
    assert "checkpoint_selection_performed" in evaluator
    assert "N0FullEnvelopeTrainableSystemV1" in evaluator


def test_successor_trainer_cannot_bypass_explicit_training_plan_authority() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    assert plan["authorization"]["optimizer"] is False
    assert plan["authorization"]["gradient"] is False
    assert plan["authorization"]["gpu_training"] is False

    trainer=(
        ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    ).read_text()
    assert 'parser.add_argument("--training-plan",required=True)' in trainer
    assert 'parser.add_argument("--training-authorization")' in trainer
    assert 'if args.execute_gradient:' in trainer
    assert 'authority.get(name) is not False' in trainer
    assert "source training authority must remain false" in trainer
    assert "runtime training authorization required for gradient" in trainer


def test_j1_dev_gate_registry_covers_every_declared_gate_without_final_authority() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    assert registry["schema"]=="alice.eipm.n0.full-envelope-dev-gate-registry.v1"
    assert registry["authority"]["checkpoint_selection_surface"]=="DEV_ONLY"
    assert registry["authority"]["final_rows_allowed"] is False
    assert registry["authority"]["final_results_allowed"] is False
    j1=registry["stages"]["J1_joint_semantic_operator"]
    assert j1["mapping_complete"] is True
    assert set(j1["gates"])==set(plan["stage_gates"]["J1"])
    allowed={"empirical","composite","static_source_proof","retained_stage_gates"}
    assert {value["kind"] for value in j1["gates"].values()} <= allowed
    for name,value in j1["gates"].items():
        if value["kind"]=="static_source_proof":
            assert value["obligation_ids"], name
        elif value["kind"]=="empirical":
            assert value["metric"] and "threshold" in value, name
        else:
            assert value["all"], name


def test_successor_dev_metric_primitives_measure_program_factor_margin_and_token_grounding() -> None:
    import torch
    from alice_personality.n0.full_envelope_dev_metrics_v1 import (
        relation_program_exact,
        event_program_exact,
        global_factor_correct,
        step_factor_exact,
        margin_success,
        token_evidence_f1,
        support_edge_f1,
        endpoint_pair_correct,
        public_judgment_correct,
        decisive_removal_success,
        irrelevant_removal_invariance_success,
        latent_noncollapse_success,
        source_view_recoverability_success,
    )
    relation_logits=torch.tensor([[
        [4.0,0.0],
        [0.0,5.0],
        [1.0,1.0],
    ]])
    mass=torch.tensor([[0.9,0.8,0.0]])
    target=torch.tensor([[0,1,0]])
    mask=torch.tensor([[True,True,False]])
    assert bool(relation_program_exact(
        relation_logits=relation_logits,
        relation_step_mass=mass,
        relation_targets=target,
        relation_step_mask=mask,
    )[0])
    events=torch.tensor([[
        [0.9,0.05,0.05],
        [0.8,0.1,0.1],
        [0.05,0.9,0.05],
    ]])
    event_target=torch.tensor([[0,0,1]])
    event_mask=torch.tensor([[True,True,True]])
    assert bool(event_program_exact(
        event_distribution=events,event_targets=event_target,event_mask=event_mask
    )[0])
    factors=global_factor_correct(
        factor_logits={"direction":torch.tensor([[0.0,4.0]])},
        factor_targets={"direction":torch.tensor([1])},
    )
    assert bool(factors["direction"][0])
    steps=step_factor_exact(
        step_factor_logits={"direction":torch.tensor([[[4.0,0.0],[0.0,4.0],[1.0,1.0]]])},
        step_factor_targets={"direction":torch.tensor([[0,1,0]])},
        step_factor_mask=mask,
    )
    assert bool(steps["direction"][0])
    margin=margin_success(
        logits=relation_logits,
        targets=target,
        counterfactual_targets=torch.tensor([[1,0,-1]]),
        valid_mask=mask,
        margin=0.2,
    )
    assert bool(margin.all())
    assert token_evidence_f1(
        predicted=torch.tensor([[0.9,0.1,0.8,0.2]]),
        target=torch.tensor([[1.0,0.0,1.0,0.0]]),
        valid_mask=torch.tensor([[True,True,True,True]]),
    )==1.0

    support_weight=torch.tensor([[0.8,0.0,0.0]])
    support_target=torch.tensor([[1.0,0.0,0.0]])
    support_valid=torch.tensor([[True,True,True]])
    assert support_edge_f1(
        edge_support_weight=support_weight,
        support_target=support_target,
        valid_mask=support_valid,
    )==1.0
    endpoint_ok=endpoint_pair_correct(
        source_weight=torch.tensor([[0.9,0.1,0.0]]),
        target_weight=torch.tensor([[0.0,0.2,0.8]]),
        source_target=torch.tensor([0]),
        target_target=torch.tensor([2]),
        active_mask=torch.tensor([True]),
    )
    assert bool(endpoint_ok[0])
    logits=torch.tensor([[3.0,1.0,-2.0]])
    valid=torch.tensor([[True,True,True]])
    target=torch.tensor([0])
    assert bool(public_judgment_correct(
        candidate_logits=logits,
        target_index=target,
        candidate_valid_mask=valid,
    )[0])
    decisive_ok=decisive_removal_success(
        normal_logits=logits,
        ablated_logits=torch.tensor([[1.0,1.0,-2.0]]),
        target_index=target,
        candidate_valid_mask=valid,
        active_mask=torch.tensor([True]),
    )
    assert bool(decisive_ok[0])
    invariant_ok=irrelevant_removal_invariance_success(
        normal_logits=logits,
        removed_logits=logits.clone(),
        candidate_valid_mask=valid,
        active_mask=torch.tensor([True]),
    )
    assert bool(invariant_ok[0])
    slots=torch.tensor([[[1.0,0.0],[0.0,1.0]]])
    assert bool(latent_noncollapse_success(latent_slots=slots)[0])
    recoverable=source_view_recoverability_success(
        latent_slots=slots,
        source_views=torch.tensor([[[1.0,0.0],[0.0,1.0],[1.0,1.0]]]),
        view_available=torch.tensor([[True,True,True]]),
        recoverable_view_mask=torch.tensor([[True,True,False]]),
        cosine_threshold=0.95,
    )
    assert recoverable.tolist()==[True,True]


def test_successor_training_and_dev_selection_bind_exact_head_static_proof_receipt() -> None:
    audit=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_proof_obligations_v1.py"
    ).read_text()
    trainer=(
        ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    ).read_text()
    assert '"source_revision": source_revision' in audit
    assert 'parser.add_argument("--static-proof-receipt",required=True)' in trainer
    assert '"PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"' in trainer
    assert "static proof receipt source revision drift" in trainer
    assert '"static_proof_receipt_sha256"' in trainer


def test_dev_gate_evaluator_fails_closed_on_missing_metrics_and_uses_static_receipt() -> None:
    from alice_personality.n0.full_envelope_dev_gate_v1 import (
        evaluate_stage_gate_registry,
    )
    registry={
        "schema":"alice.eipm.n0.full-envelope-dev-gate-registry.v1",
        "stages":{
            "J1_joint_semantic_operator":{
                "mapping_complete":True,
                "gates":{
                    "empirical":{
                        "kind":"empirical","metric":"semantic.score",
                        "comparison":">=","threshold":0.9,
                    },
                    "static":{
                        "kind":"static_source_proof",
                        "obligation_ids":["N0-X"],
                    },
                },
            },
        },
    }
    proof={"obligations":[{"id":"N0-X","kind":"STATIC_REQUIRED"}]}
    receipt={"status":"PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"}
    good=evaluate_stage_gate_registry(
        registry=registry,
        stage="J1_joint_semantic_operator",
        metrics={"semantic":{"score":0.95}},
        proof_contract=proof,
        static_receipt=receipt,
    )
    assert good["stage_gate_coverage_complete"] is True
    assert good["stage_gate_pass"] is True

    missing=evaluate_stage_gate_registry(
        registry=registry,
        stage="J1_joint_semantic_operator",
        metrics={"semantic":{}},
        proof_contract=proof,
        static_receipt=receipt,
    )
    assert missing["stage_gate_coverage_complete"] is False
    assert missing["stage_gate_pass"] is False
    assert missing["mapping_errors"]


def test_successor_dev_evaluator_executes_j1_gate_registry_and_fixed_regression_suites() -> None:
    dev=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_validation_contract_v1.json").read_text()
    )
    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert dev["dev_gate_registry"]=="configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json"
    assert dev["static_proof_receipt_required"] is True
    assert dev["fixed_regression_suites"]=={
        "core_fixed":"evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl",
        "voice_fixed":"evaluation/eipm/n0/n0_v02_voice_readiness_base_v0.1.jsonl",
        "novel_cross":"evaluation/eipm/n0/n0_v02_novel_cross_competency_base_v0.1.jsonl",
    }
    for symbol in (
        "semantic_batch_record",
        "evaluate_stage_gate_registry",
        "fixed_preference_top1",
        "source_target_pair_completion",
        "mixed_step_direction_sequence_exact",
        "bidirectional_relation_token_grounding_min_f1",
        "relation_counterfactual_margin_success",
        "PASS_DEV_STAGE_GATE",
    ):
        assert symbol in evaluator
    assert 'add_argument("--static-proof-receipt",required=True)' in evaluator
    assert "FINAL rows are forbidden in successor DEV evaluator" in evaluator
    assert "stage_gate_pass" in evaluator
    assert "stage_gate_pass=False" not in evaluator.replace(" ","")


def test_behavioral_dev_geometry_extrapolates_beyond_train_without_becoming_ceiling() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "behavioral_builder_dev_geometry",
            scripts/"build_n0_v02_full_envelope_behavioral_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        train=[
            module.materialize_row(
                split="train",
                example=i,
                seed=20260922,
                relation_count=(1,2,4,6,8)[i % 5],
                field_count=(4,6,8,12,16)[i % 5],
                answer_count=(2,3,4,6)[i % 4],
            )
            for i in range(17)
        ]
        dev_answer_points=(3,4,5,7)
        dev=[
            module.materialize_row(
                split="dev",
                example=i,
                seed=20260922,
                relation_count=(1,2,4,6,8)[i % 5],
                field_count=(4,6,8,12,16)[i % 5],
                answer_count=dev_answer_points[i % len(dev_answer_points)],
            )
            for i in range(17)
        ]
        assert max(x["runtime_candidate_answer_count"] for x in dev) > max(
            x["runtime_candidate_answer_count"] for x in train
        )
        assert max(x["runtime_reasoning_steps"] for x in dev) > max(
            x["runtime_reasoning_steps"] for x in train
        )

        final=module.materialize_row(
            split="final",
            example=17,
            seed=20260922,
            relation_count=9,
            field_count=18,
            answer_count=8,
        )
        assert final["scenario_family"]=="long_causal_chain"
        assert final["runtime_reasoning_steps"] > max(
            x["runtime_reasoning_steps"] for x in dev
        )
        assert final["runtime_candidate_answer_count"] > max(
            x["runtime_candidate_answer_count"] for x in dev
        )

        contract=json.loads(
            (ROOT/"configs/eipm/n0/n0_v02_full_envelope_behavioral_curriculum_contract_v1.json").read_text()
        )
        assert contract["split_isolation"]["dev_extrapolation_is_operating_point_not_capability_ceiling"] is True
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_deep_chain_curriculum_keeps_entities_and_candidate_answers_distinct() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "behavioral_builder_deep_chain_uniqueness",
            scripts/"build_n0_v02_full_envelope_behavioral_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for split,examples,answer_count in (
            ("train",range(14,14+17*4,17),6),
            ("dev",range(14,14+17*4,17),7),
            ("final",(17,36),8),
        ):
            for example in examples:
                row=module.materialize_row(
                    split=split,
                    example=example,
                    seed=20260922,
                    relation_count=9 if split=="final" else 8,
                    field_count=18 if split=="final" else 16,
                    answer_count=answer_count,
                )
                if row["scenario_family"] not in {"causal_chain","long_causal_chain"}:
                    continue
                assert len(row["entities"])==len(set(row["entities"]))
                assert len(row["candidate_answers"])==len(set(row["candidate_answers"]))
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_j2_j3_dev_gate_registry_precommits_every_declared_gate_before_gradient() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    allowed={"empirical","composite","static_source_proof","retained_stage_gates"}
    for stage_key,plan_key in (
        ("J2_reopened_representation_interfaces","J2"),
        ("J3_full_public_n0_coadaptation","J3"),
    ):
        stage=registry["stages"][stage_key]
        assert stage["mapping_complete"] is True, stage_key
        assert set(stage["gates"])==set(plan["stage_gates"][plan_key]), stage_key
        assert {value["kind"] for value in stage["gates"].values()} <= allowed
        for name,value in stage["gates"].items():
            kind=value["kind"]
            if kind=="static_source_proof":
                assert value["obligation_ids"], (stage_key,name)
            elif kind=="empirical":
                assert value["metric"] and "threshold" in value, (stage_key,name)
            elif kind=="composite":
                assert value["all"], (stage_key,name)
            else:
                assert value["stages"], (stage_key,name)


def test_dev_gate_evaluator_rechecks_retained_prior_stage_gates() -> None:
    from alice_personality.n0.full_envelope_dev_gate_v1 import (
        evaluate_stage_gate_registry,
    )
    registry={
        "schema":"alice.eipm.n0.full-envelope-dev-gate-registry.v1",
        "stages":{
            "J1":{
                "mapping_complete":True,
                "gates":{
                    "score":{
                        "kind":"empirical",
                        "metric":"semantic.score",
                        "comparison":">=",
                        "threshold":0.9,
                    }
                },
            },
            "J2":{
                "mapping_complete":True,
                "gates":{
                    "J1_gates_retained":{
                        "kind":"retained_stage_gates",
                        "stages":["J1"],
                    }
                },
            },
        },
    }
    proof={"obligations":[]}
    receipt={"status":"PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"}
    passed=evaluate_stage_gate_registry(
        registry=registry,stage="J2",
        metrics={"semantic":{"score":0.95}},
        proof_contract=proof,static_receipt=receipt,
    )
    assert passed["stage_gate_pass"] is True
    failed=evaluate_stage_gate_registry(
        registry=registry,stage="J2",
        metrics={"semantic":{"score":0.85}},
        proof_contract=proof,static_receipt=receipt,
    )
    assert failed["stage_gate_pass"] is False


def test_j3_dev_gate_measures_positive_source_view_recoverability() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    assert "source_view_recoverability" in plan["stage_gates"]["J3"]
    gate=registry["stages"]["J3_full_public_n0_coadaptation"]["gates"][
        "source_view_recoverability"
    ]
    assert gate["kind"]=="composite"
    expected={
        "full_fabric.full_envelope_behavioral.source_view_recoverability_success_rate",
        "full_fabric.runtime_view_supplement.source_view_recoverability_success_rate",
        "full_fabric.long_context_supplement.source_view_recoverability_success_rate",
    }
    assert {item["metric"] for item in gate["all"]}==expected
    assert all(item["comparison"]==">=" and item["threshold"]==0.9 for item in gate["all"])

    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert "source_view_recoverability_success" in evaluator
    assert "source_view_recoverability_success_rate" in evaluator


def test_j3_dev_gate_measures_irrelevant_source_removal_invariance() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    assert "irrelevant_source_removal_invariance" in plan["stage_gates"]["J3"]
    gate=registry["stages"]["J3_full_public_n0_coadaptation"]["gates"][
        "irrelevant_source_removal_invariance"
    ]
    assert gate["kind"]=="composite"
    expected={
        "full_fabric.full_envelope_behavioral.irrelevant_source_removal_invariance",
        "full_fabric.runtime_view_supplement.irrelevant_source_removal_invariance",
        "full_fabric.long_context_supplement.irrelevant_source_removal_invariance",
    }
    assert {item["metric"] for item in gate["all"]}==expected
    assert all(item["comparison"]==">=" and item["threshold"]==0.95 for item in gate["all"])


def test_dev_evaluator_binds_every_dev_lane_to_exact_public_mixture_hashes() -> None:
    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert "verify_dev_lane_bindings" in evaluator
    required={
        "semantic_operator_intervention":"semantic_rows",
        "semantic_operator_long_context":"semantic_long_rows",
        "full_envelope_behavioral":"behavioral_rows",
        "runtime_view_supplement":"runtime_view_rows",
        "long_context_supplement":"long_context_rows",
        "natural_relation":"natural_rows",
    }
    for lane,arg_name in required.items():
        assert lane in evaluator
        assert arg_name in evaluator
    assert "natural_bank" in evaluator
    assert "rows_sha256" in evaluator
    assert "bank_sha256" in evaluator
    assert "DEV lane/mixture hash drift" in evaluator


def test_dev_empirical_gate_can_require_nonempty_coverage() -> None:
    from alice_personality.n0.full_envelope_dev_gate_v1 import (
        evaluate_stage_gate_registry,
    )

    registry={
        "schema":"alice.eipm.n0.full-envelope-dev-gate-registry.v1",
        "stages":{
            "J3":{
                "mapping_complete":True,
                "gates":{
                    "subset":{
                        "kind":"empirical",
                        "metric":"fabric.accuracy",
                        "comparison":">=",
                        "threshold":0.9,
                        "coverage_metric":"fabric.count",
                        "minimum_coverage":1,
                    }
                },
            }
        },
    }
    proof={"obligations":[]}
    receipt={"status":"PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"}
    empty=evaluate_stage_gate_registry(
        registry=registry,stage="J3",
        metrics={"fabric":{"accuracy":1.0,"count":0}},
        proof_contract=proof,static_receipt=receipt,
    )
    assert empty["stage_gate_pass"] is False
    covered=evaluate_stage_gate_registry(
        registry=registry,stage="J3",
        metrics={"fabric":{"accuracy":0.95,"count":2}},
        proof_contract=proof,static_receipt=receipt,
    )
    assert covered["stage_gate_pass"] is True


def test_semantic_plurality_intervention_is_not_a_single_target_known_case() -> None:
    import importlib.util
    import random

    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_semantic_operator_intervention_curriculum_v1.py"
    spec=importlib.util.spec_from_file_location(
        "semantic_intervention_builder_plurality",
        builder_path,
    )
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    row=module.make_row(
        split="train",
        relation=module.TRAIN_RELATIONS[0],
        second=module.TRAIN_RELATIONS[1],
        example=9,
        candidates=4,
        rng=random.Random(20260931),
    )
    assert row["intervention"]=="plurality"
    valid=list(row["relation_plurality_target_indices"])
    assert len(valid)>=2
    assert len(valid)==len(set(valid))
    assert all(0 <= int(index) < len(row["relation_candidates"]) for index in valid)
    assert float(row["uncertainty_target"]) > 0.0
    assert row["plurality_supervision_required"] is True

    source=(
        ROOT/"src/alice_personality/n0/semantic_operator_batch_v1.py"
    ).read_text()
    objective=(
        ROOT/"src/alice_personality/n0/semantic_operator_objectives_v1.py"
    ).read_text()
    assert "relation_plurality_target_distribution" in source
    assert "relation_plurality_target_distribution" in objective


def test_relation_plurality_soft_target_loss_does_not_force_representative_top1() -> None:
    import torch
    from alice_personality.n0.semantic_operator_objectives_v1 import (
        relation_semantic_supervision_loss,
    )

    logits=torch.tensor([[[0.0,0.0,-8.0,-8.0]]],requires_grad=True)
    representative=torch.tensor([[0]])
    active=torch.tensor([[True]])
    target=torch.tensor([[[0.5,0.5,0.0,0.0]]])
    plural=torch.tensor([[True]])
    loss=relation_semantic_supervision_loss(
        logits,
        representative,
        active,
        relation_plurality_target_distribution=target,
        relation_plurality_mask=plural,
    )
    loss.backward()
    assert float(loss.detach()) < 0.71
    assert logits.grad is not None
    assert abs(float(logits.grad[0,0,0]-logits.grad[0,0,1])) < 1.0e-6


def test_j1_dev_gate_measures_true_semantic_plurality_without_representative_top1_pressure() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    assert "semantic_plurality" in plan["stage_gates"]["J1"]
    gate=registry["stages"]["J1_joint_semantic_operator"]["gates"]["semantic_plurality"]
    assert gate["kind"]=="composite"
    expected={
        "semantic_operator.plurality_valid_mass_mean":(">=",0.9),
        "semantic_operator.plurality_forced_top1_rate":("<=",0.1),
        "semantic_operator.plurality_uncertainty_mae":("<=",0.15),
    }
    assert {
        item["metric"]:(item["comparison"],item["threshold"])
        for item in gate["all"]
    }==expected
    assert all(
        item["coverage_metric"]=="semantic_operator.plurality_count"
        and item["minimum_coverage"]>=1
        for item in gate["all"]
    )

    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert "relation_plurality_metrics" in evaluator
    assert "plurality_valid_mass_mean" in evaluator
    assert "plurality_forced_top1_rate" in evaluator
    assert "plurality_uncertainty_mae" in evaluator
    assert '"plurality"' in evaluator
    assert 'not in {"unknown_defer","plurality"}' in evaluator


def test_plurality_dev_metric_rewards_valid_set_mass_without_hard_top1() -> None:
    import torch
    from alice_personality.n0.full_envelope_dev_metrics_v1 import (
        relation_plurality_metrics,
    )

    distribution=torch.tensor([[
        [0.49,0.49,0.01,0.01],
        [0.25,0.25,0.25,0.25],
    ]])
    target=torch.tensor([[
        [0.50,0.50,0.00,0.00],
        [0.00,0.00,0.00,0.00],
    ]])
    mask=torch.tensor([[True,False]])
    result=relation_plurality_metrics(
        relation_distribution=distribution,
        plurality_target_distribution=target,
        plurality_mask=mask,
        uncertainty=torch.tensor([0.50]),
        uncertainty_target=torch.tensor([0.50]),
    )
    assert torch.allclose(result["valid_mass"],torch.tensor([0.98]))
    assert result["forced_top1"].tolist()==[False]
    assert torch.allclose(result["uncertainty_abs_error"],torch.tensor([0.0]))


def test_semantic_plurality_row_compiles_into_optimizer_facing_multi_positive_targets() -> None:
    import torch
    import importlib.util
    import random
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "semantic_intervention_builder_plurality_compile",
            scripts/"build_n0_v02_semantic_operator_intervention_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from alice_personality.n0.semantic_operator_batch_v1 import (
            compile_semantic_operator_batch,
        )
        from test_n0_semantic_operator_foundation_v1 import _OffsetTokenizer

        class _PluralityTokenizer(_OffsetTokenizer):
            def __call__(
                self,
                texts,
                *,
                padding=True,
                truncation=False,
                return_tensors="pt",
                return_offsets_mapping=False,
                return_special_tokens_mask=False,
                max_length=None,
            ):
                return super().__call__(
                    texts,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=return_tensors,
                    return_offsets_mapping=True,
                    return_special_tokens_mask=True,
                    max_length=max_length,
                )

        row=module.make_row(
            split="train",
            relation=module.TRAIN_RELATIONS[0],
            second=module.TRAIN_RELATIONS[1],
            example=9,
            candidates=4,
            rng=random.Random(20260932),
        )
        compiled=compile_semantic_operator_batch(
            rows=[row],
            tokenizer=_PluralityTokenizer(),
        )
        targets=compiled["operator_targets"]
        plural=targets["relation_plurality_mask"]
        assert plural.shape[0]==1
        assert bool(plural[0,0])
        distribution=targets["relation_plurality_target_distribution"][0,0]
        assert int(distribution.gt(0.0).sum())>=2
        assert torch.allclose(distribution.sum(),torch.tensor(1.0))
        assert compiled["metadata"]["plurality_supervision_rows"]==1
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_decisive_evidence_removal_must_increase_public_judgment_uncertainty() -> None:
    import torch
    from alice_personality.n0.full_envelope_behavioral_objectives_v1 import (
        evidence_removal_uncertainty_margin_loss,
    )

    valid=torch.tensor([[True,True]])
    normal=torch.tensor([[4.0,0.0]],requires_grad=True)
    diffuse=torch.tensor([[0.0,0.0]],requires_grad=True)
    confidently_wrong=torch.tensor([[0.0,4.0]],requires_grad=True)

    good=evidence_removal_uncertainty_margin_loss(
        normal,
        diffuse,
        candidate_valid_mask=valid,
        active_mask=torch.tensor([True]),
        minimum_entropy_increase=0.05,
    )
    bad=evidence_removal_uncertainty_margin_loss(
        normal,
        confidently_wrong,
        candidate_valid_mask=valid,
        active_mask=torch.tensor([True]),
        minimum_entropy_increase=0.05,
    )
    assert float(good.detach())==0.0
    assert float(bad.detach())>0.0
    (good+bad).backward()
    assert normal.grad is not None
    assert torch.isfinite(normal.grad).all()


def test_j3_dev_gate_measures_uncertainty_increase_after_decisive_evidence_removal() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    assert "evidence_removal_uncertainty_increase" in plan["stage_gates"]["J3"]
    gate=registry["stages"]["J3_full_public_n0_coadaptation"]["gates"][
        "evidence_removal_uncertainty_increase"
    ]
    assert gate["kind"]=="composite"
    expected={
        "full_fabric.full_envelope_behavioral.evidence_removal_uncertainty_increase",
        "full_fabric.runtime_view_supplement.evidence_removal_uncertainty_increase",
        "full_fabric.long_context_supplement.evidence_removal_uncertainty_increase",
    }
    assert {item["metric"] for item in gate["all"]}==expected
    assert all(
        item["comparison"]==">="
        and item["threshold"]==0.05
        and item["coverage_metric"].endswith(
            ".evidence_removal_uncertainty_count"
        )
        and item["minimum_coverage"]>=1
        for item in gate["all"]
    )
    evaluator=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert "evidence_removal_uncertainty_increase" in evaluator
    assert "evidence_removal_uncertainty_count" in evaluator


def test_behavioral_dev_precommits_transfer_axes_below_sealed_final() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "behavioral_builder_transfer_axes",
            scripts/"build_n0_v02_full_envelope_behavioral_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        train=module.materialize_row(
            split="train",example=4,seed=20260922,
            relation_count=4,field_count=6,answer_count=4,
        )
        dev=module.materialize_row(
            split="dev",example=4,seed=20260922,
            relation_count=4,field_count=6,answer_count=4,
        )
        final=module.materialize_row(
            split="final",example=4,seed=20260922,
            relation_count=4,field_count=6,answer_count=4,
        )
        open_name="open_semantic_factor"
        assert len(train["factor_schemas"][open_name]) < len(dev["factor_schemas"][open_name])
        assert len(dev["factor_schemas"][open_name]) < len(final["factor_schemas"][open_name])
        assert train["type_schema"] != dev["type_schema"]
        assert dev["type_schema"] != final["type_schema"]
        assert train["domain_family"] != dev["domain_family"]
        assert dev["domain_family"] != final["domain_family"]

        dev_combo=module.materialize_row(
            split="dev",example=17,seed=20260922,
            relation_count=4,field_count=6,answer_count=4,
        )
        assert dev_combo["scenario_family"]=="heldout_reliability_temporal_combo"
        assert dev_combo["factor_target_keys"]["reliability"]=="MOD_RELIABILITY_ON"
        assert dev_combo["factor_target_keys"]["temporal"]=="MOD_TEMPORAL_ON"
        assert dev_combo["factor_target_keys"]["recency"]=="MOD_RECENCY_OFF"
        assert dev_combo["factor_target_keys"]["provenance"]=="MOD_PROVENANCE_OFF"

        final_combo=module.materialize_row(
            split="final",example=18,seed=20260922,
            relation_count=4,field_count=6,answer_count=4,
        )
        assert final_combo["scenario_family"]=="heldout_recency_provenance_combo"
        assert final_combo["factor_target_keys"]["recency"]=="MOD_RECENCY_ON"
        assert final_combo["factor_target_keys"]["provenance"]=="MOD_PROVENANCE_ON"
        assert final_combo["factor_target_keys"]["reliability"]=="MOD_RELIABILITY_OFF"
        assert final_combo["factor_target_keys"]["temporal"]=="MOD_TEMPORAL_OFF"
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_j3_dev_gates_precommit_transfer_axes_needed_by_final_v2() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_dev_gate_registry_v1.json").read_text()
    )
    required={
        "factor_cardinality_extrapolation",
        "heldout_factor_combination_transfer",
        "unseen_type_schema_transfer",
        "domain_transfer",
    }
    assert required <= set(plan["stage_gates"]["J3"])
    gates=registry["stages"]["J3_full_public_n0_coadaptation"]["gates"]
    for name in required:
        assert name in gates
        gate=gates[name]
        assert gate["kind"]=="empirical"
        assert gate["comparison"]==">="
        assert gate["coverage_metric"].endswith("_count")
        assert gate["minimum_coverage"]>=1


def test_behavioral_dev_context_swap_pairs_keep_runtime_geometry_after_transfer_mode_added() -> None:
    import importlib.util
    import sys

    scripts=ROOT/"scripts/eipm/n0"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location(
            "behavioral_builder_dev_pair_geometry",
            scripts/"build_n0_v02_full_envelope_behavioral_curriculum_v1.py",
        )
        assert spec is not None and spec.loader is not None
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        relation_points=[1,2,4,6,8]
        field_points=[4,6,8,12,16]
        answer_points=[3,4,5,7]
        for first_example in (15,33,51,69,87):
            second_example=first_example+1
            first_axis=first_example
            second_axis=first_example
            first=module.materialize_row(
                split="dev",
                example=first_example,
                seed=20260922,
                relation_count=relation_points[first_axis % len(relation_points)],
                field_count=field_points[
                    (first_axis//len(relation_points)) % len(field_points)
                ],
                answer_count=answer_points[
                    (first_axis//3) % len(answer_points)
                ],
            )
            second=module.materialize_row(
                split="dev",
                example=second_example,
                seed=20260922,
                relation_count=relation_points[second_axis % len(relation_points)],
                field_count=field_points[
                    (second_axis//len(relation_points)) % len(field_points)
                ],
                answer_count=answer_points[
                    (second_axis//3) % len(answer_points)
                ],
            )
            assert first["candidate_context_swap_pair_id"] == second[
                "candidate_context_swap_pair_id"
            ]
            assert first["candidate_answers"] == second["candidate_answers"]
            assert first["relation_sequence_target"] == second[
                "relation_sequence_target"
            ]
            assert first["runtime_relation_count"] == second[
                "runtime_relation_count"
            ]
            assert first["runtime_field_count"] == second[
                "runtime_field_count"
            ]
            assert first["runtime_candidate_answer_count"] == second[
                "runtime_candidate_answer_count"
            ]
            assert first["public_target_index"] != second["public_target_index"]
            assert [
                edge["reliability"] for edge in first["edges"][:2]
            ] != [
                edge["reliability"] for edge in second["edges"][:2]
            ]
    finally:
        if sys.path and sys.path[0]==str(scripts):
            sys.path.pop(0)


def test_successor_stage_transition_binds_dev_receipt_to_exact_checkpoint_state() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "accelerator_state_tree_sha256" in trainer
    assert "candidate_checkpoint_receipt_sha256" in trainer
    assert "candidate_system_sha256" in trainer
    assert "sha256_tree(args.resume_accelerator_state)" in trainer
    assert "sha256_file(args.resume_objective_state)" in trainer
    assert "predecessor DEV receipt/checkpoint receipt hash drift" in trainer
    assert "predecessor DEV receipt/system hash drift" in trainer
    assert "resume accelerator-state hash drift" in trainer
    assert "resume objective-state hash drift" in trainer
    assert "predecessor source revision drift" in trainer



def test_full_public_mixture_audit_is_bound_to_exact_manifest_consumed_by_runtime() -> None:
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_public_mixture_manifest_v1.py"
    ).read_text()
    trainer=(
        ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py"
    ).read_text()
    gpu=(
        ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py"
    ).read_text()
    assert '"manifest_sha256":sha256(Path(args.manifest))' in auditor
    assert 'mixture_audit.get("manifest_sha256")!=sha256_file(' in trainer
    assert 'mixture_audit.get("manifest_sha256")!=sha256_file(' in gpu
    assert "full public mixture audit/manifest hash drift" in trainer
    assert "full public mixture audit/manifest hash drift" in gpu



def test_dev_evaluator_requires_exact_candidate_source_checkout() -> None:
    source=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py"
    ).read_text()
    assert 'subprocess.check_output(["git","rev-parse","HEAD"]' in source
    assert 'subprocess.check_output(["git","status","--porcelain"]' in source
    assert "--untracked-files=no" not in source
    assert "DEV evaluator source revision does not match candidate" in source
    assert "DEV evaluator requires a clean exact-source worktree" in source



def test_successor_trainer_requires_exact_short_operator_evidence_token_receipt() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    audit=(ROOT/"scripts/eipm/n0/audit_n0_v02_operator_evidence_token_alignment_v1.py").read_text()
    assert '--operator-evidence-token-receipt' in trainer
    assert 'PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1' in trainer
    assert 'operator evidence token alignment source revision drift' in trainer
    assert 'operator evidence token alignment row hash drift' in trainer
    assert 'operator evidence token alignment tokenizer hash drift' in trainer
    assert 'rows_sha256' in audit
    assert 'tokenizer_json_sha256' in audit
    assert 'source_revision' in audit



def test_successor_trainer_binds_every_optimizer_lane_to_exact_public_mixture() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "verify_optimizer_lane_bindings" in trainer
    for lane in (
        "semantic_operator_intervention","semantic_operator_long_context",
        "full_envelope_behavioral","runtime_view_supplement",
        "long_context_supplement","natural_relation",
        "broad_semantic_replay","governed_judgment_replay",
    ):
        assert lane in trainer
    assert "optimizer lane/mixture row hash drift" in trainer
    assert "optimizer lane/mixture bank hash drift" in trainer
    assert "optimizer broad replay/mixture source-config hash drift" in trainer
    assert "optimizer broad replay/mixture corpus-receipt hash drift" in trainer
    assert "optimizer teacher replay/mixture registry hash drift" in trainer
    assert "optimizer teacher replay/mixture audit hash drift" in trainer



def test_pregradient_long_token_receipts_are_exact_artifact_bound() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    boundary=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_token_boundaries_v1.py"
    ).read_text()
    semantic_long_path=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_semantic_operator_long_token_alignment_v1.py"
    )
    assert semantic_long_path.is_file(), "P40C runtime audit implementation missing"
    semantic_long=semantic_long_path.read_text()
    for source in (boundary,semantic_long):
        assert 'source_revision' in source
        assert 'rows_sha256' in source
        assert 'manifest_sha256' in source
        assert 'tokenizer_json_sha256' in source
    assert "long-context boundary row hash drift" in trainer
    assert "long-context boundary manifest hash drift" in trainer
    assert "long-context boundary tokenizer hash drift" in trainer
    assert "semantic long-context token row hash drift" in trainer
    assert "semantic long-context token manifest hash drift" in trainer
    assert "semantic long-context token tokenizer hash drift" in trainer


def test_tokenizer_stress_receipt_is_bound_to_exact_tokenizer_and_corpus() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "tokenizer stress tokenizer hash drift" in trainer
    assert "tokenizer stress corpus-receipt hash drift" in trainer



def test_cpu_and_gpu_runtime_receipts_bind_exact_registered_artifacts() -> None:
    cpu=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py").read_text()
    gpu=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert '--topology-config' in cpu
    assert '--source-revision' in cpu
    for key in (
        "source_revision","registered_topology_sha256","qualification_config_sha256",
        "semantic_config_sha256","semantic_checkpoint_sha256","tokenizer_json_sha256",
        "source_config_sha256","corpus_receipt_sha256",
    ):
        assert key in cpu
    for key in (
        "registered_topology_sha256","qualification_config_sha256",
        "semantic_config_sha256","semantic_checkpoint_sha256","tokenizer_json_sha256",
        "source_config_sha256","corpus_receipt_sha256","mixture_manifest_sha256",
        "mixture_audit_sha256","teacher_registry_sha256","teacher_audit_sha256",
    ):
        assert key in gpu
    assert "CPU runtime topology hash drift" in trainer
    assert "CPU runtime semantic checkpoint hash drift" in trainer
    assert "CPU runtime tokenizer hash drift" in trainer
    assert "GPU memory topology hash drift" in trainer
    assert "GPU memory semantic checkpoint hash drift" in trainer
    assert "GPU memory tokenizer hash drift" in trainer
    assert "GPU memory mixture manifest hash drift" in trainer
    assert "GPU memory mixture audit hash drift" in trainer



def test_gpu_memory_dry_run_binds_exact_optimizer_facing_mixture_lanes() -> None:
    source=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    for lane in (
        "semantic_operator_intervention","semantic_operator_long_context",
        "full_envelope_behavioral","runtime_view_supplement",
        "long_context_supplement","natural_relation",
        "broad_semantic_replay","governed_judgment_replay",
    ):
        assert lane in source
    assert "GPU memory lane/mixture row hash drift" in source
    assert "GPU memory lane/mixture bank hash drift" in source
    assert "GPU memory broad replay source-config hash drift" in source
    assert "GPU memory broad replay corpus-receipt hash drift" in source
    assert "GPU memory teacher registry hash drift" in source
    assert "GPU memory teacher audit hash drift" in source
    assert "GPU memory qualification source revision drift" in source



def test_runtime_training_authorization_breaks_no_gradient_source_mutation_cycle() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    # Source authority remains false permanently. Runtime evidence opens the
    # optimizer without a source edit that would invalidate exact-head receipts.
    assert plan["authorization"]["optimizer"] is False
    assert plan["authorization"]["gradient"] is False
    assert plan["authorization"]["gpu_training"] is False
    assert plan["authorization"]["runtime_training_authorization_receipt_required"] is True
    authorizer=ROOT/"scripts/eipm/n0/authorize_n0_v02_full_envelope_training_v1.py"
    assert authorizer.is_file()
    source=authorizer.read_text()
    assert "AUTHORIZED_N0_FULL_ENVELOPE_TRAINING_FROM_EXACT_RUNTIME_RECEIPTS" in source
    assert "verify_pre_gradient_runtime" in source
    assert "verify_optimizer_lane_bindings" in source
    assert "verify_public_corpus_v021" in source
    assert "verify_teacher_registry" in source
    assert "training_authorizer_sha256" in source
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert 'parser.add_argument("--training-authorization")' in trainer
    assert "runtime training authorization required for gradient" in trainer
    assert "training authorization source revision drift" in trainer
    assert "training authorization/mixture manifest hash drift" in trainer
    assert "training authorization/GPU receipt hash drift" in trainer
    assert "training authorization authorizer hash drift" in trainer



def test_stage_training_can_resume_same_stage_without_treating_step_budget_as_completion() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert 'resume_kind="same_stage"' in trainer
    assert 'resume_kind="stage_transition"' in trainer
    assert 'start_optimizer_step=int(prior.get("optimizer_step",0))+1' in trainer
    assert 'range(start_optimizer_step,args.max_optimizer_steps+1)' in trainer
    assert "same-stage resume must not supply predecessor DEV receipt" in trainer
    assert "same-stage resume checkpoint stage drift" in trainer
    assert "same-stage resume already reached requested optimizer-step operating point" in trainer
    assert 'if all_resume:' in trainer
    assert 'resume_data_seed_offset=(start_optimizer_step-1 if resume_kind=="same_stage" else 0)' in trainer
    assert 'segment_seed=args.seed+resume_data_seed_offset*100003' in trainer
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    assert plan["optimization_strategy"]["fixed_total_steps"] is None
    assert "do not stop merely because an arbitrary step budget expires" in plan[
        "optimization_strategy"
    ]["stop_rule"]



def test_dev_checkpoint_selection_enforces_first_passing_chain_member() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    assert plan["optimization_strategy"]["checkpoint_selection"].startswith(
        "first checkpoint on the precommitted"
    )
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "stage_checkpoint_parent_receipt_sha256" in trainer
    assert 'parser.add_argument("--predecessor-selection-receipt")' in trainer
    selector=ROOT/"scripts/eipm/n0/select_n0_v02_full_envelope_dev_checkpoint_v1.py"
    assert selector.is_file()
    source=selector.read_text()
    assert 'p.add_argument("--training-plan",required=True)' in source
    assert "checkpoint_evaluation_cadence_steps" in source
    assert "checkpoint receipt cadence drift" in source
    assert "checkpoint step violates precommitted DEV cadence" in source
    assert "SELECTED_FIRST_PASSING_N0_DEV_CHECKPOINT" in source
    assert "stage_checkpoint_parent_receipt_sha256" in source
    assert "candidate_checkpoint_receipt_sha256" in source
    assert "first_passing_checkpoint" in source
    assert "missing DEV receipt for checkpoint chain member" in source
    assert "earlier passing checkpoint exists" in source
    assert "off-chain stage checkpoint exists in checkpoint root" in source
    final_open=(ROOT/"scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py").read_text()
    assert 'p.add_argument("--selection-receipt",required=True)' in final_open
    assert "FINAL opening selection receipt drift" in final_open



def test_p41_public_corpus_runtime_revalidation_checks_license_and_source_balance_metadata() -> None:
    source=(ROOT/"src/alice_personality/n0/v02_training.py").read_text()
    assert 'source.get("repo_id") != spec.get("repo_id")' in source
    assert 'source.get("category") != spec.get("category")' in source
    assert 'source.get("allowed_license_values")' in source
    assert 'spec.get("allowed_license_values")' in source
    assert "source allowed-license set mismatch" in source
    assert "activated source target shares must sum to one" in source
    authorizer=(
        ROOT/"scripts/eipm/n0/authorize_n0_v02_full_envelope_training_v1.py"
    ).read_text()
    assert "verify_public_corpus_v021" in authorizer
    assert '"public_corpus_runtime_revalidated":True' in authorizer



def test_n0_contract_ci_watches_shared_runtime_data_and_verifier_dependencies() -> None:
    workflow=(
        ROOT/".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml"
    ).read_text()
    for path in (
        "src/alice_personality/n0/v02_training.py",
        "src/alice_personality/n0/data.py",
        "src/alice_personality/n0/curriculum_data.py",
    ):
        assert f'- "{path}"' in workflow
        assert path+" \\" in workflow



def test_stage_transition_restarts_optimizer_scheduler_without_losing_selected_model_state() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert 'if resume_kind=="same_stage":' in trainer
    assert 'elif resume_kind=="stage_transition":' in trainer
    assert 'predecessor_system_path=Path(args.resume_receipt).parent/"full_system.safetensors"' in trainer
    assert "stage-transition predecessor system hash drift" in trainer
    assert "system.load_state_dict(predecessor_system_state,strict=True)" in trainer
    assert 'accelerator.load_state(args.resume_accelerator_state)' in trainer
    assert 'if resume_kind=="same_stage":\n        accelerator.load_state' in trainer
    assert 'stage_transition_optimizer_state_restored":False' in trainer
    assert 'stage_transition_scheduler_state_restored":False' in trainer
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    policy=plan["optimization_strategy"]["stage_transition_optimizer_policy"]
    assert policy["model_weights"]=="selected_predecessor_checkpoint"
    assert policy["objective_balancer_state"]=="preserve"
    assert policy["optimizer_state"]=="restart"
    assert policy["scheduler_state"]=="restart_with_stage_local_warmup"



def test_stage_transition_checkpoint_receipt_preserves_dev_selection_authority_lineage() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    for field in (
        "stage_transition_predecessor_checkpoint_receipt_sha256",
        "stage_transition_predecessor_dev_receipt_sha256",
        "stage_transition_predecessor_selection_receipt_sha256",
    ):
        assert field in trainer
    assert 'if prior_stage==args.stage:' in trainer
    assert 'transition_predecessor_checkpoint_receipt_sha256=prior.get(' in trainer
    assert 'transition_predecessor_dev_receipt_sha256=prior.get(' in trainer
    assert 'transition_predecessor_selection_receipt_sha256=prior.get(' in trainer
    checkpoint=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_checkpoint_contract_v1.json").read_text()
    )
    for field in (
        "stage_transition_predecessor_checkpoint_receipt_sha256",
        "stage_transition_predecessor_dev_receipt_sha256",
        "stage_transition_predecessor_selection_receipt_sha256",
    ):
        assert field in checkpoint["required_lineage"]



def test_stage_scheduler_horizon_cannot_zero_learning_rate_before_dev_completion() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "minimum_lr_scale" in trainer
    assert "post-horizon scheduler floor must be strictly positive" in trainer
    assert "minimum_lr_scale+(1.0-minimum_lr_scale)" in trainer
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    scheduler=plan["optimization_strategy"]["scheduler_policy"]
    assert scheduler["family"]=="linear_warmup_cosine_decay_to_nonzero_floor"
    assert 0.0 < float(scheduler["minimum_lr_scale"]) < 1.0
    assert scheduler["horizon_is_stage_completion"] is False
    assert scheduler["post_horizon_behavior"]=="hold_precommitted_nonzero_floor_until_dev_decision"
    checkpoint=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_checkpoint_contract_v1.json").read_text()
    )
    assert checkpoint["scheduler_contract"]["minimum_lr_scale"]==scheduler["minimum_lr_scale"]



def test_first_passing_dev_selection_uses_precommitted_checkpoint_cadence() -> None:
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "checkpoint_evaluation_cadence_steps" in trainer
    assert "save cadence must match precommitted DEV selection cadence" in trainer
    assert '"checkpoint_evaluation_cadence_steps":int(save_every)' in trainer
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    cadence=int(plan["optimization_strategy"]["checkpoint_evaluation_cadence_steps"])
    assert cadence>0
    assert plan["optimization_strategy"]["checkpoint_cadence_fixed_before_gradient"] is True
    checkpoint=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_checkpoint_contract_v1.json").read_text()
    )
    assert checkpoint["stage_policy"]["checkpoint_evaluation_cadence_steps"]==cadence
    assert checkpoint["stage_policy"]["checkpoint_cadence_fixed_before_gradient"] is True



def test_n0_authority_scripts_reject_untracked_source_shadowing() -> None:
    for relative in (
        "scripts/eipm/n0/select_n0_v02_full_envelope_dev_checkpoint_v1.py",
        "scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py",
    ):
        source=(ROOT/relative).read_text()
        assert '["git","status","--porcelain"]' in source
        assert "--untracked-files=no" not in source
        assert "clean exact-source worktree" in source



def test_dev_evaluator_binds_candidate_to_runtime_training_authorization() -> None:
    source=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py").read_text()
    assert 'p.add_argument("--training-authorization",required=True)' in source
    assert "AUTHORIZED_N0_FULL_ENVELOPE_TRAINING_FROM_EXACT_RUNTIME_RECEIPTS" in source
    assert "candidate/training authorization receipt hash drift" in source
    assert "training authorization source revision drift" in source
    assert "training authorization/mixture manifest drift" in source
    assert "training authorization/mixture audit drift" in source
    assert "candidate/operator-token authorization lineage drift" in source
    assert "candidate/teacher-audit lineage drift" in source
    assert "candidate/public-corpus lineage drift" in source



def test_joint_trainer_enables_precommitted_backbone_gradient_checkpointing() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    assert plan["optimization_strategy"]["gradient_checkpointing"] is True
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "def enable_precommitted_gradient_checkpointing" in trainer
    assert "gradient_checkpointing_enable" in trainer
    assert "is_gradient_checkpointing" in trainer
    assert "precommitted backbone gradient checkpointing unavailable" in trainer
    assert '"gradient_checkpointing_enabled":bool(gradient_checkpointing_enabled)' in trainer
    assert "gradient_checkpointing_enabled=True" in trainer



def test_trainer_binds_optimizer_and_runtime_route_to_plan_and_p43_receipt() -> None:
    plan=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json").read_text()
    )
    strategy=plan["optimization_strategy"]
    assert strategy["runtime_hyperparameters_fixed_before_gradient"] is True
    assert strategy["training_seed"]==20260922
    assert strategy["scheduler_policy"]["operating_horizon_steps"]==1000
    assert strategy["warmup_fraction"]==0.05
    gpu_cfg=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json").read_text()
    )
    route=gpu_cfg["route"]
    assert route["microbatch_size"]==1
    assert route["teacher_batch_size"]==2
    assert route["replay_sequence_length"]==512
    assert route["training_mixed_precision"]=="fp16"
    qualifier=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    for field in (
        "teacher_batch_size","replay_sequence_length","training_mixed_precision",
    ):
        assert f'"{field}"' in qualifier
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    for message in (
        "backbone learning rate drift from precommitted plan",
        "interface learning rate drift from precommitted plan",
        "backbone weight decay drift from precommitted plan",
        "interface weight decay drift from precommitted plan",
        "gradient clip drift from precommitted plan",
        "training seed drift from precommitted plan",
        "scheduler horizon drift from precommitted plan",
        "warmup steps drift from precommitted plan",
        "P43 lane microbatch drift",
        "P43 MLM microbatch drift",
        "P43 teacher batch drift",
        "P43 replay sequence length drift",
        "P43 gradient accumulation drift",
        "P43 mixed precision drift",
        "P43 world-size drift",
    ):
        assert message in trainer



def test_p42_p43_require_canonical_clean_exact_source_configs() -> None:
    for relative in (
        "scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py",
        "scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py",
    ):
        source=(ROOT/relative).read_text()
        assert "require_canonical_source_file" in source
        assert '["git","status","--porcelain"]' in source
        assert "clean exact-source worktree" in source
        assert "registered topology config" in source
        assert "semantic config" in source
        assert "public source config" in source



def test_p40_tokenizer_gate_executes_named_stress_families_and_is_exact_head_bound() -> None:
    audit=(ROOT/"scripts/eipm/n0/audit_tokenizer_v02.py").read_text()
    runner=(ROOT/"scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh").read_text()
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "PASS_N0_TOKENIZER_STRESS_V1" in audit
    assert 'parser.add_argument("--source-revision",required=True)' in audit or 'parser.add_argument("--source-revision", required=True)' in audit
    for family in (
        "byte_fallback_oov",
        "unicode_normalization",
        "heldout_relation_factor_fragmentation",
        "long_entity",
        "punctuation_code_math",
        "multilingual",
    ):
        assert family in audit
    assert "stress_family_coverage" in audit
    assert "stress_case_count" in audit
    assert "fragmentation_limits" in audit
    assert "tokenizer_stress.json" in runner
    assert "audit_tokenizer_v02.py" in runner
    assert '--source-revision "$HEAD"' in runner
    assert "PASS_N0_TOKENIZER_STRESS_V1" in runner
    assert 'PASS_TOKENIZER="PASS_N0_TOKENIZER_STRESS_V1"' in trainer
    assert "tokenizer stress source revision drift" in trainer
    assert "tokenizer stress family coverage incomplete" in trainer



def test_cpu_runtime_runner_rejects_untracked_source_shadowing_before_p40_receipts() -> None:
    runner=(ROOT/"scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh").read_text()
    assert 'git status --porcelain)' in runner
    assert '--untracked-files=no' not in runner
    assert 'STOP: repo is not clean' in runner
    assert runner.index('git status --porcelain') < runner.index('audit_tokenizer_v02.py')



def test_p40_runtime_auditors_verify_claimed_revision_against_clean_checkout() -> None:
    helper=(ROOT/"src/alice_personality/n0/source_authority_v1.py").read_text()
    assert "def require_clean_exact_revision(" in helper
    assert '["git","status","--porcelain"]' in helper
    assert '["git","rev-parse","HEAD"]' in helper
    for relative in (
        "scripts/eipm/n0/audit_tokenizer_v02.py",
        "scripts/eipm/n0/audit_n0_v02_operator_evidence_token_alignment_v1.py",
        "scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_token_boundaries_v1.py",
        "scripts/eipm/n0/audit_n0_v02_semantic_operator_long_token_alignment_v1.py",
    ):
        source=(ROOT/relative).read_text()
        assert "require_clean_exact_revision" in source
        assert "expected_revision=source_revision" in source



def test_n0_contract_ci_watches_source_authority_helper() -> None:
    workflow=(ROOT/".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml").read_text()
    path="src/alice_personality/n0/source_authority_v1.py"
    assert f'- "{path}"' in workflow
    assert path+" \\\"" not in workflow  # guard accidental quoted escape form
    assert path+" \\" in workflow



def test_p43_gpu_memory_stresses_independent_full_fabric_axes() -> None:
    qualifier=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    config=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json").read_text()
    )
    required=set(config["qualification"]["required_full_fabric_memory_cases"])
    assert required=={
        "max_candidate_cardinality",
        "max_field_cardinality",
        "max_edge_cardinality",
        "max_view_cardinality",
        "max_reasoning_depth",
        "long_additional_view_source",
    }
    assert "choose_full_fabric_cases" in qualifier
    assert "full_fabric_case_receipts" in qualifier
    assert "required_full_fabric_memory_cases" in qualifier
    for case in required:
        assert case in qualifier
    assert "missing required full-fabric GPU memory case" in qualifier



def test_p43_gpu_memory_stresses_semantic_factor_cardinality_separately() -> None:
    qualifier=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    config=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json").read_text()
    )
    required=set(config["qualification"]["required_semantic_memory_cases"])
    assert required=={
        "max_runtime_axes",
        "max_factor_cardinality",
        "long_context_semantic",
    }
    assert "choose_max_factor_semantic" in qualifier
    assert "required_semantic_memory_cases" in qualifier
    assert "semantic_factor_candidate_count" in qualifier
    assert "missing required semantic GPU memory case" in qualifier



def test_n0_contract_ci_syntax_checks_authoritative_runtime_shell_entrypoints() -> None:
    workflow=(ROOT/".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml").read_text()
    for path in (
        "scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh",
        "scripts/eipm/n0/magnolia_cpu_n0_v02_full_envelope_runtime_v1.sbatch",
        "scripts/eipm/n0/magnolia_p100x2_n0_v02_full_envelope_gpu_memory_v1.sbatch",
    ):
        assert f'bash -n {path}' in workflow



def test_p43_gpu_memory_wraps_actual_ddp_replica_route_before_measurement() -> None:
    qualifier=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "DistributedDataParallel" in qualifier
    assert 'route.get("ddp_replica_topology") is not True' in qualifier
    assert "ddp_replica_wrapped" in qualifier
    assert "find_unused_parameters=True" in qualifier
    assert "P43 DDP replica route was not measured" in trainer



def test_p43_magnolia_handoff_asserts_expanded_memory_coverage_and_clean_source() -> None:
    sbatch=(ROOT/"scripts/eipm/n0/magnolia_p100x2_n0_v02_full_envelope_gpu_memory_v1.sbatch").read_text()
    assert 'git status --porcelain)' in sbatch
    assert '--untracked-files=no' not in sbatch
    assert 'set(r["semantic_case_receipts"])=={' in sbatch
    assert '"max_factor_cardinality"' in sbatch
    assert 'set(r["full_fabric_case_receipts"])==set(r["required_full_fabric_memory_cases"])' in sbatch
    assert 'r["stress_pair_count"]==18' in sbatch
    assert 'r["world_size"]==2' in sbatch
    assert 'r["ddp_replica_wrapped"] is True' in sbatch



def test_n0_contract_ci_watches_magnolia_udocker_runtime_wrapper() -> None:
    workflow=(ROOT/".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml").read_text()
    path="scripts/eipm/n0/magnolia_udocker_exec.sh"
    assert f'- "{path}"' in workflow
    assert f'bash -n {path}' in workflow



def test_magnolia_udocker_wrapper_mounts_exact_repo_root() -> None:
    wrapper=(ROOT/"scripts/eipm/n0/magnolia_udocker_exec.sh").read_text()
    assert '--volume="$ROOT:$ROOT"' in wrapper
    assert '--workdir="$ROOT"' in wrapper
    assert '--env="ALICE_N0_REPO_ROOT=$ROOT"' in wrapper



def test_p43_memory_projection_includes_measured_resident_runtime_overhead() -> None:
    qualifier=(ROOT/"scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").read_text()
    assert "projection_baseline_bytes=max(model_bytes,resident_before)" in qualifier
    assert "projection_baseline_bytes" in qualifier
    assert "projected=(\n        projection_baseline_bytes" in qualifier
    assert '"projection_baseline_bytes":projection_baseline_bytes' in qualifier
    assert '"measured_resident_overhead_bytes":max(0,resident_before-model_bytes)' in qualifier



def test_static_proof_receipt_is_canonical_and_clean_exact_source_bound() -> None:
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_proof_obligations_v1.py"
    ).read_text()
    assert "require_clean_exact_revision" in auditor
    assert "require_canonical_source_file" in auditor
    assert 'configs/eipm/n0/n0_v02_full_envelope_proof_obligations_v1.json' in auditor
    assert 'configs/eipm/n0/n0_v02_full_envelope_supersession_map_v1.json' in auditor
    assert 'configs/eipm/n0/n0_v02_full_envelope_retrospective_audit_v1.json' in auditor
    assert '"proof_contract_sha256"' in auditor
    assert '"supersession_sha256"' in auditor
    assert '"retrospective_sha256"' in auditor
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "static proof contract hash drift" in trainer
    assert "static proof supersession hash drift" in trainer
    assert "static proof retrospective hash drift" in trainer



def test_full_public_mixture_audit_verifies_clean_exact_source_and_canonical_contract() -> None:
    source=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_public_mixture_manifest_v1.py"
    ).read_text()
    assert "require_clean_exact_revision" in source
    assert "require_canonical_source_file" in source
    assert 'configs/eipm/n0/n0_v02_full_public_mixture_contract_v1.json' in source
    assert '"contract_sha256"' in source
    assert '"auditor_sha256"' in source
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "mixture audit contract hash drift" in trainer
    assert "mixture audit implementation hash drift" in trainer



def test_static_proof_receipt_executes_all_registered_static_tests() -> None:
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_proof_obligations_v1.py"
    ).read_text()
    assert 'subprocess.run(' in auditor
    assert '"-m","pytest","-q"' in auditor
    assert 'contract.get("static_test_files", [])' in auditor
    assert '"static_suite_executed": True' in auditor
    assert '"static_suite_pass": bool(static_suite_pass)' in auditor
    assert '"static_suite_exit_code": int(static_suite_exit_code)' in auditor
    assert '"executed_static_test_files": static_test_files' in auditor
    assert "static proof suite failed" in auditor
    assert 'label="static proof audit after pytest"' in auditor
    assert "tree.body" in auditor
    assert "required_static_nodeids" in auditor
    assert '"--junitxml"' in auditor
    assert '"pytest_skipped_cases"' in auditor
    assert "static proof suite skipped required cases" in auditor
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "static proof suite was not executed" in trainer
    assert "static proof suite did not pass" in trainer
    assert "static proof suite file coverage drift" in trainer
    assert "static proof suite skipped required cases" in trainer

    dev=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py").read_text()
    assert "DEV static proof suite was not executed" in dev
    assert "DEV static proof suite did not pass" in dev
    assert "DEV static proof suite file coverage drift" in dev
    assert "DEV static proof suite skipped required cases" in dev
    final=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py").read_text()
    assert "FINAL static proof suite was not executed" in final
    assert "FINAL static proof suite did not pass" in final
    assert "FINAL static proof suite file coverage drift" in final
    assert "FINAL static proof suite skipped required cases" in final



def test_checkpoint_receipt_consumers_require_canonical_schema_and_status() -> None:
    expected_schema="alice.eipm.n0.full-envelope-checkpoint-receipt.v1"
    expected_status="TRAINED_PUBLIC_N0_CANDIDATE_REQUIRES_DEV_SELECTION"
    for relative in (
        "scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py",
        "scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py",
        "scripts/eipm/n0/select_n0_v02_full_envelope_dev_checkpoint_v1.py",
        "scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py",
        "scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py",
    ):
        source=(ROOT/relative).read_text()
        assert expected_schema in source
        assert expected_status in source
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "resume checkpoint receipt schema drift" in trainer
    assert "resume checkpoint receipt status drift" in trainer
    dev=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py").read_text()
    assert "candidate checkpoint receipt schema drift" in dev
    assert "candidate checkpoint receipt status drift" in dev
    selector=(ROOT/"scripts/eipm/n0/select_n0_v02_full_envelope_dev_checkpoint_v1.py").read_text()
    assert "checkpoint receipt status drift" in selector
    opening=(ROOT/"scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py").read_text()
    assert "FINAL opening candidate checkpoint receipt schema drift" in opening
    assert "FINAL opening candidate checkpoint receipt status drift" in opening
    final=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py").read_text()
    assert "FINAL candidate checkpoint receipt schema drift" in final
    assert "FINAL candidate checkpoint receipt status drift" in final



def test_dev_receipt_is_bound_to_canonical_evaluator_and_gate_contracts() -> None:
    evaluator=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py").read_text()
    for field in (
        "dev_evaluator_sha256",
        "dev_contract_sha256",
        "gate_registry_sha256",
        "proof_contract_sha256",
        "training_plan_sha256",
    ):
        assert f'"{field}"' in evaluator
    selector=(ROOT/"scripts/eipm/n0/select_n0_v02_full_envelope_dev_checkpoint_v1.py").read_text()
    assert "DEV evaluator implementation hash drift" in selector
    assert "DEV contract hash drift" in selector
    assert "DEV gate registry hash drift" in selector
    assert "DEV proof contract hash drift" in selector
    assert "DEV training plan hash drift" in selector
    assert '"selected_dev_evaluator_sha256"' in selector
    opening=(ROOT/"scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py").read_text()
    assert "FINAL opening DEV evaluator implementation hash drift" in opening
    final=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py").read_text()
    assert "FINAL DEV evaluator implementation hash drift" in final



def test_static_proof_receipt_binds_all_ci_required_receipts_to_authoritative_workflow() -> None:
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_proof_obligations_v1.py"
    ).read_text()
    assert 'n0-full-envelope-foundation-build-v1-contract.yml' in auditor
    assert '"ci_workflow_sha256"' in auditor
    assert '"ci_required_receipts"' in auditor
    assert "CI_REQUIRED receipt missing from authoritative workflow" in auditor
    trainer=(ROOT/"scripts/eipm/n0/train_n0_v02_full_envelope_joint_v1.py").read_text()
    assert "static proof CI workflow hash drift" in trainer
    assert "static proof CI receipt coverage drift" in trainer
    dev=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_dev_v1.py").read_text()
    assert "DEV static proof CI workflow hash drift" in dev
    final=(ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py").read_text()
    assert "FINAL static proof CI workflow hash drift" in final

