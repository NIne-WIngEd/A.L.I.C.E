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
