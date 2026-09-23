from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_successor_final_v2_package_exists_and_legacy_final_cannot_satisfy_it() -> None:
    final_contract=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_validation_contract_v2.json").read_text()
    )
    package_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json"
    assert package_path.is_file(), "successor final-v2 package contract is missing"
    package=json.loads(package_path.read_text())
    assert package["schema"]=="alice.eipm.n0.full-envelope-final-package.v1"
    assert package["final_contract"]=="configs/eipm/n0/n0_v02_full_envelope_final_validation_contract_v2.json"
    assert package["registered_system"]=="src/alice_personality/n0/n0_full_envelope_trainable_system_v1.py"
    assert package["legacy_final_v1_authority"] is False
    assert package["legacy_final_v3_evaluator_authority"] is False
    assert package["training_authorized"] is False
    assert package["model_selection_authorized"] is False
    assert package["final_opening_authorized"] is False
    assert final_contract["corpus"]["full_final_corpus_build_and_hash_freeze_required"] is True

    required=[
        "scripts/eipm/n0/build_n0_v02_full_envelope_final_v2_package.py",
        "scripts/eipm/n0/audit_n0_v02_full_envelope_final_v2_package.py",
        "scripts/eipm/n0/freeze_n0_v02_full_envelope_final_v2_package.py",
        "configs/eipm/n0/n0_v02_full_envelope_final_v2_evaluator_contract_v1.json",
    ]
    for relative in required:
        assert (ROOT/relative).is_file(), relative

    evaluator=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_v2_evaluator_contract_v1.json").read_text()
    )
    assert evaluator["registered_system"]=="N0FullEnvelopeTrainableSystemV1"
    assert evaluator["legacy_final_self_validation_v1_forbidden"] is True
    assert evaluator["legacy_final_self_validation_v3_forbidden"] is True
    assert evaluator["results_observed"] is False
    assert evaluator["final_opening_authorized"] is False


def test_final_v2_evaluator_is_executable_precommitted_and_freeze_bound() -> None:
    package=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json").read_text()
    )
    evaluator=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_v2_evaluator_contract_v1.json").read_text()
    )
    final_contract=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_validation_contract_v2.json").read_text()
    )
    implementation=ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py"
    gate_registry=ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_v2_gate_registry_v1.json"
    freezer=(
        ROOT/"scripts/eipm/n0/freeze_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()

    assert implementation.is_file(), "successor FINAL-v2 evaluator implementation missing"
    assert gate_registry.is_file(), "successor FINAL-v2 gate registry missing"
    assert package["evaluator_implementation"] == str(
        implementation.relative_to(ROOT)
    )
    assert package["evaluator_gate_registry"] == str(
        gate_registry.relative_to(ROOT)
    )
    assert evaluator["evaluator_implementation"] == str(
        implementation.relative_to(ROOT)
    )
    assert evaluator["gate_registry"] == str(gate_registry.relative_to(ROOT))

    registry=json.loads(gate_registry.read_text())
    assert registry["schema"]=="alice.eipm.n0.full-envelope-final-v2-gate-registry.v1"
    required_sections=set(evaluator["required_gate_sections"])
    assert set(registry["sections"])==required_sections
    for section in sorted(required_sections):
        expected_keys=set(final_contract[section])
        mapped_keys=set(registry["sections"][section])
        assert mapped_keys==expected_keys, (
            section,
            sorted(expected_keys-mapped_keys),
            sorted(mapped_keys-expected_keys),
        )

    source=implementation.read_text()
    assert "FINAL_V2_EVALUATION_COMPLETE" in source
    assert "checkpoint_selection_performed" in source
    assert "automatic_repair_or_rerun" in source
    assert "threshold_changes_after_results" in source
    assert "final_results_observed" in source
    assert "N0FullEnvelopeTrainableSystemV1" in source

    # The evaluator and exact gate mapping are part of the frozen package,
    # not mutable post-result implementation details.
    assert "--evaluator-implementation" in freezer
    assert "--gate-registry" in freezer
    assert '"evaluator_implementation"' in freezer
    assert '"gate_registry"' in freezer


def test_final_v2_has_independent_synthetic_semantic_operator_component() -> None:
    package=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json").read_text()
    )
    component=package["components"].get("synthetic_semantic_operator")
    assert isinstance(component,dict), (
        "FINAL-v2 cannot measure synthetic held-out relation families, "
        "semantic plurality, schema-evidence causality, or mixed-step operator "
        "semantics from the behavioral component alone"
    )
    assert component["row_schema"]=="alice.eipm.n0.semantic-operator-intervention-row.v1"
    assert component["relation_families_heldout_from_train_dev"] is True
    assert component["plurality_rows_required"] is True
    assert component["source_target_pairs_required"] is True
    assert component["ordered_composition_required"] is True
    assert component["mixed_step_direction_required"] is True
    assert component["mixed_step_modifier_required"] is True
    assert component["unknown_defer_required"] is True

    builder=(
        ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    freezer=(
        ROOT/"scripts/eipm/n0/freeze_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    assert "--semantic-final-output" in builder
    assert "--semantic-final-rows" in auditor
    assert "--semantic-final-rows" in freezer
    assert "FINAL_SEMANTIC_RELATIONS" in builder


def test_semantic_operator_row_builder_supports_sealed_final_authority_without_changing_train_dev_defaults() -> None:
    import importlib.util
    import random

    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_semantic_operator_intervention_curriculum_v1.py"
    spec=importlib.util.spec_from_file_location("semantic_builder_final_authority",builder_path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    final_relations=[
        module.rel(
            "z001","final_attestation",
            "The source artifact receives an attestation from the target authority.",
            "artifact receiving attestation","attesting authority",
            ["receives attestation from","is attested by"],
        ),
        module.rel(
            "z002","final_reservation",
            "The source resource is reserved for the target activity.",
            "reserved resource","activity holding the reservation",
            ["is reserved for","is held for"],
        ),
    ]
    row=module.make_row(
        split="final",
        relation=final_relations[0],
        second=final_relations[1],
        example=0,
        candidates=2,
        rng=random.Random(7),
        entity_pool=["FinalA","FinalB","FinalC","FinalD"],
        candidate_pool=final_relations,
        relation_partition="sealed_final_relation_family",
        template_partition="sealed_final_templates",
        data_origin="sealed_final_test",
    )
    assert row["split"]=="final"
    assert row["training_authorized"] is False
    assert row["model_selection_authorized"] is False
    assert row["final_validation_only"] is True
    assert row["relation_partition"]=="sealed_final_relation_family"
    assert row["template_partition"]=="sealed_final_templates"
    assert row["data_origin"]=="sealed_final_test"


def test_final_v2_contains_heterogeneous_runtime_view_and_long_context_challenges() -> None:
    package=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json").read_text()
    )
    runtime=package["components"].get("heterogeneous_runtime_views")
    long_context=package["components"].get("long_context")
    assert isinstance(runtime,dict), (
        "FINAL-v2 has runtime-view gates but no sealed additional-view challenge"
    )
    assert runtime["fixed_view_taxonomy"] is False
    assert runtime["relevance_not_availability_required"] is True
    assert runtime["reliability_reversal_required"] is True
    assert runtime["unavailable_view_required"] is True
    assert isinstance(long_context,dict), (
        "FINAL-v2 has long/cross-window gates but no sealed long-text component"
    )
    assert set(long_context["required_surfaces"]) == {
        "query","relation_schema","factor_schema","type_schema",
        "field_text","field_descriptor","candidate_text",
        "internal_view_descriptor","additional_view_descriptor",
        "additional_view_source",
    }
    assert long_context["boundary_shift_pairs_required"] is True
    assert long_context["distant_decisive_semantics_required"] is True
    builder=(
        ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    freezer=(
        ROOT/"scripts/eipm/n0/freeze_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    for flag in ("--runtime-view-final-output","--long-context-final-output"):
        assert flag in builder
    for flag in ("--runtime-view-final-rows","--long-context-final-rows"):
        assert flag in auditor
        assert flag in freezer


def test_final_v2_precommits_paraphrase_pairs_and_cross_window_conflict_probe() -> None:
    package=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json").read_text()
    )
    assert package["components"]["adversarial_synthetic"]["paraphrase_pair_required"] is True
    assert package["components"]["long_context"]["cross_window_conflict_required"] is True

    behavioral=(
        ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_behavioral_curriculum_v1.py"
    ).read_text()
    final_builder=(
        ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    auditor=(
        ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_final_v2_package.py"
    ).read_text()
    assert "final_paraphrase_query" in behavioral
    assert "example_override=8" in final_builder
    assert "conflict_plurality" in auditor
    assert "final_paraphrase_query" in auditor


def test_final_directional_endpoint_reversal_gate_uses_end_to_end_behavior() -> None:
    """A semantic reverse-program score cannot stand in for endpoint behavior."""
    registry=json.loads(
        (ROOT/"configs/eipm/n0/n0_v02_full_envelope_final_v2_gate_registry_v1.json").read_text()
    )
    gate=registry["sections"]["deterministic_behavior_gate"][
        "directional_relation_endpoint_reversal_sensitivity_min"
    ]
    assert gate["kind"]=="metric"
    assert gate["path"]==(
        "fabric.scenario_public_accuracy.reverse_traversal"
    )
    assert gate["coverage_path"]==(
        "fabric.scenario_public_count.reverse_traversal"
    )



def test_final_v2_evaluator_requires_exact_candidate_source_checkout() -> None:
    source=(
        ROOT/"scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py"
    ).read_text()
    assert 'subprocess.check_output(["git","rev-parse","HEAD"]' in source
    assert 'subprocess.check_output(["git","status","--porcelain","--untracked-files=no"]' in source
    assert "FINAL evaluator source revision does not match candidate" in source
    assert "FINAL evaluator requires a clean tracked-source worktree" in source

