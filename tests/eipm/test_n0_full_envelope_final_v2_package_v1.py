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
