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
