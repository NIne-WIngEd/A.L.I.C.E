from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_runtime_view_train_dev_curriculum_is_executable_and_not_internal_view_only() -> None:
    contract_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_runtime_view_curriculum_contract_v1.json"
    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_runtime_view_curriculum_v1.py"
    audit_path=ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_runtime_view_curriculum_v1.py"
    assert contract_path.is_file(), "additional runtime-view TRAIN/DEV contract missing"
    assert builder_path.is_file(), "additional runtime-view TRAIN/DEV builder missing"
    assert audit_path.is_file(), "additional runtime-view TRAIN/DEV audit missing"

    contract=json.loads(contract_path.read_text())
    assert contract["schema"]=="alice.eipm.n0.full-envelope-runtime-view-curriculum-contract.v1"
    assert contract["authority"]["private_identity_data"] is False
    assert contract["authority"]["final_rows_training_allowed"] is False
    assert contract["runtime_views"]["fixed_view_taxonomy"] is False
    assert contract["runtime_views"]["view_count_ceiling"] is None
    assert contract["runtime_views"]["source_text_training_adapter_required"] is True
    assert contract["required_interventions"]["relevant_vs_merely_available"] is True
    assert contract["required_interventions"]["constant_reliability_relevance_flip"] is True
    assert contract["required_interventions"]["unavailable_view_exact_inertness"] is True
    assert contract["required_interventions"]["recoverability_not_equal_availability"] is True
    assert contract["validation_before_gradient"]["materialized_train_dev_rows_required"] is True
    assert contract["validation_before_gradient"]["row_to_registered_system_compiler_required"] is True
