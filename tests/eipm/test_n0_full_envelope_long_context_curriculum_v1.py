from __future__ import annotations

import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_long_context_train_dev_supplement_is_executable_and_covers_every_governed_text_surface() -> None:
    contract_path=ROOT/"configs/eipm/n0/n0_v02_full_envelope_long_context_curriculum_contract_v1.json"
    builder_path=ROOT/"scripts/eipm/n0/build_n0_v02_full_envelope_long_context_curriculum_v1.py"
    audit_path=ROOT/"scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_curriculum_v1.py"
    assert contract_path.is_file(), "long-context TRAIN/DEV contract missing"
    assert builder_path.is_file(), "long-context TRAIN/DEV builder missing"
    assert audit_path.is_file(), "long-context TRAIN/DEV audit missing"

    contract=json.loads(contract_path.read_text())
    assert contract["schema"]=="alice.eipm.n0.full-envelope-long-context-curriculum-contract.v1"
    assert contract["authority"]["final_rows_training_allowed"] is False
    assert contract["authority"]["private_identity_data"] is False
    assert contract["operating_point"]["native_window_tokens"]==4096
    assert contract["operating_point"]["long_word_target"] > 4096
    assert contract["operating_point"]["long_word_target_is_product_ceiling"] is False
    assert set(contract["required_surfaces"]) == {
        "query",
        "relation_schema",
        "factor_schema",
        "type_schema",
        "field_text",
        "field_descriptor",
        "candidate_text",
        "internal_view_descriptor",
    }
    assert contract["semantics"]["decisive_content_after_long_prefix_required"] is True
    assert contract["validation_before_gradient"]["materialized_train_dev_rows_required"] is True
    assert contract["validation_before_gradient"]["base_target_equivalence_audit_required"] is True
    assert contract["validation_before_gradient"]["surface_coverage_receipt_required"] is True
