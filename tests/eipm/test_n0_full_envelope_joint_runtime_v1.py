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
