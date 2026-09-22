from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/eipm/n0/audit_n0_v02_full_envelope_shortcuts_v1.py"
WORKFLOW = ROOT / ".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml"
CONTRACT = ROOT / "configs/eipm/n0/n0_v02_semantic_operator_curriculum_contract_v1.json"


def test_full_envelope_shortcut_preflight_is_current_successor_authority() -> None:
    contract=json.loads(CONTRACT.read_text(encoding="utf-8"))
    controls=contract["shortcut_controls"]
    required={
        "template_only_prediction_probe_required",
        "lexical_overlap_baseline_required",
        "metadata_only_baseline_required",
        "schema_only_without_query_baseline_required",
        "query_only_without_schema_baseline_required",
        "random_candidate_permutation_consistency_required",
        "hard_negative_minimum_per_positive",
        "counterfactual_pairs_required",
        "no_single_shortcut_baseline_may_clear_operator_gate",
    }
    assert all(controls.get(name) is True for name in required)

    assert SCRIPT.exists(), (
        "full-envelope shortcut requirements are declared but no current "
        "successor shortcut auditor exists"
    )
    source=SCRIPT.read_text(encoding="utf-8")
    for token in (
        "template_only",
        "lexical_overlap",
        "metadata_only",
        "schema_only_without_query",
        "query_only_without_schema",
        "candidate_permutation",
        "hard_negative",
        "counterfactual",
        "PASS_N0_FULL_ENVELOPE_SHORTCUT_PREFLIGHT_V1",
    ):
        assert token in source, token

    workflow=WORKFLOW.read_text(encoding="utf-8")
    assert "audit_n0_v02_full_envelope_shortcuts_v1.py" in workflow
    assert "PASS_N0_FULL_ENVELOPE_SHORTCUT_PREFLIGHT_V1" in workflow
    assert "audit_n0_v02_qsre_t2_shortcuts_v0_1.py" not in workflow
