from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIGHLIGHT = "OWNER-RATIFIED FLAGSHIP CAPABILITY RULE"

HIGHLIGHT_DOCUMENTS = [
    "README.md",
    "docs/ALICE_FRIDAY_FLAGSHIP_GOVERNANCE.md",
    "docs/ALICE_FRIDAY_SEPARATION_PLAN.md",
    "docs/FRIDAY_ARCHITECTURE.md",
    "docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md",
    "docs/FRIDAY_PRODUCTION_GOVERNANCE.md",
    "docs/FRIDAY_PRODUCT_VISION.md",
    "docs/FRIDAY_ROADMAP.md",
    "docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md",
    "docs/ROADMAP.md",
    "docs/SHARED_KERNEL_EXTRACTION_STANDARD.md",
    "docs/decisions/ADR-011-friday-independent-repository-dual-approval.md",
]

GOVERNANCE_DOCUMENTS = [
    "docs/ALICE_FRIDAY_SEPARATION_PLAN.md",
    "docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md",
    "docs/FRIDAY_PRODUCTION_GOVERNANCE.md",
    "docs/FRIDAY_PRODUCT_VISION.md",
    "docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md",
    "docs/decisions/ADR-011-friday-independent-repository-dual-approval.md",
]

FORBIDDEN_LEGACY_PHRASES = [
    "Friday development teams may research, design, prototype, implement, test, and propose changes independently.",
    "A future Friday team may independently research, design, implement, test, and propose features.",
    "The team does not need prior permission to think, research, prototype, or prepare candidates.",
    "The team may go its own way in research and implementation.",
    "A dedicated Friday team may diverge in research and product experience",
    "a Friday team may develop independently but cannot unilaterally ship",
    "A future Friday team is free to invent, research, prototype, test, and propose product-specific work",
    "Friday's production team is free to invent, investigate, implement, and argue for its own product direction.",
    "release and evolve products independently while sharing versioned contracts",
]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def existing_documents(paths: list[str]) -> list[Path]:
    return [ROOT / relative for relative in paths if (ROOT / relative).is_file()]


def test_alice_is_flagship_and_default_upstream() -> None:
    policy = load("policies/alice_friday_flagship_governance.json")
    assert policy["flagship_product"] == "alice"
    assert policy["default_downstream_product"] == "friday"
    assert policy["default_capability_flow"] == [
        "alice",
        "personal-cognitive-kernel",
        "friday",
    ]


def test_parity_is_mandatory_through_phase_15() -> None:
    parity = load("policies/alice_friday_flagship_governance.json")["parity"]
    assert parity["required_through_alice_phase"] == "15"
    assert parity["all_transferable_alice_capabilities_required"] is True
    assert parity["permanent_omission_before_phase_15_allowed"] is False


def test_friday_first_capability_is_rejected_by_default() -> None:
    gate = load(
        "policies/alice_friday_flagship_governance.json"
    )["default_new_capability_gate"]
    assert gate["friday_first_capability_allowed"] is False
    assert (
        gate[
            "alice_existing_implemented_evaluated_approved_and_gained_capability_required"
        ]
        is True
    )
    assert gate["friday_discovered_capability_must_be_proposed_upstream"] is True
    assert gate["alice_must_gain_capability_before_friday_eligibility"] is True
    assert gate["alice_permission_required"] is True
    assert gate["rayan_permission_required"] is True


def test_handover_does_not_create_capability_sovereignty() -> None:
    handover = load(
        "policies/alice_friday_flagship_governance.json"
    )["handover"]
    assert handover["formal_owner_handover_required"] is True
    assert (
        handover["pre_handover_friday_independent_capability_governance_allowed"]
        is False
    )
    assert (
        handover["post_handover_friday_independent_maintenance_and_roadmap_allowed"]
        is True
    )
    assert handover["post_handover_new_capability_dual_permission_required"] is True


def test_rayan_retains_supreme_override_authority() -> None:
    override = load(
        "policies/alice_friday_flagship_governance.json"
    )["owner_override"]
    assert override["authority"] == "MK Rayan"
    assert override["may_override_any_product_family_policy"] is True
    assert override["may_authorize_friday_first_capability"] is True
    assert override["explicit_record_required"] is True
    assert override["exact_scope_and_artifact_binding_required"] is True
    assert override["non_precedential_by_default"] is True


def test_product_manifest_and_parity_ledger_agree() -> None:
    product_lines = load("policies/product_lines.json")
    parity = load("policies/capability_parity_ledger.json")["flagship_governance"]
    assert product_lines["flagship_governance"]["flagship_product"] == "alice"
    assert (
        product_lines["flagship_governance"][
            "minimum_full_parity_through_alice_phase"
        ]
        == "15"
    )
    assert (
        product_lines["flagship_governance"][
            "friday_first_capability_allowed_by_default"
        ]
        is False
    )
    assert parity["flagship_product"] == "alice"
    assert parity["minimum_full_parity_through_alice_phase"] == "15"
    assert parity["owner_override_allowed"] is True


def test_pre_phase_65_state_is_foundation_only() -> None:
    gate = load("policies/friday_pre_phase_6_5_gate.json")
    assert gate["state"] == "foundation_only_waiting_for_alice_phase_6_5"
    assert gate["repository_exists"] is True
    assert gate["phase_6_5_is_readiness_certification_not_repository_creation"] is True
    assert "production_runtime_activation" in gate["prohibited_before_gate"]


def test_production_policy_is_unambiguous_and_deny_by_default() -> None:
    production = load("policies/friday_production_governance.json")
    candidate = production["candidate_work"]
    assert production["policy_version"] == "1.1.0"
    assert candidate["maintenance_allowed"] is True
    assert candidate["product_experience_research_allowed"] is True
    assert candidate["upstream_proposal_research_allowed"] is True
    assert candidate["eligible_capability_productization_allowed"] is True
    assert candidate["independent_research_allowed"] is False
    assert candidate["independent_design_allowed"] is False
    assert candidate["independent_implementation_allowed"] is False
    assert candidate["independent_testing_allowed"] is False
    assert candidate["new_capability_implementation_as_friday_capability_allowed"] is False
    assert candidate["new_capability_testing_as_friday_capability_allowed"] is False
    assert candidate["unilateral_production_promotion_allowed"] is False
    amendment = production["owner_ratified_flagship_amendment"]
    assert amendment["supersedes_candidate_work_for_new_capabilities"] is True
    assert amendment["legacy_candidate_work_fields_are_deny_by_default"] is True
    assert amendment["uneligible_capability_implementation_or_testing_allowed"] is False
    assert amendment["alice_must_gain_capability_before_friday_eligibility"] is True
    assert amendment["rayan_owner_override_allowed"] is True


def test_owner_override_schema_is_exact_scope_bound() -> None:
    schema = load("policies/rayan_owner_override_schema.json")
    assert schema["properties"]["decision_owner"]["const"] == "MK Rayan"
    assert "waived_requirements" in schema["required"]
    assert "effective_scope" in schema["required"]
    assert "risk_disposition" in schema["required"]


def test_governance_callouts_render_as_github_admonitions() -> None:
    documents = existing_documents(HIGHLIGHT_DOCUMENTS)
    assert documents
    for path in documents:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert HIGHLIGHT in text, path.relative_to(ROOT)
        assert "\\n>" not in "\n".join(lines[:12]), path.relative_to(ROOT)
        marker_index = lines.index("> [!IMPORTANT]")
        assert lines[marker_index + 1].startswith("> **" + HIGHLIGHT)


def test_conflicting_friday_first_language_is_removed() -> None:
    documents = existing_documents(GOVERNANCE_DOCUMENTS)
    assert documents
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN_LEGACY_PHRASES:
            assert phrase not in text, f"{phrase!r} remains in {path.relative_to(ROOT)}"
