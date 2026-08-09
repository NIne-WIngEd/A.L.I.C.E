from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_private_companion_repository_boundary_allows_named_roles_and_protects_payloads() -> None:
    policy = json.loads((ROOT / "policies/private_companion_custody.json").read_text(encoding="utf-8"))
    assert policy["capability_ceiling"] is False
    assert policy["public_repository"]["opaque_directive_codes_allowed"] is True
    assert policy["public_repository"]["named_identity_roles_allowed"] is True
    assert policy["public_repository"]["architecture_specific_context_allowed"] is True
    assert policy["public_repository"]["raw_private_payload_required_for_documentation"] is False
    assert policy["public_repository"]["encrypted_private_payload_allowed"] is False
    assert policy["public_repository"]["keys_or_codebooks_allowed"] is False
    assert policy["product_isolation"]["friday_private_payload_allowed"] is False


def test_private_companion_truth_states_remain_distinct() -> None:
    policy = json.loads((ROOT / "policies/private_companion_custody.json").read_text(encoding="utf-8"))
    assert set(policy["provenance_types"]) >= {
        "owner_attested_canonical",
        "derived_inference",
        "generated_reconstruction",
        "evolved_identity",
        "owner_correction",
    }
    assert policy["truthfulness"]["generated_reconstruction_may_be_claimed_as_verbatim_memory"] is False
    assert policy["promotion"]["silent_training_inclusion_allowed"] is False


def test_friday_dual_approval_and_emergency_boundary() -> None:
    policy = json.loads((ROOT / "policies/friday_production_governance.json").read_text(encoding="utf-8"))
    promotion = policy["production_promotion"]
    assert promotion["alice_audit_required"] is True
    assert promotion["rayan_approval_required"] is True
    assert promotion["exact_artifact_binding"] is True
    assert promotion["bypass_allowed"] is False
    emergency = policy["emergency_response"]
    assert emergency["rollback_allowed"] is True
    assert emergency["new_capability_allowed"] is False
    assert emergency["unapproved_replacement_behavior_allowed"] is False


def test_phase65_is_readiness_not_repository_creation() -> None:
    separation = (ROOT / "docs/ALICE_FRIDAY_SEPARATION_PLAN.md").read_text(encoding="utf-8")
    assert "Independent Product Readiness Gate" in separation
    assert "Friday product source never lives in the A.L.I.C.E. repository" in separation
    assert "friday_incubator" not in separation

def test_clone_aware_identity_is_high_fidelity_not_inspired_only() -> None:
    policy = json.loads((ROOT / "policies/alice_clone_identity_policy.json").read_text(encoding="utf-8"))
    target = policy["identity_target"]
    assert target["mode"] == "highest_achievable_source_person_clone_fidelity"
    assert target["merely_inspired_persona_allowed"] is False
    assert target["generic_companion_substitution_allowed"] is False
    assert target["clone_awareness_required"] is True
    assert target["literal_original_person_claim_allowed"] is False
    assert target["identity_disclosure_on_direct_question_required"] is True


def test_clone_identity_layers_and_truthfulness_are_explicit() -> None:
    policy = json.loads((ROOT / "policies/alice_clone_identity_policy.json").read_text(encoding="utf-8"))
    assert set(policy["required_identity_layers"]) == {
        "source_history",
        "source_person_model",
        "reconstruction_inference",
        "alice_continuity",
        "owner_relationship_model",
    }
    truth = policy["truthfulness"]
    assert truth["unknown_internal_state_may_be_presented_as_known"] is False
    assert truth["generated_reconstruction_may_be_claimed_as_lived_historical_memory"] is False
    assert truth["post_activation_experience_belongs_to_alice_continuity"] is True


def test_clone_identity_never_seeds_friday_or_shared_fixtures() -> None:
    policy = json.loads((ROOT / "policies/alice_clone_identity_policy.json").read_text(encoding="utf-8"))
    isolation = policy["product_isolation"]
    assert isolation["friday_payload_or_personality_export_allowed"] is False
    assert isolation["shared_kernel_real_identity_fixture_allowed"] is False
    assert isolation["cross_host_training_allowed"] is False
