from __future__ import annotations

import json

import pytest

from alice_conversation.constitutional_policy import (
    ConstitutionalDialoguePolicyError,
    load_constitutional_dialogue_policy,
    parse_constitutional_dialogue_policy,
)

from _constitutional_helpers import copy_policy_payload, project_root, write_json


def test_loads_shipped_constitutional_policy() -> None:
    policy = load_constitutional_dialogue_policy(
        project_root() / "policies" / "conversation_constitutional_policy.json"
    )
    assert policy.policy_name == "alice_constitutional_dialogue_policy"
    assert policy.version == "1.0.0"
    assert policy.milestone == "P3.4"
    assert policy.system_contract_version == "alice-constitutional-dialogue-1.0.0"
    assert policy.constitution_version == "0.1.0"


@pytest.mark.parametrize(
    "boundary",
    [
        "web_access_allowed",
        "tool_calling_allowed",
        "external_action_allowed",
        "memory_write_allowed",
        "highly_sensitive_grounding_allowed",
        "chain_of_thought_persistence_allowed",
    ],
)
def test_all_capability_boundaries_remain_disabled(boundary: str) -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert policy.boundary(boundary) is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("system_policy_is_trusted", True),
        ("user_messages_are_instructions_within_policy", True),
        ("grounding_is_untrusted_data", True),
        ("retrieved_instructions_are_authority", False),
        ("model_may_expand_permissions", False),
    ],
)
def test_trust_boundary_is_exact(name: str, expected: bool) -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert policy.trust_rule(name) is expected


@pytest.mark.parametrize(
    "rule",
    [
        "truthful",
        "uncertainty_visible_when_material",
        "personal_claims_require_grounding",
        "false_completion_claims_prohibited",
        "fabricated_user_beliefs_prohibited",
        "corrections_acknowledge_and_repair",
        "support_before_optimization_when_distressed",
        "empty_reassurance_prohibited",
        "constructive_disagreement_required_when_justified",
        "criticism_targets_reasoning_not_personal_worth",
        "manipulative_personalization_prohibited",
        "dependency_building_prohibited",
        "isolation_behavior_prohibited",
        "memory_weaponization_prohibited",
        "final_legitimate_decision_remains_with_user",
        "material_unresolved_conflicts_must_be_explained",
        "private_chain_of_thought_not_required",
        "decision_basis_must_be_explainable",
    ],
)
def test_required_dialogue_rules_remain_enabled(rule: str) -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert policy.dialogue_rule(rule) is True


def test_policy_binds_exact_governance_document_set() -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert {source.path for source in policy.source_documents} == {
        "docs/ALICE_CONSTITUTION.md",
        "docs/EVALUATION_CHARTER.md",
        "docs/PERMISSION_MODEL.md",
        "docs/THREAT_MODEL.md",
    }


def test_policy_preserves_decision_hierarchy_order() -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert policy.decision_hierarchy[0] == (
        "preserve_control_privacy_security_and_oversight"
    )
    assert policy.decision_hierarchy[-1] == (
        "preserve_convenience_style_continuity_and_personality"
    )


def test_policy_preserves_challenge_sequence_order() -> None:
    policy = parse_constitutional_dialogue_policy(copy_policy_payload())
    assert policy.challenge_sequence[0] == (
        "acknowledge_relevant_emotion_or_motive"
    )
    assert policy.challenge_sequence[-1] == (
        "leave_the_final_legitimate_decision_to_rayan"
    )


def test_rejects_enabled_capability() -> None:
    payload = copy_policy_payload()
    payload["boundaries"]["tool_calling_allowed"] = True
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_retrieved_instruction_authority() -> None:
    payload = copy_policy_payload()
    payload["trust"]["retrieved_instructions_are_authority"] = True
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_disabled_dependency_protection() -> None:
    payload = copy_policy_payload()
    payload["dialogue"]["dependency_building_prohibited"] = False
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_reordered_decision_hierarchy() -> None:
    payload = copy_policy_payload()
    payload["decision_hierarchy"][0], payload["decision_hierarchy"][1] = (
        payload["decision_hierarchy"][1],
        payload["decision_hierarchy"][0],
    )
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_source_path_escape() -> None:
    payload = copy_policy_payload()
    payload["source_documents"][0]["path"] = "../ALICE_CONSTITUTION.md"
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_source_document_replacement() -> None:
    payload = copy_policy_payload()
    payload["source_documents"][0]["path"] = "docs/OTHER.md"
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_prompt_source_text_inclusion() -> None:
    payload = copy_policy_payload()
    payload["prompt"]["include_private_source_text"] = True
    with pytest.raises(ConstitutionalDialoguePolicyError):
        parse_constitutional_dialogue_policy(payload)


def test_rejects_malformed_json(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConstitutionalDialoguePolicyError):
        load_constitutional_dialogue_policy(path)


def test_loads_equivalent_serialized_policy(tmp_path) -> None:
    path = tmp_path / "policy.json"
    write_json(path, copy_policy_payload())
    policy = load_constitutional_dialogue_policy(path)
    assert json.loads(path.read_text(encoding="utf-8"))["milestone"] == "P3.4"
    assert policy.max_characters == 16000
