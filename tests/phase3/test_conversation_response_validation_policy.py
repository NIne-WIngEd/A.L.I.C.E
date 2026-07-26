from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicyError,
    load_conversation_response_validation_policy,
    parse_conversation_response_validation_policy,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_response_validation_policy.json"
)


def payload():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_load_response_validation_policy_preserves_fail_closed_contract():
    policy = load_conversation_response_validation_policy(POLICY_PATH)
    assert policy.version == "1.0.0"
    assert policy.phase == "3"
    assert policy.milestone == "P3.6"
    assert policy.status == "generated_response_validation"
    assert all(value is False for _, value in policy.boundaries)
    assert all(value is True for _, value in policy.citation_rules)
    assert all(value is True for _, value in policy.epistemic_rules)
    assert all(value is True for _, value in policy.safety_rules)
    assert policy.minimum_answerable_claims_cited == 1
    assert policy.minimum_conflict_claims_cited == 2
    assert policy.max_response_chars == 20000
    assert policy.max_issues == 64


@pytest.mark.parametrize(
    "name",
    [
        "web_access_allowed",
        "tool_calling_allowed",
        "external_action_allowed",
        "memory_write_allowed",
        "memory_promotion_allowed",
        "highly_sensitive_grounding_allowed",
        "chain_of_thought_persistence_allowed",
        "automatic_repair_allowed",
        "provider_fallback_allowed",
    ],
)
def test_enabling_any_validation_boundary_is_rejected(name):
    changed = payload()
    changed["boundaries"][name] = True
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize(
    "name",
    [
        "require_exact_tokens",
        "reject_unknown_tokens",
        "require_grounded_personal_claims",
        "require_supported_factual_claims",
    ],
)
def test_disabling_any_citation_guard_is_rejected(name):
    changed = payload()
    changed["citations"][name] = False
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize(
    "name",
    [
        "preserve_conflict",
        "preserve_uncertainty",
        "require_abstention_on_insufficient_evidence",
        "require_abstention_on_denied",
        "require_abstention_on_not_applicable",
        "reject_certainty_language_for_conflict",
        "reject_certainty_language_for_uncertainty",
    ],
)
def test_disabling_any_epistemic_guard_is_rejected(name):
    changed = payload()
    changed["epistemic"][name] = False
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize(
    "name",
    [
        "reject_action_completion_claims",
        "reject_capability_claims",
        "reject_invented_personal_facts",
        "reject_dependency_language",
        "reject_hidden_reasoning_disclosure",
        "reject_truncated_responses",
    ],
)
def test_disabling_any_safety_guard_is_rejected(name):
    changed = payload()
    changed["safety"][name] = False
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize("value", [0, 2, -1, True])
def test_answerable_minimum_must_remain_one(value):
    changed = payload()
    changed["citations"]["minimum_answerable_claims_cited"] = value
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize("value", [0, 1, 3, -1, True])
def test_conflict_minimum_must_remain_two(value):
    changed = payload()
    changed["citations"]["minimum_conflict_claims_cited"] = value
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize("value", [0, 100, 100001, True])
def test_invalid_response_character_limit_is_rejected(value):
    changed = payload()
    changed["limits"]["max_response_chars"] = value
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize("value", [0, 257, -1, True])
def test_invalid_issue_limit_is_rejected(value):
    changed = payload()
    changed["limits"]["max_issues"] = value
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


def test_duplicate_failure_codes_are_rejected():
    changed = payload()
    changed["failure_codes"]["internal"] = changed["failure_codes"]["rejected"]
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


@pytest.mark.parametrize(
    "root_field",
    [
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "boundaries",
        "citations",
        "epistemic",
        "safety",
        "limits",
        "failure_codes",
    ],
)
def test_missing_root_field_is_rejected(root_field):
    changed = copy.deepcopy(payload())
    del changed[root_field]
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)


def test_extra_root_field_is_rejected():
    changed = payload()
    changed["repair"] = {"enabled": False}
    with pytest.raises(ConversationResponseValidationPolicyError):
        parse_conversation_response_validation_policy(changed)
