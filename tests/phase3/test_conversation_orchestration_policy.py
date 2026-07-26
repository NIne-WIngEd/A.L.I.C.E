from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alice_conversation.orchestration_policy import (
    ConversationOrchestrationPolicyError,
    load_conversation_orchestration_policy,
    parse_conversation_orchestration_policy,
)


POLICY_PATH = Path(__file__).resolve().parents[2] / "policies/conversation_orchestration_policy.json"


def payload():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_load_orchestration_policy_preserves_fail_closed_values():
    policy = load_conversation_orchestration_policy(POLICY_PATH)
    assert policy.version == "1.0.0"
    assert policy.milestone == "P3.5"
    assert policy.max_output_tokens == 1024
    assert policy.temperature == 0.0
    assert all(value is False for _, value in policy.boundaries)
    assert policy.lifecycle_value("automatic_retry_count") == 0
    assert policy.lifecycle_value("provider_fallback_allowed") is False
    assert policy.lifecycle_value("final_grounding_validation_enabled") is False


@pytest.mark.parametrize("name", [
    "web_access_allowed",
    "tool_calling_allowed",
    "external_action_allowed",
    "memory_write_allowed",
    "memory_promotion_allowed",
    "highly_sensitive_grounding_allowed",
    "chain_of_thought_persistence_allowed",
])
def test_enabling_any_boundary_is_rejected(name):
    changed = payload()
    changed["boundaries"][name] = True
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


@pytest.mark.parametrize("name", [
    "constitutional_contract_required",
    "prebuilt_grounding_only",
    "model_registry_resolution_required",
    "generation_attempt_recording_required",
    "atomic_state_transitions_required",
    "response_identity_match_required",
])
def test_disabling_required_lifecycle_guards_is_rejected(name):
    changed = payload()
    changed["lifecycle"][name] = False
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


@pytest.mark.parametrize("name", [
    "live_retrieval_allowed",
    "duplicate_assistant_messages_allowed",
    "provider_fallback_allowed",
    "final_grounding_validation_enabled",
])
def test_enabling_deferred_features_is_rejected(name):
    changed = payload()
    changed["lifecycle"][name] = True
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


def test_automatic_retry_is_rejected():
    changed = payload()
    changed["lifecycle"]["automatic_retry_count"] = 1
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


@pytest.mark.parametrize("temperature", [0.1, 1, -0.1, 2.0])
def test_nondeterministic_temperature_is_rejected(temperature):
    changed = payload()
    changed["request"]["temperature"] = temperature
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


@pytest.mark.parametrize("tokens", [0, -1, 8193, True])
def test_invalid_output_token_budget_is_rejected(tokens):
    changed = payload()
    changed["request"]["max_output_tokens"] = tokens
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


def test_duplicate_failure_codes_are_rejected():
    changed = payload()
    changed["failure_codes"]["timeout"] = changed["failure_codes"]["budget"]
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)


@pytest.mark.parametrize("root_field", [
    "policy_name", "version", "phase", "milestone", "status", "boundaries",
    "lifecycle", "request", "failure_codes"
])
def test_missing_root_field_is_rejected(root_field):
    changed = copy.deepcopy(payload())
    del changed[root_field]
    with pytest.raises(ConversationOrchestrationPolicyError):
        parse_conversation_orchestration_policy(changed)
