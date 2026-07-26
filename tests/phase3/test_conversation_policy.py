"""P3.0 public conversation-policy validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alice_conversation.policy import (
    ConversationPolicyError,
    load_conversation_policy,
    parse_conversation_policy,
)

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_policy.json"
)


def _payload() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_repository_policy_loads_with_zero_capabilities() -> None:
    policy = load_conversation_policy(POLICY_PATH)
    assert policy.phase == "3"
    assert policy.milestone == "P3.0"
    assert policy.default_data_classification == "PRIVATE"
    assert policy.default_retention == "session_only"
    assert policy.ordinary_grounding_classifications == (
        "PUBLIC",
        "INTERNAL",
        "PRIVATE",
    )
    policy.capabilities.validate()


def test_policy_rejects_tools_or_external_actions() -> None:
    payload = _payload()
    payload["allowed_tools"] = ["web_search"]
    with pytest.raises(ConversationPolicyError, match="empty list"):
        parse_conversation_policy(payload)
    payload = _payload()
    payload["boundaries"]["external_action_allowed"] = True
    with pytest.raises(ConversationPolicyError, match="must remain false"):
        parse_conversation_policy(payload)


def test_policy_rejects_chain_of_thought_persistence() -> None:
    payload = _payload()
    payload["conversation_state"]["persist_chain_of_thought"] = True
    with pytest.raises(ConversationPolicyError, match="must remain false"):
        parse_conversation_policy(payload)


def test_policy_preserves_phase2_memory_promotion_boundary() -> None:
    payload = _payload()
    payload["conversation_state"]["durable_memory_promotion_path"] = "direct"
    with pytest.raises(ConversationPolicyError, match="Phase 2 candidate"):
        parse_conversation_policy(payload)


def test_policy_rejects_highly_sensitive_ordinary_grounding() -> None:
    payload = _payload()
    payload["grounding"]["ordinary_classifications"].append(
        "HIGHLY_SENSITIVE"
    )
    with pytest.raises(ConversationPolicyError, match="classification boundary"):
        parse_conversation_policy(payload)
