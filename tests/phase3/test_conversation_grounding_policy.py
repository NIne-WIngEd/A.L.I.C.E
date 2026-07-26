from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alice_conversation.grounding_policy import (
    ConversationGroundingPolicyError,
    load_conversation_grounding_policy,
    parse_conversation_grounding_policy,
)


def payload() -> dict:
    path = Path(__file__).resolve().parents[2] / "policies" / "conversation_grounding_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_loads_default_policy() -> None:
    policy = load_conversation_grounding_policy()
    assert policy.milestone == "P3.3"
    assert policy.maximum_phase1_sources == 12
    assert policy.phase1_default_confidence == 0.5


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("boundaries", "memory_write_allowed", True),
        ("boundaries", "external_action_allowed", True),
        ("boundaries", "tool_calling_allowed", True),
        ("boundaries", "web_access_allowed", True),
        ("boundaries", "highly_sensitive_grounding_allowed", True),
        ("integrity", "source_text_is_untrusted_data", False),
        ("integrity", "preserve_conflicts", False),
        ("integrity", "preserve_uncertainty", False),
        ("integrity", "require_exact_citation_tokens", False),
    ],
)
def test_rejects_weakened_boundaries(section: str, key: str, value: object) -> None:
    value_payload = payload()
    value_payload[section][key] = value
    with pytest.raises(ConversationGroundingPolicyError):
        parse_conversation_grounding_policy(value_payload)


def test_rejects_changed_classification_order() -> None:
    value = payload()
    value["ordinary_classifications"] = ["PRIVATE", "INTERNAL", "PUBLIC"]
    with pytest.raises(ConversationGroundingPolicyError):
        parse_conversation_grounding_policy(value)


def test_rejects_overwide_phase1_source_limit() -> None:
    value = payload()
    value["phase1"]["maximum_sources"] = 13
    with pytest.raises(ConversationGroundingPolicyError):
        parse_conversation_grounding_policy(value)


def test_rejects_phase1_truth_overstatement() -> None:
    value = payload()
    value["phase1"]["default_knowledge_status"] = "verified_fact"
    with pytest.raises(ConversationGroundingPolicyError):
        parse_conversation_grounding_policy(value)


def test_rejects_phase1_confidence_change() -> None:
    value = copy.deepcopy(payload())
    value["phase1"]["default_confidence"] = 1.0
    with pytest.raises(ConversationGroundingPolicyError):
        parse_conversation_grounding_policy(value)
