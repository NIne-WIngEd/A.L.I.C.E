from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alice_conversation.context_policy import (
    ConversationContextPolicyError,
    load_conversation_context_policy,
    parse_conversation_context_policy,
)
from _context_helpers import context_policy, context_policy_payload


def test_context_policy_accepts_exact_p38_contract() -> None:
    policy = context_policy()
    assert policy.version == "1.0.0"
    assert policy.max_prior_turns == 12
    assert policy.max_prior_messages == 24
    assert policy.max_prior_characters == 12000
    assert policy.boundary("same_session_only") is True
    assert policy.boundary("cross_session_content_allowed") is False
    assert policy.failure_code("integrity") == "context_integrity_failed"


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("boundaries", "same_session_only", False),
        ("boundaries", "completed_turns_only", False),
        ("boundaries", "accepted_or_abstained_only", False),
        ("boundaries", "whole_turn_pairs_only", False),
        ("boundaries", "exclude_current_turn", False),
        ("boundaries", "integrity_verification_required", False),
        ("boundaries", "hidden_reasoning_allowed", True),
        ("boundaries", "rejected_output_allowed", True),
        ("boundaries", "failed_turn_content_allowed", True),
        ("boundaries", "cross_session_content_allowed", True),
        ("boundaries", "message_identifiers_rendered_to_model", True),
        ("boundaries", "semantic_summarization_allowed", True),
        ("boundaries", "memory_write_allowed", True),
        ("truncation", "drop_oldest_first", False),
        ("truncation", "partial_turn_allowed", True),
        ("truncation", "partial_message_allowed", True),
    ],
)
def test_context_policy_rejects_weakened_boolean_boundaries(
    section: str, name: str, value: bool
) -> None:
    payload = copy.deepcopy(context_policy_payload())
    payload[section][name] = value
    with pytest.raises(ConversationContextPolicyError):
        parse_conversation_context_policy(payload)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_prior_turns", 0),
        ("max_prior_turns", 129),
        ("max_prior_messages", 3),
        ("max_prior_messages", 258),
        ("max_prior_characters", 1023),
        ("max_prior_characters", 100001),
    ],
)
def test_context_policy_rejects_invalid_limits(name: str, value: int) -> None:
    payload = copy.deepcopy(context_policy_payload())
    payload["limits"][name] = value
    with pytest.raises(ConversationContextPolicyError):
        parse_conversation_context_policy(payload)


def test_context_policy_allows_independent_even_message_budget() -> None:
    policy = parse_conversation_context_policy(
        context_policy_payload(max_prior_turns=3, max_prior_messages=4)
    )
    assert policy.max_prior_turns == 3
    assert policy.max_prior_messages == 4


def test_context_policy_rejects_unknown_root_field() -> None:
    payload = context_policy_payload()
    payload["unexpected"] = True
    with pytest.raises(ConversationContextPolicyError):
        parse_conversation_context_policy(payload)


def test_context_policy_loads_repository_json(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(context_policy_payload()), encoding="utf-8")
    loaded = load_conversation_context_policy(path)
    assert loaded == context_policy()
