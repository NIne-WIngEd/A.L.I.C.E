from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from alice_conversation.repair_policy import (
    ConversationResponseRepairPolicyError,
    load_conversation_response_repair_policy,
    parse_conversation_response_repair_policy,
)

from _repair_helpers import POLICIES, enabled_repair_policy


def payload():
    return json.loads((POLICIES / "conversation_response_repair_policy.json").read_text())


def test_repository_policy_is_valid_and_disabled_by_default():
    policy = load_conversation_response_repair_policy(
        POLICIES / "conversation_response_repair_policy.json"
    )
    assert policy.version == "1.0.0"
    assert policy.enabled is False
    assert policy.max_repair_attempts == 1
    assert policy.boundary("same_provider_model_required") is True
    assert policy.boundary("provider_fallback_allowed") is False


def test_policy_can_explicitly_enable_controlled_repair():
    data = payload()
    data["enabled"] = True
    policy = parse_conversation_response_repair_policy(data)
    assert policy.enabled is True


@pytest.mark.parametrize("name", sorted(payload()["boundaries"]))
def test_policy_rejects_any_weakened_boundary(name):
    data = payload()
    data["boundaries"][name] = not data["boundaries"][name]
    with pytest.raises(ConversationResponseRepairPolicyError):
        parse_conversation_response_repair_policy(data)


@pytest.mark.parametrize(
    "path,value",
    [
        (("policy_name",), "wrong"),
        (("phase",), "4"),
        (("milestone",), "P3.10"),
        (("status",), "uncontrolled"),
        (("limits", "max_repair_attempts"), 2),
        (("limits", "max_issue_codes"), 0),
        (("limits", "max_repair_prompt_chars"), 100),
        (("limits", "max_repair_output_tokens"), 0),
        (("limits", "max_total_output_tokens"), 100),
        (("limits", "max_total_elapsed_seconds"), 0),
        (("failure_codes", "exhausted"), "not safe"),
    ],
)
def test_policy_rejects_contract_drift(path, value):
    data = payload()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConversationResponseRepairPolicyError):
        parse_conversation_response_repair_policy(data)


def test_policy_rejects_extra_fields():
    data = payload()
    data["extra"] = True
    with pytest.raises(ConversationResponseRepairPolicyError):
        parse_conversation_response_repair_policy(data)


def test_enabled_policy_remains_exactly_one_attempt():
    policy = enabled_repair_policy()
    with pytest.raises(ConversationResponseRepairPolicyError):
        replace(policy, max_repair_attempts=3).validate()
