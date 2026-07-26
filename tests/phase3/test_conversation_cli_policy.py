from __future__ import annotations

from copy import deepcopy

import pytest

from alice_conversation.cli_policy import (
    ConversationCliPolicyError,
    parse_conversation_cli_policy,
)

from _cli_helpers import cli_policy_payload


def test_cli_policy_parses_exact_contract():
    policy = parse_conversation_cli_policy(cli_policy_payload())
    assert policy.version == "1.0.0"
    assert policy.allowed_providers == ("ollama-local",)
    assert policy.default_retention == "session_only"
    assert policy.commands[-1] == ":exit"


@pytest.mark.parametrize("field", sorted(cli_policy_payload()["boundaries"]))
def test_cli_policy_rejects_weakened_boundaries(field):
    payload = cli_policy_payload()
    payload["boundaries"][field] = not payload["boundaries"][field]
    with pytest.raises(ConversationCliPolicyError):
        parse_conversation_cli_policy(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("policy_name",), "wrong"),
        (("phase",), "2"),
        (("milestone",), "P3.8"),
        (("status",), "other"),
        (("runtime", "allowed_retentions"), ["retained", "session_only"]),
        (("runtime", "default_retention"), "retained"),
        (("runtime", "allowed_providers"), ["deterministic-test"]),
        (("runtime", "explicit_provider_required"), False),
        (("runtime", "explicit_model_required"), False),
        (("runtime", "prebuilt_grounding_file_allowed"), False),
        (("commands",), [":exit"]),
        (("limits", "max_input_chars"), 100),
        (("limits", "max_grounding_file_bytes"), 100),
    ],
)
def test_cli_policy_rejects_contract_drift(path, value):
    payload = deepcopy(cli_policy_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConversationCliPolicyError):
        parse_conversation_cli_policy(payload)


@pytest.mark.parametrize("extra", ["extra", "tools", "memory"])
def test_cli_policy_rejects_extra_root_fields(extra):
    payload = cli_policy_payload()
    payload[extra] = True
    with pytest.raises(ConversationCliPolicyError):
        parse_conversation_cli_policy(payload)
