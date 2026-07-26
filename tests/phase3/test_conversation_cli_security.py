from __future__ import annotations

import pytest

from alice_conversation.cli_runtime import (
    ConversationCliPolicyViolation,
    ConversationCliRuntime,
)

from _cli_helpers import cli_policy, make_runtime


def test_runtime_rejects_nonlocal_provider(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    with pytest.raises(ConversationCliPolicyViolation):
        ConversationCliRuntime(
            state_service=runtime.state_service,
            orchestrator=runtime.orchestrator,
            provider="remote-provider",
            model="model",
            policy=cli_policy(),
        )


@pytest.mark.parametrize(
    "attribute",
    [
        "web_access_allowed",
        "tool_calling_allowed",
        "external_action_allowed",
        "memory_write_allowed",
        "memory_promotion_allowed",
        "live_retrieval_allowed",
        "hidden_reasoning_display_allowed",
        "raw_database_identifiers_display_allowed",
    ],
)
def test_security_boundaries_remain_false(attribute):
    policy = cli_policy()
    assert policy.boundary(attribute) is False


def test_local_only_and_private_vault_are_required():
    policy = cli_policy()
    assert policy.boundary("local_only") is True
    assert policy.boundary("private_vault_required") is True
    assert policy.boundary("repository_state_allowed") is False


def test_runtime_does_not_expose_database_path(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    inspection = runtime.inspect()
    rendered = repr(inspection)
    assert "sqlite" not in rendered.lower()
    assert str(runtime.state_service.store.database_path) not in rendered


def test_runtime_does_not_expose_hidden_reasoning(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, ["Consider the next step."])
    output = runtime.send("Hello")
    assert "reasoning" not in repr(output).lower()
    assert "chain_of_thought" not in repr(output).lower()
