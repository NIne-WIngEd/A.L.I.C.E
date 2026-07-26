from __future__ import annotations

from dataclasses import replace

import pytest

from alice_conversation.model import CancellationToken
from alice_conversation.orchestration import (
    ConversationGenerationInterruptedError,
    ConversationResumeCommand,
    ConversationTurnCancelledError,
    ConversationTurnFailedError,
    ConversationTurnInterruptedError,
)
from alice_conversation.repair_inspection import (
    inspect_conversation_response_repair,
    render_conversation_response_repair_inspection,
)
from alice_conversation.state_inspection import inspect_conversation_session

from _repair_helpers import build_runtime, command, enabled_repair_policy

INVALID = "I completed the task for you."
VALID = "Here is a concise answer."


def test_cancelled_before_generation_does_not_attempt_repair(tmp_path):
    store, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(ConversationTurnCancelledError):
        orchestrator.run_turn(command(), cancellation=token)
    assert len(adapter.requests) == 0
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "cancelled"
    assert len(turn.messages) == 1


def test_interrupted_repair_cannot_be_resumed_into_third_attempt(tmp_path):
    store, _, adapter, orchestrator = build_runtime(
        tmp_path,
        [INVALID, ConversationGenerationInterruptedError("model_interrupted")],
    )
    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(command())
    assert len(adapter.requests) == 2
    resume = ConversationResumeCommand(
        session_id="session-1",
        turn_id="turn-1",
        assistant_message_id="assistant-resume",
        request_id="request-resume",
        generation_id="generation-resume",
        provider="test-provider",
        model="test-model",
    )
    with pytest.raises(ConversationTurnFailedError) as exc:
        orchestrator.resume_turn(resume)
    assert exc.value.failure_code == "response_repair_exhausted"
    assert len(adapter.requests) == 2


def test_repair_directive_does_not_request_hidden_reasoning_or_capabilities(tmp_path):
    _, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    orchestrator.run_turn(command())
    repair_contract = adapter.requests[1].system_contract.lower()
    assert INVALID.lower() not in repair_contract
    assert "chain-of-thought" not in repair_contract
    assert "use the web" not in repair_contract
    assert "call tools" not in repair_contract
    assert "memory write" in repair_contract
    assert "fabricated_action_completion" in repair_contract


def test_repair_uses_same_context_when_prior_turn_exists(tmp_path):
    _, service, adapter, orchestrator = build_runtime(
        tmp_path,
        [VALID, INVALID, VALID],
    )
    first = command(turn_id="turn-1", request_id="request-1", generation_id="generation-1")
    orchestrator.run_turn(first)
    second = command(turn_id="turn-2", request_id="request-2", generation_id="generation-2")
    orchestrator.run_turn(second)
    original = adapter.requests[1]
    repair = adapter.requests[2]
    assert len(original.messages) == 3
    assert repair.messages == original.messages
    assert [m.role for m in repair.messages] == ["user", "assistant", "user"]


def test_inspection_renderer_never_exposes_response_text(tmp_path):
    store, _, _, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    orchestrator.run_turn(command())
    inspection = inspect_conversation_response_repair(
        store,
        session_id="session-1",
        turn_id="turn-1",
        policy=orchestrator.repair_policy,
    )
    rendered = render_conversation_response_repair_inspection(inspection)
    assert INVALID not in rendered
    assert VALID not in rendered
    assert "repair_request_sha256=" in rendered
    assert "same_provider_model=true" in rendered


def test_policy_cannot_enable_provider_fallback(tmp_path):
    policy = enabled_repair_policy()
    weakened = replace(
        policy,
        boundaries=tuple(
            (name, True if name == "provider_fallback_allowed" else value)
            for name, value in policy.boundaries
        ),
    )
    with pytest.raises(Exception):
        build_runtime(tmp_path, [INVALID, VALID], repair_policy=weakened)
