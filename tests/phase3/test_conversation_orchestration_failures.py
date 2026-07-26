from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from alice_conversation.contracts import ModelRequest, ModelResponse
from alice_conversation.model import (
    CancellationToken,
    ConversationModelBudgetError,
    ConversationModelCancelledError,
    ConversationModelConfigurationError,
    ConversationModelProtocolError,
    ConversationModelProviderError,
    ConversationModelTimeoutError,
    ProviderFailure,
)
from alice_conversation.orchestration import (
    ConversationGenerationInterruptedError,
    ConversationTurnCancelledError,
    ConversationTurnCommand,
    ConversationTurnFailedError,
    ConversationTurnInterruptedError,
)
from alice_conversation.state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)

from _orchestration_helpers import RecordingModel, make_orchestrator


def command() -> ConversationTurnCommand:
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id="turn-1",
        user_message_id="user-1",
        assistant_message_id="assistant-1",
        request_id="request-1",
        generation_id="generation-1",
        provider="deterministic-test",
        model="orchestration-v1",
        user_content="Explain the current state.",
    )


@dataclass
class RaisingModel:
    error: Exception
    provider: str = "deterministic-test"
    model: str = "orchestration-v1"
    calls: int = 0
    requests: list[ModelRequest] = field(default_factory=list)

    def generate(self, request, cancellation=None):
        self.calls += 1
        self.requests.append(request)
        raise self.error


@pytest.mark.parametrize(
    "error,expected_code",
    [
        (ConversationModelTimeoutError("late"), "model_timeout"),
        (ConversationModelBudgetError("large"), "model_budget"),
        (ConversationModelConfigurationError("bad config"), "model_configuration"),
        (ConversationModelProtocolError("bad protocol"), "model_protocol"),
    ],
)
def test_known_model_failures_are_recorded_once(tmp_path, error, expected_code):
    model = RaisingModel(error)
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnFailedError) as raised:
        orchestrator.run_turn(command())
    assert raised.value.failure_code == expected_code
    assert model.calls == 1
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "failed"
    assert turn.failure_code == expected_code
    assert len(turn.messages) == 1
    assert len(turn.generations) == 1
    assert turn.generations[0].status == "failed"
    assert turn.generations[0].failure_code == expected_code
    assert verify_conversation_session_integrity(
        store, session_id="session-1"
    ).valid


def test_provider_failure_is_sanitized_in_state(tmp_path):
    failure = ProviderFailure(
        provider="deterministic-test",
        model="orchestration-v1",
        code="secret-provider-code",
        message="private transport detail must not be persisted",
        retryable=True,
    )
    model = RaisingModel(ConversationModelProviderError(failure))
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnFailedError) as raised:
        orchestrator.run_turn(command())
    assert raised.value.failure_code == "provider_failure"
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    turn = inspection.turns[0]
    assert turn.failure_code == "provider_failure"
    assert turn.generations[0].failure_code == "provider_failure"
    serialized = repr(inspection)
    assert "private transport detail" not in serialized
    assert "secret-provider-code" not in serialized


def test_pre_cancelled_token_records_cancelled_generation(tmp_path):
    model = RecordingModel()
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(ConversationTurnCancelledError) as raised:
        orchestrator.run_turn(command(), cancellation=token)
    assert raised.value.failure_code == "model_cancelled"
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "cancelled"
    assert turn.failure_code == "model_cancelled"
    assert len(turn.messages) == 1
    assert turn.generations[0].status == "cancelled"


def test_explicit_model_cancellation_records_cancelled_generation(tmp_path):
    model = RaisingModel(ConversationModelCancelledError("cancelled"))
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnCancelledError):
        orchestrator.run_turn(command())
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "cancelled"
    assert turn.generations[0].status == "cancelled"


def test_interruption_records_resumable_state_without_assistant_message(tmp_path):
    model = RaisingModel(ConversationGenerationInterruptedError())
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnInterruptedError) as raised:
        orchestrator.run_turn(command())
    assert raised.value.failure_code == "model_interrupted"
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    assert inspection.status == "interrupted"
    turn = inspection.turns[0]
    assert turn.status == "interrupted"
    assert turn.interruption_count == 1
    assert len(turn.messages) == 1
    assert turn.generations[0].status == "interrupted"


def test_unregistered_model_fails_after_context_without_generation_attempt(tmp_path):
    orchestrator, store, _, model, _, _ = make_orchestrator(tmp_path)
    selected = ConversationTurnCommand(
        **{**command().__dict__, "model": "missing-model"}
    )

    with pytest.raises(ConversationTurnFailedError) as raised:
        orchestrator.run_turn(selected)
    assert raised.value.failure_code == "model_configuration"
    assert model.calls == 0
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "failed"
    assert turn.failure_code == "model_configuration"
    assert len(turn.generations) == 0


@dataclass
class WrongIdentityModel:
    mode: str
    provider: str = "deterministic-test"
    model: str = "orchestration-v1"
    calls: int = 0

    def generate(self, request, cancellation=None):
        self.calls += 1
        request_id = request.request_id
        provider = self.provider
        model = self.model
        if self.mode == "request":
            request_id = "wrong-request"
        elif self.mode == "provider":
            provider = "wrong-provider"
        elif self.mode == "model":
            model = "wrong-model"
        elif self.mode == "empty":
            return ModelResponse(
                request_id=request_id,
                provider=provider,
                model=model,
                content="",
                finish_reason="stop",
                created_at="2026-07-26T05:00:30Z",
            )
        return ModelResponse(
            request_id=request_id,
            provider=provider,
            model=model,
            content="response",
            finish_reason="stop",
            created_at="2026-07-26T05:00:30Z",
        )


@pytest.mark.parametrize("mode", ["request", "provider", "model", "empty"])
def test_response_contract_violations_fail_without_assistant_message(tmp_path, mode):
    model = WrongIdentityModel(mode)
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnFailedError) as raised:
        orchestrator.run_turn(command())
    assert raised.value.failure_code == "model_protocol"
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "failed"
    assert len(turn.messages) == 1
    assert turn.generations[0].status == "failed"


def test_unexpected_exception_is_recorded_without_exception_text(tmp_path):
    model = RaisingModel(RuntimeError("secret internal detail"))
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnFailedError) as raised:
        orchestrator.run_turn(command())
    assert raised.value.failure_code == "orchestration_internal"
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    assert "secret internal detail" not in repr(inspection)
    assert inspection.turns[0].failure_code == "orchestration_internal"
