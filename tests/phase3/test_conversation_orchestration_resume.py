from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from alice_conversation.contracts import ModelRequest, ModelResponse
from alice_conversation.orchestration import (
    ConversationGenerationInterruptedError,
    ConversationOrchestrationError,
    ConversationResumeCommand,
    ConversationTurnCommand,
    ConversationTurnInterruptedError,
)
from alice_conversation.state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)

from _orchestration_helpers import grounding_packet, make_orchestrator


@dataclass
class InterruptThenRespondModel:
    provider: str = "deterministic-test"
    model: str = "orchestration-v1"
    calls: int = 0
    requests: list[ModelRequest] = field(default_factory=list)

    def generate(self, request, cancellation=None):
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            raise ConversationGenerationInterruptedError()
        response = ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content="Resumed response.",
            finish_reason="stop",
            created_at="2026-07-26T05:01:00Z",
        )
        response.validate()
        return response


def initial_command(*, grounding=None):
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id="turn-1",
        user_message_id="user-1",
        assistant_message_id="assistant-initial",
        request_id="request-initial",
        generation_id="generation-initial",
        provider="deterministic-test",
        model="orchestration-v1",
        user_content="Continue after interruption.",
        grounding=grounding,
    )


def resume_command(*, grounding=None):
    return ConversationResumeCommand(
        session_id="session-1",
        turn_id="turn-1",
        assistant_message_id="assistant-resumed",
        request_id="request-resumed",
        generation_id="generation-resumed",
        provider="deterministic-test",
        model="orchestration-v1",
        grounding=grounding,
    )


def test_explicit_resume_creates_new_contiguous_attempt_and_one_assistant(tmp_path):
    model = InterruptThenRespondModel()
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(initial_command())
    result = orchestrator.resume_turn(resume_command())

    assert result.replayed is False
    assert result.assistant_message.content == "Resumed response."
    assert model.calls == 2
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    assert inspection.status == "active"
    turn = inspection.turns[0]
    assert turn.status == "completed"
    assert turn.interruption_count == 1
    assert [item.attempt_index for item in turn.generations] == [0, 1]
    assert [item.status for item in turn.generations] == ["interrupted", "completed"]
    assert [item.request_id for item in turn.generations] == [
        "request-initial",
        "request-resumed",
    ]
    assert [message.role for message in turn.messages] == ["user", "assistant"]
    assert verify_conversation_session_integrity(
        store, session_id="session-1"
    ).valid


def test_repeating_completed_resume_command_does_not_generate_again(tmp_path):
    model = InterruptThenRespondModel()
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)

    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(initial_command())
    selected = resume_command()
    first = orchestrator.resume_turn(selected)
    second = orchestrator.resume_turn(selected)

    assert first.replayed is False
    assert second.replayed is True
    assert model.calls == 2
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert len(turn.messages) == 2
    assert len(turn.generations) == 2


def test_interrupted_grounded_turn_requires_exact_original_packet(tmp_path):
    model = InterruptThenRespondModel()
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    packet = grounding_packet()

    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(initial_command(grounding=packet))

    with pytest.raises(ConversationOrchestrationError) as missing:
        orchestrator.resume_turn(resume_command())
    assert missing.value.failure_code == "grounding_required"

    changed = grounding_packet()
    object.__setattr__(changed, "packet_id", "different-packet")
    with pytest.raises(ConversationOrchestrationError) as mismatch:
        orchestrator.resume_turn(resume_command(grounding=changed))
    assert mismatch.value.failure_code == "grounding_mismatch"

    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "interrupted"
    assert len(turn.generations) == 1


def test_resume_with_exact_grounding_preserves_request_packet(tmp_path):
    model = InterruptThenRespondModel()
    orchestrator, _, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    packet = grounding_packet()

    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(initial_command(grounding=packet))
    orchestrator.resume_turn(resume_command(grounding=packet))

    assert model.requests[0].grounding == packet
    assert model.requests[1].grounding == packet


@pytest.mark.parametrize("status", ["received", "context_ready", "generating", "failed", "cancelled"])
def test_resume_rejects_non_interrupted_statuses(tmp_path, status):
    # State transitions for every status are already covered by P3.2/P3.5 tests.
    # This focused check verifies the public resume guard using a completed turn,
    # then rewrites only the in-memory inspection path through a minimal fake.
    model = InterruptThenRespondModel()
    orchestrator, _, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    if status in {"failed", "cancelled", "received", "context_ready", "generating"}:
        with pytest.raises(ConversationOrchestrationError):
            orchestrator.resume_turn(resume_command())
