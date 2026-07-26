from __future__ import annotations

from pathlib import Path

import pytest

from alice_conversation.orchestration import (
    ConversationResumeCommand,
    ConversationTurnFailedError,
    ConversationTurnInterruptedError,
)
from _context_helpers import RecordingModel, command, make_orchestrator


def test_orchestrator_passes_bounded_history_to_model(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    for index in range(1, 4):
        orchestrator.run_turn(command(index))
    third = model.requests[-1]
    assert tuple(message.role for message in third.messages) == (
        "user", "assistant", "user", "assistant", "user"
    )


def test_resume_reassembles_same_completed_history_without_duplication(tmp_path: Path) -> None:
    model = RecordingModel(interrupt_calls={2})
    orchestrator, _, _, model = make_orchestrator(tmp_path, model=model)
    orchestrator.run_turn(command(1))
    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(command(2))
    resumed = orchestrator.resume_turn(
        ConversationResumeCommand(
            session_id="session-1",
            turn_id="turn-2",
            assistant_message_id="assistant-2",
            request_id="request-2-resume",
            generation_id="generation-2-resume",
            provider="deterministic-test",
            model="context-v1",
        )
    )
    assert resumed.replayed is False
    assert [message.turn_id for message in model.requests[-1].messages] == [
        "turn-1", "turn-1", "turn-2"
    ]


def test_context_integrity_failure_fails_current_turn_with_sanitized_code(tmp_path: Path, monkeypatch) -> None:
    orchestrator, _, _, _ = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1))

    from alice_conversation import orchestration as module
    from alice_conversation.context_assembly import ConversationContextAssemblyError

    def fail(*args, **kwargs):
        raise ConversationContextAssemblyError(
            "private details must not escape", failure_code="context_integrity_failed"
        )

    monkeypatch.setattr(module, "assemble_conversation_context", fail)
    with pytest.raises(ConversationTurnFailedError) as caught:
        orchestrator.run_turn(command(2))
    assert caught.value.failure_code == "context_integrity_failed"
    assert "private details" not in str(caught.value)
