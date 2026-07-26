from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alice_conversation.context_assembly import (
    ConversationContextAssemblyError,
    assemble_conversation_context,
)
from alice_conversation.orchestration import (
    ConversationTurnInterruptedError,
    ConversationTurnValidationError,
)
from _context_helpers import RecordingModel
from _context_helpers import command, make_orchestrator


def test_rejected_assistant_output_never_enters_later_context(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    model.response_text = "I completed the external action for you."
    with pytest.raises(ConversationTurnValidationError):
        orchestrator.run_turn(command(1, content="Please act."))
    model.response_text = "Acknowledged."
    orchestrator.run_turn(command(2))
    assert [message.turn_id for message in model.requests[-1].messages] == ["turn-2"]
    assert all("completed the external action" not in message.content for message in model.requests[-1].messages)


def test_failed_turn_user_message_is_excluded_from_later_context(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    model.response_text = "I completed the external action for you."
    with pytest.raises(ConversationTurnValidationError):
        orchestrator.run_turn(command(1, content="sensitive failed input"))
    model.response_text = "Acknowledged."
    orchestrator.run_turn(command(2))
    assert all("sensitive failed input" not in message.content for message in model.requests[-1].messages)


def test_context_never_crosses_session_boundary(tmp_path: Path) -> None:
    orchestrator, _, service, model = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1, content="session one secret"))
    service.create_session(
        session_id="session-2",
        created_at="2026-07-26T08:00:00Z",
        retention="retained",
        data_classification="PRIVATE",
    )
    orchestrator.run_turn(command(2, session_id="session-2", content="session two message"))
    request = model.requests[-1]
    assert [message.content for message in request.messages] == ["session two message"]


def test_context_digest_does_not_depend_on_internal_identifiers(tmp_path: Path) -> None:
    first, store1, _, _ = make_orchestrator(tmp_path / "a", session_id="session-a")
    second, store2, _, _ = make_orchestrator(tmp_path / "b", session_id="session-b")
    first.run_turn(command(1, session_id="session-a", content="same content"))
    second.run_turn(command(99, session_id="session-b", content="same content"))
    context1 = assemble_conversation_context(store1, session_id="session-a", policy=first.context_policy)
    context2 = assemble_conversation_context(store2, session_id="session-b", policy=second.context_policy)
    assert context1.context_sha256 == context2.context_sha256


def test_integrity_tampering_blocks_context_assembly(tmp_path: Path) -> None:
    orchestrator, store, _, _ = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1))
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE conversation_messages SET content_sha256 = ? WHERE message_id = ?",
            ("0" * 64, "assistant-1"),
        )
        connection.commit()
    with pytest.raises(ConversationContextAssemblyError) as caught:
        assemble_conversation_context(store, session_id="session-1", policy=orchestrator.context_policy)
    assert caught.value.failure_code == "context_integrity_failed"


def test_cancelled_interrupted_turn_is_excluded_from_later_context(tmp_path: Path) -> None:
    model = RecordingModel(interrupt_calls={1})
    orchestrator, _, service, model = make_orchestrator(tmp_path, model=model)
    with pytest.raises(ConversationTurnInterruptedError):
        orchestrator.run_turn(command(1, content="interrupted private input"))
    service.cancel_turn(
        turn_id="turn-1",
        cancelled_at="2026-07-26T08:00:00Z",
        reason_code="user_cancelled",
    )
    orchestrator.run_turn(command(2))
    assert [item.turn_id for item in model.requests[-1].messages] == ["turn-2"]
    assert all(
        "interrupted private input" not in item.content
        for item in model.requests[-1].messages
    )
