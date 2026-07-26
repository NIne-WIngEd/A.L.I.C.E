from __future__ import annotations

from pathlib import Path

import pytest

from alice_conversation.state_inspection import inspect_conversation_session
from alice_conversation.state_service import ConversationStateError

from _state_helpers import TIMES, make_store, message, response


def _generating_turn(tmp_path: Path):
    store, service, _, _ = make_store(tmp_path)
    service.create_session(
        session_id="session-1",
        created_at=TIMES[0],
        retention="retained",
    )
    user = message(
        message_id="user-1",
        turn_id="turn-1",
        role="user",
        content="Continue",
        created_at=TIMES[1],
    )
    service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    )
    service.set_turn_context(
        turn_id="turn-1", references=(), updated_at=TIMES[2]
    )
    service.start_generation(
        turn_id="turn-1",
        generation_id="generation-1",
        request_id="request-1",
        provider="deterministic-test",
        model="fixed-response-v1",
        started_at=TIMES[3],
    )
    return store, service


def test_interrupted_turn_can_resume_with_new_generation(tmp_path: Path) -> None:
    store, service = _generating_turn(tmp_path)
    service.interrupt_turn(
        turn_id="turn-1",
        request_id="request-1",
        interrupted_at=TIMES[4],
        reason_code="user_interrupt",
    )
    interrupted = inspect_conversation_session(store, session_id="session-1")
    assert interrupted.status == "interrupted"
    assert interrupted.turns[0].status == "interrupted"
    assert interrupted.turns[0].interruption_count == 1
    assert interrupted.turns[0].generations[0].status == "interrupted"

    service.resume_turn(turn_id="turn-1", resumed_at=TIMES[5])
    assert service.start_generation(
        turn_id="turn-1",
        generation_id="generation-2",
        request_id="request-2",
        provider="deterministic-test",
        model="fixed-response-v1",
        started_at=TIMES[6],
    ) == 1
    assistant = message(
        message_id="assistant-1",
        turn_id="turn-1",
        role="assistant",
        content="Resumed answer",
        created_at=TIMES[7],
    )
    service.complete_turn(
        turn_id="turn-1",
        request_id="request-2",
        response=response(
            request_id="request-2",
            content=assistant.content,
            created_at=TIMES[7],
        ),
        assistant_message=assistant,
        completed_at=TIMES[7],
    )
    completed = inspect_conversation_session(store, session_id="session-1")
    assert completed.status == "active"
    assert completed.turns[0].status == "completed"
    assert [generation.status for generation in completed.turns[0].generations] == [
        "interrupted",
        "completed",
    ]


def test_received_turn_can_be_cancelled(tmp_path: Path) -> None:
    store, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="user-1",
        turn_id="turn-1",
        role="user",
        content="Cancel this",
        created_at=TIMES[1],
    )
    service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    )
    service.cancel_turn(
        turn_id="turn-1",
        cancelled_at=TIMES[2],
        reason_code="user_cancelled",
    )
    inspection = inspect_conversation_session(store, session_id="session-1")
    assert inspection.status == "active"
    assert inspection.turns[0].status == "cancelled"
    assert inspection.turns[0].failure_code == "user_cancelled"


def test_generating_turn_can_fail_with_sanitized_code(tmp_path: Path) -> None:
    store, service = _generating_turn(tmp_path)
    service.fail_turn(
        turn_id="turn-1",
        failed_at=TIMES[4],
        failure_code="provider_timeout",
    )
    inspection = inspect_conversation_session(store, session_id="session-1")
    turn = inspection.turns[0]
    assert turn.status == "failed"
    assert turn.failure_code == "provider_timeout"
    assert turn.generations[0].status == "failed"
    assert turn.generations[0].failure_code == "provider_timeout"


def test_free_form_failure_text_is_rejected(tmp_path: Path) -> None:
    _, service = _generating_turn(tmp_path)
    with pytest.raises(ConversationStateError):
        service.fail_turn(
            turn_id="turn-1",
            failed_at=TIMES[4],
            failure_code="provider failed because user said a private sentence",
        )


def test_session_cannot_close_while_turn_is_open(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="user-1",
        turn_id="turn-1",
        role="user",
        content="Open",
        created_at=TIMES[1],
    )
    service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    )
    with pytest.raises(ConversationStateError):
        service.close_session(session_id="session-1", closed_at=TIMES[2])
