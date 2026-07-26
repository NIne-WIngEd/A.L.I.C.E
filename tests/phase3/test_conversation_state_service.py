from __future__ import annotations

from pathlib import Path

import pytest

from alice_conversation.state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)
from alice_conversation.state_service import (
    ConversationStateError,
    ConversationStateReference,
)

from _state_helpers import TIMES, digest, make_store, message, response


def _complete_retained_turn(tmp_path: Path):
    store, service, _, _ = make_store(tmp_path)
    service.create_session(
        session_id="session-1",
        created_at=TIMES[0],
        retention="retained",
    )
    user = message(
        message_id="message-user-1",
        turn_id="turn-1",
        role="user",
        content="What did I complete?",
        created_at=TIMES[1],
    )
    assert service.start_turn(
        session_id="session-1",
        turn_id="turn-1",
        user_message=user,
    ) == 0
    reference = ConversationStateReference(
        reference_id="reference-1",
        source_kind="memory_source",
        source_ref="memory-source-1",
        citation_token="[M1:S1]",
        content_sha256=digest("source record"),
        data_classification="PRIVATE",
        created_at=TIMES[2],
    )
    service.set_turn_context(
        turn_id="turn-1",
        references=(reference,),
        grounding_packet_id="packet-1",
        grounding_packet_sha256=digest("packet-1"),
        updated_at=TIMES[2],
    )
    assert service.start_generation(
        turn_id="turn-1",
        generation_id="generation-1",
        request_id="request-1",
        provider="deterministic-test",
        model="fixed-response-v1",
        started_at=TIMES[3],
    ) == 0
    assistant = message(
        message_id="message-assistant-1",
        turn_id="turn-1",
        role="assistant",
        content="You completed Phase 2.",
        created_at=TIMES[4],
    )
    service.complete_turn(
        turn_id="turn-1",
        request_id="request-1",
        response=response(
            request_id="request-1",
            content=assistant.content,
            created_at=TIMES[4],
        ),
        assistant_message=assistant,
        completed_at=TIMES[4],
    )
    return store, service


def test_retained_session_records_grounded_completed_turn(tmp_path: Path) -> None:
    store, service = _complete_retained_turn(tmp_path)
    redacted = inspect_conversation_session(
        store,
        session_id="session-1",
        include_content=False,
    )
    assert redacted.status == "active"
    assert redacted.turns[0].messages[0].content is None
    assert redacted.turns[0].grounding_packet_id == "packet-1"
    assert redacted.turns[0].references[0].source_ref == "memory-source-1"
    assert redacted.turns[0].generations[0].provider == "deterministic-test"
    assert redacted.turns[0].generations[0].reasoning_status == "not_persisted"

    visible = inspect_conversation_session(
        store,
        session_id="session-1",
        include_content=True,
    )
    assert [message.content for message in visible.turns[0].messages] == [
        "What did I complete?",
        "You completed Phase 2.",
    ]
    report = verify_conversation_session_integrity(
        store,
        session_id="session-1",
    )
    assert report.valid is True
    assert report.errors == ()
    assert report.turn_count == 1
    assert report.message_count == 2
    assert report.reference_count == 1
    assert report.generation_count == 1

    assert service.close_session(
        session_id="session-1",
        closed_at=TIMES[5],
    ) is None
    closed = inspect_conversation_session(
        store,
        session_id="session-1",
        include_content=False,
    )
    assert closed.status == "completed"
    assert closed.closed_at == TIMES[5]


def test_session_and_turn_creation_are_exactly_idempotent(tmp_path: Path) -> None:
    store, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Hello",
        created_at=TIMES[1],
    )
    assert service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    ) == 0
    assert service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    ) == 0
    inspection = inspect_conversation_session(store, session_id="session-1")
    assert len(inspection.turns) == 1
    assert len(inspection.turns[0].messages) == 1


def test_session_rejects_more_sensitive_message(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(
        session_id="session-1",
        created_at=TIMES[0],
        data_classification="INTERNAL",
    )
    user = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Private content",
        created_at=TIMES[1],
        classification="PRIVATE",
    )
    with pytest.raises(ConversationStateError):
        service.start_turn(
            session_id="session-1",
            turn_id="turn-1",
            user_message=user,
        )


def test_session_rejects_second_nonterminal_turn(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    first = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="First",
        created_at=TIMES[1],
    )
    second = message(
        message_id="message-2",
        turn_id="turn-2",
        role="user",
        content="Second",
        created_at=TIMES[2],
    )
    service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=first
    )
    with pytest.raises(ConversationStateError):
        service.start_turn(
            session_id="session-1", turn_id="turn-2", user_message=second
        )


def test_context_rejects_duplicate_logical_references(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Question",
        created_at=TIMES[1],
    )
    service.start_turn(
        session_id="session-1", turn_id="turn-1", user_message=user
    )
    refs = tuple(
        ConversationStateReference(
            reference_id=f"reference-{index}",
            source_kind="memory",
            source_ref="same-memory",
            data_classification="PRIVATE",
            created_at=TIMES[2],
        )
        for index in range(2)
    )
    with pytest.raises(ConversationStateError):
        service.set_turn_context(
            turn_id="turn-1",
            references=refs,
            updated_at=TIMES[2],
        )


def test_completion_rejects_mismatched_response_content(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Question",
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
    assistant = message(
        message_id="assistant-1",
        turn_id="turn-1",
        role="assistant",
        content="Visible answer",
        created_at=TIMES[4],
    )
    with pytest.raises(ConversationStateError):
        service.complete_turn(
            turn_id="turn-1",
            request_id="request-1",
            response=response(
                request_id="request-1",
                content="Different answer",
                created_at=TIMES[4],
            ),
            assistant_message=assistant,
            completed_at=TIMES[4],
        )


def test_completion_rejects_cancelled_finish_reason(tmp_path: Path) -> None:
    _, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    user = message(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Question",
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
    assistant = message(
        message_id="assistant-1",
        turn_id="turn-1",
        role="assistant",
        content="Cancelled output",
        created_at=TIMES[4],
    )
    cancelled = response(
        request_id="request-1",
        content=assistant.content,
        created_at=TIMES[4],
    )
    cancelled = type(cancelled)(
        request_id=cancelled.request_id,
        provider=cancelled.provider,
        model=cancelled.model,
        content=cancelled.content,
        finish_reason="cancelled",
        created_at=cancelled.created_at,
    )
    with pytest.raises(ConversationStateError):
        service.complete_turn(
            turn_id="turn-1",
            request_id="request-1",
            response=cancelled,
            assistant_message=assistant,
            completed_at=TIMES[4],
        )
