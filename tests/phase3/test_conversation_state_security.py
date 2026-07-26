from __future__ import annotations

from pathlib import Path

import pytest

from alice_conversation.state_inspection import (
    inspect_conversation_session,
    inspect_conversation_tombstone,
    verify_conversation_session_integrity,
)
from alice_conversation.contracts import ConversationContractError
from alice_conversation.state_service import ConversationStateError

from _state_helpers import TIMES, make_store, message


def _terminal_session(tmp_path: Path, *, retention: str):
    store, service, _, _ = make_store(tmp_path)
    service.create_session(
        session_id="session-1",
        created_at=TIMES[0],
        retention=retention,
    )
    user = message(
        message_id="user-1",
        turn_id="turn-1",
        role="user",
        content="Temporary content",
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
    return store, service


def test_session_only_close_physically_purges_content(tmp_path: Path) -> None:
    store, service = _terminal_session(tmp_path, retention="session_only")
    tombstone = service.close_session(
        session_id="session-1",
        closed_at=TIMES[3],
    )
    assert tombstone is not None
    assert tombstone.turn_count == 1
    assert tombstone.message_count == 1
    with pytest.raises(ConversationStateError):
        inspect_conversation_session(store, session_id="session-1")
    inspected = inspect_conversation_tombstone(store, session_id="session-1")
    assert inspected == tombstone
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_sessions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_messages"
        ).fetchone()[0] == 0
        row = connection.execute(
            "SELECT * FROM conversation_session_tombstones WHERE session_id = ?",
            ("session-1",),
        ).fetchone()
        assert "Temporary content" not in repr(dict(row))
    assert service.close_session(
        session_id="session-1", closed_at=TIMES[3]
    ) == tombstone


def test_retained_session_can_be_explicitly_deleted(tmp_path: Path) -> None:
    store, service = _terminal_session(tmp_path, retention="retained")
    service.close_session(session_id="session-1", closed_at=TIMES[3])
    tombstone = service.delete_session(
        session_id="session-1",
        deleted_at=TIMES[4],
    )
    assert tombstone.retention == "retained"
    with pytest.raises(ConversationStateError):
        inspect_conversation_session(store, session_id="session-1")
    assert service.delete_session(
        session_id="session-1", deleted_at=TIMES[4]
    ) == tombstone


def test_tampered_message_digest_fails_integrity_check(tmp_path: Path) -> None:
    store, service = _terminal_session(tmp_path, retention="retained")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE conversation_messages SET content = 'tampered' WHERE message_id = 'user-1'"
        )
    report = verify_conversation_session_integrity(
        store,
        session_id="session-1",
    )
    assert report.valid is False
    assert any("Message digest mismatch" in error for error in report.errors)


def test_highly_sensitive_message_is_rejected_before_storage(tmp_path: Path) -> None:
    store, service, _, _ = make_store(tmp_path)
    service.create_session(session_id="session-1", created_at=TIMES[0])
    with pytest.raises(ConversationContractError):
        user = message(
            message_id="user-1",
            turn_id="turn-1",
            role="user",
            content="Sensitive",
            created_at=TIMES[1],
            classification="HIGHLY_SENSITIVE",
        )
        service.start_turn(
            session_id="session-1", turn_id="turn-1", user_message=user
        )
    with store.read_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversation_messages"
        ).fetchone()[0] == 0


def test_purged_session_id_cannot_be_reused(tmp_path: Path) -> None:
    _, service = _terminal_session(tmp_path, retention="session_only")
    service.close_session(session_id="session-1", closed_at=TIMES[3])
    with pytest.raises(ConversationStateError):
        service.create_session(session_id="session-1", created_at=TIMES[4])
