from __future__ import annotations

import sqlite3
from pathlib import Path

from alice_conversation.state_policy import load_conversation_state_policy
from alice_conversation.state_schema import (
    MIGRATION_1_SQL,
    REFERENCE_KINDS,
    SCHEMA_VERSION,
)
from alice_conversation.state_service import ConversationStateReference
from alice_conversation.state_store import ConversationStateStore


TIMESTAMP = "2026-07-31T00:00:00Z"


def _store(tmp_path: Path, *, initialize: bool) -> ConversationStateStore:
    vault = tmp_path / "vault"
    repository = tmp_path / "repository"
    repository.mkdir(parents=True)
    store = ConversationStateStore.from_paths(
        database_path=vault / "conversation.sqlite3",
        allowed_root=vault,
        repository_root=repository,
        policy=load_conversation_state_policy(),
    )
    if initialize:
        store.initialize()
    return store


def _insert_session_and_turn(connection: sqlite3.Connection, *, suffix: str) -> str:
    session_id = f"session-{suffix}"
    turn_id = f"turn-{suffix}"
    connection.execute(
        """
        INSERT INTO conversation_sessions(
            session_id, status, retention, data_classification,
            created_at, updated_at, closed_at
        ) VALUES (?, 'active', 'session_only', 'PUBLIC', ?, ?, NULL)
        """,
        (session_id, TIMESTAMP, TIMESTAMP),
    )
    connection.execute(
        """
        INSERT INTO conversation_turns(
            turn_id, session_id, turn_index, status,
            grounding_packet_id, grounding_packet_sha256,
            interruption_count, created_at, updated_at,
            completed_at, failure_code
        ) VALUES (?, ?, 0, 'received', NULL, NULL, 0, ?, ?, NULL, NULL)
        """,
        (turn_id, session_id, TIMESTAMP, TIMESTAMP),
    )
    return turn_id


def _insert_reference(
    connection: sqlite3.Connection,
    *,
    turn_id: str,
    reference_id: str,
    source_kind: str,
    source_ref: str,
) -> None:
    connection.execute(
        """
        INSERT INTO conversation_turn_references(
            reference_id, turn_id, reference_index, source_kind,
            source_ref, citation_token, content_sha256,
            data_classification, created_at
        ) VALUES (?, ?, 0, ?, ?, '[WEB:S1]', NULL, 'PUBLIC', ?)
        """,
        (reference_id, turn_id, source_kind, source_ref, TIMESTAMP),
    )


def test_fresh_schema_accepts_validated_web_source_reference(tmp_path: Path) -> None:
    store = _store(tmp_path, initialize=True)
    assert SCHEMA_VERSION == 3
    assert "web_source" in REFERENCE_KINDS
    assert store.schema_version() == 3
    reference = ConversationStateReference(
        reference_id="web-reference-fresh",
        source_kind="web_source",
        source_ref="https://www.python.org/#alice-source-sha256=" + "a" * 64,
        citation_token="[WEB:S1]",
        content_sha256=None,
        data_classification="PUBLIC",
        created_at=TIMESTAMP,
    )
    reference.validate()
    with store.transaction() as connection:
        turn_id = _insert_session_and_turn(connection, suffix="fresh")
        _insert_reference(
            connection,
            turn_id=turn_id,
            reference_id=reference.reference_id,
            source_kind=reference.source_kind,
            source_ref=reference.source_ref,
        )
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT source_kind, source_ref FROM conversation_turn_references"
        ).fetchone()
    assert tuple(row) == ("web_source", reference.source_ref)
    assert store.integrity_check() == ("ok",)


def test_schema_v1_migrates_without_losing_existing_references(tmp_path: Path) -> None:
    store = _store(tmp_path, initialize=False)
    store.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE conversation_schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        connection.executescript(MIGRATION_1_SQL)
        connection.execute(
            """
            INSERT INTO conversation_schema_migrations(version, applied_at)
            VALUES (1, '2026-07-25T00:00:00Z')
            """
        )
        legacy_turn = _insert_session_and_turn(connection, suffix="legacy")
        _insert_reference(
            connection,
            turn_id=legacy_turn,
            reference_id="legacy-reference",
            source_kind="phase1_source",
            source_ref="phase1://source/S1",
        )
    store.initialize()
    assert store.schema_version() == 3
    with store.transaction() as connection:
        rows = connection.execute(
            """
            SELECT version FROM conversation_schema_migrations ORDER BY version
            """
        ).fetchall()
        assert [int(row[0]) for row in rows] == [1, 2, 3]
        legacy = connection.execute(
            """
            SELECT source_kind, source_ref
            FROM conversation_turn_references
            WHERE reference_id = 'legacy-reference'
            """
        ).fetchone()
        assert tuple(legacy) == ("phase1_source", "phase1://source/S1")
        migrated_turn = _insert_session_and_turn(connection, suffix="migrated")
        _insert_reference(
            connection,
            turn_id=migrated_turn,
            reference_id="web-reference-migrated",
            source_kind="web_source",
            source_ref="https://docs.python.org/#alice-source-sha256=" + "b" * 64,
        )
    assert store.integrity_check() == ("ok",)
