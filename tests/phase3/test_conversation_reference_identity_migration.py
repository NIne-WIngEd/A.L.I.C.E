from __future__ import annotations

import sqlite3
from pathlib import Path

from alice_conversation.state_policy import load_conversation_state_policy
from alice_conversation.state_schema import (
    MIGRATION_1_SQL,
    MIGRATION_2_SQL,
    SCHEMA_VERSION,
)
from alice_conversation.state_store import ConversationStateStore


TIMESTAMP = "2026-07-31T00:00:00Z"
REFERENCE_ID = "shared-web-citation"


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


def _insert_session_turn(
    connection: sqlite3.Connection,
    *,
    suffix: str,
) -> str:
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
    source_ref: str,
) -> None:
    connection.execute(
        """
        INSERT INTO conversation_turn_references(
            reference_id, turn_id, reference_index, source_kind,
            source_ref, citation_token, content_sha256,
            data_classification, created_at
        ) VALUES (?, ?, 0, 'web_source', ?, '[WEB:S1]', NULL, 'PUBLIC', ?)
        """,
        (REFERENCE_ID, turn_id, source_ref, TIMESTAMP),
    )


def test_fresh_schema_scopes_reference_identity_to_turn(tmp_path: Path) -> None:
    store = _store(tmp_path, initialize=True)
    assert SCHEMA_VERSION == 3
    assert store.schema_version() == 3
    with store.transaction() as connection:
        first_turn = _insert_session_turn(connection, suffix="fresh-first")
        second_turn = _insert_session_turn(connection, suffix="fresh-second")
        _insert_reference(
            connection,
            turn_id=first_turn,
            source_ref="https://www.python.org/#first",
        )
        _insert_reference(
            connection,
            turn_id=second_turn,
            source_ref="https://www.python.org/#second",
        )
    with store.read_connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM conversation_turn_references
            WHERE reference_id = ?
            """,
            (REFERENCE_ID,),
        ).fetchone()[0]
        table = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table'
              AND name = 'conversation_turn_references'
            """
        ).fetchone()[0]
    assert int(count) == 2
    normalized = " ".join(str(table).split())
    assert "PRIMARY KEY (turn_id, reference_id)" in normalized
    assert store.integrity_check() == ("ok",)


def test_schema_v2_migrates_and_preserves_existing_reference(tmp_path: Path) -> None:
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
        connection.executescript(MIGRATION_2_SQL)
        connection.execute(
            """
            INSERT INTO conversation_schema_migrations(version, applied_at)
            VALUES (2, '2026-07-31T00:00:00Z')
            """
        )
        legacy_turn = _insert_session_turn(connection, suffix="legacy")
        _insert_reference(
            connection,
            turn_id=legacy_turn,
            source_ref="https://www.python.org/#legacy",
        )

    store.initialize()
    assert store.schema_version() == 3

    with store.transaction() as connection:
        migrations = connection.execute(
            """
            SELECT version
            FROM conversation_schema_migrations
            ORDER BY version
            """
        ).fetchall()
        assert [int(row[0]) for row in migrations] == [1, 2, 3]
        existing = connection.execute(
            """
            SELECT turn_id, source_ref
            FROM conversation_turn_references
            WHERE reference_id = ?
            """,
            (REFERENCE_ID,),
        ).fetchall()
        assert [tuple(row) for row in existing] == [
            (legacy_turn, "https://www.python.org/#legacy")
        ]
        new_turn = _insert_session_turn(connection, suffix="post-migration")
        _insert_reference(
            connection,
            turn_id=new_turn,
            source_ref="https://www.python.org/#post-migration",
        )

    with store.read_connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM conversation_turn_references
            WHERE reference_id = ?
            """,
            (REFERENCE_ID,),
        ).fetchone()[0]
    assert int(count) == 2
    assert store.integrity_check() == ("ok",)
