"""P2.6a schema-v2 migration tests."""

from __future__ import annotations

import sqlite3

import pytest

from alice_memory.migrations import (
    UnsafeSensitivePlaintextMigrationError,
    ensure_current_schema,
)
from alice_memory.schema import (
    SCHEMA_V1_DDL_STATEMENTS,
    SCHEMA_VERSION,
    configure_connection,
    current_schema_version,
)


def _v1_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    configure_connection(connection)
    for statement in SCHEMA_V1_DDL_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        """
        INSERT INTO schema_migrations (version, applied_at)
        VALUES (1, '2026-07-21T00:00:00Z')
        """
    )
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def test_v1_database_migrates_explicitly_to_v2() -> None:
    connection = _v1_connection()

    version = ensure_current_schema(
        connection,
        applied_at="2026-07-22T00:00:00Z",
    )

    assert version == SCHEMA_VERSION == 2
    assert current_schema_version(connection) == 2
    assert "memory_sensitive_payloads" in _table_names(connection)
    assert "sensitive_memory_access_events" in _table_names(connection)
    rows = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert rows == [(1,), (2,)]


def test_v1_migration_fails_closed_if_sensitive_plaintext_exists() -> None:
    connection = _v1_connection()
    connection.execute(
        """
        INSERT INTO memories (
            memory_id,
            schema_version,
            content,
            content_sha256,
            category,
            knowledge_status,
            confidence,
            data_classification,
            recorded_at,
            rayan_confirmed,
            validity_state,
            retention_state,
            deletion_state,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "legacy-sensitive",
            1,
            "legacy sensitive plaintext",
            "a" * 64,
            "episodic",
            "rayan_statement",
            1.0,
            "HIGHLY_SENSITIVE",
            "2026-07-21T00:00:00Z",
            1,
            "current",
            "durable",
            "active",
            "2026-07-21T00:00:00Z",
            "2026-07-21T00:00:00Z",
        ),
    )

    with pytest.raises(UnsafeSensitivePlaintextMigrationError):
        ensure_current_schema(
            connection,
            applied_at="2026-07-22T00:00:00Z",
        )

    assert current_schema_version(connection) == 1
    assert "memory_sensitive_payloads" not in _table_names(connection)
