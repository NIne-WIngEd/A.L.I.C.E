"""P2.7a schema-v3 candidate-staging migration tests."""

from __future__ import annotations

import sqlite3

import pytest

from alice_memory.migrations import (
    UnsupportedSchemaVersionError,
    ensure_current_schema,
)
from alice_memory.schema import (
    SCHEMA_V1_DDL_STATEMENTS,
    SCHEMA_V2_DDL_STATEMENTS,
    SCHEMA_VERSION,
    configure_connection,
    current_schema_version,
)


def _v2_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    configure_connection(connection)
    for statement in SCHEMA_V1_DDL_STATEMENTS:
        connection.execute(statement)
    for statement in SCHEMA_V2_DDL_STATEMENTS:
        connection.execute(statement)
    connection.execute(
        """
        INSERT INTO schema_migrations (version, applied_at)
        VALUES (1, '2026-07-21T00:00:00Z')
        """
    )
    connection.execute(
        """
        INSERT INTO schema_migrations (version, applied_at)
        VALUES (2, '2026-07-22T00:00:00Z')
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


def test_v2_database_migrates_to_candidate_staging_schema() -> None:
    connection = _v2_connection()

    version = ensure_current_schema(
        connection,
        applied_at="2026-07-24T00:00:00Z",
    )

    assert version == SCHEMA_VERSION == 3
    assert current_schema_version(connection) == 3
    assert {
        "memory_candidates",
        "memory_candidate_sources",
        "memory_candidate_events",
    }.issubset(_table_names(connection))

    rows = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    assert rows == [(1,), (2,), (3,)]


def test_v2_migration_preserves_existing_authoritative_memory() -> None:
    connection = _v2_connection()
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
            "existing-memory",
            2,
            "Existing authoritative memory",
            "a" * 64,
            "project",
            "verified_fact",
            1.0,
            "PRIVATE",
            "2026-07-22T00:00:00Z",
            1,
            "current",
            "durable",
            "active",
            "2026-07-22T00:00:00Z",
            "2026-07-22T00:00:00Z",
        ),
    )

    ensure_current_schema(
        connection,
        applied_at="2026-07-24T00:00:00Z",
    )

    row = connection.execute(
        "SELECT content FROM memories WHERE memory_id = ?",
        ("existing-memory",),
    ).fetchone()
    assert row == ("Existing authoritative memory",)


def test_v2_migration_fails_if_required_sensitive_table_is_missing() -> None:
    connection = _v2_connection()
    connection.execute("DROP TABLE sensitive_memory_access_events")

    with pytest.raises(UnsupportedSchemaVersionError):
        ensure_current_schema(
            connection,
            applied_at="2026-07-24T00:00:00Z",
        )

    assert current_schema_version(connection) == 2
    assert "memory_candidates" not in _table_names(connection)
