"""P2.0 schema-contract tests for the A.L.I.C.E. Memory Core."""

from __future__ import annotations

import sqlite3

import pytest

from alice_memory.schema import (
    DATA_CLASSIFICATIONS,
    KNOWLEDGE_STATUSES,
    MEMORY_CATEGORIES,
    MEMORY_STORABLE_CLASSIFICATIONS,
    SCHEMA_VERSION,
    configure_connection,
    current_schema_version,
    initialize_schema,
)

REQUIRED_TABLES = {
    "schema_migrations",
    "memories",
    "memory_sources",
    "memory_relations",
    "memory_derivations",
    "memory_entities",
    "memory_events",
    "memory_tombstones",
}


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    configure_connection(connection)
    return connection


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def test_schema_initializes_required_tables_and_version() -> None:
    connection = _connection()

    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:00Z",
    )

    assert REQUIRED_TABLES.issubset(_table_names(connection))
    assert current_schema_version(connection) == SCHEMA_VERSION


def test_schema_initialization_is_idempotent() -> None:
    connection = _connection()

    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:00Z",
    )
    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:01Z",
    )

    rows = connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()

    assert rows == [(SCHEMA_VERSION,)]


def test_foreign_keys_are_enabled() -> None:
    connection = _connection()

    enabled = connection.execute("PRAGMA foreign_keys").fetchone()

    assert enabled == (1,)


def test_memory_source_cannot_reference_missing_memory() -> None:
    connection = _connection()
    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:00Z",
    )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO memory_sources (
                memory_source_id,
                memory_id,
                source_type,
                source_ref,
                support_relation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "source-1",
                "missing-memory",
                "phase1_chunk",
                "phase1:test",
                "supports",
                "2026-07-21T00:00:00Z",
            ),
        )


@pytest.mark.parametrize("category", MEMORY_CATEGORIES)
def test_all_policy_memory_categories_are_schema_valid(category: str) -> None:
    assert category in MEMORY_CATEGORIES


@pytest.mark.parametrize("knowledge_status", KNOWLEDGE_STATUSES)
def test_all_policy_knowledge_statuses_are_schema_valid(
    knowledge_status: str,
) -> None:
    assert knowledge_status in KNOWLEDGE_STATUSES


def test_secrets_class_is_recognized_but_not_storable_as_memory() -> None:
    assert "SECRETS" in DATA_CLASSIFICATIONS
    assert "SECRETS" not in MEMORY_STORABLE_CLASSIFICATIONS


def test_secrets_memory_insert_is_rejected_by_database() -> None:
    connection = _connection()
    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:00Z",
    )

    with pytest.raises(sqlite3.IntegrityError):
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
                "secret-memory-1",
                SCHEMA_VERSION,
                "This must never be stored as ordinary memory.",
                "b" * 64,
                "profile",
                "rayan_statement",
                1.0,
                "SECRETS",
                "2026-07-21T00:00:00Z",
                1,
                "current",
                "durable",
                "active",
                "2026-07-21T00:00:00Z",
                "2026-07-21T00:00:00Z",
            ),
        )


def test_invalid_category_is_rejected_by_database() -> None:
    connection = _connection()
    initialize_schema(
        connection,
        applied_at="2026-07-21T00:00:00Z",
    )

    with pytest.raises(sqlite3.IntegrityError):
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
                "memory-1",
                SCHEMA_VERSION,
                "Test memory",
                "a" * 64,
                "invalid-category",
                "rayan_statement",
                1.0,
                "PRIVATE",
                "2026-07-21T00:00:00Z",
                1,
                "current",
                "durable",
                "active",
                "2026-07-21T00:00:00Z",
                "2026-07-21T00:00:00Z",
            ),
        )
