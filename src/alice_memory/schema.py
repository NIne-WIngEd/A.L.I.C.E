"""Versioned SQLite schema contract for the A.L.I.C.E. Phase 2 Memory Core.

P2.0 intentionally defines and initializes an isolated schema only. It does not
create or open the live private memory database and does not implement memory
writes, retrieval, or access-control decisions.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

SCHEMA_VERSION = 1

MEMORY_CATEGORIES = (
    "working",
    "profile",
    "episodic",
    "project",
    "goal",
    "procedural",
    "relationship",
    "reflective",
)

KNOWLEDGE_STATUSES = (
    "verified_fact",
    "rayan_statement",
    "external_claim",
    "alice_inference",
    "estimate",
    "uncertain",
    "disputed",
    "historical",
    "superseded",
)

DATA_CLASSIFICATIONS = (
    "PUBLIC",
    "INTERNAL",
    "PRIVATE",
    "HIGHLY_SENSITIVE",
    "SECRETS",
)

MEMORY_STORABLE_CLASSIFICATIONS = tuple(
    classification
    for classification in DATA_CLASSIFICATIONS
    if classification != "SECRETS"
)

VALIDITY_STATES = (
    "current",
    "historical",
    "disputed",
    "unknown",
)

RETENTION_STATES = (
    "durable",
    "review_due",
    "archived",
)

DELETION_STATES = (
    "active",
    "pending_deletion",
)

RELATION_TYPES = (
    "supersedes",
    "conflicts_with",
    "supports",
    "duplicates",
    "derived_from",
    "corrects",
)

SOURCE_TYPES = (
    "phase1_chunk",
    "phase1_source",
    "rayan_direct_statement",
    "approved_manual_entry",
    "external_source",
    "alice_inference",
)

SUPPORT_RELATIONS = (
    "supports",
    "contradicts",
    "context",
    "derived_from",
)

DERIVATION_TYPES = (
    "explicit_user",
    "deterministic_import",
    "model_proposed",
    "human_confirmed",
    "derived_inference",
)

EVENT_TYPES = (
    "created",
    "inspected",
    "corrected",
    "superseded",
    "conflict_marked",
    "reclassified",
    "archived",
    "deletion_requested",
    "deleted",
    "index_rebuilt",
)


def _sql_values(values: Iterable[str]) -> str:
    """Return safely quoted SQL literals for static controlled vocabularies."""
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


DDL_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        memory_key TEXT,
        category TEXT NOT NULL CHECK (category IN ({_sql_values(MEMORY_CATEGORIES)})),
        knowledge_status TEXT NOT NULL
            CHECK (knowledge_status IN ({_sql_values(KNOWLEDGE_STATUSES)})),
        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        data_classification TEXT NOT NULL
            CHECK (
                data_classification IN (
                    {_sql_values(MEMORY_STORABLE_CLASSIFICATIONS)}
                )
            ),
        valid_from TEXT,
        valid_to TEXT,
        time_precision TEXT,
        recorded_at TEXT NOT NULL,
        verified_at TEXT,
        rayan_confirmed INTEGER NOT NULL DEFAULT 0
            CHECK (rayan_confirmed IN (0, 1)),
        validity_state TEXT NOT NULL
            CHECK (validity_state IN ({_sql_values(VALIDITY_STATES)})),
        retention_state TEXT NOT NULL
            CHECK (retention_state IN ({_sql_values(RETENTION_STATES)})),
        deletion_state TEXT NOT NULL DEFAULT 'active'
            CHECK (deletion_state IN ({_sql_values(DELETION_STATES)})),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (length(content_sha256) = 64),
        CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS memory_sources (
        memory_source_id TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        source_type TEXT NOT NULL CHECK (source_type IN ({_sql_values(SOURCE_TYPES)})),
        source_ref TEXT NOT NULL,
        source_content_sha256 TEXT,
        source_text_sha256 TEXT,
        chunk_id TEXT,
        file_id TEXT,
        source_date TEXT,
        support_relation TEXT NOT NULL
            CHECK (support_relation IN ({_sql_values(SUPPORT_RELATIONS)})),
        created_at TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        UNIQUE (memory_id, source_type, source_ref, chunk_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS memory_relations (
        relation_id TEXT PRIMARY KEY,
        from_memory_id TEXT NOT NULL,
        to_memory_id TEXT NOT NULL,
        relation_type TEXT NOT NULL
            CHECK (relation_type IN ({_sql_values(RELATION_TYPES)})),
        created_at TEXT NOT NULL,
        FOREIGN KEY (from_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        FOREIGN KEY (to_memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        CHECK (from_memory_id <> to_memory_id),
        UNIQUE (from_memory_id, to_memory_id, relation_type)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS memory_derivations (
        derivation_id TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        derivation_type TEXT NOT NULL
            CHECK (derivation_type IN ({_sql_values(DERIVATION_TYPES)})),
        policy_version TEXT,
        model TEXT,
        model_version TEXT,
        prompt_version TEXT,
        run_id TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_entities (
        memory_entity_id TEXT PRIMARY KEY,
        memory_id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        entity_value TEXT NOT NULL,
        normalized_value TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE,
        UNIQUE (memory_id, entity_type, entity_value)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS memory_events (
        event_id TEXT PRIMARY KEY,
        memory_id TEXT,
        event_type TEXT NOT NULL CHECK (event_type IN ({_sql_values(EVENT_TYPES)})),
        actor TEXT NOT NULL,
        details_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_tombstones (
        tombstone_id TEXT PRIMARY KEY,
        deleted_memory_id TEXT NOT NULL UNIQUE,
        content_sha256 TEXT NOT NULL,
        deleted_at TEXT NOT NULL,
        deletion_scope TEXT NOT NULL,
        event_id TEXT,
        CHECK (length(content_sha256) = 64),
        FOREIGN KEY (event_id) REFERENCES memory_events(event_id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_knowledge_status ON memories(knowledge_status)",
    "CREATE INDEX IF NOT EXISTS idx_memories_classification ON memories(data_classification)",
    "CREATE INDEX IF NOT EXISTS idx_memories_validity_state ON memories(validity_state)",
    "CREATE INDEX IF NOT EXISTS idx_memories_memory_key ON memories(memory_key)",
    "CREATE INDEX IF NOT EXISTS idx_memories_valid_range ON memories(valid_from, valid_to)",
    "CREATE INDEX IF NOT EXISTS idx_memory_sources_memory_id ON memory_sources(memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_relations_from ON memory_relations(from_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_relations_to ON memory_relations(to_memory_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_entities_lookup "
    "ON memory_entities(entity_type, normalized_value)",
)


def configure_connection(connection: sqlite3.Connection) -> None:
    """Enable deterministic safety settings required by the schema contract."""
    connection.execute("PRAGMA foreign_keys = ON")


def initialize_schema(
    connection: sqlite3.Connection,
    *,
    applied_at: str,
) -> None:
    """Initialize schema version 1 transactionally on an existing connection."""
    configure_connection(connection)

    try:
        connection.execute("BEGIN")
        for statement in DDL_STATEMENTS:
            connection.execute(statement)

        connection.execute(
            """
            INSERT OR IGNORE INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (SCHEMA_VERSION, applied_at),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def current_schema_version(connection: sqlite3.Connection) -> int | None:
    """Return the highest initialized schema version, or None if uninitialized."""
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'schema_migrations'
        """
    ).fetchone()
    if row is None:
        return None

    version_row = connection.execute(
        "SELECT MAX(version) FROM schema_migrations"
    ).fetchone()
    if version_row is None or version_row[0] is None:
        return None

    return int(version_row[0])
