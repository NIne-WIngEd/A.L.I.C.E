"""Explicit schema migrations for the A.L.I.C.E. Phase 2 Memory Core."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .schema import (
    SCHEMA_V2_DDL_STATEMENTS,
    SCHEMA_VERSION,
    configure_connection,
    current_schema_version,
    initialize_schema,
)


class MemorySchemaError(RuntimeError):
    """Base error for Memory Core schema validation."""


class FutureSchemaVersionError(MemorySchemaError):
    """Raised when a database was created by newer unsupported code."""


class UnsupportedSchemaVersionError(MemorySchemaError):
    """Raised when an older database cannot be migrated safely."""


class UnsafeSensitivePlaintextMigrationError(UnsupportedSchemaVersionError):
    """Raised when a v1 database contains unprotected HIGHLY_SENSITIVE rows."""


class MemoryDatabaseIntegrityError(MemorySchemaError):
    """Raised when SQLite reports database-integrity problems."""


_V1_REQUIRED_TABLES = {
    "schema_migrations",
    "memories",
    "memory_sources",
    "memory_relations",
    "memory_derivations",
    "memory_entities",
    "memory_events",
    "memory_tombstones",
}


def utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_database_integrity(connection: sqlite3.Connection) -> None:
    """Fail closed unless SQLite reports a clean integrity check."""
    row = connection.execute("PRAGMA integrity_check").fetchone()
    result = None if row is None else str(row[0])

    if result != "ok":
        raise MemoryDatabaseIntegrityError(
            f"Memory database integrity check failed: {result!r}"
        )


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def migrate_v1_to_v2(
    connection: sqlite3.Connection,
    *,
    applied_at: str,
) -> None:
    """Add protected sensitive-payload storage to a valid schema-v1 database.

    Version 1 normally contains no HIGHLY_SENSITIVE rows because the service
    blocked their creation. If direct SQL or legacy code inserted such rows,
    migration fails closed rather than preserving sensitive plaintext.
    """
    if current_schema_version(connection) != 1:
        raise UnsupportedSchemaVersionError(
            "The v1-to-v2 migration requires a schema version 1 database."
        )

    missing = _V1_REQUIRED_TABLES - _table_names(connection)
    if missing:
        raise UnsupportedSchemaVersionError(
            "Schema version 1 database is missing required tables: "
            + ", ".join(sorted(missing))
        )

    sensitive_count = connection.execute(
        """
        SELECT COUNT(*)
        FROM memories
        WHERE data_classification = 'HIGHLY_SENSITIVE'
        """
    ).fetchone()[0]
    if int(sensitive_count) != 0:
        raise UnsafeSensitivePlaintextMigrationError(
            "Refusing automatic v1-to-v2 migration because the database "
            "contains HIGHLY_SENSITIVE rows that may be stored as plaintext."
        )

    try:
        connection.execute("BEGIN")
        for statement in SCHEMA_V2_DDL_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO schema_migrations (version, applied_at)
            VALUES (?, ?)
            """,
            (2, applied_at),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def ensure_current_schema(
    connection: sqlite3.Connection,
    *,
    applied_at: str | None = None,
) -> int:
    """Initialize a fresh database or apply supported ordered migrations."""
    configure_connection(connection)
    version = current_schema_version(connection)
    timestamp = applied_at or utc_now()

    if version is None:
        initialize_schema(connection, applied_at=timestamp)
        version = current_schema_version(connection)

    if version is None:
        raise MemorySchemaError(
            "Memory schema initialization completed without a schema version."
        )

    if version > SCHEMA_VERSION:
        raise FutureSchemaVersionError(
            "Memory database schema version "
            f"{version} is newer than supported version {SCHEMA_VERSION}."
        )

    while version < SCHEMA_VERSION:
        if version == 1:
            migrate_v1_to_v2(connection, applied_at=timestamp)
        else:
            raise UnsupportedSchemaVersionError(
                "Memory database schema version "
                f"{version} has no supported migration path to version "
                f"{SCHEMA_VERSION}."
            )
        version = current_schema_version(connection)
        if version is None:
            raise MemorySchemaError(
                "Memory migration completed without a schema version."
            )

    verify_database_integrity(connection)
    return version
