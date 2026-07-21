"""Schema-version enforcement for the A.L.I.C.E. Phase 2 Memory Core."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .schema import (
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
    """Raised when an older database requires an unavailable migration."""


class MemoryDatabaseIntegrityError(MemorySchemaError):
    """Raised when SQLite reports database-integrity problems."""


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


def ensure_current_schema(
    connection: sqlite3.Connection,
    *,
    applied_at: str | None = None,
) -> int:
    """Initialize a fresh database or validate an existing schema version.

    P2.1 supports only schema version 1. Future migrations must be explicit,
    ordered, tested, and versioned rather than silently rewriting the database.
    """
    configure_connection(connection)
    version = current_schema_version(connection)

    if version is None:
        initialize_schema(
            connection,
            applied_at=applied_at or utc_now(),
        )
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

    if version < SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            "Memory database schema version "
            f"{version} requires an explicit migration to version "
            f"{SCHEMA_VERSION}; no automatic migration is available."
        )

    verify_database_integrity(connection)
    return version
