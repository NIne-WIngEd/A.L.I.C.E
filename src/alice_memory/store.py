"""Private authoritative SQLite store for the A.L.I.C.E. Memory Core.

P2.1 establishes database path safety, connection configuration, schema
initialization, transaction handling, and integrity validation. It intentionally
does not implement memory formation, automatic memory writes, model calls,
vector indexes, or Phase 1 mutation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .migrations import ensure_current_schema
from .schema import configure_connection

MEMORY_DATABASE_RELATIVE_PATH = Path(
    "memory",
    "phase2",
    "memory-core.sqlite3",
)


class MemoryStoreError(RuntimeError):
    """Base error for Memory Core store operations."""


class UnsafeMemoryStorePathError(MemoryStoreError):
    """Raised when a live private database path is inside the public repo."""


class MemoryStoreConfigurationError(MemoryStoreError):
    """Raised when required SQLite runtime configuration cannot be applied."""


def default_repository_root() -> Path:
    """Return the repository/package root inferred from this source file."""
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    """Return True when path is root itself or is contained by root."""
    return path == root or root in path.parents


def validate_private_database_path(
    database_path: Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Resolve and validate that a live database is outside the public repo."""
    resolved_database = database_path.expanduser().resolve(strict=False)
    resolved_repository = (
        repository_root or default_repository_root()
    ).expanduser().resolve(strict=True)

    if _is_within(resolved_database, resolved_repository):
        raise UnsafeMemoryStorePathError(
            "Refusing to create or open the private Memory Core database "
            f"inside the public repository: {resolved_database}"
        )

    return resolved_database


def memory_database_path(
    vault_root: Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Return the canonical private Memory Core database path for a vault."""
    resolved_vault = vault_root.expanduser().resolve(strict=True)
    candidate = resolved_vault / MEMORY_DATABASE_RELATIVE_PATH

    return validate_private_database_path(
        candidate,
        repository_root=repository_root,
    )


def _configure_mutable_store(connection: sqlite3.Connection) -> None:
    """Apply SQLite settings used for the mutable authoritative store."""
    configure_connection(connection)
    connection.row_factory = sqlite3.Row

    journal_mode_row = connection.execute(
        "PRAGMA journal_mode=WAL"
    ).fetchone()
    journal_mode = (
        None
        if journal_mode_row is None
        else str(journal_mode_row[0]).lower()
    )
    if journal_mode != "wal":
        raise MemoryStoreConfigurationError(
            "Memory Core requires SQLite WAL journal mode; "
            f"received {journal_mode!r}."
        )

    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")


@contextmanager
def open_memory_store(
    vault_root: Path,
    *,
    repository_root: Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open the private authoritative Memory Core database safely.

    The vault root must already exist. The Memory Core subdirectory may be
    created. The database is refused if it would reside inside the repository.
    """
    database = memory_database_path(
        vault_root,
        repository_root=repository_root,
    )
    database.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        database,
        isolation_level=None,
    )

    try:
        _configure_mutable_store(connection)
        ensure_current_schema(connection)
        yield connection
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


@contextmanager
def transaction(
    connection: sqlite3.Connection,
) -> Iterator[sqlite3.Connection]:
    """Run an explicit atomic write transaction with rollback on failure."""
    if connection.in_transaction:
        raise MemoryStoreError(
            "Nested Memory Core transactions are not supported."
        )

    connection.execute("BEGIN IMMEDIATE")

    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()
