"""P2.1 authoritative-store tests for the A.L.I.C.E. Memory Core."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alice_memory.migrations import (
    FutureSchemaVersionError,
    UnsupportedSchemaVersionError,
    ensure_current_schema,
)
from alice_memory.schema import SCHEMA_VERSION, current_schema_version
from alice_memory.store import (
    MEMORY_DATABASE_RELATIVE_PATH,
    MemoryStoreError,
    UnsafeMemoryStorePathError,
    memory_database_path,
    open_memory_store,
    transaction,
    validate_private_database_path,
)


def test_memory_database_path_uses_private_phase2_layout(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    database = memory_database_path(
        vault,
        repository_root=repository,
    )

    assert database == (
        vault.resolve() / MEMORY_DATABASE_RELATIVE_PATH
    )


def test_repository_local_database_path_is_rejected(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    database = repository / "memory" / "phase2" / "memory-core.sqlite3"

    with pytest.raises(UnsafeMemoryStorePathError):
        validate_private_database_path(
            database,
            repository_root=repository,
        )


def test_repository_local_vault_is_rejected(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = repository / "vault"
    repository.mkdir()
    vault.mkdir()

    with pytest.raises(UnsafeMemoryStorePathError):
        memory_database_path(
            vault,
            repository_root=repository,
        )


def test_vault_root_must_already_exist(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    missing_vault = tmp_path / "missing-vault"

    with pytest.raises(FileNotFoundError):
        memory_database_path(
            missing_vault,
            repository_root=repository,
        )


def test_open_memory_store_creates_database_and_schema(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    expected_database = (
        vault.resolve() / MEMORY_DATABASE_RELATIVE_PATH
    )

    assert not expected_database.exists()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        assert expected_database.is_file()
        assert current_schema_version(connection) == SCHEMA_VERSION

    assert expected_database.is_file()


def test_open_memory_store_applies_required_pragmas(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        assert connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0] == 1
        assert str(
            connection.execute(
                "PRAGMA journal_mode"
            ).fetchone()[0]
        ).lower() == "wal"
        assert connection.execute(
            "PRAGMA busy_timeout"
        ).fetchone()[0] == 5000


def test_open_memory_store_uses_row_factory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        row = connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchone()

        assert isinstance(row, sqlite3.Row)
        assert row["version"] == SCHEMA_VERSION


def test_open_memory_store_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as first:
        assert current_schema_version(first) == SCHEMA_VERSION

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as second:
        rows = second.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()

        assert [row["version"] for row in rows] == [SCHEMA_VERSION]


def test_connection_is_closed_after_context_exit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        connection.execute("SELECT 1").fetchone()

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1").fetchone()


def test_transaction_commits_on_success(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            connection.execute(
                """
                INSERT INTO memory_events (
                    event_id,
                    memory_id,
                    event_type,
                    actor,
                    details_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "event-commit",
                    None,
                    "created",
                    "test",
                    None,
                    "2026-07-21T00:00:00Z",
                ),
            )

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_events
            WHERE event_id = ?
            """,
            ("event-commit",),
        ).fetchone()[0]

        assert count == 1


def test_transaction_rolls_back_on_failure(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with pytest.raises(RuntimeError):
            with transaction(connection):
                connection.execute(
                    """
                    INSERT INTO memory_events (
                        event_id,
                        memory_id,
                        event_type,
                        actor,
                        details_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "event-rollback",
                        None,
                        "created",
                        "test",
                        None,
                        "2026-07-21T00:00:00Z",
                    ),
                )
                raise RuntimeError("force rollback")

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_events
            WHERE event_id = ?
            """,
            ("event-rollback",),
        ).fetchone()[0]

        assert count == 0


def test_nested_transactions_are_rejected(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            with pytest.raises(MemoryStoreError):
                with transaction(connection):
                    pass


def test_future_schema_version_fails_closed() -> None:
    connection = sqlite3.connect(
        ":memory:",
        isolation_level=None,
    )
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO schema_migrations (version, applied_at)
        VALUES (?, ?)
        """,
        (
            SCHEMA_VERSION + 1,
            "2026-07-21T00:00:00Z",
        ),
    )

    with pytest.raises(FutureSchemaVersionError):
        ensure_current_schema(connection)


def test_older_schema_version_requires_explicit_migration() -> None:
    connection = sqlite3.connect(
        ":memory:",
        isolation_level=None,
    )
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO schema_migrations (version, applied_at)
        VALUES (?, ?)
        """,
        (
            SCHEMA_VERSION - 1,
            "2026-07-21T00:00:00Z",
        ),
    )

    with pytest.raises(UnsupportedSchemaVersionError):
        ensure_current_schema(connection)
