"""Private SQLite store and migration runner for A.L.I.C.E. P3.2."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .state_policy import (
    ConversationStatePolicy,
    ConversationStatePolicyError,
    load_conversation_state_policy,
    resolve_conversation_state_database_path,
)
from .state_schema import MIGRATION_1_SQL, REQUIRED_TABLES, SCHEMA_VERSION


class ConversationStateStoreError(RuntimeError):
    """Raised when private conversation-state storage fails closed."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sql_statements(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    pending = ""
    for line in script.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                statements.append(statement)
            pending = ""
    if pending.strip():
        raise ConversationStateStoreError("Conversation schema SQL is incomplete.")
    return tuple(statements)


@dataclass(frozen=True)
class ConversationStateStore:
    database_path: Path
    allowed_root: Path
    repository_root: Path
    policy: ConversationStatePolicy

    @classmethod
    def for_vault(
        cls,
        *,
        vault_root: str | Path,
        repository_root: str | Path,
        policy: ConversationStatePolicy | None = None,
    ) -> "ConversationStateStore":
        selected_policy = policy or load_conversation_state_policy()
        database_path = resolve_conversation_state_database_path(
            policy=selected_policy,
            vault_root=vault_root,
            repository_root=repository_root,
        )
        return cls.from_paths(
            database_path=database_path,
            allowed_root=vault_root,
            repository_root=repository_root,
            policy=selected_policy,
        )

    @classmethod
    def from_paths(
        cls,
        *,
        database_path: str | Path,
        allowed_root: str | Path,
        repository_root: str | Path,
        policy: ConversationStatePolicy,
    ) -> "ConversationStateStore":
        database = Path(database_path).expanduser().resolve(strict=False)
        allowed = Path(allowed_root).expanduser().resolve(strict=False)
        repository = Path(repository_root).expanduser().resolve(strict=False)
        if _is_within(allowed, repository):
            raise ConversationStatePolicyError(
                "The approved private state root cannot be inside the repository."
            )
        if not _is_within(database, allowed):
            raise ConversationStatePolicyError(
                "Conversation-state database must remain under its approved private root."
            )
        if _is_within(database, repository):
            raise ConversationStatePolicyError(
                "Conversation-state database cannot be stored in the repository."
            )
        return cls(
            database_path=database,
            allowed_root=allowed,
            repository_root=repository,
            policy=policy,
        )

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute(f"PRAGMA journal_mode = {self.policy.journal_mode}")
        connection.execute(f"PRAGMA synchronous = {self.policy.synchronous}")
        foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
        if foreign_keys != 1:
            connection.close()
            raise ConversationStateStoreError(
                "SQLite foreign-key enforcement could not be enabled."
            )
        return connection

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT MAX(version) AS version FROM conversation_schema_migrations"
            ).fetchone()
            current = int(row["version"] or 0)
            if current > SCHEMA_VERSION:
                raise ConversationStateStoreError(
                    "Conversation-state database uses a newer unsupported schema."
                )
            if current == 0:
                for statement in _sql_statements(MIGRATION_1_SQL):
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO conversation_schema_migrations(version, applied_at)
                    VALUES (?, ?)
                    """,
                    (SCHEMA_VERSION, "2026-07-25T00:00:00Z"),
                )
            elif current != SCHEMA_VERSION:
                raise ConversationStateStoreError(
                    "No approved migration path exists for this state database."
                )
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        self.verify_schema()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM conversation_schema_migrations"
            ).fetchone()
            return int(row["version"] or 0)

    def integrity_check(self) -> tuple[str, ...]:
        with self.read_connection() as connection:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
            return tuple(str(row[0]) for row in rows)

    def verify_schema(self) -> None:
        with self.read_connection() as connection:
            version_row = connection.execute(
                "SELECT MAX(version) AS version FROM conversation_schema_migrations"
            ).fetchone()
            version = int(version_row["version"] or 0)
            if version != SCHEMA_VERSION:
                raise ConversationStateStoreError(
                    f"Expected conversation schema {SCHEMA_VERSION}; found {version}."
                )
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            tables = tuple(str(row["name"]) for row in rows)
            if set(tables) != set(REQUIRED_TABLES):
                raise ConversationStateStoreError(
                    "Conversation-state database table set does not match policy."
                )
            forbidden_columns = {
                "chain_of_thought",
                "chain_of_thought_content",
                "reasoning_content",
                "hidden_reasoning",
                "scratchpad",
            }
            for table in tables:
                columns = connection.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
                names = {str(column["name"]) for column in columns}
                if names & forbidden_columns:
                    raise ConversationStateStoreError(
                        "Conversation schema contains prohibited reasoning storage."
                    )
            check_rows = connection.execute("PRAGMA integrity_check").fetchall()
            checks = tuple(str(row[0]) for row in check_rows)
            if checks != ("ok",):
                raise ConversationStateStoreError(
                    "SQLite integrity check failed: " + "; ".join(checks)
                )
