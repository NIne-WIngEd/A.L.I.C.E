from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from alice_conversation.state_policy import (
    ConversationStatePolicyError,
    load_conversation_state_policy,
)
from alice_conversation.state_schema import REQUIRED_TABLES, SCHEMA_VERSION
from alice_conversation.state_store import (
    ConversationStateStore,
    ConversationStateStoreError,
)

from _state_helpers import make_store


def test_store_initializes_versioned_schema(tmp_path: Path) -> None:
    store, _, _, _ = make_store(tmp_path)
    assert store.schema_version() == SCHEMA_VERSION
    assert store.integrity_check() == ("ok",)
    with store.read_connection() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            if not str(row[0]).startswith("sqlite_")
        }
        assert tables == set(REQUIRED_TABLES)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_store_initialization_is_idempotent(tmp_path: Path) -> None:
    store, _, _, _ = make_store(tmp_path)
    store.initialize()
    assert store.schema_version() == SCHEMA_VERSION
    assert store.integrity_check() == ("ok",)


def test_store_rejects_database_outside_private_root(tmp_path: Path) -> None:
    policy = load_conversation_state_policy()
    vault = tmp_path / "vault"
    repository = tmp_path / "repository"
    outside = tmp_path / "outside.sqlite3"
    with pytest.raises(ConversationStatePolicyError):
        ConversationStateStore.from_paths(
            database_path=outside,
            allowed_root=vault,
            repository_root=repository,
            policy=policy,
        )


def test_store_rejects_private_root_inside_repository(tmp_path: Path) -> None:
    policy = load_conversation_state_policy()
    repository = tmp_path / "repository"
    vault = repository / "vault"
    with pytest.raises(ConversationStatePolicyError):
        ConversationStateStore.from_paths(
            database_path=vault / "conversation.sqlite3",
            allowed_root=vault,
            repository_root=repository,
            policy=policy,
        )


def test_store_rejects_newer_schema_version(tmp_path: Path) -> None:
    store, _, _, _ = make_store(tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE conversation_schema_migrations SET version = 99"
        )
    with pytest.raises(ConversationStateStoreError):
        store.initialize()


def test_schema_has_no_reasoning_content_columns(tmp_path: Path) -> None:
    store, _, _, _ = make_store(tmp_path)
    forbidden = {
        "chain_of_thought",
        "chain_of_thought_content",
        "reasoning_content",
        "hidden_reasoning",
        "scratchpad",
    }
    with store.read_connection() as connection:
        for table in REQUIRED_TABLES:
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            assert not columns & forbidden
