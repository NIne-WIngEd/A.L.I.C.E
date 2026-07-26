from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_conversation.state_policy import (
    ConversationStatePolicyError,
    load_conversation_state_policy,
    parse_conversation_state_policy,
    resolve_conversation_state_database_path,
)


def _payload() -> dict:
    path = Path("policies/conversation_state_policy.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_state_policy_preserves_p32_boundaries() -> None:
    policy = load_conversation_state_policy()
    assert policy.phase == "3"
    assert policy.milestone == "P3.2"
    assert policy.default_retention == "session_only"
    assert policy.allowed_retentions == ("session_only", "retained")
    assert policy.repository_storage_allowed is False
    assert policy.private_output_only is True
    assert policy.chain_of_thought_persistence_allowed is False
    assert policy.memory_write_allowed is False
    assert policy.external_action_allowed is False
    assert policy.web_access_allowed is False
    assert policy.tool_calling_allowed is False


def test_policy_rejects_repository_storage() -> None:
    payload = _payload()
    payload["storage"]["repository_storage_allowed"] = True
    with pytest.raises(ConversationStatePolicyError):
        parse_conversation_state_policy(payload)


def test_policy_rejects_chain_of_thought_persistence() -> None:
    payload = _payload()
    payload["boundaries"]["chain_of_thought_persistence_allowed"] = True
    with pytest.raises(ConversationStatePolicyError):
        parse_conversation_state_policy(payload)


def test_policy_rejects_highly_sensitive_state() -> None:
    payload = _payload()
    payload["boundaries"]["highly_sensitive_allowed"] = True
    with pytest.raises(ConversationStatePolicyError):
        parse_conversation_state_policy(payload)


def test_policy_rejects_parent_traversal_database_path() -> None:
    payload = _payload()
    payload["storage"]["database_relative_path"] = "../alice.sqlite3"
    with pytest.raises(ConversationStatePolicyError):
        parse_conversation_state_policy(payload)


def test_database_path_resolves_under_vault_outside_repo(tmp_path: Path) -> None:
    policy = load_conversation_state_policy()
    vault = tmp_path / "vault"
    repository = tmp_path / "repository"
    database = resolve_conversation_state_database_path(
        policy=policy,
        vault_root=vault,
        repository_root=repository,
    )
    assert database == (vault / "conversation/alice-conversation.sqlite3").resolve()


def test_database_path_rejects_vault_inside_repository(tmp_path: Path) -> None:
    policy = load_conversation_state_policy()
    repository = tmp_path / "repository"
    vault = repository / "private"
    with pytest.raises(ConversationStatePolicyError):
        resolve_conversation_state_database_path(
            policy=policy,
            vault_root=vault,
            repository_root=repository,
        )
