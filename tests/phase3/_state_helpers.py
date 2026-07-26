from __future__ import annotations

from itertools import count
from pathlib import Path

from alice_conversation.contracts import (
    ConversationMessage,
    ModelResponse,
    sha256_text,
)
from alice_conversation.state_policy import load_conversation_state_policy
from alice_conversation.state_service import ConversationStateService
from alice_conversation.state_store import ConversationStateStore

TIMES = [
    "2026-07-25T20:00:00Z",
    "2026-07-25T20:00:01Z",
    "2026-07-25T20:00:02Z",
    "2026-07-25T20:00:03Z",
    "2026-07-25T20:00:04Z",
    "2026-07-25T20:00:05Z",
    "2026-07-25T20:00:06Z",
    "2026-07-25T20:00:07Z",
    "2026-07-25T20:00:08Z",
]


def make_store(tmp_path: Path):
    repository = tmp_path / "repository"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    policy = load_conversation_state_policy()
    store = ConversationStateStore.for_vault(
        vault_root=vault,
        repository_root=repository,
        policy=policy,
    )
    store.initialize()
    sequence = count()
    service = ConversationStateService(
        store,
        event_id_factory=lambda: f"event-{next(sequence)}",
    )
    return store, service, repository, vault


def message(
    *,
    message_id: str,
    turn_id: str,
    role: str,
    content: str,
    created_at: str,
    classification: str = "PRIVATE",
) -> ConversationMessage:
    return ConversationMessage.create(
        message_id=message_id,
        turn_id=turn_id,
        role=role,
        content=content,
        created_at=created_at,
        data_classification=classification,
    )


def response(
    *,
    request_id: str,
    content: str,
    created_at: str,
    provider: str = "deterministic-test",
    model: str = "fixed-response-v1",
) -> ModelResponse:
    value = ModelResponse(
        request_id=request_id,
        provider=provider,
        model=model,
        content=content,
        finish_reason="stop",
        created_at=created_at,
    )
    value.validate()
    return value


def digest(text: str) -> str:
    return sha256_text(text)
