"""P3.1 provider registry and cancellation tests."""

from __future__ import annotations

import pytest
from alice_conversation.contracts import ConversationMessage, ModelRequest
from alice_conversation.model import (
    CancellationToken,
    ConversationModelCancelledError,
    ConversationModelConfigurationError,
    DeterministicConversationModel,
)
from alice_conversation.registry import ConversationModelRegistry

NOW = "2026-07-26T03:00:00Z"


def _request() -> ModelRequest:
    return ModelRequest(
        request_id="request-registry-001",
        session_id="session-registry-001",
        turn_id="turn-registry-001",
        system_contract_version="alice-constitution-0.1.0",
        system_contract="Be truthful.",
        messages=(
            ConversationMessage.create(
                message_id="message-registry-001",
                turn_id="turn-registry-001",
                role="user",
                content="Reply deterministically.",
                created_at=NOW,
            ),
        ),
        grounding=None,
    )


def test_registry_resolves_only_exact_registered_identity() -> None:
    registry = ConversationModelRegistry()
    model = DeterministicConversationModel(response_text="ok")
    registry.register(model)
    assert registry.resolve(
        provider="deterministic-test",
        model="fixed-response-v1",
    ) is model
    assert registry.identities() == (
        ("deterministic-test", "fixed-response-v1"),
    )
    with pytest.raises(ConversationModelConfigurationError, match="not registered"):
        registry.resolve(provider="ollama-local", model="qwen3:8b")


def test_registry_rejects_duplicate_or_invalid_adapter() -> None:
    registry = ConversationModelRegistry()
    model = DeterministicConversationModel(response_text="ok")
    registry.register(model)
    with pytest.raises(ConversationModelConfigurationError, match="already"):
        registry.register(model)

    class InvalidAdapter:
        provider = "invalid"
        model = "missing-generate"

    with pytest.raises(ConversationModelConfigurationError, match="generate"):
        registry.register(InvalidAdapter())  # type: ignore[arg-type]


def test_deterministic_model_honors_cooperative_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    model = DeterministicConversationModel(response_text="not returned")
    with pytest.raises(ConversationModelCancelledError, match="cancelled"):
        model.generate(_request(), cancellation=token)
    assert model.requests == []
