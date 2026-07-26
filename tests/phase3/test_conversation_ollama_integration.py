"""Optional live P3.1 Ollama integration test.

This test is skipped unless ALICE_RUN_OLLAMA_INTEGRATION=1 is set explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alice_conversation.contracts import ConversationMessage, ModelRequest
from alice_conversation.model_policy import load_conversation_model_policy
from alice_conversation.ollama import OllamaConversationModel, OllamaModelConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("ALICE_RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set ALICE_RUN_OLLAMA_INTEGRATION=1 for the live local test.",
)

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_model_policy.json"
)


def test_live_local_ollama_generation_has_governed_identity() -> None:
    policy = load_conversation_model_policy(POLICY_PATH)
    config = OllamaModelConfig.from_policy(policy.provider("ollama-local"))
    model = OllamaConversationModel(config=config)
    request = ModelRequest(
        request_id="request-live-ollama-001",
        session_id="session-live-ollama-001",
        turn_id="turn-live-ollama-001",
        system_contract_version="alice-constitution-0.1.0",
        system_contract=(
            "Be truthful. Do not call tools. Reply with one short sentence."
        ),
        messages=(
            ConversationMessage.create(
                message_id="message-live-ollama-001",
                turn_id="turn-live-ollama-001",
                role="user",
                content="Confirm that the local model adapter is responding.",
                created_at="2026-07-26T03:00:00Z",
            ),
        ),
        grounding=None,
        max_output_tokens=64,
        temperature=0.0,
    )
    response = model.generate(request)
    assert response.provider == "ollama-local"
    assert response.model == config.model
    assert response.content.strip()
