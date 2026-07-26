"""P3.1 public model-provider policy validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alice_conversation.model import ConversationModelConfigurationError
from alice_conversation.model_policy import (
    ConversationModelPolicyError,
    load_conversation_model_policy,
    parse_conversation_model_policy,
)
from alice_conversation.ollama import OllamaModelConfig

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_model_policy.json"
)


def _payload() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_repository_model_policy_is_local_bounded_and_tool_free() -> None:
    policy = load_conversation_model_policy(POLICY_PATH)
    assert policy.phase == "3"
    assert policy.milestone == "P3.1"
    assert policy.default_provider == "ollama-local"
    assert tuple(provider.provider_id for provider in policy.providers) == (
        "deterministic-test",
        "ollama-local",
    )
    ollama = policy.provider("ollama-local")
    assert ollama.local_only is True
    assert ollama.base_url == "http://127.0.0.1:11434"
    assert ollama.stream is False
    assert ollama.think is False
    assert ollama.tools_allowed is False
    assert ollama.request_timeout_seconds == 600.0
    assert ollama.max_context_chars == 32768
    assert ollama.max_output_tokens == 4096


def test_model_policy_rejects_remote_or_path_bearing_endpoints() -> None:
    payload = _payload()
    payload["providers"]["ollama-local"]["base_url"] = (
        "https://models.example.com"
    )
    with pytest.raises(ConversationModelPolicyError, match="loopback|local"):
        parse_conversation_model_policy(payload)

    payload = _payload()
    payload["providers"]["ollama-local"]["base_url"] = (
        "http://127.0.0.1:11434/api"
    )
    with pytest.raises(ConversationModelPolicyError, match="API path"):
        parse_conversation_model_policy(payload)


@pytest.mark.parametrize("field", ["stream", "think", "tools_allowed"])
def test_model_policy_rejects_expanded_ollama_surfaces(field: str) -> None:
    payload = _payload()
    payload["providers"]["ollama-local"][field] = True
    with pytest.raises(ConversationModelPolicyError, match="must remain false"):
        parse_conversation_model_policy(payload)


def test_model_policy_rejects_unapproved_provider_or_default_model() -> None:
    payload = _payload()
    payload["providers"]["remote-cloud"] = {
        "enabled": True,
        "local_only": False,
    }
    with pytest.raises(ConversationModelPolicyError, match="exactly"):
        parse_conversation_model_policy(payload)

    payload = _payload()
    payload["providers"]["ollama-local"]["default_model"] = "unknown"
    with pytest.raises(ConversationModelPolicyError, match="allowed_models"):
        parse_conversation_model_policy(payload)


def test_ollama_configuration_is_projected_only_from_approved_policy() -> None:
    policy = load_conversation_model_policy(POLICY_PATH)
    ollama_policy = policy.provider("ollama-local")
    config = OllamaModelConfig.from_policy(ollama_policy)
    assert config.provider == "ollama-local"
    assert config.model == "qwen3:8b"
    assert config.request_timeout_seconds == 600.0

    with pytest.raises(
        ConversationModelConfigurationError,
        match="not approved",
    ):
        OllamaModelConfig.from_policy(ollama_policy, model="unapproved:latest")
