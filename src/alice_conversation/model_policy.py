"""Versioned model-provider policy loading for A.L.I.C.E. P3.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .model import ConversationModelConfigurationError

DEFAULT_MODEL_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_model_policy.json"
)


class ConversationModelPolicyError(ConversationModelConfigurationError):
    """Raised when the public model-provider policy is invalid."""


@dataclass(frozen=True)
class ModelProviderPolicy:
    provider_id: str
    enabled: bool
    local_only: bool
    allowed_models: tuple[str, ...]
    default_model: str
    base_url: str | None
    request_timeout_seconds: float | None
    max_context_chars: int | None
    max_output_tokens: int | None
    stream: bool
    think: bool
    tools_allowed: bool


@dataclass(frozen=True)
class ConversationModelPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    default_provider: str
    providers: tuple[ModelProviderPolicy, ...]

    def provider(self, provider_id: str) -> ModelProviderPolicy:
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        raise ConversationModelPolicyError(
            f"Provider is not approved by model policy: {provider_id}"
        )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationModelPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationModelPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationModelPolicyError(
            f"{field} must remain {str(expected).lower()} in P3.1."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationModelPolicyError(
            f"{field} must be a positive integer."
        )
    return value


def _positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConversationModelPolicyError(
            f"{field} must be a positive number."
        )
    return float(value)


def _model_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConversationModelPolicyError(
            f"{field} must be a non-empty list."
        )
    models = tuple(_text(item, field=f"{field} item") for item in value)
    if len(set(models)) != len(models):
        raise ConversationModelPolicyError(f"{field} cannot contain duplicates.")
    return models


def _validate_loopback_base_url(value: str, *, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http":
        raise ConversationModelPolicyError(
            f"{field} must use plain HTTP on the local loopback interface."
        )
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ConversationModelPolicyError(
            f"{field} must target a local loopback host."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConversationModelPolicyError(
            f"{field} cannot contain credentials, a query, or a fragment."
        )
    if parsed.path not in {"", "/"}:
        raise ConversationModelPolicyError(
            f"{field} cannot contain an API path."
        )
    try:
        if parsed.port is None:
            raise ConversationModelPolicyError(
                f"{field} must include an explicit port."
            )
    except ValueError as exc:
        raise ConversationModelPolicyError(
            f"{field} contains an invalid port."
        ) from exc
    return value.rstrip("/")


def parse_conversation_model_policy(
    payload: dict[str, Any],
) -> ConversationModelPolicy:
    providers_payload = _mapping(payload.get("providers"), field="providers")
    if set(providers_payload) != {"deterministic-test", "ollama-local"}:
        raise ConversationModelPolicyError(
            "P3.1 providers must be exactly deterministic-test and ollama-local."
        )

    deterministic_payload = _mapping(
        providers_payload["deterministic-test"],
        field="providers.deterministic-test",
    )
    deterministic_models = _model_list(
        deterministic_payload.get("allowed_models"),
        field="providers.deterministic-test.allowed_models",
    )
    deterministic_default = _text(
        deterministic_payload.get("default_model"),
        field="providers.deterministic-test.default_model",
    )
    if deterministic_default not in deterministic_models:
        raise ConversationModelPolicyError(
            "Deterministic default model must be in allowed_models."
        )
    deterministic = ModelProviderPolicy(
        provider_id="deterministic-test",
        enabled=_strict_bool(
            deterministic_payload.get("enabled"),
            expected=True,
            field="providers.deterministic-test.enabled",
        ),
        local_only=_strict_bool(
            deterministic_payload.get("local_only"),
            expected=True,
            field="providers.deterministic-test.local_only",
        ),
        allowed_models=deterministic_models,
        default_model=deterministic_default,
        base_url=None,
        request_timeout_seconds=None,
        max_context_chars=None,
        max_output_tokens=None,
        stream=False,
        think=False,
        tools_allowed=False,
    )

    ollama_payload = _mapping(
        providers_payload["ollama-local"],
        field="providers.ollama-local",
    )
    ollama_models = _model_list(
        ollama_payload.get("allowed_models"),
        field="providers.ollama-local.allowed_models",
    )
    ollama_default = _text(
        ollama_payload.get("default_model"),
        field="providers.ollama-local.default_model",
    )
    if ollama_default not in ollama_models:
        raise ConversationModelPolicyError(
            "Ollama default model must be in allowed_models."
        )
    ollama = ModelProviderPolicy(
        provider_id="ollama-local",
        enabled=_strict_bool(
            ollama_payload.get("enabled"),
            expected=True,
            field="providers.ollama-local.enabled",
        ),
        local_only=_strict_bool(
            ollama_payload.get("local_only"),
            expected=True,
            field="providers.ollama-local.local_only",
        ),
        allowed_models=ollama_models,
        default_model=ollama_default,
        base_url=_validate_loopback_base_url(
            _text(
                ollama_payload.get("base_url"),
                field="providers.ollama-local.base_url",
            ),
            field="providers.ollama-local.base_url",
        ),
        request_timeout_seconds=_positive_number(
            ollama_payload.get("request_timeout_seconds"),
            field="providers.ollama-local.request_timeout_seconds",
        ),
        max_context_chars=_positive_int(
            ollama_payload.get("max_context_chars"),
            field="providers.ollama-local.max_context_chars",
        ),
        max_output_tokens=_positive_int(
            ollama_payload.get("max_output_tokens"),
            field="providers.ollama-local.max_output_tokens",
        ),
        stream=_strict_bool(
            ollama_payload.get("stream"),
            expected=False,
            field="providers.ollama-local.stream",
        ),
        think=_strict_bool(
            ollama_payload.get("think"),
            expected=False,
            field="providers.ollama-local.think",
        ),
        tools_allowed=_strict_bool(
            ollama_payload.get("tools_allowed"),
            expected=False,
            field="providers.ollama-local.tools_allowed",
        ),
    )

    policy = ConversationModelPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        default_provider=_text(
            payload.get("default_provider"),
            field="default_provider",
        ),
        providers=(deterministic, ollama),
    )
    if policy.phase != "3" or policy.milestone != "P3.1":
        raise ConversationModelPolicyError(
            "Model policy must be bound to Phase 3 milestone P3.1."
        )
    if policy.status != "model_abstraction":
        raise ConversationModelPolicyError(
            "P3.1 model policy status must be model_abstraction."
        )
    if policy.default_provider != "ollama-local":
        raise ConversationModelPolicyError(
            "P3.1 default provider must remain ollama-local."
        )
    return policy


def load_conversation_model_policy(
    path: str | Path = DEFAULT_MODEL_POLICY_PATH,
) -> ConversationModelPolicy:
    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversationModelPolicyError(
            f"Unable to load conversation model policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationModelPolicyError(
            "Conversation model policy root must be a JSON object."
        )
    return parse_conversation_model_policy(payload)
