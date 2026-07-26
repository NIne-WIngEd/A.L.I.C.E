"""Governed local Ollama adapter for A.L.I.C.E. Phase 3 P3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import ModelRequest, ModelResponse
from .model import (
    CancellationToken,
    ConversationModelBudgetError,
    ConversationModelConfigurationError,
    ConversationModelProtocolError,
    ConversationModelProviderError,
    ProviderFailure,
)
from .model_policy import ModelProviderPolicy
from .transport import (
    ConversationTransportError,
    JsonHttpResponse,
    JsonHttpTransport,
    UrllibJsonHttpTransport,
)

_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_ALLOWED_DONE_REASONS = {"stop", "length"}


@dataclass(frozen=True)
class OllamaModelConfig:
    """Validated local-only configuration projected from model policy."""

    model: str
    base_url: str
    request_timeout_seconds: float
    max_context_chars: int
    max_output_tokens: int
    provider: str = "ollama-local"

    @classmethod
    def from_policy(
        cls,
        policy: ModelProviderPolicy,
        *,
        model: str | None = None,
    ) -> "OllamaModelConfig":
        if policy.provider_id != "ollama-local":
            raise ConversationModelConfigurationError(
                "Ollama configuration requires the ollama-local policy."
            )
        selected_model = model or policy.default_model
        if selected_model not in policy.allowed_models:
            raise ConversationModelConfigurationError(
                f"Ollama model is not approved by policy: {selected_model}"
            )
        if not policy.enabled or not policy.local_only:
            raise ConversationModelConfigurationError(
                "Ollama policy must remain enabled and local-only."
            )
        if policy.stream or policy.think or policy.tools_allowed:
            raise ConversationModelConfigurationError(
                "P3.1 Ollama policy must disable streaming, thinking, and tools."
            )
        if (
            policy.base_url is None
            or policy.request_timeout_seconds is None
            or policy.max_context_chars is None
            or policy.max_output_tokens is None
        ):
            raise ConversationModelConfigurationError(
                "Ollama policy is missing required bounded configuration."
            )
        return cls(
            model=selected_model,
            base_url=policy.base_url,
            request_timeout_seconds=policy.request_timeout_seconds,
            max_context_chars=policy.max_context_chars,
            max_output_tokens=policy.max_output_tokens,
        )


@dataclass
class OllamaConversationModel:
    """Non-streaming local Ollama adapter with no tool-call surface."""

    config: OllamaModelConfig
    transport: JsonHttpTransport = UrllibJsonHttpTransport()

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def generate(
        self,
        request: ModelRequest,
        cancellation: CancellationToken | None = None,
    ) -> ModelResponse:
        request.validate()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        messages = self._messages(request)
        context_chars = sum(len(message["content"]) for message in messages)
        if context_chars > self.config.max_context_chars:
            raise ConversationModelBudgetError(
                "Model request exceeds the approved context-character budget: "
                f"{context_chars} > {self.config.max_context_chars}."
            )
        if request.max_output_tokens > self.config.max_output_tokens:
            raise ConversationModelBudgetError(
                "Model request exceeds the approved output-token budget: "
                f"{request.max_output_tokens} > {self.config.max_output_tokens}."
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        try:
            response = self.transport.post_json(
                url=f"{self.config.base_url}/api/chat",
                payload=payload,
                timeout_seconds=self.config.request_timeout_seconds,
                cancellation=cancellation,
            )
        except ConversationTransportError as exc:
            raise ConversationModelProviderError(
                ProviderFailure(
                    provider=self.provider,
                    model=self.model,
                    code="transport_error",
                    message=str(exc),
                    retryable=True,
                )
            ) from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return self._parse_response(request=request, response=response)

    def _messages(self, request: ModelRequest) -> list[dict[str, str]]:
        system_parts = [
            f"System contract version: {request.system_contract_version}",
            request.system_contract,
        ]
        if request.grounding is not None:
            system_parts.append(request.grounding.render_for_model())
        messages = [
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        ]
        messages.extend(
            {"role": message.role, "content": message.content}
            for message in request.messages
        )
        return messages

    def _parse_response(
        self,
        *,
        request: ModelRequest,
        response: JsonHttpResponse,
    ) -> ModelResponse:
        try:
            payload = response.json_object()
        except ValueError as exc:
            raise ConversationModelProtocolError(str(exc)) from exc
        if not 200 <= response.status_code < 300:
            message = self._provider_error_message(payload)
            raise ConversationModelProviderError(
                ProviderFailure(
                    provider=self.provider,
                    model=self.model,
                    code="http_error",
                    message=message,
                    retryable=response.status_code in _RETRYABLE_HTTP_STATUSES,
                    http_status=response.status_code,
                )
            )
        error = payload.get("error")
        if error is not None:
            raise ConversationModelProviderError(
                ProviderFailure(
                    provider=self.provider,
                    model=self.model,
                    code="provider_error",
                    message=self._error_text(error),
                    retryable=False,
                )
            )
        if payload.get("done") is not True:
            raise ConversationModelProtocolError(
                "Non-streaming Ollama response must be complete."
            )
        response_model = payload.get("model")
        if response_model != self.model:
            raise ConversationModelProtocolError(
                "Ollama response model identity does not match configuration."
            )
        done_reason = payload.get("done_reason")
        if done_reason not in _ALLOWED_DONE_REASONS:
            raise ConversationModelProtocolError(
                f"Unsupported Ollama done reason: {done_reason!r}"
            )
        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ConversationModelProtocolError(
                "Ollama response must include created_at."
            )
        message = payload.get("message")
        if not isinstance(message, dict):
            raise ConversationModelProtocolError(
                "Ollama response message must be an object."
            )
        if message.get("role") != "assistant":
            raise ConversationModelProtocolError(
                "Ollama response message role must be assistant."
            )
        if self._nonempty(message.get("tool_calls")):
            raise ConversationModelProtocolError(
                "Ollama returned tool calls while tools are prohibited."
            )
        if self._nonempty(message.get("thinking")):
            raise ConversationModelProtocolError(
                "Ollama returned hidden thinking while thinking is disabled."
            )
        if self._nonempty(message.get("images")):
            raise ConversationModelProtocolError(
                "Ollama returned images in the text-only P3.1 adapter."
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ConversationModelProtocolError(
                "Ollama response content must be non-empty text."
            )
        model_response = ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=content,
            finish_reason=done_reason,
            created_at=created_at,
        )
        model_response.validate()
        return model_response

    @staticmethod
    def _nonempty(value: Any) -> bool:
        return value not in (None, "", [], {})

    @staticmethod
    def _error_text(value: Any) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "Local model provider returned an unspecified error."

    @classmethod
    def _provider_error_message(cls, payload: Mapping[str, Any]) -> str:
        return cls._error_text(payload.get("error"))
