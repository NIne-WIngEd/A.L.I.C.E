"""Provider-neutral model boundary for A.L.I.C.E. Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Protocol

from .contracts import ModelRequest, ModelResponse, utc_now_text


class ConversationModelError(RuntimeError):
    """Base error for conversational model adapters."""


class ConversationModelConfigurationError(ConversationModelError):
    """Raised when a model adapter or registry configuration is invalid."""


class ConversationModelTimeoutError(ConversationModelError):
    """Raised when a provider exceeds its approved timeout."""


class ConversationModelCancelledError(ConversationModelError):
    """Raised when a generation is cancelled before completion."""


class ConversationModelBudgetError(ConversationModelError):
    """Raised when a request exceeds an approved model budget."""


class ConversationModelProtocolError(ConversationModelError):
    """Raised when a provider response violates the model contract."""


@dataclass(frozen=True)
class ProviderFailure:
    """Structured, sanitized provider failure metadata."""

    provider: str
    model: str
    code: str
    message: str
    retryable: bool
    http_status: int | None = None

    def validate(self) -> None:
        for field_name in ("provider", "model", "code", "message"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConversationModelConfigurationError(
                    f"Provider failure {field_name} must be non-empty text."
                )
        if not isinstance(self.retryable, bool):
            raise ConversationModelConfigurationError(
                "Provider failure retryable must be boolean."
            )
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ConversationModelConfigurationError(
                "Provider failure HTTP status must be between 100 and 599."
            )


class ConversationModelProviderError(ConversationModelError):
    """Raised for a sanitized provider-side or transport-side failure."""

    def __init__(self, failure: ProviderFailure) -> None:
        failure.validate()
        self.failure = failure
        super().__init__(
            f"{failure.provider}/{failure.model} {failure.code}: "
            f"{failure.message}"
        )


class CancellationToken:
    """Thread-safe cooperative cancellation token for model generation."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ConversationModelCancelledError(
                "Conversation model generation was cancelled."
            )


class ConversationModel(Protocol):
    """Minimal model interface. Tool execution is intentionally absent."""

    provider: str
    model: str

    def generate(
        self,
        request: ModelRequest,
        cancellation: CancellationToken | None = None,
    ) -> ModelResponse:
        """Generate one response for a validated request."""


@dataclass
class DeterministicConversationModel:
    """Deterministic adapter used for tests and orchestration evaluation."""

    response_text: str
    provider: str = "deterministic-test"
    model: str = "fixed-response-v1"
    requests: list[ModelRequest] = field(default_factory=list, init=False)

    def generate(
        self,
        request: ModelRequest,
        cancellation: CancellationToken | None = None,
    ) -> ModelResponse:
        request.validate()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if not self.response_text.strip():
            raise ConversationModelError(
                "Deterministic model response must be non-empty."
            )
        self.requests.append(request)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        response = ModelResponse(
            request_id=request.request_id,
            provider=self.provider,
            model=self.model,
            content=self.response_text,
            finish_reason="stop",
            created_at=utc_now_text(),
        )
        response.validate()
        return response
