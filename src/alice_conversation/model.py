"""Provider-neutral model boundary for A.L.I.C.E. Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .contracts import ModelRequest, ModelResponse, utc_now_text


class ConversationModelError(RuntimeError):
    """Base error for conversational model adapters."""


class ConversationModelTimeoutError(ConversationModelError):
    """Raised when a provider exceeds its approved timeout."""


class ConversationModelCancelledError(ConversationModelError):
    """Raised when a generation is cancelled before completion."""


class ConversationModel(Protocol):
    """Minimal model interface. Tool execution is intentionally absent."""

    provider: str
    model: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response for a validated request."""


@dataclass
class DeterministicConversationModel:
    """Deterministic adapter used for tests before local inference is connected."""

    response_text: str
    provider: str = "deterministic-test"
    model: str = "fixed-response-v1"
    requests: list[ModelRequest] = field(default_factory=list, init=False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        request.validate()
        if not self.response_text.strip():
            raise ConversationModelError(
                "Deterministic model response must be non-empty."
            )
        self.requests.append(request)
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
