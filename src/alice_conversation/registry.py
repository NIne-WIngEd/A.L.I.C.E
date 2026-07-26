"""Fail-closed model adapter registry for A.L.I.C.E. Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import ConversationModel, ConversationModelConfigurationError


def _identity(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationModelConfigurationError(
            f"Model adapter {field_name} must be non-empty text."
        )
    return value.strip()


@dataclass
class ConversationModelRegistry:
    """Explicit registry keyed by exact provider and model identity."""

    _models: dict[tuple[str, str], ConversationModel] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def register(self, adapter: ConversationModel) -> None:
        provider = _identity(getattr(adapter, "provider", None), field_name="provider")
        model = _identity(getattr(adapter, "model", None), field_name="model")
        generate = getattr(adapter, "generate", None)
        if not callable(generate):
            raise ConversationModelConfigurationError(
                "Model adapter must provide a callable generate method."
            )
        key = (provider, model)
        if key in self._models:
            raise ConversationModelConfigurationError(
                f"Model adapter already registered: {provider}/{model}"
            )
        self._models[key] = adapter

    def resolve(self, *, provider: str, model: str) -> ConversationModel:
        key = (
            _identity(provider, field_name="provider"),
            _identity(model, field_name="model"),
        )
        try:
            return self._models[key]
        except KeyError as exc:
            raise ConversationModelConfigurationError(
                f"Model adapter is not registered: {key[0]}/{key[1]}"
            ) from exc

    def identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._models))
