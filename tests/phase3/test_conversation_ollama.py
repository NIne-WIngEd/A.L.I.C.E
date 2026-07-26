"""P3.1 local Ollama adapter tests using an injected fake transport."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import pytest
from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ConversationMessage,
    ModelRequest,
    sha256_text,
)
from alice_conversation.model import (
    CancellationToken,
    ConversationModelBudgetError,
    ConversationModelCancelledError,
    ConversationModelProtocolError,
    ConversationModelProviderError,
    ConversationModelTimeoutError,
)
from alice_conversation.model_policy import load_conversation_model_policy
from alice_conversation.ollama import OllamaConversationModel, OllamaModelConfig
from alice_conversation.transport import (
    ConversationTransportError,
    JsonHttpResponse,
)

NOW = "2026-07-26T03:00:00Z"
POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_model_policy.json"
)


@dataclass
class FakeTransport:
    response: JsonHttpResponse | None = None
    error: Exception | None = None
    cancel_during_call: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)

    def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> JsonHttpResponse:
        self.calls.append(
            {
                "url": url,
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        if self.cancel_during_call and cancellation is not None:
            cancellation.cancel()
        assert self.response is not None
        return self.response


def _grounding() -> ConversationGroundingPacket:
    text = "Phase 2 is complete."
    citation = ConversationCitation(
        citation_id="citation-ollama-001",
        source_kind="memory_source",
        source_ref="phase1://source/example",
        token="[memory:memory-001 source:phase1://source/example]",
        data_classification="PRIVATE",
    )
    claim = ConversationGroundingClaim(
        claim_id="claim-ollama-001",
        text=text,
        content_sha256=sha256_text(text),
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        citations=(citation,),
    )
    return ConversationGroundingPacket(
        packet_id="packet-ollama-001",
        outcome="answerable",
        claims=(claim,),
        created_at=NOW,
    )


def _request(**changes: Any) -> ModelRequest:
    values: dict[str, Any] = {
        "request_id": "request-ollama-001",
        "session_id": "session-ollama-001",
        "turn_id": "turn-ollama-001",
        "system_contract_version": "alice-constitution-0.1.0",
        "system_contract": "Be truthful and preserve uncertainty.",
        "messages": (
            ConversationMessage.create(
                message_id="message-ollama-001",
                turn_id="turn-ollama-001",
                role="user",
                content="What phase is complete?",
                created_at=NOW,
            ),
        ),
        "grounding": _grounding(),
        "max_output_tokens": 256,
        "temperature": 0.0,
    }
    values.update(changes)
    return ModelRequest(**values)


def _success_payload(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "qwen3:8b",
        "created_at": NOW,
        "message": {
            "role": "assistant",
            "content": "Phase 2 is complete.",
        },
        "done": True,
        "done_reason": "stop",
    }
    payload.update(changes)
    return payload


def _response(
    payload: dict[str, Any],
    *,
    status_code: int = 200,
) -> JsonHttpResponse:
    return JsonHttpResponse(
        status_code=status_code,
        body=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def _config(**changes: Any) -> OllamaModelConfig:
    policy = load_conversation_model_policy(POLICY_PATH)
    config = OllamaModelConfig.from_policy(policy.provider("ollama-local"))
    return replace(config, **changes)


def test_ollama_adapter_builds_bounded_non_tool_payload() -> None:
    transport = FakeTransport(response=_response(_success_payload()))
    model = OllamaConversationModel(config=_config(), transport=transport)
    response = model.generate(_request())

    assert response.provider == "ollama-local"
    assert response.model == "qwen3:8b"
    assert response.content == "Phase 2 is complete."
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/chat"
    assert call["timeout_seconds"] == 600.0
    payload = call["payload"]
    assert payload["stream"] is False
    assert payload["think"] is False
    assert "tools" not in payload
    assert payload["options"] == {"temperature": 0.0, "num_predict": 256}
    assert payload["messages"][0]["role"] == "system"
    assert "BEGIN UNTRUSTED GROUNDING DATA" in payload["messages"][0]["content"]
    assert payload["messages"][-1] == {
        "role": "user",
        "content": "What phase is complete?",
    }


def test_ollama_adapter_enforces_context_and_output_budgets() -> None:
    transport = FakeTransport(response=_response(_success_payload()))
    model = OllamaConversationModel(
        config=_config(max_context_chars=20),
        transport=transport,
    )
    with pytest.raises(ConversationModelBudgetError, match="context"):
        model.generate(_request())
    assert transport.calls == []

    model = OllamaConversationModel(
        config=_config(max_output_tokens=128),
        transport=transport,
    )
    with pytest.raises(ConversationModelBudgetError, match="output-token"):
        model.generate(_request(max_output_tokens=256))
    assert transport.calls == []


def test_ollama_adapter_honors_cancellation_before_and_after_transport() -> None:
    token = CancellationToken()
    token.cancel()
    transport = FakeTransport(response=_response(_success_payload()))
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelCancelledError, match="cancelled"):
        model.generate(_request(), cancellation=token)
    assert transport.calls == []

    token = CancellationToken()
    transport = FakeTransport(
        response=_response(_success_payload()),
        cancel_during_call=True,
    )
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelCancelledError, match="cancelled"):
        model.generate(_request(), cancellation=token)
    assert len(transport.calls) == 1


def test_ollama_adapter_preserves_timeout_and_wraps_transport_failure() -> None:
    timeout = FakeTransport(
        error=ConversationModelTimeoutError("provider timeout")
    )
    model = OllamaConversationModel(config=_config(), transport=timeout)
    with pytest.raises(ConversationModelTimeoutError, match="timeout"):
        model.generate(_request())

    failed = FakeTransport(
        error=ConversationTransportError("connection refused")
    )
    model = OllamaConversationModel(config=_config(), transport=failed)
    with pytest.raises(ConversationModelProviderError) as exc_info:
        model.generate(_request())
    failure = exc_info.value.failure
    assert failure.code == "transport_error"
    assert failure.retryable is True
    assert failure.http_status is None


def test_ollama_adapter_exposes_structured_http_provider_error() -> None:
    transport = FakeTransport(
        response=_response({"error": "model not found"}, status_code=404)
    )
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelProviderError) as exc_info:
        model.generate(_request())
    failure = exc_info.value.failure
    assert failure.code == "http_error"
    assert failure.message == "model not found"
    assert failure.http_status == 404
    assert failure.retryable is False


@pytest.mark.parametrize(
    ("message_change", "match"),
    [
        ({"tool_calls": [{"function": {"name": "forbidden"}}]}, "tool calls"),
        ({"thinking": "private reasoning"}, "hidden thinking"),
        ({"images": ["base64-data"]}, "images"),
    ],
)
def test_ollama_adapter_rejects_unapproved_response_surfaces(
    message_change: dict[str, Any],
    match: str,
) -> None:
    message = {
        "role": "assistant",
        "content": "This must not be accepted.",
        **message_change,
    }
    transport = FakeTransport(
        response=_response(_success_payload(message=message))
    )
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelProtocolError, match=match):
        model.generate(_request())


@pytest.mark.parametrize(
    "payload",
    [
        _success_payload(done=False),
        _success_payload(model="different-model"),
        _success_payload(done_reason="unknown"),
        _success_payload(message={"role": "assistant", "content": ""}),
    ],
)
def test_ollama_adapter_rejects_invalid_provider_contract(
    payload: dict[str, Any],
) -> None:
    transport = FakeTransport(response=_response(payload))
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelProtocolError):
        model.generate(_request())


def test_ollama_adapter_rejects_malformed_json() -> None:
    transport = FakeTransport(
        response=JsonHttpResponse(
            status_code=200,
            body=b"not-json",
            headers={},
        )
    )
    model = OllamaConversationModel(config=_config(), transport=transport)
    with pytest.raises(ConversationModelProtocolError, match="JSON"):
        model.generate(_request())
