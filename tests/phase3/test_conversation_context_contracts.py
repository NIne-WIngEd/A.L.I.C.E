from __future__ import annotations

from dataclasses import replace

import pytest

from alice_conversation.contracts import (
    ConversationCapabilities,
    ConversationContractError,
    ConversationMessage,
    ModelRequest,
)


def message(turn: int, role: str, content: str | None = None) -> ConversationMessage:
    return ConversationMessage.create(
        message_id=f"{role}-{turn}",
        turn_id=f"turn-{turn}",
        role=role,
        content=content or f"{role} content {turn}",
        created_at=f"2026-07-26T06:00:{turn:02d}Z",
        data_classification="PRIVATE",
    )


def request(messages: tuple[ConversationMessage, ...]) -> ModelRequest:
    return ModelRequest(
        request_id="request-current",
        session_id="session-1",
        turn_id="turn-3",
        system_contract_version="contract-v1",
        system_contract="Trusted system contract.",
        messages=messages,
        grounding=None,
        capabilities=ConversationCapabilities(),
        max_output_tokens=100,
        temperature=0.0,
    )


def test_model_request_accepts_current_turn_only() -> None:
    request((message(3, "user"),)).validate()


def test_model_request_accepts_ordered_prior_pairs() -> None:
    request(
        (
            message(1, "user"),
            message(1, "assistant"),
            message(2, "user"),
            message(2, "assistant"),
            message(3, "user"),
        )
    ).validate()


@pytest.mark.parametrize(
    "messages",
    [
        (message(1, "user"), message(3, "user")),
        (message(1, "assistant"), message(1, "user"), message(3, "user")),
        (message(1, "user"), message(2, "assistant"), message(3, "user")),
        (message(1, "user"), message(1, "assistant"), message(1, "user"), message(1, "assistant"), message(3, "user")),
        (message(1, "user"), message(1, "assistant"), message(3, "assistant")),
    ],
)
def test_model_request_rejects_invalid_history_shape(
    messages: tuple[ConversationMessage, ...]
) -> None:
    with pytest.raises(ConversationContractError):
        request(messages).validate()


def test_model_request_rejects_duplicate_message_identifier() -> None:
    first = message(1, "user")
    duplicate = replace(message(1, "assistant"), message_id=first.message_id)
    with pytest.raises(ConversationContractError):
        request((first, duplicate, message(3, "user"))).validate()


def test_model_request_requires_current_user_message_last() -> None:
    wrong = replace(message(2, "user"), turn_id="turn-2")
    with pytest.raises(ConversationContractError):
        request((wrong,)).validate()
