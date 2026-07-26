from __future__ import annotations

import pytest

from alice_conversation.contracts import ConversationCapabilities, ConversationMessage, ModelResponse
from alice_conversation.state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)
from alice_conversation.state_service import ConversationStateError

from _repair_helpers import build_runtime


def _started(service):
    user = ConversationMessage.create(
        message_id="user-1",
        turn_id="turn-1",
        role="user",
        content="Please answer.",
        created_at="2026-07-26T00:00:01Z",
        data_classification="PRIVATE",
    )
    service.start_turn(session_id="session-1", turn_id="turn-1", user_message=user)
    service.set_turn_context(
        turn_id="turn-1",
        references=(),
        updated_at="2026-07-26T00:00:02Z",
        grounding_packet_id=None,
        grounding_packet_sha256=None,
    )
    service.start_generation(
        turn_id="turn-1",
        generation_id="generation-1",
        request_id="request-1",
        provider="test-provider",
        model="test-model",
        started_at="2026-07-26T00:00:03Z",
    )
    return ModelResponse(
        request_id="request-1",
        provider="test-provider",
        model="test-model",
        content="I completed the task for you.",
        finish_reason="stop",
        created_at="2026-07-26T00:00:04Z",
    )


def test_rejected_generation_can_reopen_turn_without_raw_text(tmp_path):
    store, service, _, _ = build_runtime(tmp_path, [])
    response = _started(service)
    service.reject_generation(
        turn_id="turn-1",
        request_id="request-1",
        response=response,
        rejected_at="2026-07-26T00:00:05Z",
        failure_code="response_validation_rejected",
        allow_repair=True,
    )
    inspection = inspect_conversation_session(store, session_id="session-1", include_content=True)
    turn = inspection.turns[0]
    assert turn.status == "context_ready"
    assert turn.failure_code is None
    assert len(turn.generations) == 1
    assert turn.generations[0].status == "failed"
    assert turn.generations[0].validation_outcome == "rejected"
    assert turn.generations[0].response_sha256 is not None
    assert len(turn.messages) == 1
    assert response.content not in repr(inspection)
    assert verify_conversation_session_integrity(store, session_id="session-1").valid


def test_rejected_generation_can_fail_turn_terminally(tmp_path):
    store, service, _, _ = build_runtime(tmp_path, [])
    response = _started(service)
    service.reject_generation(
        turn_id="turn-1",
        request_id="request-1",
        response=response,
        rejected_at="2026-07-26T00:00:05Z",
        failure_code="response_repair_exhausted",
        allow_repair=False,
    )
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "failed"
    assert turn.failure_code == "response_repair_exhausted"
    assert len(turn.messages) == 1
    assert verify_conversation_session_integrity(store, session_id="session-1").valid


def test_rejection_requires_matching_active_request(tmp_path):
    _, service, _, _ = build_runtime(tmp_path, [])
    response = _started(service)
    with pytest.raises(ConversationStateError):
        service.reject_generation(
            turn_id="turn-1",
            request_id="other-request",
            response=response,
            rejected_at="2026-07-26T00:00:05Z",
            failure_code="response_validation_rejected",
            allow_repair=True,
        )


def test_rejection_requires_same_provider_model(tmp_path):
    _, service, _, _ = build_runtime(tmp_path, [])
    response = _started(service)
    wrong = ModelResponse(
        request_id=response.request_id,
        provider="other-provider",
        model=response.model,
        content=response.content,
        finish_reason="stop",
        created_at=response.created_at,
    )
    with pytest.raises(ConversationStateError):
        service.reject_generation(
            turn_id="turn-1",
            request_id="request-1",
            response=wrong,
            rejected_at="2026-07-26T00:00:05Z",
            failure_code="response_validation_rejected",
            allow_repair=True,
        )
