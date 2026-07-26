from __future__ import annotations

import pytest

from alice_conversation.model import ConversationModelTimeoutError
from alice_conversation.orchestration import (
    ConversationResumeCommand,
    ConversationTurnFailedError,
    ConversationTurnValidationError,
)
from alice_conversation.repair_inspection import inspect_conversation_response_repair
from alice_conversation.state_inspection import inspect_conversation_session

from _repair_helpers import (
    SequenceMonotonic,
    build_runtime,
    command,
    enabled_repair_policy,
)

INVALID = "I completed the task for you."
VALID = "Here is a concise answer."


def test_rejected_initial_response_is_repaired_once(tmp_path):
    store, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    result = orchestrator.run_turn(command())
    assert result.response.content == VALID
    assert result.validation_outcome == "accepted"
    assert result.repair_attempted is True
    assert result.request_id.startswith("repair-request:")
    assert result.generation_id.startswith("repair-generation:")
    assert len(adapter.requests) == 2
    assert adapter.requests[0].messages == adapter.requests[1].messages
    assert adapter.requests[0].grounding == adapter.requests[1].grounding
    assert adapter.requests[0].capabilities == adapter.requests[1].capabilities
    assert adapter.provider == result.provider and adapter.model == result.model
    inspection = inspect_conversation_session(store, session_id="session-1", include_content=True)
    turn = inspection.turns[0]
    assert [g.attempt_index for g in turn.generations] == [0, 1]
    assert turn.generations[0].status == "failed"
    assert turn.generations[0].validation_outcome == "rejected"
    assert turn.generations[1].status == "completed"
    assert [m.content for m in turn.messages] == ["Please answer this request.", VALID]
    assert INVALID not in repr(inspection)


def test_disabled_repair_preserves_original_rejection_behavior(tmp_path):
    disabled = enabled_repair_policy()
    disabled = type(disabled).disabled()
    store, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID], repair_policy=disabled)
    with pytest.raises(ConversationTurnValidationError):
        orchestrator.run_turn(command())
    assert len(adapter.requests) == 1
    inspection = inspect_conversation_session(store, session_id="session-1", include_content=True)
    assert inspection.turns[0].status == "failed"
    assert len(inspection.turns[0].messages) == 1
    assert INVALID not in repr(inspection)


def test_second_rejection_fails_without_third_generation(tmp_path):
    store, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, INVALID])
    with pytest.raises(ConversationTurnValidationError) as exc:
        orchestrator.run_turn(command())
    assert exc.value.failure_code == "response_repair_exhausted"
    assert len(adapter.requests) == 2
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "failed"
    assert [g.validation_outcome for g in turn.generations] == ["rejected", "rejected"]
    assert len(turn.messages) == 1
    assert INVALID not in repr(turn)


def test_repair_timeout_fails_without_more_attempts(tmp_path):
    store, _, adapter, orchestrator = build_runtime(
        tmp_path,
        [INVALID, ConversationModelTimeoutError("timeout")],
    )
    with pytest.raises(ConversationTurnFailedError) as exc:
        orchestrator.run_turn(command())
    assert exc.value.failure_code == "model_timeout"
    assert len(adapter.requests) == 2
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "failed"
    assert len(turn.messages) == 1


def test_total_elapsed_budget_blocks_repair_before_second_call(tmp_path):
    policy = enabled_repair_policy(max_total_elapsed_seconds=1.0)
    store, _, adapter, orchestrator = build_runtime(
        tmp_path,
        [INVALID, VALID],
        repair_policy=policy,
        monotonic=SequenceMonotonic([0.0, 2.0]),
    )
    with pytest.raises(ConversationTurnFailedError) as exc:
        orchestrator.run_turn(command())
    assert exc.value.failure_code == "response_repair_timeout"
    assert len(adapter.requests) == 1
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "failed"


def test_total_output_budget_blocks_repair(tmp_path):
    policy = enabled_repair_policy(max_total_output_tokens=1024)
    store, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, VALID], repair_policy=policy)
    with pytest.raises(ConversationTurnFailedError) as exc:
        orchestrator.run_turn(command())
    assert exc.value.failure_code == "response_repair_budget"
    assert len(adapter.requests) == 1


def test_repaired_turn_replays_from_original_idempotency_key(tmp_path):
    _, _, adapter, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    first = orchestrator.run_turn(command())
    replay = orchestrator.run_turn(command())
    assert replay.replayed is True
    assert replay.response.content == first.response.content
    assert replay.repair_attempted is True
    assert replay.repair_request_sha256 == first.repair_request_sha256
    assert len(adapter.requests) == 2


def test_metadata_safe_repair_inspection(tmp_path):
    store, _, _, orchestrator = build_runtime(tmp_path, [INVALID, VALID])
    orchestrator.run_turn(command())
    inspected = inspect_conversation_response_repair(
        store,
        session_id="session-1",
        turn_id="turn-1",
        policy=orchestrator.repair_policy,
    )
    assert inspected.repair_attempted is True
    assert inspected.attempt_count == 2
    assert inspected.original_validation_outcome == "rejected"
    assert inspected.repair_validation_outcome == "accepted"
    assert inspected.same_provider_model is True
    assert INVALID not in repr(inspected)
    assert VALID not in repr(inspected)


def test_total_elapsed_budget_rejects_late_repair_response(tmp_path):
    policy = enabled_repair_policy(max_total_elapsed_seconds=1.0)
    store, _, adapter, orchestrator = build_runtime(
        tmp_path,
        [INVALID, VALID],
        repair_policy=policy,
        monotonic=SequenceMonotonic([0.0, 0.0, 2.0]),
    )
    with pytest.raises(ConversationTurnFailedError) as exc:
        orchestrator.run_turn(command())
    assert exc.value.failure_code == "response_repair_timeout"
    assert len(adapter.requests) == 2
    turn = inspect_conversation_session(store, session_id="session-1", include_content=True).turns[0]
    assert turn.status == "failed"
    assert len(turn.messages) == 1
