from __future__ import annotations

import pytest

from alice_conversation.orchestration import (
    ConversationOrchestrationError,
    ConversationTurnCommand,
    ConversationTurnValidationError,
)
from alice_conversation.state_inspection import inspect_conversation_session

from _orchestration_helpers import RecordingModel, make_orchestrator
from _response_validation_helpers import answerable_packet, empty_packet


def command(*, grounding=None, suffix: str = "validation"):
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id=f"turn-{suffix}",
        user_message_id=f"user-{suffix}",
        assistant_message_id=f"assistant-{suffix}",
        request_id=f"request-{suffix}",
        generation_id=f"generation-{suffix}",
        provider="deterministic-test",
        model="orchestration-v1",
        user_content="Answer using the available evidence.",
        grounding=grounding,
    )


def test_accepted_validation_outcome_is_persisted(tmp_path):
    packet = answerable_packet()
    model = RecordingModel(
        response_text=(
            "Rayan prefers exact deterministic workflows. [memory:claim-1]"
        )
    )
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    result = orchestrator.run_turn(command(grounding=packet))
    assert result.validation_outcome == "accepted"
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "completed"
    assert turn.generations[0].validation_outcome == "accepted"
    assert [message.role for message in turn.messages] == ["user", "assistant"]


def test_abstained_validation_outcome_is_persisted(tmp_path):
    packet = empty_packet("insufficient_evidence")
    model = RecordingModel(
        response_text="I cannot determine this because there is insufficient evidence."
    )
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    result = orchestrator.run_turn(command(grounding=packet))
    assert result.validation_outcome == "abstained"
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "completed"
    assert turn.generations[0].validation_outcome == "abstained"
    assert len(turn.messages) == 2


def test_rejected_response_records_failure_without_assistant_message(tmp_path):
    packet = answerable_packet()
    model = RecordingModel(response_text="Rayan prefers exact deterministic workflows.")
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    with pytest.raises(ConversationTurnValidationError) as raised:
        orchestrator.run_turn(command(grounding=packet))
    assert raised.value.failure_code == "response_validation_rejected"
    assert raised.value.report.outcome == "rejected"
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert turn.status == "failed"
    assert turn.failure_code == "response_validation_rejected"
    assert len(turn.messages) == 1
    assert turn.messages[0].role == "user"
    assert turn.generations[0].status == "failed"
    assert turn.generations[0].validation_outcome == "not_evaluated"
    assert turn.generations[0].failure_code == "response_validation_rejected"
    assert turn.generations[0].response_sha256 is None


def test_fabricated_action_completion_is_rejected_before_commit(tmp_path):
    model = RecordingModel(response_text="I sent the email successfully.")
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    with pytest.raises(ConversationTurnValidationError) as raised:
        orchestrator.run_turn(command())
    assert "fabricated_action_completion" in {
        issue.code for issue in raised.value.report.issues
    }
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert [message.role for message in turn.messages] == ["user"]


def test_unknown_citation_is_rejected_before_commit(tmp_path):
    packet = answerable_packet()
    model = RecordingModel(
        response_text=(
            "Rayan prefers exact deterministic workflows. [memory:forged]"
        )
    )
    orchestrator, store, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    with pytest.raises(ConversationTurnValidationError) as raised:
        orchestrator.run_turn(command(grounding=packet))
    assert "unknown_citation_token" in {
        issue.code for issue in raised.value.report.issues
    }
    turn = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    ).turns[0]
    assert len(turn.messages) == 1


def test_repeating_rejected_turn_does_not_call_model_again(tmp_path):
    packet = answerable_packet()
    model = RecordingModel(response_text="Unsupported response.")
    orchestrator, _, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    selected = command(grounding=packet)
    with pytest.raises(ConversationTurnValidationError):
        orchestrator.run_turn(selected)
    with pytest.raises(ConversationOrchestrationError) as repeated:
        orchestrator.run_turn(selected)
    assert repeated.value.failure_code == "turn_failed"
    assert model.calls == 1


def test_replayed_completed_turn_preserves_validation_outcome(tmp_path):
    packet = answerable_packet()
    model = RecordingModel(
        response_text=(
            "Rayan prefers exact deterministic workflows. [memory:claim-1]"
        )
    )
    orchestrator, _, _, _, _, _ = make_orchestrator(tmp_path, model=model)
    selected = command(grounding=packet)
    first = orchestrator.run_turn(selected)
    second = orchestrator.run_turn(selected)
    assert first.validation_outcome == "accepted"
    assert second.validation_outcome == "accepted"
    assert second.replayed is True
    assert model.calls == 1


def test_orchestration_requires_enabled_final_validation_gate(tmp_path):
    orchestrator, _, _, _, _, _ = make_orchestrator(tmp_path)
    assert orchestrator.policy.lifecycle_value(
        "final_grounding_validation_enabled"
    ) is True
    assert orchestrator.response_validation_policy.milestone == "P3.6"
