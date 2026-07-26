from __future__ import annotations

import pytest

from alice_conversation.contracts import ConversationCapabilities
from alice_conversation.orchestration import (
    ConversationOrchestrationError,
    ConversationTurnCommand,
)
from alice_conversation.state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)

from _orchestration_helpers import grounding_packet, make_orchestrator


def command(*, grounding=None, suffix: str = "1") -> ConversationTurnCommand:
    return ConversationTurnCommand(
        session_id="session-1",
        turn_id=f"turn-{suffix}",
        user_message_id=f"user-{suffix}",
        assistant_message_id=f"assistant-{suffix}",
        request_id=f"request-{suffix}",
        generation_id=f"generation-{suffix}",
        provider="deterministic-test",
        model="orchestration-v1",
        user_content="Explain the current state.",
        grounding=grounding,
    )


def test_successful_turn_records_one_generation_and_assistant_message(tmp_path):
    orchestrator, store, _, model, _, _ = make_orchestrator(tmp_path)
    result = orchestrator.run_turn(command())

    assert result.replayed is False
    assert result.assistant_message.content == "Grounded response."
    assert model.calls == 1
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    turn = inspection.turns[0]
    assert turn.status == "completed"
    assert [message.role for message in turn.messages] == ["user", "assistant"]
    assert len(turn.generations) == 1
    assert turn.generations[0].status == "completed"
    assert turn.generations[0].provider == "deterministic-test"
    assert turn.generations[0].model == "orchestration-v1"
    assert turn.generations[0].reasoning_status == "not_persisted"
    assert verify_conversation_session_integrity(
        store, session_id="session-1"
    ).valid


def test_grounding_is_attached_and_only_metadata_references_are_persisted(tmp_path):
    orchestrator, store, _, model, _, _ = make_orchestrator(tmp_path)
    packet = grounding_packet()
    result = orchestrator.run_turn(command(grounding=packet))

    assert result.grounding_packet_id == packet.packet_id
    assert result.grounding_packet_sha256 is not None
    request = model.requests[0]
    assert request.grounding == packet
    assert "BEGIN UNTRUSTED GROUNDING DATA" in request.grounding.render_for_model()
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    turn = inspection.turns[0]
    assert turn.grounding_packet_id == packet.packet_id
    assert len(turn.references) == 1
    assert turn.references[0].source_ref == "memory-1"
    assert turn.references[0].citation_token == "[memory:memory-1]"
    assert turn.references[0].content_sha256 is None


def test_model_request_uses_compiled_constitutional_contract(tmp_path):
    orchestrator, _, _, model, _, _ = make_orchestrator(tmp_path)
    orchestrator.run_turn(command())

    request = model.requests[0]
    assert request.system_contract_version == orchestrator.system_contract.version
    assert request.system_contract == orchestrator.system_contract.content
    assert request.capabilities == ConversationCapabilities()
    request.capabilities.validate()
    assert request.max_output_tokens == 1024
    assert request.temperature == 0.0
    assert request.messages[-1].role == "user"


def test_repeating_the_same_completed_command_is_idempotent(tmp_path):
    orchestrator, store, _, model, _, _ = make_orchestrator(tmp_path)
    selected = command()
    first = orchestrator.run_turn(selected)
    second = orchestrator.run_turn(selected)

    assert first.replayed is False
    assert second.replayed is True
    assert second.assistant_message == first.assistant_message
    assert model.calls == 1
    inspection = inspect_conversation_session(
        store, session_id="session-1", include_content=True
    )
    assert len(inspection.turns[0].messages) == 2
    assert len(inspection.turns[0].generations) == 1


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("assistant_message_id", "assistant-other"),
        ("request_id", "request-other"),
        ("generation_id", "generation-other"),
        ("provider", "provider-other"),
        ("model", "model-other"),
        ("user_content", "Different content."),
    ],
)
def test_completed_turn_rejects_conflicting_idempotency_keys(
    tmp_path, field, replacement
):
    orchestrator, _, _, model, _, _ = make_orchestrator(tmp_path)
    selected = command()
    orchestrator.run_turn(selected)
    conflicting = ConversationTurnCommand(
        **{**selected.__dict__, field: replacement}
    )

    with pytest.raises(ConversationOrchestrationError) as raised:
        orchestrator.run_turn(conflicting)
    assert raised.value.failure_code == "idempotency_conflict"
    assert model.calls == 1


def test_nonexistent_session_fails_before_model_generation(tmp_path):
    orchestrator, _, _, model, _, _ = make_orchestrator(tmp_path)
    missing = ConversationTurnCommand(
        **{**command().__dict__, "session_id": "missing-session"}
    )
    with pytest.raises(Exception):
        orchestrator.run_turn(missing)
    assert model.calls == 0
