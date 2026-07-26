from __future__ import annotations

from pathlib import Path

from alice_conversation.contracts import ConversationGroundingPacket
from alice_conversation.context_assembly import (
    assemble_conversation_context,
    conversation_context_sha256,
)
from _context_helpers import command, context_policy, make_orchestrator


def test_first_turn_has_empty_context(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1))
    assert [(item.role, item.content) for item in model.requests[0].messages] == [
        ("user", "User message 1.")
    ]


def test_second_turn_receives_complete_prior_pair(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1))
    orchestrator.run_turn(command(2))
    assert [(item.role, item.content) for item in model.requests[1].messages] == [
        ("user", "User message 1."),
        ("assistant", "Acknowledged."),
        ("user", "User message 2."),
    ]


def test_context_preserves_chronological_pair_order(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    for index in range(1, 5):
        orchestrator.run_turn(command(index))
    assert [item.turn_id for item in model.requests[-1].messages] == [
        "turn-1", "turn-1", "turn-2", "turn-2", "turn-3", "turn-3", "turn-4"
    ]


def test_turn_budget_keeps_newest_contiguous_suffix(tmp_path: Path) -> None:
    policy = context_policy(max_prior_turns=2, max_prior_messages=4)
    orchestrator, _, _, model = make_orchestrator(
        tmp_path, selected_context_policy=policy
    )
    for index in range(1, 5):
        orchestrator.run_turn(command(index))
    assert [item.turn_id for item in model.requests[-1].messages] == [
        "turn-2", "turn-2", "turn-3", "turn-3", "turn-4"
    ]


def test_character_budget_never_slices_a_turn_pair(tmp_path: Path) -> None:
    policy = context_policy(max_prior_turns=4, max_prior_messages=8, max_prior_characters=1024)
    orchestrator, _, _, model = make_orchestrator(
        tmp_path, selected_context_policy=policy
    )
    orchestrator.run_turn(command(1, content="A" * 600))
    orchestrator.run_turn(command(2, content="B" * 600))
    orchestrator.run_turn(command(3))
    request = model.requests[-1]
    assert [item.turn_id for item in request.messages] == ["turn-2", "turn-2", "turn-3"]
    assert len(request.messages[0].content) == 600


def test_oversized_newest_pair_blocks_older_pair_selection(tmp_path: Path) -> None:
    policy = context_policy(max_prior_turns=4, max_prior_messages=8, max_prior_characters=1024)
    orchestrator, _, _, model = make_orchestrator(
        tmp_path, selected_context_policy=policy
    )
    orchestrator.run_turn(command(1, content="short"))
    orchestrator.run_turn(command(2, content="X" * 1100))
    orchestrator.run_turn(command(3))
    assert [item.turn_id for item in model.requests[-1].messages] == ["turn-3"]


def test_preview_assembly_reports_deterministic_digest(tmp_path: Path) -> None:
    orchestrator, store, _, _ = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1))
    orchestrator.run_turn(command(2))
    first = assemble_conversation_context(store, session_id="session-1", policy=orchestrator.context_policy)
    second = assemble_conversation_context(store, session_id="session-1", policy=orchestrator.context_policy)
    assert first == second
    assert first.context_sha256 == conversation_context_sha256(
        first.messages, policy_version=first.policy_version
    )
    assert first.included_turn_count == 2
    assert first.excluded_turn_count == 0


def test_abstained_completed_turn_is_eligible_context(tmp_path: Path) -> None:
    orchestrator, _, _, model = make_orchestrator(tmp_path)
    packet = ConversationGroundingPacket(
        packet_id="packet-empty",
        outcome="insufficient_evidence",
        claims=(),
        created_at="2026-07-26T06:00:00Z",
    )
    model.response_text = (
        "I cannot determine this because there is insufficient evidence."
    )
    result = orchestrator.run_turn(command(1, grounding=packet))
    assert result.validation_outcome == "abstained"
    model.response_text = "Acknowledged."
    orchestrator.run_turn(command(2))
    assert [item.turn_id for item in model.requests[-1].messages] == [
        "turn-1", "turn-1", "turn-2"
    ]
