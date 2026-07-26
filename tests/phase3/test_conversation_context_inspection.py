from __future__ import annotations

from pathlib import Path

from alice_conversation.context_assembly import assemble_conversation_context
from alice_conversation.context_inspection import (
    inspect_conversation_context,
    render_conversation_context_inspection,
)
from _context_helpers import command, context_policy, make_orchestrator


def test_context_inspection_is_metadata_only(tmp_path: Path) -> None:
    orchestrator, store, _, _ = make_orchestrator(tmp_path)
    orchestrator.run_turn(command(1, content="private user words"))
    assembly = assemble_conversation_context(store, session_id="session-1", policy=orchestrator.context_policy)
    inspection = inspect_conversation_context(assembly, policy=orchestrator.context_policy)
    rendered = render_conversation_context_inspection(inspection)
    assert inspection.roles == ("user", "assistant")
    assert "private user words" not in rendered
    assert "User message" not in rendered
    assert "session-1" not in rendered
    assert "turn-1" not in rendered
    assert "user-1" not in rendered
    assert "assistant-1" not in rendered
    assert assembly.context_sha256 in rendered


def test_context_inspection_reports_truncation(tmp_path: Path) -> None:
    policy = context_policy(max_prior_turns=1, max_prior_messages=2)
    orchestrator, store, _, _ = make_orchestrator(tmp_path, selected_context_policy=policy)
    orchestrator.run_turn(command(1))
    orchestrator.run_turn(command(2))
    assembly = assemble_conversation_context(store, session_id="session-1", policy=policy)
    inspection = inspect_conversation_context(assembly, policy=policy)
    assert inspection.eligible_turn_count == 2
    assert inspection.included_turn_count == 1
    assert inspection.omitted_turn_count == 1
    assert inspection.truncated is True
