from __future__ import annotations

from itertools import count
from pathlib import Path

from alice_conversation.cli import _render_inspection
from alice_conversation.cli_policy import ConversationCliPolicy
from alice_conversation.cli_runtime import ConversationCliRuntime
from _context_helpers import RecordingModel, make_orchestrator


def cli_policy() -> ConversationCliPolicy:
    return ConversationCliPolicy(
        policy_name="alice_conversation_cli_policy",
        version="1.0.0",
        phase="3",
        milestone="P3.7",
        status="local_conversational_runtime",
        boundaries=(
            ("local_only", True),
            ("private_vault_required", True),
            ("repository_state_allowed", False),
            ("web_access_allowed", False),
            ("tool_calling_allowed", False),
            ("external_action_allowed", False),
            ("memory_write_allowed", False),
            ("memory_promotion_allowed", False),
            ("live_retrieval_allowed", False),
            ("hidden_reasoning_display_allowed", False),
            ("raw_database_identifiers_display_allowed", False),
            ("automatic_response_repair_allowed", False),
            ("provider_fallback_allowed", False),
        ),
        allowed_retentions=("session_only", "retained"),
        default_retention="session_only",
        allowed_providers=("ollama-local",),
        explicit_provider_required=True,
        explicit_model_required=True,
        prebuilt_grounding_file_allowed=True,
        commands=(":help", ":new", ":close", ":inspect", ":cancel", ":resume", ":grounding", ":exit"),
        max_input_chars=100000,
        max_grounding_file_bytes=1048576,
    )


def runtime(tmp_path: Path) -> tuple[ConversationCliRuntime, RecordingModel]:
    model = RecordingModel(provider="ollama-local", model="qwen3:8b")
    orchestrator, _, service, model = make_orchestrator(
        tmp_path, model=model, session_id="bootstrap-session"
    )
    # The helper creates a bootstrap session for direct orchestration tests. The CLI
    # creates its own globally unique private session and never reads the bootstrap.
    ids = count()
    value = ConversationCliRuntime(
        state_service=service,
        orchestrator=orchestrator,
        provider="ollama-local",
        model="qwen3:8b",
        policy=cli_policy(),
        clock=lambda: "2026-07-26T09:00:00Z",
        id_factory=lambda: f"id-{next(ids)}",
    )
    value.new_session("retained")
    return value, model


def test_cli_later_turn_uses_prior_completed_pair(tmp_path: Path) -> None:
    value, model = runtime(tmp_path)
    value.send("first private turn")
    value.send("second private turn")
    assert [message.content for message in model.requests[-1].messages] == [
        "first private turn",
        "Acknowledged.",
        "second private turn",
    ]


def test_cli_inspection_exposes_context_metadata_not_content_or_ids(tmp_path: Path) -> None:
    value, _ = runtime(tmp_path)
    value.send("private words must not render")
    inspection = value.inspect()
    assert inspection.context_turn_count == 1
    assert inspection.context_message_count == 2
    assert len(inspection.context_sha256) == 64

    output: list[str] = []
    _render_inspection(value, output.append)
    rendered = "\n".join(output)
    assert "private words must not render" not in rendered
    assert "session-id" not in rendered
    assert "turn-id" not in rendered
    assert inspection.context_sha256 in rendered
