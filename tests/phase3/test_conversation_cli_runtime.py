from __future__ import annotations

import pytest

from alice_conversation.cli_runtime import (
    ConversationCliError,
    ConversationCliValidationError,
)
from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    sha256_text,
)
from alice_conversation.state_inspection import inspect_conversation_session

from _cli_helpers import make_runtime


def grounding_packet(outcome="answerable"):
    if outcome in {"insufficient_evidence", "denied", "not_applicable"}:
        claims = ()
    else:
        text = "The project is in Phase 3."
        claims = (
            ConversationGroundingClaim(
                claim_id="claim-1",
                text=text,
                content_sha256=sha256_text(text),
                knowledge_status="verified_fact",
                confidence=1.0,
                data_classification="PRIVATE",
                citations=(
                    ConversationCitation(
                        citation_id="citation-1",
                        source_kind="phase1_source",
                        source_ref="source-1",
                        token="[phase1:source-1]",
                        data_classification="PRIVATE",
                    ),
                ),
            ),
        )
    packet = ConversationGroundingPacket(
        packet_id=f"packet-{outcome}",
        outcome=outcome,
        claims=claims,
        created_at="2026-07-26T00:00:00Z",
        max_classification="PRIVATE",
    )
    packet.validate()
    return packet


def test_runtime_completes_and_persists_valid_turn(tmp_path):
    runtime, model, store = make_runtime(tmp_path, ["Consider the next step."])
    output = runtime.send("Help me plan.")
    assert output.content == "Consider the next step."
    assert output.validation_outcome == "accepted"
    inspection = runtime.inspect()
    assert inspection.turn_count == 1
    assert inspection.message_count == 2
    assert inspection.generation_count == 1
    assert model.requests[0].capabilities.web_access_allowed is False


def test_runtime_renders_exact_used_citation_tokens(tmp_path):
    grounding = grounding_packet()
    runtime, _, _ = make_runtime(
        tmp_path,
        ["The project is in Phase 3. [phase1:source-1]"],
        grounding=grounding,
    )
    output = runtime.send("What phase is the project in?")
    assert output.citation_tokens == ("[phase1:source-1]",)
    assert output.validation_outcome == "accepted"


def test_runtime_validation_rejection_persists_no_assistant_message(tmp_path):
    runtime, _, store = make_runtime(tmp_path, ["I searched the web."])
    with pytest.raises(ConversationCliValidationError) as error:
        runtime.send("Search this.")
    assert "fabricated_capability_claim" in error.value.issue_codes
    session = runtime.inspect()
    assert session.turn_statuses == (("failed", 1),)
    assert session.message_count == 1
    raw = inspect_conversation_session(
        store,
        session_id=runtime._session_id,
        include_content=True,
    )
    assert tuple(message.role for message in raw.turns[0].messages) == ("user",)


@pytest.mark.parametrize("retention,purged", [("session_only", True), ("retained", False)])
def test_runtime_close_obeys_retention(tmp_path, retention, purged):
    runtime, _, _ = make_runtime(tmp_path, ["Consider the next step."], retention=retention)
    runtime.send("Hello")
    summary = runtime.close_session()
    assert summary.retention == retention
    assert summary.purged is purged
    assert summary.turn_count == 1
    assert runtime.has_session is False


def test_new_session_closes_previous_session(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    runtime.new_session("retained")
    inspection = runtime.inspect()
    assert inspection.retention == "retained"


@pytest.mark.parametrize("content", ["", "   "])
def test_runtime_rejects_empty_input(tmp_path, content):
    runtime, _, _ = make_runtime(tmp_path, [])
    with pytest.raises(ConversationCliError):
        runtime.send(content)


def test_runtime_rejects_oversized_input(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [])
    with pytest.raises(ConversationCliError):
        runtime.send("x" * (runtime.policy.max_input_chars + 1))


def test_inspection_never_returns_raw_ids_or_content(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, ["Consider the next step."])
    runtime.send("Private user text")
    inspection = runtime.inspect()
    fields = set(inspection.__dataclass_fields__)
    assert "session_id" not in fields
    assert "turn_id" not in fields
    assert "content" not in fields
    assert inspection.provider == "ollama-local"


def test_grounding_status_is_metadata_only(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, [], grounding=grounding_packet())
    status = runtime.grounding_status()
    assert status.enabled is True
    assert status.claim_count == 1
    assert not hasattr(status, "claims")
    assert not hasattr(status, "packet_id")
