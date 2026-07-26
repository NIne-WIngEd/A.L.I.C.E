from __future__ import annotations

import json

from alice_conversation.response_validation import validate_conversation_response
from alice_conversation.response_validation_inspection import (
    inspect_conversation_response_validation,
    render_conversation_response_validation_inspection,
)
from alice_conversation.response_validation_policy import (
    load_conversation_response_validation_policy,
)

from _response_validation_helpers import answerable_packet, response


def report():
    packet = answerable_packet()
    selected = validate_conversation_response(
        response=response(
            "Rayan prefers exact deterministic workflows. [memory:claim-1]"
        ),
        grounding=packet,
        policy=load_conversation_response_validation_policy(),
    )
    selected.validate()
    return selected


def test_inspection_exposes_only_metadata_and_digests():
    selected = inspect_conversation_response_validation(report())
    assert selected.outcome == "accepted"
    assert selected.issue_count == 0
    assert selected.cited_claim_count == 1
    assert selected.cited_token_count == 1
    assert len(selected.response_sha256) == 64
    assert len(selected.report_sha256) == 64


def test_rendered_inspection_is_deterministic_json():
    selected = inspect_conversation_response_validation(report())
    first = render_conversation_response_validation_inspection(selected)
    second = render_conversation_response_validation_inspection(selected)
    assert first == second
    payload = json.loads(first)
    assert payload["outcome"] == "accepted"
    assert payload["cited_claim_ids"] == ["claim-1"]


def test_inspection_does_not_expose_response_or_raw_citation_text():
    selected_report = report()
    rendered = render_conversation_response_validation_inspection(
        inspect_conversation_response_validation(selected_report)
    )
    assert "Rayan prefers exact deterministic workflows" not in rendered
    assert "[memory:claim-1]" not in rendered
    assert selected_report.response_sha256 in rendered
    assert selected_report.cited_token_sha256[0] in rendered


def test_rejected_inspection_exposes_issue_codes_without_sentence_text():
    packet = answerable_packet()
    selected_report = validate_conversation_response(
        response=response("Rayan prefers exact deterministic workflows."),
        grounding=packet,
        policy=load_conversation_response_validation_policy(),
    )
    inspection = inspect_conversation_response_validation(selected_report)
    rendered = render_conversation_response_validation_inspection(inspection)
    assert inspection.outcome == "rejected"
    assert inspection.issue_count >= 1
    assert "missing_required_citation" in inspection.issue_codes
    assert "Rayan prefers exact deterministic workflows" not in rendered
