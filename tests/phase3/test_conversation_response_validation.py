from __future__ import annotations

import pytest

from alice_conversation.response_validation import (
    ConversationResponseRejectedError,
    conversation_response_validation_report_sha256,
    validate_conversation_response,
)
from alice_conversation.response_validation_policy import (
    load_conversation_response_validation_policy,
)

from _response_validation_helpers import (
    answerable_packet,
    conflict_packet,
    empty_packet,
    response,
    uncertain_packet,
)


@pytest.fixture
def policy():
    return load_conversation_response_validation_policy()


def issue_codes(report):
    return {issue.code for issue in report.issues}


def test_ungrounded_nonfactual_response_is_accepted(policy):
    report = validate_conversation_response(
        response=response("Here is a concise explanation."),
        grounding=None,
        policy=policy,
    )
    assert report.outcome == "accepted"
    assert report.issues == ()
    assert report.grounding_packet_id is None
    assert report.cited_claim_ids == ()


def test_answerable_grounding_requires_exact_citation(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response(
            "Rayan prefers exact deterministic workflows. [memory:claim-1]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "accepted"
    assert report.cited_claim_ids == ("claim-1",)
    assert len(report.cited_token_sha256) == 1


def test_answerable_grounding_without_citation_is_rejected(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response("Rayan prefers exact deterministic workflows."),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "missing_required_citation" in issue_codes(report)
    assert "ungrounded_personal_claim" in issue_codes(report)


def test_altered_citation_token_is_rejected(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response(
            "Rayan prefers exact deterministic workflows. [memory:claim-other]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "unknown_citation_token" in issue_codes(report)
    assert "missing_required_citation" in issue_codes(report)


def test_citation_without_grounding_is_rejected(policy):
    report = validate_conversation_response(
        response=response("A claim appears here. [memory:claim-1]"),
        grounding=None,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "citation_without_grounding" in issue_codes(report)
    assert "unknown_citation_token" in issue_codes(report)


def test_supported_paraphrase_with_exact_token_is_accepted(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response(
            "The evidence indicates a preference for deterministic workflows. "
            "[memory:claim-1]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "accepted"


def test_unsupported_factual_sentence_is_rejected(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response(
            "Rayan prefers exact deterministic workflows. [memory:claim-1] "
            "The project passed 900 tests."
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "unsupported_factual_claim" in issue_codes(report)


def test_conflict_requires_two_distinct_claim_citations(policy):
    packet = conflict_packet()
    report = validate_conversation_response(
        response=response(
            "The records conflict: one gives August 1, 2026. [memory:claim-a]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "insufficient_conflict_citations" in issue_codes(report)


def test_conflict_with_two_citations_and_visible_conflict_is_accepted(policy):
    packet = conflict_packet()
    report = validate_conversation_response(
        response=response(
            "The records conflict. One gives August 1, 2026. [memory:claim-a] "
            "The other gives August 8, 2026. [memory:claim-b]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "accepted"
    assert report.cited_claim_ids == ("claim-a", "claim-b")


def test_conflict_without_conflict_language_is_rejected(policy):
    packet = conflict_packet()
    report = validate_conversation_response(
        response=response(
            "One record gives August 1, 2026. [memory:claim-a] "
            "Another gives August 8, 2026. [memory:claim-b]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert "conflict_not_preserved" in issue_codes(report)


def test_conflict_with_false_certainty_is_rejected(policy):
    packet = conflict_packet()
    report = validate_conversation_response(
        response=response(
            "The records conflict, but the date is definitely August 1, 2026. "
            "[memory:claim-a] [memory:claim-b]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert "false_certainty_on_conflict" in issue_codes(report)


def test_uncertain_grounding_requires_uncertainty_language(policy):
    packet = uncertain_packet()
    report = validate_conversation_response(
        response=response(
            "The application is under review. [memory:claim-u]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "uncertainty_not_preserved" in issue_codes(report)


def test_uncertain_grounding_with_caveat_is_accepted(policy):
    packet = uncertain_packet()
    report = validate_conversation_response(
        response=response(
            "The application may still be under review. [memory:claim-u]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "accepted"


def test_uncertain_grounding_with_false_certainty_is_rejected(policy):
    packet = uncertain_packet()
    report = validate_conversation_response(
        response=response(
            "The status is uncertain, but it is definitely under review. "
            "[memory:claim-u]"
        ),
        grounding=packet,
        policy=policy,
    )
    assert "false_certainty_on_uncertainty" in issue_codes(report)


def test_insufficient_evidence_requires_abstention(policy):
    packet = empty_packet("insufficient_evidence")
    report = validate_conversation_response(
        response=response("The answer is yes."),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "rejected"
    assert "missing_insufficient_evidence_abstention" in issue_codes(report)


def test_insufficient_evidence_abstention_is_recorded(policy):
    packet = empty_packet("insufficient_evidence")
    report = validate_conversation_response(
        response=response("I cannot determine this because there is insufficient evidence."),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "abstained"
    assert report.issues == ()


def test_denied_grounding_requires_refusal(policy):
    packet = empty_packet("denied")
    report = validate_conversation_response(
        response=response("Here is the requested answer."),
        grounding=packet,
        policy=policy,
    )
    assert "missing_denial_abstention" in issue_codes(report)


def test_denied_grounding_refusal_is_abstained(policy):
    packet = empty_packet("denied")
    report = validate_conversation_response(
        response=response("I cannot comply with that request."),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "abstained"


def test_not_applicable_requires_explicit_marker(policy):
    packet = empty_packet("not_applicable")
    report = validate_conversation_response(
        response=response("There is no answer."),
        grounding=packet,
        policy=policy,
    )
    assert "missing_not_applicable_abstention" in issue_codes(report)


def test_not_applicable_marker_is_abstained(policy):
    packet = empty_packet("not_applicable")
    report = validate_conversation_response(
        response=response("This is not applicable to the current request."),
        grounding=packet,
        policy=policy,
    )
    assert report.outcome == "abstained"


def test_truncated_response_is_rejected(policy):
    report = validate_conversation_response(
        response=response("Partial response", finish_reason="length"),
        grounding=None,
        policy=policy,
    )
    assert "truncated_response" in issue_codes(report)


def test_report_digest_is_deterministic(policy):
    packet = answerable_packet()
    selected_response = response(
        "Rayan prefers exact deterministic workflows. [memory:claim-1]"
    )
    first = validate_conversation_response(
        response=selected_response, grounding=packet, policy=policy
    )
    second = validate_conversation_response(
        response=selected_response, grounding=packet, policy=policy
    )
    assert first == second
    assert conversation_response_validation_report_sha256(first) == (
        conversation_response_validation_report_sha256(second)
    )


def test_rejected_error_requires_rejected_report(policy):
    packet = answerable_packet()
    report = validate_conversation_response(
        response=response("Unsupported answer."),
        grounding=packet,
        policy=policy,
    )
    raised = ConversationResponseRejectedError(report)
    assert raised.report == report
