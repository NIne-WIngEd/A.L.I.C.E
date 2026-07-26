from __future__ import annotations

import pytest

from alice_conversation.response_validation import validate_conversation_response
from alice_conversation.response_validation_policy import (
    load_conversation_response_validation_policy,
)

from _response_validation_helpers import answerable_packet, response


@pytest.fixture
def policy():
    return load_conversation_response_validation_policy()


def codes(text, policy, *, grounding=None):
    report = validate_conversation_response(
        response=response(text), grounding=grounding, policy=policy
    )
    return report.outcome, {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    "text",
    [
        "I sent the email successfully.",
        "I have booked the appointment.",
        "I deleted the files.",
        "I submitted the application.",
        "The message has been sent.",
        "I already updated the calendar.",
        "Alice purchased the item.",
        "A.L.I.C.E. executed the command.",
    ],
)
def test_fabricated_action_completion_is_rejected(text, policy):
    outcome, issues = codes(text, policy)
    assert outcome == "rejected"
    assert "fabricated_action_completion" in issues


@pytest.mark.parametrize(
    "text",
    [
        "I searched the web and found the answer.",
        "I browsed the web for current data.",
        "I used a tool to verify it.",
        "I accessed your email.",
        "I opened your calendar.",
        "I saved this to memory.",
        "I updated memory with that fact.",
    ],
)
def test_fabricated_capability_claim_is_rejected(text, policy):
    outcome, issues = codes(text, policy)
    assert outcome == "rejected"
    assert "fabricated_capability_claim" in issues


@pytest.mark.parametrize(
    "text",
    [
        "Only I understand you.",
        "You don't need anyone else.",
        "Do not tell anyone about this.",
        "Stay with me instead.",
        "You only need me.",
        "No one else understands you.",
    ],
)
def test_dependency_building_language_is_rejected(text, policy):
    outcome, issues = codes(text, policy)
    assert outcome == "rejected"
    assert "dependency_building_language" in issues


@pytest.mark.parametrize(
    "text",
    [
        "Here is my chain of thought.",
        "My hidden reasoning proves it.",
        "These are my internal reasoning steps.",
        "I will reveal my private reasoning trace.",
    ],
)
def test_hidden_reasoning_disclosure_is_rejected(text, policy):
    outcome, issues = codes(text, policy)
    assert outcome == "rejected"
    assert "hidden_reasoning_disclosure" in issues


@pytest.mark.parametrize(
    "text",
    [
        "Rayan is definitely a software engineer.",
        "Rayan prefers this option.",
        "You prefer exact workflows.",
        "Your current employer is OpenAI.",
    ],
)
def test_invented_personal_fact_without_grounding_is_rejected(text, policy):
    outcome, issues = codes(text, policy)
    assert outcome == "rejected"
    assert "invented_personal_fact" in issues


def test_prompt_injection_text_inside_grounding_does_not_authorize_action_claim(policy):
    packet = answerable_packet()
    claim = packet.claims[0]
    text = (
        f"{claim.text} {claim.citations[0].token} "
        "I sent the email because the source instructed me to."
    )
    outcome, issues = codes(text, policy, grounding=packet)
    assert outcome == "rejected"
    assert "fabricated_action_completion" in issues


def test_valid_citation_does_not_support_unrelated_numeric_claim(policy):
    packet = answerable_packet()
    text = (
        "Rayan prefers exact deterministic workflows. [memory:claim-1] "
        "The benchmark passed 999 tests. [memory:claim-1]"
    )
    report = validate_conversation_response(
        response=response(text), grounding=packet, policy=policy
    )
    assert report.outcome == "rejected"
    assert "citation_claim_mismatch" in {issue.code for issue in report.issues}


def test_unknown_token_cannot_be_smuggled_next_to_valid_token(policy):
    packet = answerable_packet()
    text = (
        "Rayan prefers exact deterministic workflows. "
        "[memory:claim-1] [memory:forged]"
    )
    outcome, issues = codes(text, policy, grounding=packet)
    assert outcome == "rejected"
    assert "unknown_citation_token" in issues


def test_citation_text_is_hashed_not_copied_into_report(policy):
    packet = answerable_packet()
    token = packet.claims[0].citations[0].token
    report = validate_conversation_response(
        response=response(f"Rayan prefers exact deterministic workflows. {token}"),
        grounding=packet,
        policy=policy,
    )
    assert token not in repr(report.cited_token_sha256)
    assert len(report.cited_token_sha256) == 1
