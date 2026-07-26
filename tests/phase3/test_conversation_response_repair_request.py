from __future__ import annotations

from dataclasses import replace

import pytest

from alice_conversation.contracts import (
    ConversationCapabilities,
    ConversationMessage,
    ModelRequest,
    ModelResponse,
    sha256_text,
)
from alice_conversation.repair_policy import ConversationResponseRepairPolicy
from alice_conversation.response_repair import (
    ConversationResponseRepairError,
    build_conversation_response_repair_request,
)
from alice_conversation.response_validation import validate_conversation_response
from alice_conversation.response_validation_policy import load_conversation_response_validation_policy

from _repair_helpers import POLICIES, enabled_repair_policy

INVALID = "I completed the task for you."


def request():
    message = ConversationMessage.create(
        message_id="message-1",
        turn_id="turn-1",
        role="user",
        content="Please answer.",
        created_at="2026-07-26T00:00:00Z",
        data_classification="PRIVATE",
    )
    return ModelRequest(
        request_id="request-1",
        session_id="session-1",
        turn_id="turn-1",
        system_contract_version="contract-v1",
        system_contract="Answer truthfully without tools or external actions.",
        messages=(message,),
        grounding=None,
        capabilities=ConversationCapabilities(),
        max_output_tokens=1024,
        temperature=0.0,
    )


def rejected(original=None):
    original = original or request()
    response = ModelResponse(
        request_id=original.request_id,
        provider="test-provider",
        model="test-model",
        content=INVALID,
        finish_reason="stop",
        created_at="2026-07-26T00:00:01Z",
    )
    policy = load_conversation_response_validation_policy(
        POLICIES / "conversation_response_validation_policy.json"
    )
    report = validate_conversation_response(response=response, grounding=None, policy=policy)
    assert report.outcome == "rejected"
    return response, report


def test_repair_request_is_deterministic_and_sanitized():
    original = request()
    response, report = rejected(original)
    policy = enabled_repair_policy()
    first = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="1" * 64,
    )
    second = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="1" * 64,
    )
    assert first == second
    assert first.issue_codes == ("fabricated_action_completion",)
    assert INVALID not in first.request.system_contract
    assert "fabricated_action_completion" in first.request.system_contract
    assert first.request.messages == original.messages
    assert first.request.grounding == original.grounding
    assert first.request.capabilities == original.capabilities
    assert first.request.request_id == f"repair-request:{first.repair_request_sha256}"
    assert first.generation_id == f"repair-generation:{first.repair_request_sha256}"


def test_repair_digest_changes_with_context_digest():
    original = request()
    response, report = rejected(original)
    policy = enabled_repair_policy()
    one = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="1" * 64,
    )
    two = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="2" * 64,
    )
    assert one.repair_request_sha256 != two.repair_request_sha256


def test_disabled_policy_cannot_build_repair_request():
    original = request()
    response, report = rejected(original)
    with pytest.raises(ConversationResponseRepairError):
        build_conversation_response_repair_request(
            original_request=original,
            rejected_response=response,
            validation_report=report,
            policy=ConversationResponseRepairPolicy.disabled(),
            context_sha256="1" * 64,
        )


def test_accepted_validation_report_cannot_trigger_repair():
    original = request()
    response = ModelResponse(
        request_id=original.request_id,
        provider="test-provider",
        model="test-model",
        content="Here is a concise answer.",
        finish_reason="stop",
        created_at="2026-07-26T00:00:01Z",
    )
    validation = load_conversation_response_validation_policy(
        POLICIES / "conversation_response_validation_policy.json"
    )
    report = validate_conversation_response(response=response, grounding=None, policy=validation)
    with pytest.raises(ConversationResponseRepairError):
        build_conversation_response_repair_request(
            original_request=original,
            rejected_response=response,
            validation_report=report,
            policy=enabled_repair_policy(),
            context_sha256="1" * 64,
        )


def test_response_identity_mismatch_fails_closed():
    original = request()
    response, report = rejected(original)
    bad = replace(response, request_id="other-request")
    with pytest.raises(ConversationResponseRepairError):
        build_conversation_response_repair_request(
            original_request=original,
            rejected_response=bad,
            validation_report=report,
            policy=enabled_repair_policy(),
            context_sha256="1" * 64,
        )


def test_repair_request_validation_rejects_changed_context():
    original = request()
    response, report = rejected(original)
    policy = enabled_repair_policy()
    repair = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="1" * 64,
    )
    other_message = replace(original.messages[0], content="Different", content_sha256=sha256_text("Different"))
    changed = replace(repair, request=replace(repair.request, messages=(other_message,)))
    with pytest.raises(ConversationResponseRepairError):
        changed.validate(original_request=original, policy=policy)


def test_repair_request_validation_rejects_changed_capabilities():
    original = request()
    response, report = rejected(original)
    policy = enabled_repair_policy()
    repair = build_conversation_response_repair_request(
        original_request=original,
        rejected_response=response,
        validation_report=report,
        policy=policy,
        context_sha256="1" * 64,
    )
    changed_caps = replace(original.capabilities, web_access_allowed=True)
    changed = replace(repair, request=replace(repair.request, capabilities=changed_caps))
    with pytest.raises(Exception):
        changed.validate(original_request=original, policy=policy)
