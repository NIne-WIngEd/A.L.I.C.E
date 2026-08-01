import pytest

from cognitive_kernel import (
    AuthorityDecision,
    AuthorityRequest,
    CognitiveKernelContractError,
    SpeakerContext,
    required_authority_for,
)
from speaker_guest_authority_helpers import guest_context, grant, host_context, request, scope


def test_private_data_requires_stronger_authentication():
    context = host_context()
    value = request(context, "private_memory_read")
    assert value.required_authority == "host_verified"
    assert value.stronger_authentication_required is True
    approved = AuthorityDecision.create(
        decision_key="approve-private",
        request=value,
        speaker_context=context,
        decided_at="2026-08-01T20:11:00Z",
        decision="approve",
        evidence_class="stronger_authentication",
        granted_authority="host_verified",
        reason_codes=("host_verified",),
    )
    assert approved.decision == "approve"


def test_voice_only_cannot_approve_privileged_action():
    context = SpeakerContext.create(
        context_key="voice",
        scope=scope(),
        observed_at="2026-08-01T20:00:00Z",
        speaker_state="recognized",
        trust_state="host_context_recognized",
        evidence_class="local_voice_match",
        confidence=0.9,
        authority_ceiling="host_context",
    )
    value = request(context, "private_file_access")
    with pytest.raises(CognitiveKernelContractError):
        AuthorityDecision.create(
            decision_key="bad-voice-approval",
            request=value,
            speaker_context=context,
            decided_at="2026-08-01T20:11:00Z",
            decision="approve",
            evidence_class="local_voice_match",
            granted_authority="host_verified",
        )


def test_guest_approval_requires_exact_active_grant():
    context = guest_context()
    value_grant = grant()
    value = request(
        context,
        "media_control",
        mission_id="mission-demo",
        guest_grant_id=value_grant.grant_id,
    )
    approved = AuthorityDecision.create(
        decision_key="approve-guest",
        request=value,
        speaker_context=context,
        decided_at="2026-08-01T21:00:00Z",
        decision="approve",
        evidence_class="explicit_guest_grant",
        granted_authority="guest_scoped",
        guest_grant=value_grant,
    )
    assert approved.guest_grant_id == value_grant.grant_id


def test_guest_cannot_use_grant_for_unapproved_capability():
    context = guest_context()
    value_grant = grant()
    value = request(
        context,
        "general_questions",
        mission_id="mission-demo",
        guest_grant_id=value_grant.grant_id,
    )
    with pytest.raises(CognitiveKernelContractError):
        AuthorityDecision.create(
            decision_key="bad-guest",
            request=value,
            speaker_context=context,
            decided_at="2026-08-01T21:00:00Z",
            decision="approve",
            evidence_class="explicit_guest_grant",
            granted_authority="guest_scoped",
            guest_grant=value_grant,
        )


def test_owner_only_capability_requires_explicit_owner_approval():
    context = host_context()
    value = request(context, "production_promotion")
    assert required_authority_for("production_promotion") == "owner_verified"
    with pytest.raises(CognitiveKernelContractError):
        AuthorityDecision.create(
            decision_key="bad-owner",
            request=value,
            speaker_context=context,
            decided_at="2026-08-01T20:11:00Z",
            decision="approve",
            evidence_class="stronger_authentication",
            granted_authority="owner_verified",
        )
    approved = AuthorityDecision.create(
        decision_key="owner-approved",
        request=value,
        speaker_context=context,
        decided_at="2026-08-01T20:12:00Z",
        decision="approve",
        evidence_class="owner_explicit_approval",
        granted_authority="owner_verified",
    )
    assert approved.decision == "approve"


def test_denial_grants_no_authority():
    context = host_context()
    value = request(context, "messaging")
    denied = AuthorityDecision.create(
        decision_key="deny-message",
        request=value,
        speaker_context=context,
        decided_at="2026-08-01T20:11:00Z",
        decision="deny",
        evidence_class="none",
        granted_authority="none",
        reason_codes=("not_approved",),
    )
    assert denied.granted_authority == "none"


def test_authority_request_rejects_guest_without_grant_id():
    context = guest_context()
    with pytest.raises(CognitiveKernelContractError):
        AuthorityRequest.create(
            request_key="missing-grant",
            scope=context.scope,
            requested_at="2026-08-01T20:10:00Z",
            actor_context_id=context.context_id,
            actor_trust_state=context.trust_state,
            capability="media_control",
        )
