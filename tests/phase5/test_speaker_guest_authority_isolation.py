import pytest

from cognitive_kernel import AuthorityDecision, CognitiveKernelContractError
from speaker_guest_authority_helpers import guest_context, grant, request, scope


def test_authority_decision_rejects_cross_host_speaker_context():
    context = guest_context()
    value_grant = grant()
    value = request(
        context,
        "media_control",
        mission_id="mission-demo",
        guest_grant_id=value_grant.grant_id,
    )
    other_context = guest_context(
        scope=scope(product="friday", host="host-b")
    )
    with pytest.raises(CognitiveKernelContractError):
        AuthorityDecision.create(
            decision_key="cross-host",
            request=value,
            speaker_context=other_context,
            decided_at="2026-08-01T21:00:00Z",
            decision="approve",
            evidence_class="explicit_guest_grant",
            granted_authority="guest_scoped",
            guest_grant=value_grant,
        )


def test_authority_decision_rejects_cross_host_guest_grant():
    context = guest_context()
    other_grant = grant(scope=scope(product="friday", host="host-b"))
    value = request(
        context,
        "media_control",
        mission_id="mission-demo",
        guest_grant_id=other_grant.grant_id,
    )
    with pytest.raises(CognitiveKernelContractError):
        AuthorityDecision.create(
            decision_key="cross-host-grant",
            request=value,
            speaker_context=context,
            decided_at="2026-08-01T21:00:00Z",
            decision="approve",
            evidence_class="explicit_guest_grant",
            granted_authority="guest_scoped",
            guest_grant=other_grant,
        )
