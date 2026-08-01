from dataclasses import replace

import pytest

from cognitive_kernel import CognitiveKernelContractError, GuestGrant
from speaker_guest_authority_helpers import grant, scope


def test_guest_grant_is_scoped_expiring_and_tamper_evident():
    value = grant()
    assert value.authorizes(
        "media_control", at="2026-08-01T21:00:00Z", mission_id="mission-demo"
    )
    assert not value.authorizes(
        "media_control", at="2026-08-01T23:00:00Z", mission_id="mission-demo"
    )
    with pytest.raises(CognitiveKernelContractError):
        replace(value, capabilities=("purchases",)).validate()


def test_guest_grant_requires_exact_mission_scope():
    value = grant()
    assert not value.authorizes(
        "timers", at="2026-08-01T21:00:00Z", mission_id="mission-other"
    )


def test_guest_grant_cannot_delegate_or_self_expand():
    with pytest.raises(CognitiveKernelContractError):
        grant(non_delegable=False)
    with pytest.raises(CognitiveKernelContractError):
        grant(self_expansion_allowed=True)


def test_revoked_guest_grant_is_inactive():
    value = grant(status="revoked", revoked_at="2026-08-01T20:30:00Z")
    assert not value.is_active_at("2026-08-01T21:00:00Z")


def test_cross_host_grant_scope_changes_identity():
    first = grant()
    second = grant(scope=scope(product="friday", host="host-b"))
    assert first.grant_id != second.grant_id
    assert first.grant_sha256 != second.grant_sha256
