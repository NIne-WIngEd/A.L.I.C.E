import pytest

from cognitive_kernel import CognitiveKernelContractError, GuestSession
from speaker_guest_authority_helpers import scope


def test_delegated_guest_session_requires_visible_grant_and_privacy_guards():
    session = GuestSession.create(
        session_key="living-room",
        scope=scope(),
        guest_reference_id="speaker-guest",
        mode="delegated_guest",
        status="active",
        started_at="2026-08-01T20:00:00Z",
        expires_at="2026-08-01T22:00:00Z",
        grant_ids=("guest-grant-1",),
    )
    assert session.private_views_hidden is True
    assert session.local_action_logging_required is True
    assert session.persistent_authority is False


def test_delegated_guest_session_without_grant_is_rejected():
    with pytest.raises(CognitiveKernelContractError):
        GuestSession.create(
            session_key="bad",
            scope=scope(),
            guest_reference_id="speaker-guest",
            mode="delegated_guest",
            status="active",
            started_at="2026-08-01T20:00:00Z",
            expires_at="2026-08-01T22:00:00Z",
        )


def test_guest_session_cannot_expose_private_views_or_persist_authority():
    base = dict(
        session_key="bad",
        scope=scope(),
        guest_reference_id="speaker-guest",
        mode="guest",
        status="active",
        started_at="2026-08-01T20:00:00Z",
        expires_at="2026-08-01T22:00:00Z",
    )
    with pytest.raises(CognitiveKernelContractError):
        GuestSession.create(**base, private_views_hidden=False)
    with pytest.raises(CognitiveKernelContractError):
        GuestSession.create(**base, persistent_authority=True)
