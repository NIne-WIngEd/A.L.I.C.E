import hashlib

from cognitive_kernel import (
    AuthorityRequest,
    GuestGrant,
    ProductHostScope,
    SpeakerContext,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def scope(product: str = "alice", host: str = "host-a") -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product,
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain=f"{host}-private",
    )


def host_context(**overrides) -> SpeakerContext:
    values = dict(
        context_key="host-context",
        scope=scope(),
        observed_at="2026-08-01T20:00:00Z",
        speaker_state="recognized",
        trust_state="host_privilege_verified",
        evidence_class="stronger_authentication",
        confidence=0.99,
        speaker_reference_id="speaker-host",
        stronger_authentication_verified=True,
        authority_ceiling="host_verified",
        reason_codes=("local_authentication",),
    )
    values.update(overrides)
    return SpeakerContext.create(**values)


def guest_context(session_id: str = "guest-session-1", **overrides) -> SpeakerContext:
    values = dict(
        context_key="guest-context",
        scope=scope(),
        observed_at="2026-08-01T20:00:00Z",
        speaker_state="recognized",
        trust_state="delegated_guest_session",
        evidence_class="explicit_guest_grant",
        confidence=0.9,
        speaker_reference_id="speaker-guest",
        session_reference_id=session_id,
        authority_ceiling="guest_scoped",
        reason_codes=("guest_grant",),
    )
    values.update(overrides)
    return SpeakerContext.create(**values)


def grant(**overrides) -> GuestGrant:
    values = dict(
        grant_key="guest-grant-one",
        scope=scope(),
        guest_reference_id="speaker-guest",
        session_reference_id="guest-session-1",
        purpose_code="living-room-demo",
        capabilities=("media_control", "timers"),
        mission_ids=("mission-demo",),
        grantor_authority="host_verified",
        issued_at="2026-08-01T20:00:00Z",
        expires_at="2026-08-01T22:00:00Z",
        policy_bindings=("guest_grant.v1",),
    )
    values.update(overrides)
    return GuestGrant.create(**values)


def request(context: SpeakerContext, capability: str, **overrides) -> AuthorityRequest:
    values = dict(
        request_key=f"request-{capability}",
        scope=context.scope,
        requested_at="2026-08-01T20:10:00Z",
        actor_context_id=context.context_id,
        actor_trust_state=context.trust_state,
        capability=capability,
        reason_codes=("explicit_request",),
    )
    values.update(overrides)
    return AuthorityRequest.create(**values)
