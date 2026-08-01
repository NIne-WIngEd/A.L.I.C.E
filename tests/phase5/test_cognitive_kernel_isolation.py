from __future__ import annotations

from cognitive_kernel import (
    ExperienceEvent,
    ProductHostScope,
    ProvenanceReference,
)


def make_event(product: str, host: str, domain: str) -> ExperienceEvent:
    scope = ProductHostScope.create(
        product_id=product,
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain=domain,
    )
    provenance = ProvenanceReference.create(
        provenance_type="evolved_identity",
        responsible_component="synthetic-host-runtime",
    )
    return ExperienceEvent.create(
        event_type="synthetic.host-event",
        scope=scope,
        occurred_at="2026-08-01T07:00:00Z",
        content_digest="b" * 64,
        provenance=provenance,
        retention_class="high_value_experience",
        storage_tier="ledger",
    )


def test_two_friday_hosts_never_share_event_identity() -> None:
    host_a = make_event("friday", "host-a", "private-a")
    host_b = make_event("friday", "host-b", "private-b")
    assert host_a.scope.storage_scope() != host_b.scope.storage_scope()
    assert host_a.event_id != host_b.event_id
    assert host_a.event_sha256 != host_b.event_sha256


def test_alice_and_friday_never_share_event_identity() -> None:
    alice = make_event("alice", "host-a", "private-a")
    friday = make_event("friday", "host-a", "private-a")
    assert alice.scope.storage_scope() != friday.scope.storage_scope()
    assert alice.event_id != friday.event_id
    assert alice.event_sha256 != friday.event_sha256
