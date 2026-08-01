from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    ExperienceEvent,
    ProductHostScope,
    ProvenanceReference,
    canonical_sha256,
)


def scope(product: str, host: str) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product,
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain="host-private",
    )


def provenance() -> ProvenanceReference:
    return ProvenanceReference.create(
        provenance_type="derived_inference",
        source_reference_ids=("source-ref-1",),
        derivation_activity_id="derivation-1",
        responsible_component="synthetic-evaluator",
        confidence=0.8,
    )


def event(product: str = "alice", host: str = "host-a") -> ExperienceEvent:
    return ExperienceEvent.create(
        event_type="conversation.outcome",
        scope=scope(product, host),
        occurred_at="2026-08-01T07:00:00Z",
        content_digest=canonical_sha256({"synthetic": True}),
        provenance=provenance(),
        retention_class="ordinary_experience",
        storage_tier="ledger",
        deletion_lineage=("deletion-lineage-1",),
        parent_event_ids=("parent-event-1",),
        outcome_reference_ids=("outcome-1",),
        policy_bindings=("foundation-policy-1",),
        payload_reference="blob-ref-1",
    )


def test_event_is_tamper_evident_and_round_trips() -> None:
    original = event()
    restored = ExperienceEvent.from_metadata_record(
        original.metadata_record()
    )
    assert restored == original
    assert restored.event_id.startswith("experience-")
    assert len(restored.event_sha256) == 64

    with pytest.raises(CognitiveKernelContractError):
        replace(original, content_digest="0" * 64).validate()


def test_event_identity_is_host_and_product_scoped() -> None:
    alice = event("alice", "host-a")
    friday = event("friday", "host-a")
    other_host = event("alice", "host-b")
    assert alice.event_id != friday.event_id
    assert alice.event_id != other_host.event_id
    assert friday.event_id != other_host.event_id


def test_event_record_contains_no_raw_payload() -> None:
    record = event().metadata_record()
    assert "payload" not in record
    assert "content" not in record
    assert record["payload_reference"] == "blob-ref-1"


def test_deleted_event_cannot_keep_payload_reference() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ExperienceEvent.create(
            event_type="storage.deletion",
            scope=scope("alice", "host-a"),
            occurred_at="2026-08-01T07:00:00Z",
            content_digest="a" * 64,
            provenance=provenance(),
            retention_class="ordinary_experience",
            storage_tier="deleted",
            payload_reference="blob-ref-1",
        )


def test_quarantine_retention_requires_quarantine_tier() -> None:
    with pytest.raises(CognitiveKernelContractError):
        ExperienceEvent.create(
            event_type="security.quarantine",
            scope=scope("alice", "host-a"),
            occurred_at="2026-08-01T07:00:00Z",
            content_digest="a" * 64,
            provenance=provenance(),
            retention_class="quarantine",
            storage_tier="ledger",
        )
