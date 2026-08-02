from __future__ import annotations

from pathlib import Path

from cognitive_kernel import (
    ExperienceEvent,
    ProductHostScope,
    ProvenanceReference,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CREATED_AT = "2026-08-02T16:00:00Z"
COMMITTED_AT = "2026-08-02T16:01:00Z"


def make_scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "owner-host",
    schema_version: str = "1.0.0",
    encryption_domain: str = "owner-domain",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version=schema_version,
        encryption_domain=encryption_domain,
    )


def make_provenance() -> ProvenanceReference:
    return ProvenanceReference.create(
        provenance_type="derived_inference",
        source_reference_ids=("source-1",),
        derivation_activity_id="activity-1",
        responsible_component="kernel-test",
        model_id="model-1",
        confidence=0.9,
    )


def make_event(
    *,
    event_type: str = "tool-result",
    content_digest: str = SHA_A,
    occurred_at: str = "2026-08-02T15:59:00Z",
    scope: ProductHostScope | None = None,
    storage_tier: str = "hot",
    payload_reference: str | None = "payload-ref-1",
) -> ExperienceEvent:
    return ExperienceEvent.create(
        event_type=event_type,
        scope=scope or make_scope(),
        occurred_at=occurred_at,
        content_digest=content_digest,
        provenance=make_provenance(),
        retention_class="ordinary_experience",
        storage_tier=storage_tier,
        policy_bindings=("storage-policy-1",),
        payload_reference=payload_reference,
    )


def paths(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    return repository, vault / "experience-ledger.sqlite3"
