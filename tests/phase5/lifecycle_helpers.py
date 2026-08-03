from __future__ import annotations

from pathlib import Path

from cognitive_kernel import (
    LifecycleDecision,
    ProductHostScope,
    RetentionBlockerRecord,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
CREATED_AT = "2026-08-03T04:30:00Z"
COMMITTED_AT = "2026-08-03T04:31:00Z"


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


def make_decision(
    *,
    decision_key: str = "decision-1",
    scope: ProductHostScope | None = None,
    content_digest: str = SHA_A,
    decision_type: str = "transition",
    current_tier: str = "hot",
    proposed_tier: str = "warm",
    decided_at: str = "2026-08-03T04:29:00Z",
    authority_level: str = "host_context",
    authority_decision_id: str | None = "authority-decision-1",
    parent_decision_id: str | None = None,
    outcome: str = "approved",
) -> LifecycleDecision:
    return LifecycleDecision.create(
        decision_key=decision_key,
        scope=scope or make_scope(),
        subject_reference="logical-record-1",
        content_digest=content_digest,
        decision_type=decision_type,
        current_tier=current_tier,
        proposed_tier=proposed_tier,
        retention_class="ordinary_experience",
        decided_at=decided_at,
        actor_id="learning-substrate",
        authority_level=authority_level,
        authority_decision_id=authority_decision_id,
        provenance_reference_id="provenance-1",
        parent_decision_id=parent_decision_id,
        reason_codes=("policy-review",),
        policy_bindings=("alice.storage-lifecycle",),
        outcome=outcome,
    )


def make_open_blocker(
    *,
    blocker_key: str = "blocker-1",
    scope: ProductHostScope | None = None,
    blocker_type: str = "active_project",
    authority_level: str = "host_context",
    authority_decision_id: str = "authority-decision-2",
) -> RetentionBlockerRecord:
    return RetentionBlockerRecord.create(
        blocker_key=blocker_key,
        scope=scope or make_scope(),
        subject_reference="logical-record-1",
        content_digest=SHA_A,
        blocker_type=blocker_type,
        state="open",
        opened_at="2026-08-03T04:28:00Z",
        recorded_at="2026-08-03T04:28:00Z",
        actor_id="retention-controller",
        authority_level=authority_level,
        authority_decision_id=authority_decision_id,
        evidence_reference_id="evidence-1",
        reason_codes=("dependency-active",),
        policy_bindings=("alice.storage-lifecycle",),
    )


def make_resolution(
    open_record: RetentionBlockerRecord,
) -> RetentionBlockerRecord:
    return RetentionBlockerRecord.create(
        blocker_key=open_record.blocker_key,
        scope=open_record.scope,
        subject_reference=open_record.subject_reference,
        content_digest=open_record.content_digest,
        blocker_type=open_record.blocker_type,
        state="resolved",
        opened_at=open_record.opened_at,
        recorded_at="2026-08-03T04:30:00Z",
        actor_id="retention-controller",
        authority_level="host_verified",
        authority_decision_id="authority-decision-3",
        evidence_reference_id="evidence-2",
        parent_record_id=open_record.blocker_record_id,
        reason_codes=("dependency-closed",),
        policy_bindings=("alice.storage-lifecycle",),
    )


def paths(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    return repository, vault / "lifecycle-journal.sqlite3"
