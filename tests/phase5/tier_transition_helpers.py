from __future__ import annotations

from pathlib import Path

from cognitive_kernel import (
    LifecycleDecision,
    ProductHostScope,
    RetentionBlockerRecord,
    open_lifecycle_journal,
    open_raw_buffer_store,
    open_tier_transition_store,
)

CREATED_AT = "2026-08-03T05:40:00Z"
CAPTURED_AT = "2026-08-03T05:41:00Z"
DECIDED_AT = "2026-08-03T05:42:00Z"
COMMITTED_AT = "2026-08-03T05:42:30Z"
EXECUTED_AT = "2026-08-03T05:43:00Z"
PAYLOAD = b"synthetic-host-sealed-tier-payload"


def make_scope(
    *,
    product_id: str = "alice",
    host_instance_id: str = "owner-host",
    encryption_domain: str = "owner-domain",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host_instance_id,
        schema_version="1.0.0",
        encryption_domain=encryption_domain,
    )


def make_paths(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    vault = tmp_path / "vault"
    vault.mkdir()
    return repository, vault


def open_raw_store(
    vault: Path,
    repository: Path,
    *,
    scope: ProductHostScope | None = None,
):
    return open_raw_buffer_store(
        vault / "raw-buffer",
        scope=scope or make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    )


def open_journal(
    vault: Path,
    repository: Path,
    *,
    scope: ProductHostScope | None = None,
):
    return open_lifecycle_journal(
        vault / "lifecycle.sqlite3",
        scope=scope or make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    )


def open_tier_store(
    vault: Path,
    repository: Path,
    *,
    scope: ProductHostScope | None = None,
):
    return open_tier_transition_store(
        vault / "tier-store",
        scope=scope or make_scope(),
        repository_root=repository,
        created_at=CREATED_AT,
    )


def capture_raw(
    store,
    *,
    payload: bytes = PAYLOAD,
    logical_record_id: str = "logical-record-1",
):
    return store.capture(
        payload,
        logical_record_id=logical_record_id,
        media_type="application/octet-stream",
        sensitivity_class="private",
        retention_class="ordinary_experience",
        captured_at=CAPTURED_AT,
        host_sealed=True,
    )


def make_decision(
    *,
    content_digest: str,
    scope: ProductHostScope | None = None,
    decision_key: str = "decision-1",
    current_tier: str = "raw_buffer",
    proposed_tier: str = "hot",
    decision_type: str = "transition",
    outcome: str = "approved",
    authority_level: str = "host_context",
    parent_decision_id: str | None = None,
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
        decided_at=DECIDED_AT,
        actor_id="retention-controller",
        authority_level=authority_level,
        authority_decision_id="authority-decision-1",
        provenance_reference_id="provenance-1",
        parent_decision_id=parent_decision_id,
        reason_codes=("policy-review",),
        policy_bindings=("alice.storage-lifecycle",),
        outcome=outcome,
    )


def make_open_blocker(
    *,
    content_digest: str,
    scope: ProductHostScope | None = None,
    blocker_key: str = "blocker-1",
) -> RetentionBlockerRecord:
    return RetentionBlockerRecord.create(
        blocker_key=blocker_key,
        scope=scope or make_scope(),
        subject_reference="logical-record-1",
        content_digest=content_digest,
        blocker_type="active_project",
        state="open",
        opened_at="2026-08-03T05:39:00Z",
        recorded_at="2026-08-03T05:39:00Z",
        actor_id="retention-controller",
        authority_level="host_context",
        authority_decision_id="authority-decision-2",
        evidence_reference_id="evidence-1",
        reason_codes=("dependency-active",),
        policy_bindings=("alice.storage-lifecycle",),
    )


def make_resolution(
    opened: RetentionBlockerRecord,
) -> RetentionBlockerRecord:
    return RetentionBlockerRecord.create(
        blocker_key=opened.blocker_key,
        scope=opened.scope,
        subject_reference=opened.subject_reference,
        content_digest=opened.content_digest,
        blocker_type=opened.blocker_type,
        state="resolved",
        opened_at=opened.opened_at,
        recorded_at="2026-08-03T05:41:30Z",
        actor_id="retention-controller",
        authority_level="host_verified",
        authority_decision_id="authority-decision-3",
        evidence_reference_id="evidence-2",
        parent_record_id=opened.blocker_record_id,
        reason_codes=("dependency-closed",),
        policy_bindings=("alice.storage-lifecycle",),
    )
