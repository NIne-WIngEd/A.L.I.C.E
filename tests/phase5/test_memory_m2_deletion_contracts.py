"""Memory M2.6 deletion-contract tests."""

from dataclasses import replace

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.deletion_contracts import (
    DeletionPlaneReceipt,
    DeletionPropagationReceipt,
    RestoreFilterDecision,
)
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope

STAMP = "2026-08-06T22:50:00Z"
DIGEST = "a" * 64


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="owner-workstation",
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def envelope(
    receipt_id: str = "deletion-receipt-1",
) -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope.create(
        scope=scope(),
        record_id=receipt_id,
        record_type="deletion_propagation_receipt",
        authority_namespace_id="owner-memory",
        host_or_cluster_id="owner-workstation",
        authority_role="operational_workflow_state",
        deployment_profile="memory-m2-6",
        created_at=STAMP,
        valid_from=STAMP,
        valid_to=None,
        transaction_time=STAMP,
        logical_clock=1,
        causal_parents=(),
        source_records=("plane-1",),
        generation=1,
        state="completed",
        data_classification="owner_private",
        retention_class="owner_hold",
        deletion_state="deletion_rehearsal",
        provenance_digest=DIGEST,
        content_digest=DIGEST,
        writer="deletion-propagation-prototype",
        workflow_or_request_id="request-1",
        idempotency_namespace="memory-m2-6",
        idempotency_key="receipt-1",
        supersedes=(),
        superseded_by=(),
        rollback_reference=None,
    )


def plane(
    *,
    receipt_id: str = "plane-1",
    plane_kind: str = "claim_authority",
    component_id: str = "claim-store",
    state: str = "completed",
    completed_at: str | None = STAMP,
    target_count: int = 2,
    deleted_count: int = 2,
    blocked_count: int = 0,
    error_code: str | None = None,
) -> DeletionPlaneReceipt:
    return DeletionPlaneReceipt.create(
        plane_receipt_id=receipt_id,
        request_id="request-1",
        plane_kind=plane_kind,
        component_id=component_id,
        deletion_mode="tombstone",
        state=state,
        requested_at=STAMP,
        completed_at=completed_at,
        target_count=target_count,
        deleted_count=deleted_count,
        blocked_count=blocked_count,
        evidence_record_ids=("evidence-1",),
        error_code=error_code,
        result_content_digest=DIGEST,
    )


def test_plane_receipt_is_canonical_and_digest_bound() -> None:
    value = plane()
    value.validate()
    assert value.plane_receipt_sha256 == canonical_sha256(
        value.material_record()
    )


def test_completed_plane_must_cover_every_target() -> None:
    with pytest.raises(CognitiveKernelContractError):
        plane(deleted_count=1)


def test_failed_plane_requires_error_code() -> None:
    with pytest.raises(CognitiveKernelContractError):
        plane(
            state="failed",
            deleted_count=0,
            blocked_count=2,
            error_code=None,
        )


def test_pending_plane_may_not_have_completion_time() -> None:
    with pytest.raises(CognitiveKernelContractError):
        plane(
            state="pending",
            completed_at=STAMP,
            deleted_count=0,
        )


def test_plane_digest_tamper_is_detected() -> None:
    value = plane()
    with pytest.raises(CognitiveKernelContractError):
        replace(value, result_content_digest="b" * 64).validate()


def test_restore_filter_requires_replacement_for_replace() -> None:
    with pytest.raises(CognitiveKernelContractError):
        RestoreFilterDecision.create(
            decision_id="restore-1",
            request_id="request-1",
            target_record_id="claim-1",
            source_snapshot_id="snapshot-1",
            action="replace",
            reason_code="deleted_record",
            evaluated_at=STAMP,
            replacement_record_id=None,
            source_content_digest=DIGEST,
        )


def test_restore_filter_exclude_is_digest_bound() -> None:
    value = RestoreFilterDecision.create(
        decision_id="restore-1",
        request_id="request-1",
        target_record_id="claim-1",
        source_snapshot_id="snapshot-1",
        action="exclude",
        reason_code="deleted_record",
        evaluated_at=STAMP,
        source_content_digest=DIGEST,
    )
    value.validate()
    assert value.decision_sha256 == canonical_sha256(
        value.material_record()
    )


def test_completed_propagation_requires_successful_planes() -> None:
    blocked = plane(
        state="blocked",
        deleted_count=0,
        blocked_count=2,
        error_code="retention_hold",
    )
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationReceipt.create(
            envelope=envelope(),
            receipt_id="deletion-receipt-1",
            request_id="request-1",
            deletion_mode="mixed",
            propagation_state="completed",
            target_record_ids=("claim-1", "claim-2"),
            reason_code="owner_delete",
            authority_decision_id="decision-1",
            requested_by="owner",
            requested_at=STAMP,
            effective_at=STAMP,
            plane_receipts=(blocked,),
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            generation=1,
            receipt_content_digest=DIGEST,
        )


def test_completed_propagation_requires_effective_time() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationReceipt.create(
            envelope=envelope(),
            receipt_id="deletion-receipt-1",
            request_id="request-1",
            deletion_mode="mixed",
            propagation_state="completed",
            target_record_ids=("claim-1", "claim-2"),
            reason_code="owner_delete",
            authority_decision_id="decision-1",
            requested_by="owner",
            requested_at=STAMP,
            effective_at=None,
            plane_receipts=(plane(),),
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            generation=1,
            receipt_content_digest=DIGEST,
        )


def test_propagation_rejects_duplicate_plane_component() -> None:
    first = plane(receipt_id="plane-1")
    second = plane(receipt_id="plane-2")
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationReceipt.create(
            envelope=envelope(),
            receipt_id="deletion-receipt-1",
            request_id="request-1",
            deletion_mode="mixed",
            propagation_state="completed",
            target_record_ids=("claim-1", "claim-2"),
            reason_code="owner_delete",
            authority_decision_id="decision-1",
            requested_by="owner",
            requested_at=STAMP,
            effective_at=STAMP,
            plane_receipts=(first, second),
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            generation=1,
            receipt_content_digest=DIGEST,
        )


def test_generation_one_rejects_previous_receipt() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationReceipt.create(
            envelope=envelope(),
            receipt_id="deletion-receipt-1",
            request_id="request-1",
            deletion_mode="mixed",
            propagation_state="completed",
            target_record_ids=("claim-1", "claim-2"),
            reason_code="owner_delete",
            authority_decision_id="decision-1",
            requested_by="owner",
            requested_at=STAMP,
            effective_at=STAMP,
            plane_receipts=(plane(),),
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            generation=1,
            previous_receipt_id="older-receipt",
            receipt_content_digest=DIGEST,
        )


def test_valid_propagation_receipt_is_digest_bound() -> None:
    value = DeletionPropagationReceipt.create(
        envelope=envelope(),
        receipt_id="deletion-receipt-1",
        request_id="request-1",
        deletion_mode="mixed",
        propagation_state="completed",
        target_record_ids=("claim-2", "claim-1"),
        reason_code="owner_delete",
        authority_decision_id="decision-1",
        requested_by="owner",
        requested_at=STAMP,
        effective_at=STAMP,
        plane_receipts=(plane(),),
        restore_filter_decision_ids=("restore-1",),
        rollback_state="rehearsed",
        retirement_state="not_applicable",
        generation=1,
        receipt_content_digest=DIGEST,
    )
    value.validate()
    assert value.target_record_ids == ("claim-1", "claim-2")
    assert value.receipt_sha256 == canonical_sha256(
        value.material_record()
    )


def test_propagation_requires_operational_envelope() -> None:
    bad = replace(
        envelope(),
        authority_role="registered_projection",
    )
    with pytest.raises(CognitiveKernelContractError):
        DeletionPropagationReceipt.create(
            envelope=bad,
            receipt_id="deletion-receipt-1",
            request_id="request-1",
            deletion_mode="mixed",
            propagation_state="completed",
            target_record_ids=("claim-1",),
            reason_code="owner_delete",
            authority_decision_id="decision-1",
            requested_by="owner",
            requested_at=STAMP,
            effective_at=STAMP,
            plane_receipts=(
                plane(target_count=1, deleted_count=1),
            ),
            rollback_state="rehearsed",
            retirement_state="not_applicable",
            generation=1,
            receipt_content_digest=DIGEST,
        )
