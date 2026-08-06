"""M2.5 Curation Task, Receipt, Workflow, and Activity Event tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cognitive_kernel.canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
)
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.curation_contracts import (
    CURATION_RECEIPT_OUTCOMES,
    CURATION_TASK_KINDS,
    CURATION_TASK_STATES,
    WORKFLOW_ACTIVITY_KINDS,
    WORKFLOW_KINDS,
    WORKFLOW_STATES,
    CurationReceipt,
    CurationTask,
    DurableWorkflow,
    WorkflowActivityEvent,
)
from cognitive_kernel.memory_contracts import MemoryUnitEnvelope

NOW = "2026-08-06T22:30:00Z"
LATER = "2026-08-06T22:31:00Z"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def scope() -> ProductHostScope:
    return ProductHostScope.create(
        product_id="alice",
        host_instance_id="owner-primary",
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def envelope(
    record_id: str,
    record_type: str,
    content_digest: str,
    *,
    generation: int = 0,
) -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope.create(
        scope=scope(),
        record_id=record_id,
        record_type=record_type,
        authority_namespace_id="owner-primary",
        host_or_cluster_id="owner-primary",
        authority_role="operational_workflow_state",
        deployment_profile="reversible_prototype",
        created_at=NOW,
        valid_from=NOW,
        valid_to=None,
        transaction_time=NOW,
        logical_clock=generation,
        causal_parents=(),
        source_records=(),
        generation=generation,
        state="committed",
        data_classification="private",
        retention_class="active_project",
        deletion_state="active",
        provenance_digest=DIGEST_A,
        content_digest=content_digest,
        writer="durable-workflow-prototype",
        workflow_or_request_id="workflow-1",
        idempotency_namespace="m2-workflow",
        idempotency_key=f"key-{record_id}",
    )


def task(state: str = "pending") -> CurationTask:
    return CurationTask.create(
        envelope=envelope("task-1", "curation_task", DIGEST_B),
        task_id="task-1",
        workflow_id="workflow-1",
        task_kind="selective_memory",
        task_state=state,
        target_record_ids=("claim-2",),
        source_record_ids=("evidence-1",),
        priority=50,
        attempt=0,
        max_attempts=3,
        scheduled_at=NOW,
        lease_expires_at=None,
        checkpoint_digest=None,
        instruction_digest=DIGEST_A,
        task_content_digest=DIGEST_B,
    )


def event() -> WorkflowActivityEvent:
    return WorkflowActivityEvent.create(
        envelope=envelope(
            "event-1",
            "workflow_activity_event",
            DIGEST_C,
        ),
        event_id="event-1",
        workflow_id="workflow-1",
        task_id="task-1",
        activity_kind="workflow_created",
        outcome="accepted",
        sequence_number=1,
        attempt=0,
        occurred_at=NOW,
        output_digest=DIGEST_C,
        reason_codes=("workflow_created",),
        idempotency_key="event-key",
    )


def receipt(outcome: str = "completed") -> CurationReceipt:
    return CurationReceipt.create(
        envelope=envelope(
            "receipt-1",
            "curation_receipt",
            DIGEST_C,
        ),
        receipt_id="receipt-1",
        task_id="task-1",
        workflow_id="workflow-1",
        outcome=outcome,
        started_at=NOW,
        completed_at=LATER,
        input_record_ids=("evidence-1",),
        output_record_ids=("claim-2",),
        activity_event_ids=("event-1",),
        attempt=1,
        checkpoint_digest=None,
        result_content_digest=DIGEST_C,
        error_code=("failure" if outcome == "failed" else None),
        retry_scheduled_at=None,
    )


def workflow(state: str = "pending") -> DurableWorkflow:
    terminal = state in {"completed", "failed", "cancelled", "rolled_back"}
    return DurableWorkflow.create(
        envelope=envelope(
            "workflow-1",
            "durable_workflow",
            DIGEST_A,
        ),
        workflow_id="workflow-1",
        workflow_kind="curator",
        workflow_state=state,
        root_task_id="task-1",
        current_task_ids=(() if terminal else ("task-1",)),
        completed_task_ids=(("task-1",) if state == "completed" else ()),
        failed_task_ids=(("task-1",) if state == "failed" else ()),
        checkpoint_digest=None,
        signal_ids=(),
        generation=0,
        started_at=NOW,
        updated_at=NOW,
        completed_at=(LATER if terminal else None),
        workflow_content_digest=DIGEST_A,
    )


def test_registered_vocabularies_cover_m2_5() -> None:
    assert "selective_memory" in CURATION_TASK_KINDS
    assert "retry_scheduled" in CURATION_TASK_STATES
    assert "deferred" in CURATION_RECEIPT_OUTCOMES
    assert {"curator", "migration", "repair", "learning"} <= WORKFLOW_KINDS
    assert "compensating" in WORKFLOW_STATES
    assert "checkpoint_saved" in WORKFLOW_ACTIVITY_KINDS


def test_curation_task_is_digest_bound() -> None:
    value = task()
    value.validate()
    assert value.task_sha256 == canonical_sha256(value.semantic_record())


def test_running_task_requires_lease() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(task(), task_state="running").validate()


def test_terminal_task_may_not_retain_lease() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            task(),
            task_state="completed",
            lease_expires_at=LATER,
        ).validate()


def test_task_rejects_wrong_envelope_type() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            task(),
            envelope=envelope("task-1", "durable_workflow", DIGEST_B),
        ).validate()


def test_receipt_is_digest_bound() -> None:
    value = receipt()
    value.validate()
    assert value.receipt_sha256 == canonical_sha256(
        value.semantic_record()
    )


def test_failed_receipt_requires_error_code() -> None:
    value = receipt("failed")
    with pytest.raises(CognitiveKernelContractError):
        replace(value, error_code=None).validate()


def test_deferred_receipt_requires_retry_timestamp() -> None:
    value = receipt()
    with pytest.raises(CognitiveKernelContractError):
        replace(value, outcome="deferred").validate()


def test_workflow_is_digest_bound() -> None:
    value = workflow()
    value.validate()
    assert value.workflow_sha256 == canonical_sha256(
        value.semantic_record()
    )


def test_workflow_task_sets_must_be_disjoint() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(
            workflow(),
            completed_task_ids=("task-1",),
        ).validate()


def test_terminal_workflow_requires_completed_at() -> None:
    value = workflow("completed")
    with pytest.raises(CognitiveKernelContractError):
        replace(value, completed_at=None).validate()


def test_nonterminal_workflow_rejects_completed_at() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(workflow(), completed_at=LATER).validate()


def test_activity_event_is_digest_bound() -> None:
    value = event()
    value.validate()
    assert value.event_sha256 == canonical_sha256(
        value.semantic_record()
    )


def test_activity_event_sequence_must_be_positive() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(event(), sequence_number=0).validate()


def test_activity_event_rejects_unknown_kind() -> None:
    with pytest.raises(CognitiveKernelContractError):
        replace(event(), activity_kind="magic").validate()
