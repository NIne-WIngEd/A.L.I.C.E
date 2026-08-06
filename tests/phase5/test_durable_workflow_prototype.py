"""M2.5 reversible durable Curator, migration, repair, and learning workflow tests."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.durable_workflow_prototype import (
    DurableWorkflowConflictError,
    DurableWorkflowIntegrityError,
    DurableWorkflowIsolationError,
    DurableWorkflowProfile,
    UnsafeDurableWorkflowPathError,
    open_durable_workflow_prototype,
)

NOW = "2026-08-06T22:30:00Z"
LEASE = "2026-08-06T22:40:00Z"
LATER = "2026-08-06T22:31:00Z"
RETRY = "2026-08-06T22:35:00Z"


def scope(
    *,
    product_id: str = "alice",
    host: str = "owner-primary",
) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=product_id,
        host_instance_id=host,
        schema_version="1.0.0",
        encryption_domain="owner-private",
    )


def profile(
    *,
    product_id: str = "alice",
    host: str = "owner-primary",
    attempts: int = 3,
) -> DurableWorkflowProfile:
    return DurableWorkflowProfile.create(
        scope=scope(product_id=product_id, host=host),
        authority_namespace_id="owner-primary",
        profile_id="m2-durable-workflow",
        default_max_attempts=attempts,
    )


def create(
    store,
    *,
    workflow_id: str = "workflow-1",
    workflow_kind: str = "curator",
    task_id: str = "task-1",
    task_kind: str = "selective_memory",
    key: str = "create-key",
):
    return store.create_workflow(
        workflow_id=workflow_id,
        workflow_kind=workflow_kind,
        task_id=task_id,
        task_kind=task_kind,
        full_workflow_content={
            "mission": "curate durable memory",
            "workflow_kind": workflow_kind,
        },
        full_task_content={
            "instruction": "review evidence and form candidates",
            "payload": {"private": "full task content"},
        },
        target_record_ids=("claim-1",),
        source_record_ids=("evidence-1",),
        priority=60,
        now=NOW,
        idempotency_namespace="tests",
        idempotency_key=key,
    )


def start(store, *, generation: int = 0, key: str = "start-key"):
    return store.start_task(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=generation,
        now=LATER,
        lease_expires_at=LEASE,
        idempotency_namespace="tests",
        idempotency_key=key,
        full_event_content={"worker": "curator-worker"},
    )


def test_profile_rejects_production_influence() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DurableWorkflowProfile.create(
            scope=scope(),
            authority_namespace_id="owner-primary",
            profile_id="unsafe",
            production_influence=True,
        )


def test_profile_rejects_canonical_claim_authority() -> None:
    with pytest.raises(CognitiveKernelContractError):
        DurableWorkflowProfile.create(
            scope=scope(),
            authority_namespace_id="owner-primary",
            profile_id="unsafe",
            canonical_claim_authority=True,
        )


def test_database_must_remain_outside_public_git(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    with pytest.raises(UnsafeDurableWorkflowPathError):
        open_durable_workflow_prototype(
            path=repository / "workflow.sqlite3",
            profile=profile(),
            repository_root=repository,
        )


@pytest.mark.parametrize(
    ("workflow_kind", "task_kind"),
    [
        ("curator", "selective_memory"),
        ("migration", "migration"),
        ("repair", "repair"),
        ("learning", "learning"),
    ],
)
def test_supported_workflow_kinds_persist(
    tmp_path: Path,
    workflow_kind: str,
    task_kind: str,
) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / f"{workflow_kind}.sqlite3",
        profile=profile(),
    )
    receipt = create(
        store,
        workflow_kind=workflow_kind,
        task_kind=task_kind,
    )
    assert receipt.workflow.workflow_kind == workflow_kind
    assert receipt.task.task_kind == task_kind
    assert receipt.event.activity_kind == "workflow_created"
    store.close()


def test_full_workflow_and_task_content_persist(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    workflow, workflow_content = store.load_workflow("workflow-1")
    task, task_content = store.load_task("task-1")
    assert workflow.workflow_state == "pending"
    assert workflow_content["mission"] == "curate durable memory"
    assert task_content["payload"]["private"] == "full task content"
    store.close()


def test_create_is_idempotent(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    first = create(store)
    second = create(store)
    assert first.operation_sha256 == second.operation_sha256
    assert len(store.list_activity_events("workflow-1")) == 1
    store.close()


def test_changed_create_request_conflicts(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    with pytest.raises(DurableWorkflowConflictError):
        create(store, workflow_kind="repair")
    store.close()


def test_task_start_is_generation_guarded(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    receipt = start(store)
    assert receipt.workflow.workflow_state == "running"
    assert receipt.workflow.generation == 1
    assert receipt.task.task_state == "running"
    assert receipt.task.attempt == 1
    assert receipt.task.lease_expires_at == "2026-08-06T22:40:00.000000Z"
    with pytest.raises(DurableWorkflowConflictError):
        store.start_task(
            workflow_id="workflow-1",
            task_id="task-1",
            expected_generation=0,
            now=LATER,
            lease_expires_at=LEASE,
            idempotency_namespace="tests",
            idempotency_key="stale-start",
            full_event_content={"worker": "other"},
        )
    store.close()


def test_checkpoint_is_durable_and_restart_safe(tmp_path: Path) -> None:
    database = tmp_path / "workflow.sqlite3"
    first = open_durable_workflow_prototype(
        path=database,
        profile=profile(),
    )
    create(first)
    start(first)
    checkpoint = first.save_checkpoint(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        idempotency_namespace="tests",
        idempotency_key="checkpoint-key",
        full_event_content={"stage": "candidate-review"},
        checkpoint_content={
            "cursor": 12,
            "reviewed": ["evidence-1"],
        },
    )
    assert checkpoint.task.task_state == "waiting"
    digest = checkpoint.task.checkpoint_digest
    first.close()

    second = open_durable_workflow_prototype(
        path=database,
        profile=profile(),
    )
    workflow, _ = second.load_workflow("workflow-1")
    task, _ = second.load_task("task-1")
    assert workflow.workflow_state == "waiting"
    assert task.checkpoint_digest == digest
    assert len(second.list_activity_events("workflow-1")) == 3
    second.close()


def test_retryable_failure_produces_deferred_receipt(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(attempts=3),
    )
    create(store)
    start(store)
    result = store.record_failure(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        retry_scheduled_at=RETRY,
        idempotency_namespace="tests",
        idempotency_key="failure-key",
        full_event_content={"failure": "temporary"},
        full_result_content={"error": "temporary"},
        error_code="temporary_failure",
    )
    assert result.task.task_state == "retry_scheduled"
    assert result.workflow.workflow_state == "retry_scheduled"
    assert result.curation_receipt is not None
    assert result.curation_receipt.outcome == "deferred"
    assert result.curation_receipt.retry_scheduled_at == "2026-08-06T22:35:00.000000Z"
    store.close()


def test_retry_resume_increments_attempt(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(attempts=3),
    )
    create(store)
    start(store)
    store.record_failure(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        retry_scheduled_at=RETRY,
        idempotency_namespace="tests",
        idempotency_key="failure-key",
        full_event_content={"failure": "temporary"},
        full_result_content={"error": "temporary"},
        error_code="temporary_failure",
    )
    resumed = store.resume_task(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=2,
        now=RETRY,
        lease_expires_at="2026-08-06T22:45:00Z",
        idempotency_namespace="tests",
        idempotency_key="resume-key",
        full_event_content={"worker": "retry-worker"},
    )
    assert resumed.task.task_state == "running"
    assert resumed.task.attempt == 2
    assert resumed.event.activity_kind == "task_resumed"
    store.close()


def test_exhausted_attempt_fails_workflow(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(attempts=1),
    )
    create(store)
    start(store)
    result = store.record_failure(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        idempotency_namespace="tests",
        idempotency_key="final-failure",
        full_event_content={"failure": "permanent"},
        full_result_content={"error": "permanent"},
        error_code="permanent_failure",
    )
    assert result.task.task_state == "failed"
    assert result.workflow.workflow_state == "failed"
    assert result.curation_receipt is not None
    assert result.curation_receipt.outcome == "failed"
    store.close()


def test_completion_persists_curation_receipt(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    start(store)
    result = store.complete_task(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        idempotency_namespace="tests",
        idempotency_key="complete-key",
        full_event_content={"activity": "candidate accepted"},
        full_result_content={
            "candidate_ids": ["candidate-1"],
            "decision": "completed",
        },
    )
    assert result.workflow.workflow_state == "completed"
    assert result.task.task_state == "completed"
    assert result.curation_receipt is not None
    assert result.curation_receipt.outcome == "completed"
    receipts = store.list_curation_receipts("workflow-1")
    assert len(receipts) == 1
    store.close()


def test_signal_is_persisted_without_finishing_workflow(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    result = store.signal_workflow(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=0,
        now=LATER,
        idempotency_namespace="tests",
        idempotency_key="signal-key",
        full_event_content={"signal": "owner-review"},
        signal_content={"command": "pause-after-checkpoint"},
    )
    assert result.workflow.workflow_state == "pending"
    assert len(result.workflow.signal_ids) == 1
    assert result.event.activity_kind == "signal_received"
    store.close()


def test_cancel_workflow_is_durable(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    result = store.cancel_workflow(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=0,
        now=LATER,
        idempotency_namespace="tests",
        idempotency_key="cancel-key",
        full_event_content={"reason": "owner_cancelled"},
        full_result_content={"reason": "owner_cancelled"},
    )
    assert result.workflow.workflow_state == "cancelled"
    assert result.task.task_state == "cancelled"
    assert result.curation_receipt is not None
    assert result.curation_receipt.outcome == "cancelled"
    store.close()


def test_transition_idempotency_returns_same_operation(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    first = start(store)
    second = start(store)
    assert first.operation_sha256 == second.operation_sha256
    assert len(store.list_activity_events("workflow-1")) == 2
    store.close()


def test_scope_isolation_is_enforced(tmp_path: Path) -> None:
    database = tmp_path / "workflow.sqlite3"
    first = open_durable_workflow_prototype(
        path=database,
        profile=profile(),
    )
    create(first)
    first.close()
    with pytest.raises(DurableWorkflowIsolationError):
        open_durable_workflow_prototype(
            path=database,
            profile=profile(product_id="friday", host="friday-primary"),
        )


def test_integrity_report_is_healthy(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    start(store)
    result = store.complete_task(
        workflow_id="workflow-1",
        task_id="task-1",
        expected_generation=1,
        now="2026-08-06T22:32:00Z",
        idempotency_namespace="tests",
        idempotency_key="complete-key",
        full_event_content={"activity": "done"},
        full_result_content={"result": "done"},
    )
    report = store.require_integrity()
    assert report.healthy
    assert report.checked_workflows == 1
    assert report.checked_tasks == 1
    assert report.checked_events == 3
    assert report.checked_receipts == 1
    assert report.checked_operations == 3
    store.close()


def test_tampering_is_detected(tmp_path: Path) -> None:
    store = open_durable_workflow_prototype(
        path=tmp_path / "workflow.sqlite3",
        profile=profile(),
    )
    create(store)
    store._connection.execute(
        "UPDATE tasks SET full_content_json = ? WHERE task_id = ?",
        (json.dumps({"tampered": True}), "task-1"),
    )
    store._connection.commit()
    with pytest.raises(DurableWorkflowIntegrityError):
        store.require_integrity()
    store.close()


def test_reversible_removal_deletes_database(tmp_path: Path) -> None:
    database = tmp_path / "workflow.sqlite3"
    store = open_durable_workflow_prototype(
        path=database,
        profile=profile(),
    )
    create(store)
    store.remove_database()
    assert not database.exists()
