"""Memory M2.5 reversible durable-curation workflow prototype."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
)
from .contracts import ProductHostScope
from .curation_contracts import (
    CURATION_CONTRACT_SCHEMA_VERSION,
    WORKFLOW_KINDS,
    CurationReceipt,
    CurationTask,
    DurableWorkflow,
    WorkflowActivityEvent,
)
from .memory_contracts import MemoryUnitEnvelope

DURABLE_WORKFLOW_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
DURABLE_WORKFLOW_PROTOTYPE_STATE = "reversible_nonproduction"


class DurableWorkflowPrototypeError(RuntimeError):
    """Base error for the M2.5 workflow profile."""


class UnsafeDurableWorkflowPathError(DurableWorkflowPrototypeError):
    """Raised when workflow storage would enter the public repository."""


class DurableWorkflowIsolationError(DurableWorkflowPrototypeError):
    """Raised when workflow storage belongs to another scope."""


class DurableWorkflowIntegrityError(DurableWorkflowPrototypeError):
    """Raised when durable workflow state fails integrity checks."""


class DurableWorkflowConflictError(DurableWorkflowPrototypeError):
    """Raised for optimistic-concurrency or idempotency conflicts."""


class DurableWorkflowTransactionError(DurableWorkflowPrototypeError):
    """Raised when one workflow mutation cannot commit."""


@dataclass(frozen=True)
class DurableWorkflowProfile:
    """Profile-selected behavior for the reversible M2.5 workflow engine."""

    scope: ProductHostScope
    authority_namespace_id: str
    profile_id: str
    default_max_attempts: int
    production_influence: bool = False
    canonical_claim_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        authority_namespace_id: object,
        profile_id: object,
        default_max_attempts: object = 3,
        production_influence: object = False,
        canonical_claim_authority: object = False,
    ) -> "DurableWorkflowProfile":
        if not isinstance(scope, ProductHostScope):
            raise CognitiveKernelContractError(
                "scope must be ProductHostScope"
            )
        scope.validate()
        if (
            not isinstance(default_max_attempts, int)
            or isinstance(default_max_attempts, bool)
            or default_max_attempts < 1
        ):
            raise CognitiveKernelContractError(
                "default_max_attempts must be a positive integer"
            )
        if not isinstance(production_influence, bool):
            raise CognitiveKernelContractError(
                "production_influence must be boolean"
            )
        if not isinstance(canonical_claim_authority, bool):
            raise CognitiveKernelContractError(
                "canonical_claim_authority must be boolean"
            )
        if production_influence:
            raise CognitiveKernelContractError(
                "M2.5 reversible profile may not influence production"
            )
        if canonical_claim_authority:
            raise CognitiveKernelContractError(
                "M2.5 workflow profile may not write canonical Claim Authority"
            )
        value = cls(
            scope=scope,
            authority_namespace_id=require_identifier(
                authority_namespace_id,
                "authority_namespace_id",
            ),
            profile_id=require_identifier(profile_id, "profile_id"),
            default_max_attempts=default_max_attempts,
            production_influence=False,
            canonical_claim_authority=False,
        )
        value.validate()
        return value

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": DURABLE_WORKFLOW_PROTOTYPE_SCHEMA_VERSION,
            "prototype_state": DURABLE_WORKFLOW_PROTOTYPE_STATE,
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "profile_id": self.profile_id,
            "default_max_attempts": self.default_max_attempts,
            "production_influence": self.production_influence,
            "canonical_claim_authority": self.canonical_claim_authority,
        }

    def profile_sha256(self) -> str:
        return canonical_sha256(self.metadata_record())

    def validate(self) -> None:
        self.scope.validate()
        if self.default_max_attempts < 1:
            raise CognitiveKernelContractError(
                "default_max_attempts must remain positive"
            )
        if self.production_influence:
            raise CognitiveKernelContractError(
                "M2.5 reversible profile may not influence production"
            )
        if self.canonical_claim_authority:
            raise CognitiveKernelContractError(
                "M2.5 workflow profile may not write canonical Claim Authority"
            )


@dataclass(frozen=True)
class DurableWorkflowOperationReceipt:
    workflow: DurableWorkflow
    task: CurationTask
    event: WorkflowActivityEvent
    curation_receipt: CurationReceipt | None
    full_event_content: dict[str, object]
    operation_sha256: str

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": DURABLE_WORKFLOW_PROTOTYPE_SCHEMA_VERSION,
            "workflow_sha256": self.workflow.workflow_sha256,
            "task_sha256": self.task.task_sha256,
            "event_sha256": self.event.event_sha256,
            "curation_receipt_sha256": (
                self.curation_receipt.receipt_sha256
                if self.curation_receipt is not None
                else None
            ),
            "full_event_content_digest": canonical_sha256(
                self.full_event_content
            ),
            "operation_sha256": self.operation_sha256,
        }


@dataclass(frozen=True)
class DurableWorkflowIntegrityReport:
    checked_workflows: int
    checked_tasks: int
    checked_events: int
    checked_receipts: int
    checked_operations: int
    problems: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.problems


def validate_durable_workflow_path(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name in {"", ".", ".."}:
        raise UnsafeDurableWorkflowPathError(
            "workflow database path is invalid"
        )
    if repository_root is not None:
        repository = repository_root.expanduser().resolve()
        try:
            resolved.relative_to(repository)
        except ValueError:
            pass
        else:
            raise UnsafeDurableWorkflowPathError(
                "workflow database must remain outside public Git"
            )
    return resolved


def _identifier_tuple(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    normalized = tuple(
        require_identifier(value, field)
        for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return tuple(sorted(normalized))


def _make_id(prefix: str, material: object) -> str:
    return f"{prefix}-{canonical_sha256(material)[:24]}"


def _scope_from_record(record: Mapping[str, object]) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=record["product_id"],
        host_instance_id=record["host_instance_id"],
        schema_version=record["schema_version"],
        encryption_domain=record["encryption_domain"],
    )


def _envelope_from_record(record: Mapping[str, object]) -> MemoryUnitEnvelope:
    value = MemoryUnitEnvelope(
        scope=_scope_from_record(record["scope"]),
        record_id=str(record["record_id"]),
        record_type=str(record["record_type"]),
        authority_namespace_id=str(record["authority_namespace_id"]),
        host_or_cluster_id=str(record["host_or_cluster_id"]),
        authority_role=str(record["authority_role"]),
        deployment_profile=str(record["deployment_profile"]),
        created_at=str(record["created_at"]),
        valid_from=str(record["valid_from"]),
        valid_to=(
            str(record["valid_to"])
            if record["valid_to"] is not None
            else None
        ),
        transaction_time=str(record["transaction_time"]),
        logical_clock=int(record["logical_clock"]),
        causal_parents=tuple(
            str(item) for item in record["causal_parents"]
        ),
        source_records=tuple(
            str(item) for item in record["source_records"]
        ),
        generation=int(record["generation"]),
        state=str(record["state"]),
        data_classification=str(record["data_classification"]),
        retention_class=str(record["retention_class"]),
        deletion_state=str(record["deletion_state"]),
        provenance_digest=str(record["provenance_digest"]),
        content_digest=str(record["content_digest"]),
        writer=str(record["writer"]),
        workflow_or_request_id=str(
            record["workflow_or_request_id"]
        ),
        idempotency_namespace=str(
            record["idempotency_namespace"]
        ),
        idempotency_key=str(record["idempotency_key"]),
        supersedes=tuple(
            str(item) for item in record["supersedes"]
        ),
        superseded_by=tuple(
            str(item) for item in record["superseded_by"]
        ),
        rollback_reference=(
            str(record["rollback_reference"])
            if record["rollback_reference"] is not None
            else None
        ),
        envelope_sha256=str(record["envelope_sha256"]),
    )
    value.validate()
    return value


def _workflow_to_record(value: DurableWorkflow) -> dict[str, object]:
    return {
        "envelope": value.envelope.metadata_record(),
        **value.metadata_record(),
    }


def _workflow_from_record(record: Mapping[str, object]) -> DurableWorkflow:
    value = DurableWorkflow(
        envelope=_envelope_from_record(record["envelope"]),
        workflow_id=str(record["workflow_id"]),
        workflow_kind=str(record["workflow_kind"]),
        workflow_state=str(record["workflow_state"]),
        root_task_id=str(record["root_task_id"]),
        current_task_ids=tuple(
            str(item) for item in record["current_task_ids"]
        ),
        completed_task_ids=tuple(
            str(item) for item in record["completed_task_ids"]
        ),
        failed_task_ids=tuple(
            str(item) for item in record["failed_task_ids"]
        ),
        checkpoint_digest=(
            str(record["checkpoint_digest"])
            if record["checkpoint_digest"] is not None
            else None
        ),
        signal_ids=tuple(str(item) for item in record["signal_ids"]),
        generation=int(record["generation"]),
        started_at=str(record["started_at"]),
        updated_at=str(record["updated_at"]),
        completed_at=(
            str(record["completed_at"])
            if record["completed_at"] is not None
            else None
        ),
        workflow_content_digest=str(
            record["workflow_content_digest"]
        ),
        workflow_sha256=str(record["workflow_sha256"]),
    )
    value.validate()
    return value


def _task_to_record(value: CurationTask) -> dict[str, object]:
    return {
        "envelope": value.envelope.metadata_record(),
        **value.metadata_record(),
    }


def _task_from_record(record: Mapping[str, object]) -> CurationTask:
    value = CurationTask(
        envelope=_envelope_from_record(record["envelope"]),
        task_id=str(record["task_id"]),
        workflow_id=str(record["workflow_id"]),
        task_kind=str(record["task_kind"]),
        task_state=str(record["task_state"]),
        target_record_ids=tuple(
            str(item) for item in record["target_record_ids"]
        ),
        source_record_ids=tuple(
            str(item) for item in record["source_record_ids"]
        ),
        priority=int(record["priority"]),
        attempt=int(record["attempt"]),
        max_attempts=int(record["max_attempts"]),
        scheduled_at=str(record["scheduled_at"]),
        lease_expires_at=(
            str(record["lease_expires_at"])
            if record["lease_expires_at"] is not None
            else None
        ),
        checkpoint_digest=(
            str(record["checkpoint_digest"])
            if record["checkpoint_digest"] is not None
            else None
        ),
        instruction_digest=str(record["instruction_digest"]),
        task_content_digest=str(record["task_content_digest"]),
        task_sha256=str(record["task_sha256"]),
    )
    value.validate()
    return value


def _event_to_record(
    value: WorkflowActivityEvent,
) -> dict[str, object]:
    return {
        "envelope": value.envelope.metadata_record(),
        **value.metadata_record(),
    }


def _event_from_record(
    record: Mapping[str, object],
) -> WorkflowActivityEvent:
    value = WorkflowActivityEvent(
        envelope=_envelope_from_record(record["envelope"]),
        event_id=str(record["event_id"]),
        workflow_id=str(record["workflow_id"]),
        task_id=(
            str(record["task_id"])
            if record["task_id"] is not None
            else None
        ),
        activity_kind=str(record["activity_kind"]),
        outcome=str(record["outcome"]),
        sequence_number=int(record["sequence_number"]),
        attempt=int(record["attempt"]),
        occurred_at=str(record["occurred_at"]),
        input_digest=(
            str(record["input_digest"])
            if record["input_digest"] is not None
            else None
        ),
        output_digest=(
            str(record["output_digest"])
            if record["output_digest"] is not None
            else None
        ),
        checkpoint_digest=(
            str(record["checkpoint_digest"])
            if record["checkpoint_digest"] is not None
            else None
        ),
        reason_codes=tuple(
            str(item) for item in record["reason_codes"]
        ),
        idempotency_key=str(record["idempotency_key"]),
        event_sha256=str(record["event_sha256"]),
    )
    value.validate()
    return value


def _receipt_to_record(value: CurationReceipt) -> dict[str, object]:
    return {
        "envelope": value.envelope.metadata_record(),
        **value.metadata_record(),
    }


def _receipt_from_record(
    record: Mapping[str, object],
) -> CurationReceipt:
    value = CurationReceipt(
        envelope=_envelope_from_record(record["envelope"]),
        receipt_id=str(record["receipt_id"]),
        task_id=str(record["task_id"]),
        workflow_id=str(record["workflow_id"]),
        outcome=str(record["outcome"]),
        started_at=str(record["started_at"]),
        completed_at=str(record["completed_at"]),
        input_record_ids=tuple(
            str(item) for item in record["input_record_ids"]
        ),
        output_record_ids=tuple(
            str(item) for item in record["output_record_ids"]
        ),
        activity_event_ids=tuple(
            str(item) for item in record["activity_event_ids"]
        ),
        attempt=int(record["attempt"]),
        checkpoint_digest=(
            str(record["checkpoint_digest"])
            if record["checkpoint_digest"] is not None
            else None
        ),
        result_content_digest=str(
            record["result_content_digest"]
        ),
        error_code=(
            str(record["error_code"])
            if record["error_code"] is not None
            else None
        ),
        retry_scheduled_at=(
            str(record["retry_scheduled_at"])
            if record["retry_scheduled_at"] is not None
            else None
        ),
        receipt_sha256=str(record["receipt_sha256"]),
    )
    value.validate()
    return value


class DurableWorkflowPrototypeStore:
    """Persistent reversible workflow and curation prototype."""

    def __init__(
        self,
        *,
        path: Path,
        profile: DurableWorkflowProfile,
        repository_root: Path | None = None,
    ) -> None:
        profile.validate()
        self.path = validate_durable_workflow_path(
            path,
            repository_root=repository_root,
        )
        self.profile = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        self._bind_profile()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DurableWorkflowPrototypeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS prototype_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                workflow_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                workflow_sha256 TEXT NOT NULL,
                generation INTEGER NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                task_sha256 TEXT NOT NULL,
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
            );
            CREATE TABLE IF NOT EXISTS activity_events (
                event_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_id TEXT,
                sequence_number INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                event_sha256 TEXT NOT NULL,
                UNIQUE(workflow_id, sequence_number),
                FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
            );
            CREATE TABLE IF NOT EXISTS curation_receipts (
                receipt_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                full_result_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                FOREIGN KEY(workflow_id) REFERENCES workflows(workflow_id)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                checkpoint_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflow_signals (
                signal_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operation_receipts (
                operation_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                operation_json TEXT NOT NULL,
                operation_sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS idempotency (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                PRIMARY KEY(namespace, key)
            );
            """
        )
        self._connection.commit()

    def _bind_profile(self) -> None:
        expected = {
            "schema_version": DURABLE_WORKFLOW_PROTOTYPE_SCHEMA_VERSION,
            "profile_sha256": self.profile.profile_sha256(),
            "scope_sha256": self.profile.scope.scope_sha256(),
            "authority_namespace_id": self.profile.authority_namespace_id,
        }
        existing = {
            str(row["key"]): str(row["value"])
            for row in self._connection.execute(
                "SELECT key, value FROM prototype_metadata"
            )
        }
        if not existing:
            self._connection.executemany(
                "INSERT INTO prototype_metadata(key, value) VALUES (?, ?)",
                tuple(expected.items()),
            )
            self._connection.commit()
            return
        if existing != expected:
            raise DurableWorkflowIsolationError(
                "workflow database profile or scope mismatch"
            )

    def _envelope(
        self,
        *,
        record_id: str,
        record_type: str,
        content_digest: str,
        workflow_id: str,
        idempotency_namespace: str,
        idempotency_key: str,
        now: str,
        generation: int,
        source_records: Iterable[str] = (),
    ) -> MemoryUnitEnvelope:
        return MemoryUnitEnvelope.create(
            scope=self.profile.scope,
            record_id=record_id,
            record_type=record_type,
            authority_namespace_id=self.profile.authority_namespace_id,
            host_or_cluster_id=self.profile.scope.host_instance_id,
            authority_role="operational_workflow_state",
            deployment_profile="reversible_prototype",
            created_at=now,
            valid_from=now,
            valid_to=None,
            transaction_time=now,
            logical_clock=generation,
            causal_parents=(),
            source_records=tuple(source_records),
            generation=generation,
            state="committed",
            data_classification="private",
            retention_class="active_project",
            deletion_state="active",
            provenance_digest=canonical_sha256(
                {
                    "profile_sha256": self.profile.profile_sha256(),
                    "workflow_id": workflow_id,
                    "record_type": record_type,
                }
            ),
            content_digest=content_digest,
            writer="durable-workflow-prototype",
            workflow_or_request_id=workflow_id,
            idempotency_namespace=idempotency_namespace,
            idempotency_key=idempotency_key,
        )

    def _lookup_idempotency(
        self,
        *,
        namespace: str,
        key: str,
        request_digest: str,
    ) -> DurableWorkflowOperationReceipt | None:
        row = self._connection.execute(
            "SELECT request_digest, operation_id FROM idempotency "
            "WHERE namespace = ? AND key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_digest"]) != request_digest:
            raise DurableWorkflowConflictError(
                "idempotency key was reused with different content"
            )
        return self.load_operation(str(row["operation_id"]))

    def _next_sequence(self, workflow_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) AS value "
            "FROM activity_events WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        return int(row["value"]) + 1

    def _load_workflow_row(
        self,
        workflow_id: str,
    ) -> tuple[DurableWorkflow, dict[str, object]]:
        row = self._connection.execute(
            "SELECT workflow_json, full_content_json "
            "FROM workflows WHERE workflow_id = ?",
            (require_identifier(workflow_id, "workflow_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(workflow_id)
        workflow = _workflow_from_record(
            json.loads(str(row["workflow_json"]))
        )
        full_content = json.loads(str(row["full_content_json"]))
        if canonical_sha256(full_content) != workflow.workflow_content_digest:
            raise DurableWorkflowIntegrityError(
                "workflow full content digest mismatch"
            )
        return workflow, full_content

    def _load_task_row(
        self,
        task_id: str,
    ) -> tuple[CurationTask, dict[str, object]]:
        row = self._connection.execute(
            "SELECT task_json, full_content_json "
            "FROM tasks WHERE task_id = ?",
            (require_identifier(task_id, "task_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        task = _task_from_record(
            json.loads(str(row["task_json"]))
        )
        full_content = json.loads(str(row["full_content_json"]))
        if canonical_sha256(full_content) != task.task_content_digest:
            raise DurableWorkflowIntegrityError(
                "task full content digest mismatch"
            )
        return task, full_content

    def load_workflow(
        self,
        workflow_id: str,
    ) -> tuple[DurableWorkflow, dict[str, object]]:
        return self._load_workflow_row(workflow_id)

    def load_task(
        self,
        task_id: str,
    ) -> tuple[CurationTask, dict[str, object]]:
        return self._load_task_row(task_id)

    def list_activity_events(
        self,
        workflow_id: str,
    ) -> tuple[WorkflowActivityEvent, ...]:
        rows = self._connection.execute(
            "SELECT event_json FROM activity_events "
            "WHERE workflow_id = ? ORDER BY sequence_number",
            (require_identifier(workflow_id, "workflow_id"),),
        ).fetchall()
        return tuple(
            _event_from_record(json.loads(str(row["event_json"])))
            for row in rows
        )

    def list_curation_receipts(
        self,
        workflow_id: str,
    ) -> tuple[CurationReceipt, ...]:
        rows = self._connection.execute(
            "SELECT receipt_json FROM curation_receipts "
            "WHERE workflow_id = ? ORDER BY rowid",
            (require_identifier(workflow_id, "workflow_id"),),
        ).fetchall()
        return tuple(
            _receipt_from_record(
                json.loads(str(row["receipt_json"]))
            )
            for row in rows
        )

    def load_operation(
        self,
        operation_id: str,
    ) -> DurableWorkflowOperationReceipt:
        row = self._connection.execute(
            "SELECT operation_json, operation_sha256 "
            "FROM operation_receipts WHERE operation_id = ?",
            (require_identifier(operation_id, "operation_id"),),
        ).fetchone()
        if row is None:
            raise KeyError(operation_id)
        record = json.loads(str(row["operation_json"]))
        workflow = _workflow_from_record(record["workflow"])
        task = _task_from_record(record["task"])
        event = _event_from_record(record["event"])
        receipt = (
            _receipt_from_record(record["curation_receipt"])
            if record["curation_receipt"] is not None
            else None
        )
        full_event_content = dict(record["full_event_content"])
        material = {
            "workflow_sha256": workflow.workflow_sha256,
            "task_sha256": task.task_sha256,
            "event_sha256": event.event_sha256,
            "curation_receipt_sha256": (
                receipt.receipt_sha256 if receipt is not None else None
            ),
            "full_event_content_digest": canonical_sha256(
                full_event_content
            ),
        }
        operation_sha256 = canonical_sha256(material)
        if operation_sha256 != str(row["operation_sha256"]):
            raise DurableWorkflowIntegrityError(
                "operation receipt digest mismatch"
            )
        return DurableWorkflowOperationReceipt(
            workflow=workflow,
            task=task,
            event=event,
            curation_receipt=receipt,
            full_event_content=full_event_content,
            operation_sha256=operation_sha256,
        )

    def _store_operation(
        self,
        *,
        workflow: DurableWorkflow,
        workflow_content: Mapping[str, object],
        task: CurationTask,
        task_content: Mapping[str, object],
        event: WorkflowActivityEvent,
        full_event_content: Mapping[str, object],
        curation_receipt: CurationReceipt | None,
        full_result_content: Mapping[str, object] | None,
        idempotency_namespace: str,
        idempotency_key: str,
        request_digest: str,
        checkpoint: tuple[str, Mapping[str, object], str] | None = None,
        signal: tuple[str, Mapping[str, object], str] | None = None,
    ) -> DurableWorkflowOperationReceipt:
        material = {
            "workflow_sha256": workflow.workflow_sha256,
            "task_sha256": task.task_sha256,
            "event_sha256": event.event_sha256,
            "curation_receipt_sha256": (
                curation_receipt.receipt_sha256
                if curation_receipt is not None
                else None
            ),
            "full_event_content_digest": canonical_sha256(
                full_event_content
            ),
        }
        operation_sha256 = canonical_sha256(material)
        operation_id = event.event_id
        operation_record = {
            "workflow": _workflow_to_record(workflow),
            "task": _task_to_record(task),
            "event": _event_to_record(event),
            "curation_receipt": (
                _receipt_to_record(curation_receipt)
                if curation_receipt is not None
                else None
            ),
            "full_event_content": dict(full_event_content),
        }
        self._connection.execute(
            "INSERT OR REPLACE INTO workflows("
            "workflow_id, workflow_json, full_content_json, "
            "workflow_sha256, generation, state"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                workflow.workflow_id,
                canonical_json_bytes(
                    _workflow_to_record(workflow)
                ).decode("utf-8"),
                canonical_json_bytes(
                    dict(workflow_content)
                ).decode("utf-8"),
                workflow.workflow_sha256,
                workflow.generation,
                workflow.workflow_state,
            ),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO tasks("
            "task_id, workflow_id, task_json, full_content_json, "
            "task_sha256, state, attempt"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task.task_id,
                task.workflow_id,
                canonical_json_bytes(
                    _task_to_record(task)
                ).decode("utf-8"),
                canonical_json_bytes(
                    dict(task_content)
                ).decode("utf-8"),
                task.task_sha256,
                task.task_state,
                task.attempt,
            ),
        )
        self._connection.execute(
            "INSERT INTO activity_events("
            "event_id, workflow_id, task_id, sequence_number, "
            "event_json, full_content_json, event_sha256"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.workflow_id,
                event.task_id,
                event.sequence_number,
                canonical_json_bytes(
                    _event_to_record(event)
                ).decode("utf-8"),
                canonical_json_bytes(
                    dict(full_event_content)
                ).decode("utf-8"),
                event.event_sha256,
            ),
        )
        if curation_receipt is not None:
            if full_result_content is None:
                raise DurableWorkflowTransactionError(
                    "curation receipt requires full result content"
                )
            self._connection.execute(
                "INSERT INTO curation_receipts("
                "receipt_id, workflow_id, task_id, receipt_json, "
                "full_result_json, receipt_sha256"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    curation_receipt.receipt_id,
                    curation_receipt.workflow_id,
                    curation_receipt.task_id,
                    canonical_json_bytes(
                        _receipt_to_record(curation_receipt)
                    ).decode("utf-8"),
                    canonical_json_bytes(
                        dict(full_result_content)
                    ).decode("utf-8"),
                    curation_receipt.receipt_sha256,
                ),
            )
        if checkpoint is not None:
            checkpoint_id, content, created_at = checkpoint
            self._connection.execute(
                "INSERT INTO checkpoints("
                "checkpoint_id, workflow_id, task_id, "
                "full_content_json, checkpoint_digest, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (
                    checkpoint_id,
                    workflow.workflow_id,
                    task.task_id,
                    canonical_json_bytes(dict(content)).decode("utf-8"),
                    canonical_sha256(content),
                    created_at,
                ),
            )
        if signal is not None:
            signal_id, content, created_at = signal
            self._connection.execute(
                "INSERT INTO workflow_signals("
                "signal_id, workflow_id, full_content_json, "
                "content_digest, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    signal_id,
                    workflow.workflow_id,
                    canonical_json_bytes(dict(content)).decode("utf-8"),
                    canonical_sha256(content),
                    created_at,
                ),
            )
        self._connection.execute(
            "INSERT INTO operation_receipts("
            "operation_id, workflow_id, task_id, operation_json, "
            "operation_sha256"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                operation_id,
                workflow.workflow_id,
                task.task_id,
                canonical_json_bytes(operation_record).decode("utf-8"),
                operation_sha256,
            ),
        )
        self._connection.execute(
            "INSERT INTO idempotency("
            "namespace, key, request_digest, operation_id"
            ") VALUES (?, ?, ?, ?)",
            (
                idempotency_namespace,
                idempotency_key,
                request_digest,
                operation_id,
            ),
        )
        return DurableWorkflowOperationReceipt(
            workflow=workflow,
            task=task,
            event=event,
            curation_receipt=curation_receipt,
            full_event_content=dict(full_event_content),
            operation_sha256=operation_sha256,
        )

    def _workflow_value(
        self,
        *,
        prior: DurableWorkflow | None,
        workflow_id: str,
        workflow_kind: str,
        workflow_state: str,
        root_task_id: str,
        current_task_ids: Iterable[str],
        completed_task_ids: Iterable[str],
        failed_task_ids: Iterable[str],
        checkpoint_digest: str | None,
        signal_ids: Iterable[str],
        generation: int,
        started_at: str,
        updated_at: str,
        completed_at: str | None,
        workflow_content_digest: str,
        idempotency_namespace: str,
        idempotency_key: str,
    ) -> DurableWorkflow:
        envelope = self._envelope(
            record_id=workflow_id,
            record_type="durable_workflow",
            content_digest=workflow_content_digest,
            workflow_id=workflow_id,
            idempotency_namespace=idempotency_namespace,
            idempotency_key=idempotency_key,
            now=updated_at,
            generation=generation,
            source_records=(
                (prior.envelope.record_id,) if prior is not None else ()
            ),
        )
        return DurableWorkflow.create(
            envelope=envelope,
            workflow_id=workflow_id,
            workflow_kind=workflow_kind,
            workflow_state=workflow_state,
            root_task_id=root_task_id,
            current_task_ids=current_task_ids,
            completed_task_ids=completed_task_ids,
            failed_task_ids=failed_task_ids,
            checkpoint_digest=checkpoint_digest,
            signal_ids=signal_ids,
            generation=generation,
            started_at=started_at,
            updated_at=updated_at,
            completed_at=completed_at,
            workflow_content_digest=workflow_content_digest,
        )

    def _task_value(
        self,
        *,
        prior: CurationTask | None,
        task_id: str,
        workflow_id: str,
        task_kind: str,
        task_state: str,
        target_record_ids: Iterable[str],
        source_record_ids: Iterable[str],
        priority: int,
        attempt: int,
        max_attempts: int,
        scheduled_at: str,
        lease_expires_at: str | None,
        checkpoint_digest: str | None,
        instruction_digest: str,
        task_content_digest: str,
        idempotency_namespace: str,
        idempotency_key: str,
        now: str,
        generation: int,
    ) -> CurationTask:
        envelope = self._envelope(
            record_id=task_id,
            record_type="curation_task",
            content_digest=task_content_digest,
            workflow_id=workflow_id,
            idempotency_namespace=idempotency_namespace,
            idempotency_key=idempotency_key,
            now=now,
            generation=generation,
            source_records=(
                (prior.envelope.record_id,) if prior is not None else ()
            ),
        )
        return CurationTask.create(
            envelope=envelope,
            task_id=task_id,
            workflow_id=workflow_id,
            task_kind=task_kind,
            task_state=task_state,
            target_record_ids=target_record_ids,
            source_record_ids=source_record_ids,
            priority=priority,
            attempt=attempt,
            max_attempts=max_attempts,
            scheduled_at=scheduled_at,
            lease_expires_at=lease_expires_at,
            checkpoint_digest=checkpoint_digest,
            instruction_digest=instruction_digest,
            task_content_digest=task_content_digest,
        )

    def _event_value(
        self,
        *,
        workflow_id: str,
        task_id: str | None,
        activity_kind: str,
        outcome: str,
        attempt: int,
        occurred_at: str,
        full_event_content: Mapping[str, object],
        checkpoint_digest: str | None,
        reason_codes: Iterable[str],
        idempotency_namespace: str,
        idempotency_key: str,
        generation: int,
    ) -> WorkflowActivityEvent:
        sequence = self._next_sequence(workflow_id)
        event_id = _make_id(
            "event",
            {
                "workflow_id": workflow_id,
                "sequence": sequence,
                "activity_kind": activity_kind,
                "idempotency_key": idempotency_key,
            },
        )
        content_digest = canonical_sha256(full_event_content)
        envelope = self._envelope(
            record_id=event_id,
            record_type="workflow_activity_event",
            content_digest=content_digest,
            workflow_id=workflow_id,
            idempotency_namespace=idempotency_namespace,
            idempotency_key=idempotency_key,
            now=occurred_at,
            generation=generation,
            source_records=(task_id,) if task_id is not None else (),
        )
        return WorkflowActivityEvent.create(
            envelope=envelope,
            event_id=event_id,
            workflow_id=workflow_id,
            task_id=task_id,
            activity_kind=activity_kind,
            outcome=outcome,
            sequence_number=sequence,
            attempt=attempt,
            occurred_at=occurred_at,
            input_digest=None,
            output_digest=content_digest,
            checkpoint_digest=checkpoint_digest,
            reason_codes=reason_codes,
            idempotency_key=idempotency_key,
        )

    def _receipt_value(
        self,
        *,
        workflow_id: str,
        task: CurationTask,
        event: WorkflowActivityEvent,
        outcome: str,
        started_at: str,
        completed_at: str,
        full_result_content: Mapping[str, object],
        error_code: str | None,
        retry_scheduled_at: str | None,
        idempotency_namespace: str,
        idempotency_key: str,
        generation: int,
    ) -> CurationReceipt:
        receipt_id = _make_id(
            "receipt",
            {
                "workflow_id": workflow_id,
                "task_id": task.task_id,
                "attempt": task.attempt,
                "event_id": event.event_id,
            },
        )
        result_digest = canonical_sha256(full_result_content)
        envelope = self._envelope(
            record_id=receipt_id,
            record_type="curation_receipt",
            content_digest=result_digest,
            workflow_id=workflow_id,
            idempotency_namespace=idempotency_namespace,
            idempotency_key=f"{idempotency_key}-receipt",
            now=completed_at,
            generation=generation,
            source_records=(task.task_id, event.event_id),
        )
        return CurationReceipt.create(
            envelope=envelope,
            receipt_id=receipt_id,
            task_id=task.task_id,
            workflow_id=workflow_id,
            outcome=outcome,
            started_at=started_at,
            completed_at=completed_at,
            input_record_ids=task.source_record_ids,
            output_record_ids=task.target_record_ids,
            activity_event_ids=(event.event_id,),
            attempt=task.attempt,
            checkpoint_digest=task.checkpoint_digest,
            result_content_digest=result_digest,
            error_code=error_code,
            retry_scheduled_at=retry_scheduled_at,
        )

    def create_workflow(
        self,
        *,
        workflow_id: object,
        workflow_kind: object,
        task_id: object,
        task_kind: object,
        full_workflow_content: Mapping[str, object],
        full_task_content: Mapping[str, object],
        target_record_ids: Iterable[object] = (),
        source_record_ids: Iterable[object] = (),
        priority: object = 50,
        max_attempts: object | None = None,
        now: object,
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DurableWorkflowOperationReceipt:
        workflow_id_value = require_identifier(
            workflow_id,
            "workflow_id",
        )
        workflow_kind_value = require_identifier(
            workflow_kind,
            "workflow_kind",
        )
        if workflow_kind_value not in WORKFLOW_KINDS:
            raise CognitiveKernelContractError(
                "workflow_kind is not registered"
            )
        task_id_value = require_identifier(task_id, "task_id")
        task_kind_value = require_identifier(task_kind, "task_kind")
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(idempotency_key, "idempotency_key")
        timestamp = normalize_timestamp(now, "now")
        target_ids = _identifier_tuple(
            target_record_ids,
            "target_record_ids",
        )
        source_ids = _identifier_tuple(
            source_record_ids,
            "source_record_ids",
        )
        attempts = (
            self.profile.default_max_attempts
            if max_attempts is None
            else int(max_attempts)
        )
        request = {
            "action": "create_workflow",
            "workflow_id": workflow_id_value,
            "workflow_kind": workflow_kind_value,
            "task_id": task_id_value,
            "task_kind": task_kind_value,
            "workflow_content": dict(full_workflow_content),
            "task_content": dict(full_task_content),
            "target_record_ids": list(target_ids),
            "source_record_ids": list(source_ids),
            "priority": priority,
            "max_attempts": attempts,
            "now": timestamp,
        }
        request_digest = canonical_sha256(request)
        existing = self._lookup_idempotency(
            namespace=namespace,
            key=key,
            request_digest=request_digest,
        )
        if existing is not None:
            return existing
        workflow_digest = canonical_sha256(full_workflow_content)
        task_digest = canonical_sha256(full_task_content)
        workflow = self._workflow_value(
            prior=None,
            workflow_id=workflow_id_value,
            workflow_kind=workflow_kind_value,
            workflow_state="pending",
            root_task_id=task_id_value,
            current_task_ids=(task_id_value,),
            completed_task_ids=(),
            failed_task_ids=(),
            checkpoint_digest=None,
            signal_ids=(),
            generation=0,
            started_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
            workflow_content_digest=workflow_digest,
            idempotency_namespace=namespace,
            idempotency_key=key,
        )
        task = self._task_value(
            prior=None,
            task_id=task_id_value,
            workflow_id=workflow_id_value,
            task_kind=task_kind_value,
            task_state="pending",
            target_record_ids=target_ids,
            source_record_ids=source_ids,
            priority=int(priority),
            attempt=0,
            max_attempts=attempts,
            scheduled_at=timestamp,
            lease_expires_at=None,
            checkpoint_digest=None,
            instruction_digest=canonical_sha256(
                {"task_kind": task_kind_value}
            ),
            task_content_digest=task_digest,
            idempotency_namespace=namespace,
            idempotency_key=key,
            now=timestamp,
            generation=0,
        )
        event_content = {
            "action": "workflow_created",
            "workflow_kind": workflow_kind_value,
            "task_kind": task_kind_value,
        }
        event = self._event_value(
            workflow_id=workflow_id_value,
            task_id=task_id_value,
            activity_kind="workflow_created",
            outcome="accepted",
            attempt=0,
            occurred_at=timestamp,
            full_event_content=event_content,
            checkpoint_digest=None,
            reason_codes=("workflow_created",),
            idempotency_namespace=namespace,
            idempotency_key=key,
            generation=0,
        )
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            operation = self._store_operation(
                workflow=workflow,
                workflow_content=full_workflow_content,
                task=task,
                task_content=full_task_content,
                event=event,
                full_event_content=event_content,
                curation_receipt=None,
                full_result_content=None,
                idempotency_namespace=namespace,
                idempotency_key=key,
                request_digest=request_digest,
            )
            self._connection.commit()
            return operation
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DurableWorkflowTransactionError(
                "workflow creation could not commit"
            ) from exc
        except Exception:
            self._connection.rollback()
            raise

    def _transition(
        self,
        *,
        action: str,
        workflow_id: object,
        task_id: object,
        expected_generation: object,
        now: object,
        idempotency_namespace: object,
        idempotency_key: object,
        full_event_content: Mapping[str, object],
        lease_expires_at: object | None = None,
        retry_scheduled_at: object | None = None,
        checkpoint_content: Mapping[str, object] | None = None,
        signal_content: Mapping[str, object] | None = None,
        full_result_content: Mapping[str, object] | None = None,
        error_code: object | None = None,
    ) -> DurableWorkflowOperationReceipt:
        workflow_id_value = require_identifier(
            workflow_id,
            "workflow_id",
        )
        task_id_value = require_identifier(task_id, "task_id")
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(idempotency_key, "idempotency_key")
        timestamp = normalize_timestamp(now, "now")
        if isinstance(expected_generation, bool) or not isinstance(
            expected_generation,
            int,
        ):
            raise CognitiveKernelContractError(
                "expected_generation must be an integer"
            )
        lease_timestamp = (
            normalize_timestamp(lease_expires_at, "lease_expires_at")
            if lease_expires_at is not None
            else None
        )
        retry_timestamp = (
            normalize_timestamp(
                retry_scheduled_at,
                "retry_scheduled_at",
            )
            if retry_scheduled_at is not None
            else None
        )
        error_value = (
            require_identifier(error_code, "error_code")
            if error_code is not None
            else None
        )
        request = {
            "action": action,
            "workflow_id": workflow_id_value,
            "task_id": task_id_value,
            "expected_generation": expected_generation,
            "now": timestamp,
            "lease_expires_at": lease_timestamp,
            "retry_scheduled_at": retry_timestamp,
            "checkpoint_content": (
                dict(checkpoint_content)
                if checkpoint_content is not None
                else None
            ),
            "signal_content": (
                dict(signal_content)
                if signal_content is not None
                else None
            ),
            "result_content": (
                dict(full_result_content)
                if full_result_content is not None
                else None
            ),
            "event_content": dict(full_event_content),
            "error_code": error_value,
        }
        request_digest = canonical_sha256(request)
        existing = self._lookup_idempotency(
            namespace=namespace,
            key=key,
            request_digest=request_digest,
        )
        if existing is not None:
            return existing

        prior_workflow, workflow_content = self._load_workflow_row(
            workflow_id_value
        )
        prior_task, task_content = self._load_task_row(task_id_value)
        if prior_task.workflow_id != workflow_id_value:
            raise DurableWorkflowConflictError(
                "task does not belong to workflow"
            )
        if prior_workflow.generation != expected_generation:
            raise DurableWorkflowConflictError(
                "workflow generation changed"
            )
        if prior_workflow.workflow_state in {
            "completed",
            "failed",
            "cancelled",
            "rolled_back",
        }:
            raise DurableWorkflowConflictError(
                "terminal workflow may not transition"
            )

        generation = prior_workflow.generation + 1
        attempt = prior_task.attempt
        task_state = prior_task.task_state
        workflow_state = prior_workflow.workflow_state
        completed_at: str | None = None
        checkpoint_digest = prior_task.checkpoint_digest
        signal_ids = prior_workflow.signal_ids
        activity_kind = "custom"
        outcome = "accepted"
        reason_codes: tuple[str, ...] = (action,)
        receipt_outcome: str | None = None
        receipt_error: str | None = None
        receipt_retry: str | None = None
        checkpoint_row = None
        signal_row = None

        if action in {"start", "resume"}:
            if task_state not in {"pending", "retry_scheduled", "waiting"}:
                raise DurableWorkflowConflictError(
                    "task is not startable"
                )
            if lease_timestamp is None:
                raise CognitiveKernelContractError(
                    "start or resume requires lease_expires_at"
                )
            if attempt >= prior_task.max_attempts:
                raise DurableWorkflowConflictError(
                    "task attempts are exhausted"
                )
            attempt += 1
            task_state = "running"
            workflow_state = "running"
            activity_kind = (
                "task_resumed"
                if action == "resume"
                else "task_started"
            )
        elif action == "checkpoint":
            if task_state != "running":
                raise DurableWorkflowConflictError(
                    "checkpoint requires running task"
                )
            if checkpoint_content is None:
                raise CognitiveKernelContractError(
                    "checkpoint requires content"
                )
            checkpoint_digest = canonical_sha256(checkpoint_content)
            checkpoint_id = _make_id(
                "checkpoint",
                {
                    "workflow_id": workflow_id_value,
                    "task_id": task_id_value,
                    "generation": generation,
                    "digest": checkpoint_digest,
                },
            )
            checkpoint_row = (
                checkpoint_id,
                checkpoint_content,
                timestamp,
            )
            task_state = "waiting"
            workflow_state = "waiting"
            lease_timestamp = None
            activity_kind = "checkpoint_saved"
            outcome = "completed"
        elif action == "failure":
            if task_state not in {"running", "waiting"}:
                raise DurableWorkflowConflictError(
                    "failure requires active task"
                )
            if full_result_content is None or error_value is None:
                raise CognitiveKernelContractError(
                    "failure requires result content and error_code"
                )
            lease_timestamp = None
            activity_kind = "task_failed"
            outcome = "failed"
            receipt_error = error_value
            if attempt < prior_task.max_attempts:
                if retry_timestamp is None:
                    raise CognitiveKernelContractError(
                        "retryable failure requires retry_scheduled_at"
                    )
                task_state = "retry_scheduled"
                workflow_state = "retry_scheduled"
                receipt_outcome = "deferred"
                receipt_retry = retry_timestamp
                reason_codes = ("task_failed", "retry_scheduled")
            else:
                task_state = "failed"
                workflow_state = "failed"
                completed_at = timestamp
                receipt_outcome = "failed"
                reason_codes = ("task_failed", "attempts_exhausted")
        elif action == "complete":
            if task_state not in {"running", "waiting"}:
                raise DurableWorkflowConflictError(
                    "completion requires active task"
                )
            if full_result_content is None:
                raise CognitiveKernelContractError(
                    "completion requires result content"
                )
            task_state = "completed"
            workflow_state = "completed"
            completed_at = timestamp
            lease_timestamp = None
            activity_kind = "task_completed"
            outcome = "completed"
            receipt_outcome = "completed"
        elif action == "signal":
            if signal_content is None:
                raise CognitiveKernelContractError(
                    "signal requires content"
                )
            signal_id = _make_id(
                "signal",
                {
                    "workflow_id": workflow_id_value,
                    "generation": generation,
                    "content": dict(signal_content),
                },
            )
            signal_ids = (*signal_ids, signal_id)
            signal_row = (signal_id, signal_content, timestamp)
            activity_kind = "signal_received"
            outcome = "accepted"
        elif action == "cancel":
            task_state = "cancelled"
            workflow_state = "cancelled"
            completed_at = timestamp
            lease_timestamp = None
            activity_kind = "workflow_cancelled"
            outcome = "cancelled"
            receipt_outcome = "cancelled"
            if full_result_content is None:
                full_result_content = {"reason": "cancelled"}
        else:
            raise CognitiveKernelContractError(
                f"unsupported workflow action: {action}"
            )

        task = self._task_value(
            prior=prior_task,
            task_id=prior_task.task_id,
            workflow_id=prior_task.workflow_id,
            task_kind=prior_task.task_kind,
            task_state=task_state,
            target_record_ids=prior_task.target_record_ids,
            source_record_ids=prior_task.source_record_ids,
            priority=prior_task.priority,
            attempt=attempt,
            max_attempts=prior_task.max_attempts,
            scheduled_at=(
                retry_timestamp
                if task_state == "retry_scheduled"
                and retry_timestamp is not None
                else prior_task.scheduled_at
            ),
            lease_expires_at=lease_timestamp,
            checkpoint_digest=checkpoint_digest,
            instruction_digest=prior_task.instruction_digest,
            task_content_digest=prior_task.task_content_digest,
            idempotency_namespace=namespace,
            idempotency_key=key,
            now=timestamp,
            generation=generation,
        )
        current_ids = (
            ()
            if task_state in {"completed", "failed", "cancelled"}
            else (task.task_id,)
        )
        completed_ids = (
            (task.task_id,)
            if task_state == "completed"
            else prior_workflow.completed_task_ids
        )
        failed_ids = (
            (task.task_id,)
            if task_state == "failed"
            else prior_workflow.failed_task_ids
        )
        workflow = self._workflow_value(
            prior=prior_workflow,
            workflow_id=prior_workflow.workflow_id,
            workflow_kind=prior_workflow.workflow_kind,
            workflow_state=workflow_state,
            root_task_id=prior_workflow.root_task_id,
            current_task_ids=current_ids,
            completed_task_ids=completed_ids,
            failed_task_ids=failed_ids,
            checkpoint_digest=checkpoint_digest,
            signal_ids=signal_ids,
            generation=generation,
            started_at=prior_workflow.started_at,
            updated_at=timestamp,
            completed_at=completed_at,
            workflow_content_digest=prior_workflow.workflow_content_digest,
            idempotency_namespace=namespace,
            idempotency_key=key,
        )
        event = self._event_value(
            workflow_id=workflow_id_value,
            task_id=task_id_value,
            activity_kind=activity_kind,
            outcome=outcome,
            attempt=task.attempt,
            occurred_at=timestamp,
            full_event_content=full_event_content,
            checkpoint_digest=checkpoint_digest,
            reason_codes=reason_codes,
            idempotency_namespace=namespace,
            idempotency_key=key,
            generation=generation,
        )
        receipt = None
        if receipt_outcome is not None:
            assert full_result_content is not None
            receipt = self._receipt_value(
                workflow_id=workflow_id_value,
                task=task,
                event=event,
                outcome=receipt_outcome,
                started_at=prior_workflow.updated_at,
                completed_at=timestamp,
                full_result_content=full_result_content,
                error_code=receipt_error,
                retry_scheduled_at=receipt_retry,
                idempotency_namespace=namespace,
                idempotency_key=key,
                generation=generation,
            )
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            operation = self._store_operation(
                workflow=workflow,
                workflow_content=workflow_content,
                task=task,
                task_content=task_content,
                event=event,
                full_event_content=full_event_content,
                curation_receipt=receipt,
                full_result_content=full_result_content,
                idempotency_namespace=namespace,
                idempotency_key=key,
                request_digest=request_digest,
                checkpoint=checkpoint_row,
                signal=signal_row,
            )
            self._connection.commit()
            return operation
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DurableWorkflowTransactionError(
                "workflow transition could not commit"
            ) from exc
        except Exception:
            self._connection.rollback()
            raise

    def start_task(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="start", **kwargs)

    def resume_task(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="resume", **kwargs)

    def save_checkpoint(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="checkpoint", **kwargs)

    def record_failure(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="failure", **kwargs)

    def complete_task(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="complete", **kwargs)

    def signal_workflow(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="signal", **kwargs)

    def cancel_workflow(self, **kwargs: object) -> DurableWorkflowOperationReceipt:
        return self._transition(action="cancel", **kwargs)

    def verify_integrity(self) -> DurableWorkflowIntegrityReport:
        problems: list[str] = []
        workflow_rows = self._connection.execute(
            "SELECT workflow_id, workflow_json, full_content_json, "
            "workflow_sha256 FROM workflows ORDER BY workflow_id"
        ).fetchall()
        task_rows = self._connection.execute(
            "SELECT task_id, task_json, full_content_json, task_sha256 "
            "FROM tasks ORDER BY task_id"
        ).fetchall()
        event_rows = self._connection.execute(
            "SELECT event_id, workflow_id, sequence_number, event_json, "
            "full_content_json, event_sha256 "
            "FROM activity_events ORDER BY workflow_id, sequence_number"
        ).fetchall()
        receipt_rows = self._connection.execute(
            "SELECT receipt_id, receipt_json, full_result_json, "
            "receipt_sha256 FROM curation_receipts ORDER BY receipt_id"
        ).fetchall()
        operation_rows = self._connection.execute(
            "SELECT operation_id FROM operation_receipts "
            "ORDER BY operation_id"
        ).fetchall()

        for row in workflow_rows:
            try:
                workflow = _workflow_from_record(
                    json.loads(str(row["workflow_json"]))
                )
                content = json.loads(str(row["full_content_json"]))
                if canonical_sha256(content) != workflow.workflow_content_digest:
                    raise ValueError("full content digest mismatch")
                if workflow.workflow_sha256 != str(row["workflow_sha256"]):
                    raise ValueError("stored workflow digest mismatch")
            except Exception as exc:
                problems.append(
                    f"workflow:{row['workflow_id']}:{exc}"
                )
        for row in task_rows:
            try:
                task = _task_from_record(
                    json.loads(str(row["task_json"]))
                )
                content = json.loads(str(row["full_content_json"]))
                if canonical_sha256(content) != task.task_content_digest:
                    raise ValueError("full content digest mismatch")
                if task.task_sha256 != str(row["task_sha256"]):
                    raise ValueError("stored task digest mismatch")
            except Exception as exc:
                problems.append(f"task:{row['task_id']}:{exc}")
        last_sequence: dict[str, int] = {}
        for row in event_rows:
            try:
                event = _event_from_record(
                    json.loads(str(row["event_json"]))
                )
                content = json.loads(str(row["full_content_json"]))
                if canonical_sha256(content) != event.envelope.content_digest:
                    raise ValueError("full event content digest mismatch")
                if event.event_sha256 != str(row["event_sha256"]):
                    raise ValueError("stored event digest mismatch")
                expected = last_sequence.get(event.workflow_id, 0) + 1
                if event.sequence_number != expected:
                    raise ValueError("activity sequence gap")
                last_sequence[event.workflow_id] = event.sequence_number
            except Exception as exc:
                problems.append(f"event:{row['event_id']}:{exc}")
        for row in receipt_rows:
            try:
                receipt = _receipt_from_record(
                    json.loads(str(row["receipt_json"]))
                )
                content = json.loads(str(row["full_result_json"]))
                if canonical_sha256(content) != receipt.result_content_digest:
                    raise ValueError("full result digest mismatch")
                if receipt.receipt_sha256 != str(row["receipt_sha256"]):
                    raise ValueError("stored receipt digest mismatch")
            except Exception as exc:
                problems.append(
                    f"receipt:{row['receipt_id']}:{exc}"
                )
        for row in operation_rows:
            try:
                self.load_operation(str(row["operation_id"]))
            except Exception as exc:
                problems.append(
                    f"operation:{row['operation_id']}:{exc}"
                )
        return DurableWorkflowIntegrityReport(
            checked_workflows=len(workflow_rows),
            checked_tasks=len(task_rows),
            checked_events=len(event_rows),
            checked_receipts=len(receipt_rows),
            checked_operations=len(operation_rows),
            problems=tuple(problems),
        )

    def require_integrity(self) -> DurableWorkflowIntegrityReport:
        report = self.verify_integrity()
        if not report.healthy:
            raise DurableWorkflowIntegrityError(
                "; ".join(report.problems)
            )
        return report

    def remove_database(self) -> None:
        path = self.path
        self.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)


def open_durable_workflow_prototype(
    *,
    path: Path,
    profile: DurableWorkflowProfile,
    repository_root: Path | None = None,
) -> DurableWorkflowPrototypeStore:
    return DurableWorkflowPrototypeStore(
        path=path,
        profile=profile,
        repository_root=repository_root,
    )
