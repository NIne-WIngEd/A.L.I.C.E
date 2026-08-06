"""Memory M2.5 curation and durable-workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

CURATION_CONTRACT_SCHEMA_VERSION = "1.0.0"

CURATION_TASK_KINDS = frozenset(
    {
        "selective_memory",
        "deduplication",
        "conflict_review",
        "projection_refresh",
        "migration",
        "repair",
        "learning",
        "deletion_rehearsal",
        "custom",
    }
)
CURATION_TASK_STATES = frozenset(
    {
        "pending",
        "leased",
        "running",
        "waiting",
        "retry_scheduled",
        "completed",
        "failed",
        "cancelled",
        "superseded",
    }
)
CURATION_RECEIPT_OUTCOMES = frozenset(
    {
        "completed",
        "partial",
        "rejected",
        "failed",
        "cancelled",
        "deferred",
    }
)
WORKFLOW_KINDS = frozenset(
    {
        "curator",
        "migration",
        "repair",
        "learning",
        "deletion",
        "mission",
        "custom",
    }
)
WORKFLOW_STATES = frozenset(
    {
        "pending",
        "running",
        "waiting",
        "retry_scheduled",
        "completed",
        "failed",
        "cancelled",
        "compensating",
        "rolled_back",
    }
)
WORKFLOW_ACTIVITY_KINDS = frozenset(
    {
        "workflow_created",
        "task_started",
        "checkpoint_saved",
        "task_failed",
        "retry_scheduled",
        "task_resumed",
        "task_completed",
        "signal_received",
        "workflow_cancelled",
        "workflow_completed",
        "compensation_started",
        "compensation_completed",
        "custom",
    }
)
WORKFLOW_ACTIVITY_OUTCOMES = frozenset(
    {
        "accepted",
        "completed",
        "partial",
        "failed",
        "cancelled",
        "deferred",
        "compensated",
        "none",
    }
)

_TERMINAL_TASK_STATES = frozenset({"completed", "failed", "cancelled", "superseded"})
_TERMINAL_WORKFLOW_STATES = frozenset(
    {"completed", "failed", "cancelled", "rolled_back"}
)


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CognitiveKernelContractError(
            f"{field} must be a positive integer"
        )
    return value


def _priority(value: object) -> int:
    normalized = _non_negative_integer(value, "priority")
    if normalized > 100:
        raise CognitiveKernelContractError(
            "priority must be between 0 and 100"
        )
    return normalized


def _optional_identifier(
    value: object | None,
    field: str,
) -> str | None:
    return (
        require_identifier(value, field)
        if value is not None
        else None
    )


def _optional_sha256(
    value: object | None,
    field: str,
) -> str | None:
    return (
        require_sha256(value, field)
        if value is not None
        else None
    )


def _optional_timestamp(
    value: object | None,
    field: str,
) -> str | None:
    return (
        normalize_timestamp(value, field)
        if value is not None
        else None
    )


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(normalize_identifier_sequence(values, field))
    )


def _require_envelope(
    envelope: MemoryUnitEnvelope,
    *,
    record_type: str,
    record_id: str,
) -> None:
    if not isinstance(envelope, MemoryUnitEnvelope):
        raise CognitiveKernelContractError(
            "envelope must be MemoryUnitEnvelope"
        )
    envelope.validate()
    if envelope.record_type != record_type:
        raise CognitiveKernelContractError(
            f"envelope.record_type must be {record_type}"
        )
    if envelope.record_id != record_id:
        raise CognitiveKernelContractError(
            "envelope.record_id must equal the contract record id"
        )
    if envelope.authority_role != "operational_workflow_state":
        raise CognitiveKernelContractError(
            "workflow records require operational_workflow_state authority"
        )


@dataclass(frozen=True)
class CurationTask:
    """One durable, authority-scoped curation unit of work."""

    envelope: MemoryUnitEnvelope
    task_id: str
    workflow_id: str
    task_kind: str
    task_state: str
    target_record_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    priority: int
    attempt: int
    max_attempts: int
    scheduled_at: str
    lease_expires_at: str | None
    checkpoint_digest: str | None
    instruction_digest: str
    task_content_digest: str
    task_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        task_id: object,
        workflow_id: object,
        task_kind: object,
        task_state: object,
        target_record_ids: Iterable[object] = (),
        source_record_ids: Iterable[object] = (),
        priority: object = 50,
        attempt: object = 0,
        max_attempts: object = 3,
        scheduled_at: object,
        lease_expires_at: object | None = None,
        checkpoint_digest: object | None = None,
        instruction_digest: object,
        task_content_digest: object,
    ) -> "CurationTask":
        draft = cls(
            envelope=envelope,
            task_id=require_identifier(task_id, "task_id"),
            workflow_id=require_identifier(workflow_id, "workflow_id"),
            task_kind=require_identifier(task_kind, "task_kind"),
            task_state=require_identifier(task_state, "task_state"),
            target_record_ids=_sorted_identifiers(
                target_record_ids,
                "target_record_ids",
            ),
            source_record_ids=_sorted_identifiers(
                source_record_ids,
                "source_record_ids",
            ),
            priority=_priority(priority),
            attempt=_non_negative_integer(attempt, "attempt"),
            max_attempts=_positive_integer(max_attempts, "max_attempts"),
            scheduled_at=normalize_timestamp(
                scheduled_at,
                "scheduled_at",
            ),
            lease_expires_at=_optional_timestamp(
                lease_expires_at,
                "lease_expires_at",
            ),
            checkpoint_digest=_optional_sha256(
                checkpoint_digest,
                "checkpoint_digest",
            ),
            instruction_digest=require_sha256(
                instruction_digest,
                "instruction_digest",
            ),
            task_content_digest=require_sha256(
                task_content_digest,
                "task_content_digest",
            ),
            task_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "task_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": CURATION_CONTRACT_SCHEMA_VERSION,
            "envelope_sha256": self.envelope.envelope_sha256,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "task_kind": self.task_kind,
            "task_state": self.task_state,
            "target_record_ids": list(self.target_record_ids),
            "source_record_ids": list(self.source_record_ids),
            "priority": self.priority,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "scheduled_at": self.scheduled_at,
            "lease_expires_at": self.lease_expires_at,
            "checkpoint_digest": self.checkpoint_digest,
            "instruction_digest": self.instruction_digest,
            "task_content_digest": self.task_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "task_sha256": self.task_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="curation_task",
            record_id=self.task_id,
        )
        if require_identifier(self.workflow_id, "workflow_id") != self.workflow_id:
            raise CognitiveKernelContractError(
                "workflow_id is not canonical"
            )
        if self.task_kind not in CURATION_TASK_KINDS:
            raise CognitiveKernelContractError(
                "task_kind is not registered"
            )
        if self.task_state not in CURATION_TASK_STATES:
            raise CognitiveKernelContractError(
                "task_state is not registered"
            )
        _priority(self.priority)
        _non_negative_integer(self.attempt, "attempt")
        _positive_integer(self.max_attempts, "max_attempts")
        if self.attempt > self.max_attempts:
            raise CognitiveKernelContractError(
                "attempt may not exceed max_attempts"
            )
        if (
            self.attempt == self.max_attempts
            and self.task_state == "retry_scheduled"
        ):
            raise CognitiveKernelContractError(
                "exhausted task may not schedule another retry"
            )
        if self.task_state in {"leased", "running"} and self.lease_expires_at is None:
            raise CognitiveKernelContractError(
                "leased or running task requires lease_expires_at"
            )
        if self.lease_expires_at is not None:
            if normalize_timestamp(
                self.lease_expires_at,
                "lease_expires_at",
            ) != self.lease_expires_at:
                raise CognitiveKernelContractError(
                    "lease_expires_at is not canonical"
                )
        if self.task_state in _TERMINAL_TASK_STATES and self.lease_expires_at is not None:
            raise CognitiveKernelContractError(
                "terminal task may not retain an active lease"
            )
        for values, field in (
            (self.target_record_ids, "target_record_ids"),
            (self.source_record_ids, "source_record_ids"),
        ):
            if _sorted_identifiers(values, field) != values:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        require_sha256(self.instruction_digest, "instruction_digest")
        require_sha256(self.task_content_digest, "task_content_digest")
        if self.checkpoint_digest is not None:
            require_sha256(self.checkpoint_digest, "checkpoint_digest")
        if canonical_sha256(self.semantic_record()) != self.task_sha256:
            raise CognitiveKernelContractError(
                "task_sha256 does not match task content"
            )


@dataclass(frozen=True)
class CurationReceipt:
    """Immutable outcome receipt for one curation task attempt."""

    envelope: MemoryUnitEnvelope
    receipt_id: str
    task_id: str
    workflow_id: str
    outcome: str
    started_at: str
    completed_at: str
    input_record_ids: tuple[str, ...]
    output_record_ids: tuple[str, ...]
    activity_event_ids: tuple[str, ...]
    attempt: int
    checkpoint_digest: str | None
    result_content_digest: str
    error_code: str | None
    retry_scheduled_at: str | None
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        receipt_id: object,
        task_id: object,
        workflow_id: object,
        outcome: object,
        started_at: object,
        completed_at: object,
        input_record_ids: Iterable[object] = (),
        output_record_ids: Iterable[object] = (),
        activity_event_ids: Iterable[object] = (),
        attempt: object,
        checkpoint_digest: object | None = None,
        result_content_digest: object,
        error_code: object | None = None,
        retry_scheduled_at: object | None = None,
    ) -> "CurationReceipt":
        draft = cls(
            envelope=envelope,
            receipt_id=require_identifier(receipt_id, "receipt_id"),
            task_id=require_identifier(task_id, "task_id"),
            workflow_id=require_identifier(workflow_id, "workflow_id"),
            outcome=require_identifier(outcome, "outcome"),
            started_at=normalize_timestamp(started_at, "started_at"),
            completed_at=normalize_timestamp(completed_at, "completed_at"),
            input_record_ids=_sorted_identifiers(
                input_record_ids,
                "input_record_ids",
            ),
            output_record_ids=_sorted_identifiers(
                output_record_ids,
                "output_record_ids",
            ),
            activity_event_ids=tuple(
                normalize_identifier_sequence(
                    activity_event_ids,
                    "activity_event_ids",
                )
            ),
            attempt=_non_negative_integer(attempt, "attempt"),
            checkpoint_digest=_optional_sha256(
                checkpoint_digest,
                "checkpoint_digest",
            ),
            result_content_digest=require_sha256(
                result_content_digest,
                "result_content_digest",
            ),
            error_code=_optional_identifier(error_code, "error_code"),
            retry_scheduled_at=_optional_timestamp(
                retry_scheduled_at,
                "retry_scheduled_at",
            ),
            receipt_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "receipt_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": CURATION_CONTRACT_SCHEMA_VERSION,
            "envelope_sha256": self.envelope.envelope_sha256,
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "input_record_ids": list(self.input_record_ids),
            "output_record_ids": list(self.output_record_ids),
            "activity_event_ids": list(self.activity_event_ids),
            "attempt": self.attempt,
            "checkpoint_digest": self.checkpoint_digest,
            "result_content_digest": self.result_content_digest,
            "error_code": self.error_code,
            "retry_scheduled_at": self.retry_scheduled_at,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "receipt_sha256": self.receipt_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="curation_receipt",
            record_id=self.receipt_id,
        )
        if self.outcome not in CURATION_RECEIPT_OUTCOMES:
            raise CognitiveKernelContractError(
                "outcome is not registered"
            )
        if self.completed_at < self.started_at:
            raise CognitiveKernelContractError(
                "completed_at precedes started_at"
            )
        if self.outcome in {"failed", "rejected"} and self.error_code is None:
            raise CognitiveKernelContractError(
                "failed or rejected receipt requires error_code"
            )
        if self.outcome == "deferred" and self.retry_scheduled_at is None:
            raise CognitiveKernelContractError(
                "deferred receipt requires retry_scheduled_at"
            )
        if self.retry_scheduled_at is not None and self.retry_scheduled_at < self.completed_at:
            raise CognitiveKernelContractError(
                "retry_scheduled_at precedes completed_at"
            )
        _non_negative_integer(self.attempt, "attempt")
        require_sha256(
            self.result_content_digest,
            "result_content_digest",
        )
        if self.checkpoint_digest is not None:
            require_sha256(self.checkpoint_digest, "checkpoint_digest")
        if canonical_sha256(self.semantic_record()) != self.receipt_sha256:
            raise CognitiveKernelContractError(
                "receipt_sha256 does not match receipt content"
            )


@dataclass(frozen=True)
class DurableWorkflow:
    """Materialized durable state for one long-running workflow."""

    envelope: MemoryUnitEnvelope
    workflow_id: str
    workflow_kind: str
    workflow_state: str
    root_task_id: str
    current_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    checkpoint_digest: str | None
    signal_ids: tuple[str, ...]
    generation: int
    started_at: str
    updated_at: str
    completed_at: str | None
    workflow_content_digest: str
    workflow_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        workflow_id: object,
        workflow_kind: object,
        workflow_state: object,
        root_task_id: object,
        current_task_ids: Iterable[object] = (),
        completed_task_ids: Iterable[object] = (),
        failed_task_ids: Iterable[object] = (),
        checkpoint_digest: object | None = None,
        signal_ids: Iterable[object] = (),
        generation: object,
        started_at: object,
        updated_at: object,
        completed_at: object | None = None,
        workflow_content_digest: object,
    ) -> "DurableWorkflow":
        draft = cls(
            envelope=envelope,
            workflow_id=require_identifier(
                workflow_id,
                "workflow_id",
            ),
            workflow_kind=require_identifier(
                workflow_kind,
                "workflow_kind",
            ),
            workflow_state=require_identifier(
                workflow_state,
                "workflow_state",
            ),
            root_task_id=require_identifier(
                root_task_id,
                "root_task_id",
            ),
            current_task_ids=_sorted_identifiers(
                current_task_ids,
                "current_task_ids",
            ),
            completed_task_ids=_sorted_identifiers(
                completed_task_ids,
                "completed_task_ids",
            ),
            failed_task_ids=_sorted_identifiers(
                failed_task_ids,
                "failed_task_ids",
            ),
            checkpoint_digest=_optional_sha256(
                checkpoint_digest,
                "checkpoint_digest",
            ),
            signal_ids=tuple(
                normalize_identifier_sequence(
                    signal_ids,
                    "signal_ids",
                )
            ),
            generation=_non_negative_integer(
                generation,
                "generation",
            ),
            started_at=normalize_timestamp(
                started_at,
                "started_at",
            ),
            updated_at=normalize_timestamp(
                updated_at,
                "updated_at",
            ),
            completed_at=_optional_timestamp(
                completed_at,
                "completed_at",
            ),
            workflow_content_digest=require_sha256(
                workflow_content_digest,
                "workflow_content_digest",
            ),
            workflow_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "workflow_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": CURATION_CONTRACT_SCHEMA_VERSION,
            "envelope_sha256": self.envelope.envelope_sha256,
            "workflow_id": self.workflow_id,
            "workflow_kind": self.workflow_kind,
            "workflow_state": self.workflow_state,
            "root_task_id": self.root_task_id,
            "current_task_ids": list(self.current_task_ids),
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "checkpoint_digest": self.checkpoint_digest,
            "signal_ids": list(self.signal_ids),
            "generation": self.generation,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "workflow_content_digest": self.workflow_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "workflow_sha256": self.workflow_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="durable_workflow",
            record_id=self.workflow_id,
        )
        if self.workflow_kind not in WORKFLOW_KINDS:
            raise CognitiveKernelContractError(
                "workflow_kind is not registered"
            )
        if self.workflow_state not in WORKFLOW_STATES:
            raise CognitiveKernelContractError(
                "workflow_state is not registered"
            )
        groups = (
            set(self.current_task_ids),
            set(self.completed_task_ids),
            set(self.failed_task_ids),
        )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise CognitiveKernelContractError(
                "workflow task state sets must be disjoint"
            )
        if (
            self.workflow_state != "cancelled"
            and self.root_task_id not in groups[0] | groups[1] | groups[2]
        ):
            raise CognitiveKernelContractError(
                "root_task_id must appear in one workflow task set"
            )
        if self.updated_at < self.started_at:
            raise CognitiveKernelContractError(
                "updated_at precedes started_at"
            )
        if self.workflow_state in _TERMINAL_WORKFLOW_STATES:
            if self.completed_at is None:
                raise CognitiveKernelContractError(
                    "terminal workflow requires completed_at"
                )
            if self.completed_at < self.updated_at:
                raise CognitiveKernelContractError(
                    "completed_at precedes updated_at"
                )
        elif self.completed_at is not None:
            raise CognitiveKernelContractError(
                "nonterminal workflow may not set completed_at"
            )
        _non_negative_integer(self.generation, "generation")
        require_sha256(
            self.workflow_content_digest,
            "workflow_content_digest",
        )
        if self.checkpoint_digest is not None:
            require_sha256(self.checkpoint_digest, "checkpoint_digest")
        if canonical_sha256(self.semantic_record()) != self.workflow_sha256:
            raise CognitiveKernelContractError(
                "workflow_sha256 does not match workflow content"
            )


@dataclass(frozen=True)
class WorkflowActivityEvent:
    """Immutable ordered activity event in a durable workflow."""

    envelope: MemoryUnitEnvelope
    event_id: str
    workflow_id: str
    task_id: str | None
    activity_kind: str
    outcome: str
    sequence_number: int
    attempt: int
    occurred_at: str
    input_digest: str | None
    output_digest: str | None
    checkpoint_digest: str | None
    reason_codes: tuple[str, ...]
    idempotency_key: str
    event_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        event_id: object,
        workflow_id: object,
        task_id: object | None,
        activity_kind: object,
        outcome: object,
        sequence_number: object,
        attempt: object,
        occurred_at: object,
        input_digest: object | None = None,
        output_digest: object | None = None,
        checkpoint_digest: object | None = None,
        reason_codes: Iterable[object] = (),
        idempotency_key: object,
    ) -> "WorkflowActivityEvent":
        draft = cls(
            envelope=envelope,
            event_id=require_identifier(event_id, "event_id"),
            workflow_id=require_identifier(
                workflow_id,
                "workflow_id",
            ),
            task_id=_optional_identifier(task_id, "task_id"),
            activity_kind=require_identifier(
                activity_kind,
                "activity_kind",
            ),
            outcome=require_identifier(outcome, "outcome"),
            sequence_number=_positive_integer(
                sequence_number,
                "sequence_number",
            ),
            attempt=_non_negative_integer(attempt, "attempt"),
            occurred_at=normalize_timestamp(
                occurred_at,
                "occurred_at",
            ),
            input_digest=_optional_sha256(
                input_digest,
                "input_digest",
            ),
            output_digest=_optional_sha256(
                output_digest,
                "output_digest",
            ),
            checkpoint_digest=_optional_sha256(
                checkpoint_digest,
                "checkpoint_digest",
            ),
            reason_codes=_sorted_identifiers(
                reason_codes,
                "reason_codes",
            ),
            idempotency_key=require_identifier(
                idempotency_key,
                "idempotency_key",
            ),
            event_sha256="0" * 64,
        )
        value = cls(
            **{
                **draft.__dict__,
                "event_sha256": canonical_sha256(
                    draft.semantic_record()
                ),
            }
        )
        value.validate()
        return value

    @property
    def scope(self) -> ProductHostScope:
        return self.envelope.scope

    def semantic_record(self) -> dict[str, object]:
        return {
            "schema_version": CURATION_CONTRACT_SCHEMA_VERSION,
            "envelope_sha256": self.envelope.envelope_sha256,
            "event_id": self.event_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "activity_kind": self.activity_kind,
            "outcome": self.outcome,
            "sequence_number": self.sequence_number,
            "attempt": self.attempt,
            "occurred_at": self.occurred_at,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "reason_codes": list(self.reason_codes),
            "idempotency_key": self.idempotency_key,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            **self.semantic_record(),
            "event_sha256": self.event_sha256,
        }

    def validate(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="workflow_activity_event",
            record_id=self.event_id,
        )
        if self.activity_kind not in WORKFLOW_ACTIVITY_KINDS:
            raise CognitiveKernelContractError(
                "activity_kind is not registered"
            )
        if self.outcome not in WORKFLOW_ACTIVITY_OUTCOMES:
            raise CognitiveKernelContractError(
                "outcome is not registered"
            )
        _positive_integer(self.sequence_number, "sequence_number")
        _non_negative_integer(self.attempt, "attempt")
        for value, field in (
            (self.input_digest, "input_digest"),
            (self.output_digest, "output_digest"),
            (self.checkpoint_digest, "checkpoint_digest"),
        ):
            if value is not None:
                require_sha256(value, field)
        if canonical_sha256(self.semantic_record()) != self.event_sha256:
            raise CognitiveKernelContractError(
                "event_sha256 does not match activity content"
            )
