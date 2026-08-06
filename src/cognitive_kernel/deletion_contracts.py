"""Memory M2.6 deletion-propagation and restore-filter contracts."""

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

DELETION_CONTRACT_SCHEMA_VERSION = "1.0.0"

DELETION_MODES = frozenset(
    {
        "tombstone",
        "physical_delete",
        "cryptographic_erasure",
        "projection_rebuild",
        "cache_invalidation",
        "replica_purge",
        "backup_filter",
        "restore_filter",
        "influence_reduction",
        "model_retirement",
        "mixed",
    }
)
DELETION_PLANE_KINDS = frozenset(
    {
        "claim_authority",
        "experience_ledger",
        "raw_payload",
        "episode_projection",
        "graph_index",
        "vector_index",
        "lexical_index",
        "retrieval_serving",
        "workflow_state",
        "cache",
        "replica",
        "backup",
        "model_artifact",
        "training_dataset",
        "evaluation_artifact",
        "custom",
    }
)
DELETION_PLANE_STATES = frozenset(
    {
        "pending",
        "acknowledged",
        "completed",
        "failed",
        "blocked",
        "not_applicable",
        "superseded",
    }
)
DELETION_PROPAGATION_STATES = frozenset(
    {
        "requested",
        "propagating",
        "partially_completed",
        "completed",
        "blocked",
        "failed",
        "rolled_back",
        "retired",
    }
)
RESTORE_FILTER_ACTIONS = frozenset(
    {
        "allow",
        "exclude",
        "redact",
        "replace",
        "revalidate",
    }
)
ROLLBACK_STATES = frozenset(
    {
        "not_requested",
        "rehearsal_pending",
        "rehearsed",
        "available",
        "executed",
        "not_possible",
        "failed",
    }
)
RETIREMENT_STATES = frozenset(
    {
        "not_applicable",
        "evaluation_pending",
        "retrain",
        "unlearn",
        "retire",
        "replacement_required",
        "completed",
        "failed",
    }
)

_TERMINAL_PLANE_STATES = frozenset(
    {"completed", "failed", "blocked", "not_applicable", "superseded"}
)


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


def _optional_identifier(value: object | None, field: str) -> str | None:
    return require_identifier(value, field) if value is not None else None


def _optional_timestamp(value: object | None, field: str) -> str | None:
    return normalize_timestamp(value, field) if value is not None else None


def _optional_sha256(value: object | None, field: str) -> str | None:
    return require_sha256(value, field) if value is not None else None


def _sorted_identifiers(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    return tuple(sorted(normalize_identifier_sequence(values, field)))


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
            "deletion records require operational_workflow_state authority"
        )


@dataclass(frozen=True)
class DeletionPlaneReceipt:
    """One plane-local result in a cross-plane deletion request."""

    plane_receipt_id: str
    request_id: str
    plane_kind: str
    component_id: str
    deletion_mode: str
    state: str
    requested_at: str
    completed_at: str | None
    target_count: int
    deleted_count: int
    blocked_count: int
    evidence_record_ids: tuple[str, ...]
    error_code: str | None
    result_content_digest: str
    plane_receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        plane_receipt_id: object,
        request_id: object,
        plane_kind: object,
        component_id: object,
        deletion_mode: object,
        state: object,
        requested_at: object,
        completed_at: object | None,
        target_count: object,
        deleted_count: object,
        blocked_count: object,
        evidence_record_ids: Iterable[object] = (),
        error_code: object | None = None,
        result_content_digest: object,
    ) -> "DeletionPlaneReceipt":
        draft = cls(
            plane_receipt_id=require_identifier(
                plane_receipt_id, "plane_receipt_id"
            ),
            request_id=require_identifier(request_id, "request_id"),
            plane_kind=require_identifier(plane_kind, "plane_kind"),
            component_id=require_identifier(component_id, "component_id"),
            deletion_mode=require_identifier(
                deletion_mode, "deletion_mode"
            ),
            state=require_identifier(state, "state"),
            requested_at=normalize_timestamp(
                requested_at, "requested_at"
            ),
            completed_at=_optional_timestamp(
                completed_at, "completed_at"
            ),
            target_count=_non_negative_integer(
                target_count, "target_count"
            ),
            deleted_count=_non_negative_integer(
                deleted_count, "deleted_count"
            ),
            blocked_count=_non_negative_integer(
                blocked_count, "blocked_count"
            ),
            evidence_record_ids=_sorted_identifiers(
                evidence_record_ids, "evidence_record_ids"
            ),
            error_code=_optional_identifier(error_code, "error_code"),
            result_content_digest=require_sha256(
                result_content_digest, "result_content_digest"
            ),
            plane_receipt_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "plane_receipt_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        for value, field in (
            (self.plane_receipt_id, "plane_receipt_id"),
            (self.request_id, "request_id"),
            (self.plane_kind, "plane_kind"),
            (self.component_id, "component_id"),
            (self.deletion_mode, "deletion_mode"),
            (self.state, "state"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.plane_kind not in DELETION_PLANE_KINDS:
            raise CognitiveKernelContractError(
                "plane_kind is not ratified"
            )
        if self.deletion_mode not in DELETION_MODES:
            raise CognitiveKernelContractError(
                "deletion_mode is not ratified"
            )
        if self.state not in DELETION_PLANE_STATES:
            raise CognitiveKernelContractError(
                "state is not ratified"
            )
        if normalize_timestamp(
            self.requested_at, "requested_at"
        ) != self.requested_at:
            raise CognitiveKernelContractError(
                "requested_at is not canonical"
            )
        if _optional_timestamp(
            self.completed_at, "completed_at"
        ) != self.completed_at:
            raise CognitiveKernelContractError(
                "completed_at is not canonical"
            )
        for value, field in (
            (self.target_count, "target_count"),
            (self.deleted_count, "deleted_count"),
            (self.blocked_count, "blocked_count"),
        ):
            _non_negative_integer(value, field)
        if self.deleted_count + self.blocked_count > self.target_count:
            raise CognitiveKernelContractError(
                "deleted_count plus blocked_count exceeds target_count"
            )
        if (
            self.state in _TERMINAL_PLANE_STATES
            and self.completed_at is None
        ):
            raise CognitiveKernelContractError(
                "terminal plane receipt requires completed_at"
            )
        if (
            self.state in {"pending", "acknowledged"}
            and self.completed_at is not None
        ):
            raise CognitiveKernelContractError(
                "nonterminal plane receipt may not have completed_at"
            )
        if (
            self.state == "completed"
            and self.deleted_count != self.target_count
        ):
            raise CognitiveKernelContractError(
                "completed plane receipt must cover every target"
            )
        if (
            self.state in {"failed", "blocked"}
            and self.error_code is None
        ):
            raise CognitiveKernelContractError(
                "failed or blocked plane receipt requires error_code"
            )
        if (
            self.state not in {"failed", "blocked"}
            and self.error_code is not None
        ):
            raise CognitiveKernelContractError(
                "error_code is only valid for failed or blocked state"
            )
        if _sorted_identifiers(
            self.evidence_record_ids, "evidence_record_ids"
        ) != self.evidence_record_ids:
            raise CognitiveKernelContractError(
                "evidence_record_ids are not canonical"
            )
        if require_sha256(
            self.result_content_digest, "result_content_digest"
        ) != self.result_content_digest:
            raise CognitiveKernelContractError(
                "result_content_digest is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "plane_receipt_id": self.plane_receipt_id,
            "request_id": self.request_id,
            "plane_kind": self.plane_kind,
            "component_id": self.component_id,
            "deletion_mode": self.deletion_mode,
            "state": self.state,
            "requested_at": self.requested_at,
            "completed_at": self.completed_at,
            "target_count": self.target_count,
            "deleted_count": self.deleted_count,
            "blocked_count": self.blocked_count,
            "evidence_record_ids": list(self.evidence_record_ids),
            "error_code": self.error_code,
            "result_content_digest": self.result_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": DELETION_CONTRACT_SCHEMA_VERSION,
            **self.material_record(),
            "plane_receipt_sha256": self.plane_receipt_sha256,
        }

    def validate(self) -> None:
        self._validate_material()
        if require_sha256(
            self.plane_receipt_sha256, "plane_receipt_sha256"
        ) != self.plane_receipt_sha256:
            raise CognitiveKernelContractError(
                "plane_receipt_sha256 is not canonical"
            )
        if canonical_sha256(
            self.material_record()
        ) != self.plane_receipt_sha256:
            raise CognitiveKernelContractError(
                "plane_receipt_sha256 does not match material"
            )


@dataclass(frozen=True)
class RestoreFilterDecision:
    """Restore-time action for one deleted or deletion-sensitive record."""

    decision_id: str
    request_id: str
    target_record_id: str
    source_snapshot_id: str
    action: str
    reason_code: str
    evaluated_at: str
    replacement_record_id: str | None
    source_content_digest: str
    decision_sha256: str

    @classmethod
    def create(
        cls,
        *,
        decision_id: object,
        request_id: object,
        target_record_id: object,
        source_snapshot_id: object,
        action: object,
        reason_code: object,
        evaluated_at: object,
        replacement_record_id: object | None = None,
        source_content_digest: object,
    ) -> "RestoreFilterDecision":
        draft = cls(
            decision_id=require_identifier(decision_id, "decision_id"),
            request_id=require_identifier(request_id, "request_id"),
            target_record_id=require_identifier(
                target_record_id, "target_record_id"
            ),
            source_snapshot_id=require_identifier(
                source_snapshot_id, "source_snapshot_id"
            ),
            action=require_identifier(action, "action"),
            reason_code=require_identifier(reason_code, "reason_code"),
            evaluated_at=normalize_timestamp(
                evaluated_at, "evaluated_at"
            ),
            replacement_record_id=_optional_identifier(
                replacement_record_id, "replacement_record_id"
            ),
            source_content_digest=require_sha256(
                source_content_digest, "source_content_digest"
            ),
            decision_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "decision_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        for value, field in (
            (self.decision_id, "decision_id"),
            (self.request_id, "request_id"),
            (self.target_record_id, "target_record_id"),
            (self.source_snapshot_id, "source_snapshot_id"),
            (self.action, "action"),
            (self.reason_code, "reason_code"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.action not in RESTORE_FILTER_ACTIONS:
            raise CognitiveKernelContractError(
                "action is not ratified"
            )
        if (
            self.action == "replace"
            and self.replacement_record_id is None
        ):
            raise CognitiveKernelContractError(
                "replace action requires replacement_record_id"
            )
        if (
            self.action != "replace"
            and self.replacement_record_id is not None
        ):
            raise CognitiveKernelContractError(
                "replacement_record_id is only valid for replace action"
            )
        if normalize_timestamp(
            self.evaluated_at, "evaluated_at"
        ) != self.evaluated_at:
            raise CognitiveKernelContractError(
                "evaluated_at is not canonical"
            )
        if require_sha256(
            self.source_content_digest, "source_content_digest"
        ) != self.source_content_digest:
            raise CognitiveKernelContractError(
                "source_content_digest is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "target_record_id": self.target_record_id,
            "source_snapshot_id": self.source_snapshot_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "evaluated_at": self.evaluated_at,
            "replacement_record_id": self.replacement_record_id,
            "source_content_digest": self.source_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": DELETION_CONTRACT_SCHEMA_VERSION,
            **self.material_record(),
            "decision_sha256": self.decision_sha256,
        }

    def validate(self) -> None:
        self._validate_material()
        if canonical_sha256(
            self.material_record()
        ) != self.decision_sha256:
            raise CognitiveKernelContractError(
                "decision_sha256 does not match material"
            )


@dataclass(frozen=True)
class DeletionPropagationReceipt:
    """Authority-scoped receipt for one cross-plane deletion operation."""

    envelope: MemoryUnitEnvelope
    receipt_id: str
    request_id: str
    deletion_mode: str
    propagation_state: str
    target_record_ids: tuple[str, ...]
    reason_code: str
    authority_decision_id: str
    requested_by: str
    requested_at: str
    effective_at: str | None
    plane_receipts: tuple[DeletionPlaneReceipt, ...]
    restore_filter_decision_ids: tuple[str, ...]
    rollback_state: str
    retirement_state: str
    generation: int
    previous_receipt_id: str | None
    receipt_content_digest: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        envelope: MemoryUnitEnvelope,
        receipt_id: object,
        request_id: object,
        deletion_mode: object,
        propagation_state: object,
        target_record_ids: Iterable[object],
        reason_code: object,
        authority_decision_id: object,
        requested_by: object,
        requested_at: object,
        effective_at: object | None,
        plane_receipts: Iterable[DeletionPlaneReceipt],
        restore_filter_decision_ids: Iterable[object] = (),
        rollback_state: object = "not_requested",
        retirement_state: object = "not_applicable",
        generation: object = 1,
        previous_receipt_id: object | None = None,
        receipt_content_digest: object,
    ) -> "DeletionPropagationReceipt":
        normalized_receipts = tuple(plane_receipts)
        draft = cls(
            envelope=envelope,
            receipt_id=require_identifier(receipt_id, "receipt_id"),
            request_id=require_identifier(request_id, "request_id"),
            deletion_mode=require_identifier(
                deletion_mode, "deletion_mode"
            ),
            propagation_state=require_identifier(
                propagation_state, "propagation_state"
            ),
            target_record_ids=_sorted_identifiers(
                target_record_ids, "target_record_ids"
            ),
            reason_code=require_identifier(reason_code, "reason_code"),
            authority_decision_id=require_identifier(
                authority_decision_id, "authority_decision_id"
            ),
            requested_by=require_identifier(
                requested_by, "requested_by"
            ),
            requested_at=normalize_timestamp(
                requested_at, "requested_at"
            ),
            effective_at=_optional_timestamp(
                effective_at, "effective_at"
            ),
            plane_receipts=tuple(
                sorted(
                    normalized_receipts,
                    key=lambda item: (
                        item.plane_kind,
                        item.component_id,
                        item.plane_receipt_id,
                    ),
                )
            ),
            restore_filter_decision_ids=_sorted_identifiers(
                restore_filter_decision_ids,
                "restore_filter_decision_ids",
            ),
            rollback_state=require_identifier(
                rollback_state, "rollback_state"
            ),
            retirement_state=require_identifier(
                retirement_state, "retirement_state"
            ),
            generation=_non_negative_integer(
                generation, "generation"
            ),
            previous_receipt_id=_optional_identifier(
                previous_receipt_id, "previous_receipt_id"
            ),
            receipt_content_digest=require_sha256(
                receipt_content_digest, "receipt_content_digest"
            ),
            receipt_sha256="0" * 64,
        )
        draft._validate_material()
        value = cls(
            **{
                **draft.__dict__,
                "receipt_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        value.validate()
        return value

    def _validate_material(self) -> None:
        _require_envelope(
            self.envelope,
            record_type="deletion_propagation_receipt",
            record_id=self.receipt_id,
        )
        for value, field in (
            (self.receipt_id, "receipt_id"),
            (self.request_id, "request_id"),
            (self.deletion_mode, "deletion_mode"),
            (self.propagation_state, "propagation_state"),
            (self.reason_code, "reason_code"),
            (self.authority_decision_id, "authority_decision_id"),
            (self.requested_by, "requested_by"),
            (self.rollback_state, "rollback_state"),
            (self.retirement_state, "retirement_state"),
        ):
            if require_identifier(value, field) != value:
                raise CognitiveKernelContractError(
                    f"{field} is not canonical"
                )
        if self.deletion_mode not in DELETION_MODES:
            raise CognitiveKernelContractError(
                "deletion_mode is not ratified"
            )
        if self.propagation_state not in DELETION_PROPAGATION_STATES:
            raise CognitiveKernelContractError(
                "propagation_state is not ratified"
            )
        if self.rollback_state not in ROLLBACK_STATES:
            raise CognitiveKernelContractError(
                "rollback_state is not ratified"
            )
        if self.retirement_state not in RETIREMENT_STATES:
            raise CognitiveKernelContractError(
                "retirement_state is not ratified"
            )
        if not self.target_record_ids:
            raise CognitiveKernelContractError(
                "target_record_ids may not be empty"
            )
        if _sorted_identifiers(
            self.target_record_ids, "target_record_ids"
        ) != self.target_record_ids:
            raise CognitiveKernelContractError(
                "target_record_ids are not canonical"
            )
        if normalize_timestamp(
            self.requested_at, "requested_at"
        ) != self.requested_at:
            raise CognitiveKernelContractError(
                "requested_at is not canonical"
            )
        if _optional_timestamp(
            self.effective_at, "effective_at"
        ) != self.effective_at:
            raise CognitiveKernelContractError(
                "effective_at is not canonical"
            )
        if not self.plane_receipts:
            raise CognitiveKernelContractError(
                "plane_receipts may not be empty"
            )
        keys: set[tuple[str, str]] = set()
        for plane in self.plane_receipts:
            if not isinstance(plane, DeletionPlaneReceipt):
                raise CognitiveKernelContractError(
                    "plane_receipts must contain DeletionPlaneReceipt"
                )
            plane.validate()
            if plane.request_id != self.request_id:
                raise CognitiveKernelContractError(
                    "plane receipt request_id mismatch"
                )
            key = (plane.plane_kind, plane.component_id)
            if key in keys:
                raise CognitiveKernelContractError(
                    "duplicate plane/component receipt"
                )
            keys.add(key)
        canonical_planes = tuple(
            sorted(
                self.plane_receipts,
                key=lambda item: (
                    item.plane_kind,
                    item.component_id,
                    item.plane_receipt_id,
                ),
            )
        )
        if canonical_planes != self.plane_receipts:
            raise CognitiveKernelContractError(
                "plane_receipts are not canonical"
            )
        if self.propagation_state == "completed":
            if any(
                item.state not in {"completed", "not_applicable"}
                for item in self.plane_receipts
            ):
                raise CognitiveKernelContractError(
                    "completed propagation requires terminal successful planes"
                )
            if self.effective_at is None:
                raise CognitiveKernelContractError(
                    "completed propagation requires effective_at"
                )
        if self.propagation_state in {"requested", "propagating"}:
            if self.effective_at is not None:
                raise CognitiveKernelContractError(
                    "nonterminal propagation may not have effective_at"
                )
        if self.generation < 1:
            raise CognitiveKernelContractError(
                "generation must be positive"
            )
        if self.generation == 1 and self.previous_receipt_id is not None:
            raise CognitiveKernelContractError(
                "generation one may not name previous_receipt_id"
            )
        if self.generation > 1 and self.previous_receipt_id is None:
            raise CognitiveKernelContractError(
                "later generation requires previous_receipt_id"
            )
        if require_sha256(
            self.receipt_content_digest, "receipt_content_digest"
        ) != self.receipt_content_digest:
            raise CognitiveKernelContractError(
                "receipt_content_digest is not canonical"
            )

    def material_record(self) -> dict[str, object]:
        self._validate_material()
        return {
            "envelope": self.envelope.metadata_record(),
            "receipt_id": self.receipt_id,
            "request_id": self.request_id,
            "deletion_mode": self.deletion_mode,
            "propagation_state": self.propagation_state,
            "target_record_ids": list(self.target_record_ids),
            "reason_code": self.reason_code,
            "authority_decision_id": self.authority_decision_id,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "effective_at": self.effective_at,
            "plane_receipts": [
                item.metadata_record() for item in self.plane_receipts
            ],
            "restore_filter_decision_ids": list(
                self.restore_filter_decision_ids
            ),
            "rollback_state": self.rollback_state,
            "retirement_state": self.retirement_state,
            "generation": self.generation,
            "previous_receipt_id": self.previous_receipt_id,
            "receipt_content_digest": self.receipt_content_digest,
        }

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": DELETION_CONTRACT_SCHEMA_VERSION,
            **self.material_record(),
            "receipt_sha256": self.receipt_sha256,
        }

    def validate(self) -> None:
        self._validate_material()
        if require_sha256(
            self.receipt_sha256, "receipt_sha256"
        ) != self.receipt_sha256:
            raise CognitiveKernelContractError(
                "receipt_sha256 is not canonical"
            )
        if canonical_sha256(
            self.material_record()
        ) != self.receipt_sha256:
            raise CognitiveKernelContractError(
                "receipt_sha256 does not match material"
            )
