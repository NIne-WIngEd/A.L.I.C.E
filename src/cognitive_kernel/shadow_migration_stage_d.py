"""Deterministic, backend-neutral historical backfill contracts for shadow migration Stage D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_sha256,
    normalize_identifier_sequence,
    normalize_timestamp,
    require_identifier,
    require_schema_version,
    require_sha256,
)
from .contracts import ProductHostScope

SHADOW_MIGRATION_STAGE_D_SCHEMA_VERSION = "1.0.0"
BACKFILL_WORKLOAD_CLASSES = frozenset({"synthetic", "owner_authorized"})
BACKFILL_DISPOSITIONS = frozenset({"accepted", "rejected", "quarantined", "ambiguous"})
BACKFILL_DESTINATION_OUTCOMES = frozenset(
    {"applied", "duplicate", "rejected", "quarantined", "ambiguous"}
)
PROVENANCE_STATES = frozenset({"complete", "partial", "missing"})


def _require_choice(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not an allowed value")
    return normalized


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CognitiveKernelContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class HistoricalBackfillManifest:
    """Digest-bound admission manifest for one Stage D backfill stream."""

    scope: ProductHostScope
    manifest_id: str
    source_registration_id: str
    destination_candidate_id: str
    source_snapshot_sha256: str
    mapping_version: str
    workload_class: str
    authorization_reference_id: str | None
    preferred_batch_size: int
    created_at: str
    authority_effect: str
    serving_effect: str
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        manifest_id: object,
        source_registration_id: object,
        destination_candidate_id: object,
        source_snapshot_sha256: object,
        mapping_version: object,
        workload_class: object,
        preferred_batch_size: object,
        created_at: object,
        authorization_reference_id: object | None = None,
        authority_effect: object = "shadow_candidate_only",
        serving_effect: object = "none",
    ) -> "HistoricalBackfillManifest":
        scope.validate()
        workload = _require_choice(
            workload_class, "workload_class", BACKFILL_WORKLOAD_CLASSES
        )
        authorization = (
            require_identifier(
                authorization_reference_id, "authorization_reference_id"
            )
            if authorization_reference_id is not None
            else None
        )
        if workload == "owner_authorized" and authorization is None:
            raise CognitiveKernelContractError(
                "owner_authorized historical backfill requires an authorization reference"
            )
        if workload == "synthetic" and authorization is not None:
            raise CognitiveKernelContractError(
                "synthetic historical backfill may not claim owner authorization"
            )

        authority = require_identifier(authority_effect, "authority_effect")
        serving = require_identifier(serving_effect, "serving_effect")
        if authority != "shadow_candidate_only":
            raise CognitiveKernelContractError(
                "Stage D manifest authority effect must remain shadow_candidate_only"
            )
        if serving != "none":
            raise CognitiveKernelContractError(
                "Stage D manifest serving effect must remain none"
            )

        record = cls(
            scope=scope,
            manifest_id=require_identifier(manifest_id, "manifest_id"),
            source_registration_id=require_identifier(
                source_registration_id, "source_registration_id"
            ),
            destination_candidate_id=require_identifier(
                destination_candidate_id, "destination_candidate_id"
            ),
            source_snapshot_sha256=require_sha256(
                source_snapshot_sha256, "source_snapshot_sha256"
            ),
            mapping_version=require_schema_version(mapping_version),
            workload_class=workload,
            authorization_reference_id=authorization,
            preferred_batch_size=_require_positive_int(
                preferred_batch_size, "preferred_batch_size"
            ),
            created_at=normalize_timestamp(created_at, "created_at"),
            authority_effect=authority,
            serving_effect=serving,
            manifest_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "manifest_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "manifest_id": self.manifest_id,
            "source_registration_id": self.source_registration_id,
            "destination_candidate_id": self.destination_candidate_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "mapping_version": self.mapping_version,
            "workload_class": self.workload_class,
            "authorization_reference_id": self.authorization_reference_id,
            "preferred_batch_size": self.preferred_batch_size,
            "created_at": self.created_at,
            "authority_effect": self.authority_effect,
            "serving_effect": self.serving_effect,
        }
        if include_digest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def validate(self) -> None:
        recreated = HistoricalBackfillManifest.create(
            scope=self.scope,
            manifest_id=self.manifest_id,
            source_registration_id=self.source_registration_id,
            destination_candidate_id=self.destination_candidate_id,
            source_snapshot_sha256=self.source_snapshot_sha256,
            mapping_version=self.mapping_version,
            workload_class=self.workload_class,
            authorization_reference_id=self.authorization_reference_id,
            preferred_batch_size=self.preferred_batch_size,
            created_at=self.created_at,
            authority_effect=self.authority_effect,
            serving_effect=self.serving_effect,
        )
        if recreated.manifest_sha256 != require_sha256(
            self.manifest_sha256, "manifest_sha256"
        ):
            raise CognitiveKernelContractError("historical backfill manifest digest mismatch")


@dataclass(frozen=True)
class HistoricalBackfillRecord:
    """One source-to-destination mapping receipt used by a deterministic batch."""

    manifest_id: str
    source_record_id: str
    source_checkpoint: str
    source_record_sha256: str
    mapped_record_sha256: str
    mapping_version: str
    provenance_state: str
    evidence_lineage_ids: tuple[str, ...]
    deletion_lineage_ids: tuple[str, ...]
    disposition: str
    disposition_reason: str | None
    idempotency_key: str
    record_receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest_id: object,
        source_record_id: object,
        source_checkpoint: object,
        source_record_sha256: object,
        mapped_record_sha256: object,
        mapping_version: object,
        provenance_state: object,
        evidence_lineage_ids: Iterable[object] = (),
        deletion_lineage_ids: Iterable[object] = (),
        disposition: object = "accepted",
        disposition_reason: object | None = None,
    ) -> "HistoricalBackfillRecord":
        state = _require_choice(
            provenance_state, "provenance_state", PROVENANCE_STATES
        )
        disposition_value = _require_choice(
            disposition, "disposition", BACKFILL_DISPOSITIONS
        )
        reason = (
            require_identifier(disposition_reason, "disposition_reason")
            if disposition_reason is not None
            else None
        )

        if disposition_value == "accepted" and state == "missing":
            raise CognitiveKernelContractError(
                "accepted historical backfill records may not invent missing provenance"
            )
        if disposition_value != "accepted" and reason is None:
            raise CognitiveKernelContractError(
                "non-accepted historical backfill records require a disposition reason"
            )

        base: dict[str, object] = {
            "manifest_id": require_identifier(manifest_id, "manifest_id"),
            "source_record_id": require_identifier(
                source_record_id, "source_record_id"
            ),
            "source_checkpoint": require_identifier(
                source_checkpoint, "source_checkpoint"
            ),
            "source_record_sha256": require_sha256(
                source_record_sha256, "source_record_sha256"
            ),
            "mapped_record_sha256": require_sha256(
                mapped_record_sha256, "mapped_record_sha256"
            ),
            "mapping_version": require_schema_version(mapping_version),
            "provenance_state": state,
            "evidence_lineage_ids": normalize_identifier_sequence(
                tuple(evidence_lineage_ids), "evidence_lineage_ids"
            ),
            "deletion_lineage_ids": normalize_identifier_sequence(
                tuple(deletion_lineage_ids), "deletion_lineage_ids"
            ),
            "disposition": disposition_value,
            "disposition_reason": reason,
        }
        idempotency_key = canonical_sha256(
            {
                "manifest_id": base["manifest_id"],
                "source_record_id": base["source_record_id"],
                "source_checkpoint": base["source_checkpoint"],
                "source_record_sha256": base["source_record_sha256"],
                "mapped_record_sha256": base["mapped_record_sha256"],
                "mapping_version": base["mapping_version"],
            }
        )
        record = cls(
            **base,
            idempotency_key=idempotency_key,
            record_receipt_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "record_receipt_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "source_record_id": self.source_record_id,
            "source_checkpoint": self.source_checkpoint,
            "source_record_sha256": self.source_record_sha256,
            "mapped_record_sha256": self.mapped_record_sha256,
            "mapping_version": self.mapping_version,
            "provenance_state": self.provenance_state,
            "evidence_lineage_ids": list(self.evidence_lineage_ids),
            "deletion_lineage_ids": list(self.deletion_lineage_ids),
            "disposition": self.disposition,
            "disposition_reason": self.disposition_reason,
            "idempotency_key": self.idempotency_key,
        }
        if include_digest:
            payload["record_receipt_sha256"] = self.record_receipt_sha256
        return payload

    def validate(self) -> None:
        recreated = HistoricalBackfillRecord.create(
            manifest_id=self.manifest_id,
            source_record_id=self.source_record_id,
            source_checkpoint=self.source_checkpoint,
            source_record_sha256=self.source_record_sha256,
            mapped_record_sha256=self.mapped_record_sha256,
            mapping_version=self.mapping_version,
            provenance_state=self.provenance_state,
            evidence_lineage_ids=self.evidence_lineage_ids,
            deletion_lineage_ids=self.deletion_lineage_ids,
            disposition=self.disposition,
            disposition_reason=self.disposition_reason,
        )
        if recreated.idempotency_key != require_sha256(
            self.idempotency_key, "idempotency_key"
        ):
            raise CognitiveKernelContractError("historical backfill idempotency mismatch")
        if recreated.record_receipt_sha256 != require_sha256(
            self.record_receipt_sha256, "record_receipt_sha256"
        ):
            raise CognitiveKernelContractError(
                "historical backfill record receipt digest mismatch"
            )


@dataclass(frozen=True)
class BackfillDestinationResult:
    """Destination-side receipt for an accepted mapping attempt."""

    idempotency_key: str
    outcome: str
    destination_record_sha256: str | None
    detail_sha256: str
    result_sha256: str

    @classmethod
    def create(
        cls,
        *,
        idempotency_key: object,
        outcome: object,
        destination_record_sha256: object | None,
        detail: object,
    ) -> "BackfillDestinationResult":
        outcome_value = _require_choice(
            outcome, "outcome", BACKFILL_DESTINATION_OUTCOMES
        )
        destination_digest = (
            require_sha256(destination_record_sha256, "destination_record_sha256")
            if destination_record_sha256 is not None
            else None
        )
        if outcome_value in {"applied", "duplicate"} and destination_digest is None:
            raise CognitiveKernelContractError(
                "applied or duplicate destination results require a destination digest"
            )
        detail_sha256 = canonical_sha256(detail)
        record = cls(
            idempotency_key=require_sha256(idempotency_key, "idempotency_key"),
            outcome=outcome_value,
            destination_record_sha256=destination_digest,
            detail_sha256=detail_sha256,
            result_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "result_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "idempotency_key": self.idempotency_key,
            "outcome": self.outcome,
            "destination_record_sha256": self.destination_record_sha256,
            "detail_sha256": self.detail_sha256,
        }
        if include_digest:
            payload["result_sha256"] = self.result_sha256
        return payload


@dataclass(frozen=True)
class BackfillBatchReceipt:
    """Deterministic summary, lineage, and reconciliation receipt for one batch."""

    manifest_id: str
    manifest_sha256: str
    batch_id: str
    source_checkpoint_start: str
    source_checkpoint_end: str
    record_count: int
    accepted_count: int
    rejected_count: int
    quarantined_count: int
    ambiguous_count: int
    applied_count: int
    duplicate_count: int
    destination_rejected_count: int
    destination_quarantined_count: int
    destination_ambiguous_count: int
    record_receipt_sha256s: tuple[str, ...]
    destination_result_sha256s: tuple[str, ...]
    evidence_lineage_sha256: str
    deletion_lineage_sha256: str
    reconciliation_sha256: str
    completed_at: str
    batch_receipt_sha256: str

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "batch_id": self.batch_id,
            "source_checkpoint_start": self.source_checkpoint_start,
            "source_checkpoint_end": self.source_checkpoint_end,
            "record_count": self.record_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "quarantined_count": self.quarantined_count,
            "ambiguous_count": self.ambiguous_count,
            "applied_count": self.applied_count,
            "duplicate_count": self.duplicate_count,
            "destination_rejected_count": self.destination_rejected_count,
            "destination_quarantined_count": self.destination_quarantined_count,
            "destination_ambiguous_count": self.destination_ambiguous_count,
            "record_receipt_sha256s": list(self.record_receipt_sha256s),
            "destination_result_sha256s": list(self.destination_result_sha256s),
            "evidence_lineage_sha256": self.evidence_lineage_sha256,
            "deletion_lineage_sha256": self.deletion_lineage_sha256,
            "reconciliation_sha256": self.reconciliation_sha256,
            "completed_at": self.completed_at,
        }
        if include_digest:
            payload["batch_receipt_sha256"] = self.batch_receipt_sha256
        return payload


class InMemoryIdempotentBackfillSink:
    """Synthetic/research sink demonstrating replay-safe Stage D write semantics."""

    def __init__(self) -> None:
        self._records: dict[str, str] = {}

    def write(self, record: HistoricalBackfillRecord) -> BackfillDestinationResult:
        record.validate()
        existing = self._records.get(record.idempotency_key)
        if existing is not None:
            return BackfillDestinationResult.create(
                idempotency_key=record.idempotency_key,
                outcome="duplicate",
                destination_record_sha256=existing,
                detail={
                    "state": "idempotent_replay",
                    "source_record_id": record.source_record_id,
                },
            )

        self._records[record.idempotency_key] = record.mapped_record_sha256
        return BackfillDestinationResult.create(
            idempotency_key=record.idempotency_key,
            outcome="applied",
            destination_record_sha256=record.mapped_record_sha256,
            detail={
                "state": "applied_to_synthetic_shadow_sink",
                "source_record_id": record.source_record_id,
            },
        )

    @property
    def record_count(self) -> int:
        return len(self._records)


def run_historical_backfill_batch(
    *,
    manifest: HistoricalBackfillManifest,
    records: Iterable[HistoricalBackfillRecord],
    write_accepted_record: Callable[
        [HistoricalBackfillRecord], BackfillDestinationResult
    ],
    completed_at: object,
) -> BackfillBatchReceipt:
    """Execute one deterministic batch without changing canonical or serving authority."""

    manifest.validate()
    items = tuple(records)
    if not items:
        raise CognitiveKernelContractError(
            "historical backfill batch requires at least one record"
        )

    seen: set[str] = set()
    for item in items:
        item.validate()
        if item.manifest_id != manifest.manifest_id:
            raise CognitiveKernelContractError(
                "historical backfill record manifest mismatch"
            )
        if item.mapping_version != manifest.mapping_version:
            raise CognitiveKernelContractError(
                "historical backfill record mapping version mismatch"
            )
        if item.idempotency_key in seen:
            raise CognitiveKernelContractError(
                "historical backfill batch contains duplicate idempotency keys"
            )
        seen.add(item.idempotency_key)

    destination_results: list[BackfillDestinationResult] = []
    for item in items:
        if item.disposition != "accepted":
            continue
        result = write_accepted_record(item)
        if result.idempotency_key != item.idempotency_key:
            raise CognitiveKernelContractError(
                "destination result idempotency key does not match source record"
            )
        destination_results.append(result)

    count_by_disposition = {
        value: sum(1 for item in items if item.disposition == value)
        for value in BACKFILL_DISPOSITIONS
    }
    count_by_destination = {
        value: sum(1 for item in destination_results if item.outcome == value)
        for value in BACKFILL_DESTINATION_OUTCOMES
    }

    evidence_lineage_sha256 = canonical_sha256(
        [list(item.evidence_lineage_ids) for item in items]
    )
    deletion_lineage_sha256 = canonical_sha256(
        [list(item.deletion_lineage_ids) for item in items]
    )
    reconciliation_payload = {
        "manifest_sha256": manifest.manifest_sha256,
        "record_receipts": [item.record_receipt_sha256 for item in items],
        "destination_results": [item.result_sha256 for item in destination_results],
        "source_counts": count_by_disposition,
        "destination_counts": count_by_destination,
        "evidence_lineage_sha256": evidence_lineage_sha256,
        "deletion_lineage_sha256": deletion_lineage_sha256,
    }
    reconciliation_sha256 = canonical_sha256(reconciliation_payload)
    batch_id = canonical_sha256(
        {
            "manifest_sha256": manifest.manifest_sha256,
            "record_receipts": [item.record_receipt_sha256 for item in items],
        }
    )
    completed = normalize_timestamp(completed_at, "completed_at")

    record = BackfillBatchReceipt(
        manifest_id=manifest.manifest_id,
        manifest_sha256=manifest.manifest_sha256,
        batch_id=batch_id,
        source_checkpoint_start=items[0].source_checkpoint,
        source_checkpoint_end=items[-1].source_checkpoint,
        record_count=len(items),
        accepted_count=count_by_disposition["accepted"],
        rejected_count=count_by_disposition["rejected"],
        quarantined_count=count_by_disposition["quarantined"],
        ambiguous_count=count_by_disposition["ambiguous"],
        applied_count=count_by_destination["applied"],
        duplicate_count=count_by_destination["duplicate"],
        destination_rejected_count=count_by_destination["rejected"],
        destination_quarantined_count=count_by_destination["quarantined"],
        destination_ambiguous_count=count_by_destination["ambiguous"],
        record_receipt_sha256s=tuple(
            item.record_receipt_sha256 for item in items
        ),
        destination_result_sha256s=tuple(
            item.result_sha256 for item in destination_results
        ),
        evidence_lineage_sha256=evidence_lineage_sha256,
        deletion_lineage_sha256=deletion_lineage_sha256,
        reconciliation_sha256=reconciliation_sha256,
        completed_at=completed,
        batch_receipt_sha256="0" * 64,
    )
    digest = canonical_sha256(record.metadata_record(include_digest=False))
    return BackfillBatchReceipt(
        **{**record.__dict__, "batch_receipt_sha256": digest}
    )


@dataclass(frozen=True)
class HistoricalBackfillCheckpoint:
    """Digest-bound continuation state across deterministic Stage D batches."""

    manifest_id: str
    manifest_sha256: str
    completed_batch_sha256s: tuple[str, ...]
    last_source_checkpoint: str
    accepted_total: int
    rejected_total: int
    quarantined_total: int
    ambiguous_total: int
    applied_total: int
    duplicate_total: int
    checkpoint_sha256: str

    @classmethod
    def from_receipts(
        cls,
        *,
        manifest: HistoricalBackfillManifest,
        receipts: Iterable[BackfillBatchReceipt],
    ) -> "HistoricalBackfillCheckpoint":
        manifest.validate()
        items = tuple(receipts)
        if not items:
            raise CognitiveKernelContractError(
                "historical backfill checkpoint requires at least one batch receipt"
            )
        for item in items:
            if item.manifest_id != manifest.manifest_id:
                raise CognitiveKernelContractError(
                    "historical backfill checkpoint manifest mismatch"
                )
            if item.manifest_sha256 != manifest.manifest_sha256:
                raise CognitiveKernelContractError(
                    "historical backfill checkpoint manifest digest mismatch"
                )

        record = cls(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest.manifest_sha256,
            completed_batch_sha256s=tuple(
                item.batch_receipt_sha256 for item in items
            ),
            last_source_checkpoint=items[-1].source_checkpoint_end,
            accepted_total=sum(item.accepted_count for item in items),
            rejected_total=sum(item.rejected_count for item in items),
            quarantined_total=sum(item.quarantined_count for item in items),
            ambiguous_total=sum(item.ambiguous_count for item in items),
            applied_total=sum(item.applied_count for item in items),
            duplicate_total=sum(item.duplicate_count for item in items),
            checkpoint_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "checkpoint_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "completed_batch_sha256s": list(self.completed_batch_sha256s),
            "last_source_checkpoint": self.last_source_checkpoint,
            "accepted_total": self.accepted_total,
            "rejected_total": self.rejected_total,
            "quarantined_total": self.quarantined_total,
            "ambiguous_total": self.ambiguous_total,
            "applied_total": self.applied_total,
            "duplicate_total": self.duplicate_total,
        }
        if include_digest:
            payload["checkpoint_sha256"] = self.checkpoint_sha256
        return payload


def build_synthetic_stage_d_manifest(
    *,
    scope: ProductHostScope,
    created_at: object,
    destination_candidate_id: object = "m2.reversible.polyglot.candidate",
) -> HistoricalBackfillManifest:
    """Build the deterministic synthetic manifest used by the Stage D evaluator."""

    return HistoricalBackfillManifest.create(
        scope=scope,
        manifest_id="phase2.shadow.stage_d.synthetic.manifest.1",
        source_registration_id="phase2.synthetic.registration.1",
        destination_candidate_id=destination_candidate_id,
        source_snapshot_sha256=canonical_sha256(
            {"source": "phase2.synthetic.snapshot", "generation": 1}
        ),
        mapping_version="1.0.0",
        workload_class="synthetic",
        preferred_batch_size=4,
        created_at=created_at,
    )
