"""Nonproduction controlled-write mirroring and projection-generation contracts."""

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

SHADOW_MIGRATION_STAGE_F_G_SCHEMA_VERSION = "1.0.0"
MIRROR_WORKLOAD_CLASSES = frozenset({"synthetic", "owner_authorized"})
CANONICAL_CHANGE_OPERATIONS = frozenset({"upsert", "correction", "delete"})
MIRROR_DESTINATION_OUTCOMES = frozenset(
    {"applied", "duplicate", "rejected", "quarantined"}
)
PROJECTION_PLANES = frozenset({"graph", "vector", "workflow"})


def _require_choice(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not an allowed value")
    return normalized


def _require_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CognitiveKernelContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ControlledMirrorManifest:
    """Digest-bound Stage F profile for mirroring one canonical outbox stream."""

    scope: ProductHostScope
    manifest_id: str
    canonical_writer_id: str
    canonical_authority_generation: str
    destination_candidate_id: str
    outbox_stream_id: str
    mapping_version: str
    workload_class: str
    authorization_reference_id: str | None
    created_at: str
    profile_state: str
    authority_transition_state: str
    serving_state: str
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        manifest_id: object,
        canonical_writer_id: object,
        canonical_authority_generation: object,
        destination_candidate_id: object,
        outbox_stream_id: object,
        mapping_version: object,
        workload_class: object,
        created_at: object,
        authorization_reference_id: object | None = None,
        profile_state: object = "nonproduction_controlled_mirroring",
        authority_transition_state: object = "unchanged",
        serving_state: object = "shadow_only",
    ) -> "ControlledMirrorManifest":
        scope.validate()
        workload = _require_choice(
            workload_class, "workload_class", MIRROR_WORKLOAD_CLASSES
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
                "owner_authorized mirror workload requires an authorization reference"
            )
        if workload == "synthetic" and authorization is not None:
            raise CognitiveKernelContractError(
                "synthetic mirror workload may not claim owner authorization"
            )

        state = require_identifier(profile_state, "profile_state")
        transition = require_identifier(
            authority_transition_state, "authority_transition_state"
        )
        serving = require_identifier(serving_state, "serving_state")
        if state != "nonproduction_controlled_mirroring":
            raise CognitiveKernelContractError(
                "Stage F profile state must identify nonproduction controlled mirroring"
            )
        if transition != "unchanged":
            raise CognitiveKernelContractError(
                "Stage F profile does not perform an authority transition"
            )
        if serving != "shadow_only":
            raise CognitiveKernelContractError(
                "Stage F profile serves only a shadow destination"
            )

        record = cls(
            scope=scope,
            manifest_id=require_identifier(manifest_id, "manifest_id"),
            canonical_writer_id=require_identifier(
                canonical_writer_id, "canonical_writer_id"
            ),
            canonical_authority_generation=require_identifier(
                canonical_authority_generation,
                "canonical_authority_generation",
            ),
            destination_candidate_id=require_identifier(
                destination_candidate_id, "destination_candidate_id"
            ),
            outbox_stream_id=require_identifier(outbox_stream_id, "outbox_stream_id"),
            mapping_version=require_schema_version(mapping_version),
            workload_class=workload,
            authorization_reference_id=authorization,
            created_at=normalize_timestamp(created_at, "created_at"),
            profile_state=state,
            authority_transition_state=transition,
            serving_state=serving,
            manifest_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "manifest_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "manifest_id": self.manifest_id,
            "canonical_writer_id": self.canonical_writer_id,
            "canonical_authority_generation": self.canonical_authority_generation,
            "destination_candidate_id": self.destination_candidate_id,
            "outbox_stream_id": self.outbox_stream_id,
            "mapping_version": self.mapping_version,
            "workload_class": self.workload_class,
            "authorization_reference_id": self.authorization_reference_id,
            "created_at": self.created_at,
            "profile_state": self.profile_state,
            "authority_transition_state": self.authority_transition_state,
            "serving_state": self.serving_state,
        }
        if include_digest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload

    def validate(self) -> None:
        recreated = ControlledMirrorManifest.create(
            scope=self.scope,
            manifest_id=self.manifest_id,
            canonical_writer_id=self.canonical_writer_id,
            canonical_authority_generation=self.canonical_authority_generation,
            destination_candidate_id=self.destination_candidate_id,
            outbox_stream_id=self.outbox_stream_id,
            mapping_version=self.mapping_version,
            workload_class=self.workload_class,
            authorization_reference_id=self.authorization_reference_id,
            created_at=self.created_at,
            profile_state=self.profile_state,
            authority_transition_state=self.authority_transition_state,
            serving_state=self.serving_state,
        )
        if recreated.manifest_sha256 != require_sha256(
            self.manifest_sha256, "manifest_sha256"
        ):
            raise CognitiveKernelContractError("controlled mirror manifest digest mismatch")


@dataclass(frozen=True)
class CanonicalChangeEnvelope:
    """One canonical change published through a Stage F outbox."""

    manifest_id: str
    change_id: str
    authority_namespace: str
    outbox_sequence: int
    operation: str
    canonical_record_sha256: str
    mapping_version: str
    evidence_lineage_ids: tuple[str, ...]
    deletion_lineage_ids: tuple[str, ...]
    idempotency_key: str
    envelope_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest_id: object,
        change_id: object,
        authority_namespace: object,
        outbox_sequence: object,
        operation: object,
        canonical_record_sha256: object,
        mapping_version: object,
        evidence_lineage_ids: Iterable[object] = (),
        deletion_lineage_ids: Iterable[object] = (),
    ) -> "CanonicalChangeEnvelope":
        manifest = require_identifier(manifest_id, "manifest_id")
        change = require_identifier(change_id, "change_id")
        namespace = require_identifier(authority_namespace, "authority_namespace")
        sequence = _require_positive_int(outbox_sequence, "outbox_sequence")
        op = _require_choice(operation, "operation", CANONICAL_CHANGE_OPERATIONS)
        record_sha = require_sha256(
            canonical_record_sha256, "canonical_record_sha256"
        )
        mapping = require_schema_version(mapping_version)
        evidence = normalize_identifier_sequence(
            tuple(evidence_lineage_ids), "evidence_lineage_ids"
        )
        deletion = normalize_identifier_sequence(
            tuple(deletion_lineage_ids), "deletion_lineage_ids"
        )
        idempotency = canonical_sha256(
            {
                "manifest_id": manifest,
                "change_id": change,
                "authority_namespace": namespace,
                "outbox_sequence": sequence,
                "operation": op,
                "canonical_record_sha256": record_sha,
                "mapping_version": mapping,
            }
        )
        record = cls(
            manifest_id=manifest,
            change_id=change,
            authority_namespace=namespace,
            outbox_sequence=sequence,
            operation=op,
            canonical_record_sha256=record_sha,
            mapping_version=mapping,
            evidence_lineage_ids=evidence,
            deletion_lineage_ids=deletion,
            idempotency_key=idempotency,
            envelope_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "envelope_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "change_id": self.change_id,
            "authority_namespace": self.authority_namespace,
            "outbox_sequence": self.outbox_sequence,
            "operation": self.operation,
            "canonical_record_sha256": self.canonical_record_sha256,
            "mapping_version": self.mapping_version,
            "evidence_lineage_ids": list(self.evidence_lineage_ids),
            "deletion_lineage_ids": list(self.deletion_lineage_ids),
            "idempotency_key": self.idempotency_key,
        }
        if include_digest:
            payload["envelope_sha256"] = self.envelope_sha256
        return payload


@dataclass(frozen=True)
class MirrorDestinationResult:
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
        detail: dict[str, object],
    ) -> "MirrorDestinationResult":
        result = cls(
            idempotency_key=require_sha256(idempotency_key, "idempotency_key"),
            outcome=_require_choice(
                outcome, "outcome", MIRROR_DESTINATION_OUTCOMES
            ),
            destination_record_sha256=(
                require_sha256(
                    destination_record_sha256, "destination_record_sha256"
                )
                if destination_record_sha256 is not None
                else None
            ),
            detail_sha256=canonical_sha256(detail),
            result_sha256="0" * 64,
        )
        digest = canonical_sha256(result.metadata_record(include_digest=False))
        return cls(**{**result.__dict__, "result_sha256": digest})

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
class ControlledMirrorBatchReceipt:
    manifest_sha256: str
    first_outbox_sequence: int
    last_outbox_sequence: int
    change_count: int
    applied_count: int
    duplicate_count: int
    rejected_count: int
    quarantined_count: int
    canonical_writer_state: str
    authority_transition_state: str
    envelope_sha256s: tuple[str, ...]
    destination_result_sha256s: tuple[str, ...]
    deletion_lineage_sha256: str
    reconciliation_sha256: str
    completed_at: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest: ControlledMirrorManifest,
        changes: tuple[CanonicalChangeEnvelope, ...],
        results: tuple[MirrorDestinationResult, ...],
        completed_at: object,
    ) -> "ControlledMirrorBatchReceipt":
        manifest.validate()
        if not changes:
            raise CognitiveKernelContractError("mirror batch requires at least one change")
        if len(results) != len(changes):
            raise CognitiveKernelContractError(
                "mirror batch requires one destination result per change"
            )
        sequences = [item.outbox_sequence for item in changes]
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            raise CognitiveKernelContractError(
                "mirror batch outbox sequences must be unique and ordered"
            )
        for change, result in zip(changes, results):
            if change.manifest_id != manifest.manifest_id:
                raise CognitiveKernelContractError("mirror change manifest mismatch")
            if result.idempotency_key != change.idempotency_key:
                raise CognitiveKernelContractError(
                    "mirror result idempotency key mismatch"
                )
        counts = {
            outcome: sum(1 for item in results if item.outcome == outcome)
            for outcome in MIRROR_DESTINATION_OUTCOMES
        }
        deletion_sha = canonical_sha256(
            [list(item.deletion_lineage_ids) for item in changes]
        )
        reconciliation = canonical_sha256(
            {
                "manifest_sha256": manifest.manifest_sha256,
                "changes": [item.envelope_sha256 for item in changes],
                "results": [item.result_sha256 for item in results],
                "canonical_writer_state": "unchanged",
                "authority_transition_state": "unchanged",
            }
        )
        record = cls(
            manifest_sha256=manifest.manifest_sha256,
            first_outbox_sequence=sequences[0],
            last_outbox_sequence=sequences[-1],
            change_count=len(changes),
            applied_count=counts["applied"],
            duplicate_count=counts["duplicate"],
            rejected_count=counts["rejected"],
            quarantined_count=counts["quarantined"],
            canonical_writer_state="unchanged",
            authority_transition_state="unchanged",
            envelope_sha256s=tuple(item.envelope_sha256 for item in changes),
            destination_result_sha256s=tuple(
                item.result_sha256 for item in results
            ),
            deletion_lineage_sha256=deletion_sha,
            reconciliation_sha256=reconciliation,
            completed_at=normalize_timestamp(completed_at, "completed_at"),
            receipt_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "receipt_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_sha256": self.manifest_sha256,
            "first_outbox_sequence": self.first_outbox_sequence,
            "last_outbox_sequence": self.last_outbox_sequence,
            "change_count": self.change_count,
            "applied_count": self.applied_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "quarantined_count": self.quarantined_count,
            "canonical_writer_state": self.canonical_writer_state,
            "authority_transition_state": self.authority_transition_state,
            "envelope_sha256s": list(self.envelope_sha256s),
            "destination_result_sha256s": list(self.destination_result_sha256s),
            "deletion_lineage_sha256": self.deletion_lineage_sha256,
            "reconciliation_sha256": self.reconciliation_sha256,
            "completed_at": self.completed_at,
        }
        if include_digest:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


class InMemoryIdempotentMirrorSink:
    """Research sink demonstrating outbox replay and delete propagation semantics."""

    def __init__(self) -> None:
        self._records: dict[str, str] = {}

    def write(self, change: CanonicalChangeEnvelope) -> MirrorDestinationResult:
        existing = self._records.get(change.idempotency_key)
        if existing is not None:
            return MirrorDestinationResult.create(
                idempotency_key=change.idempotency_key,
                outcome="duplicate",
                destination_record_sha256=existing,
                detail={"state": "idempotent_replay", "change_id": change.change_id},
            )
        self._records[change.idempotency_key] = change.canonical_record_sha256
        return MirrorDestinationResult.create(
            idempotency_key=change.idempotency_key,
            outcome="applied",
            destination_record_sha256=change.canonical_record_sha256,
            detail={
                "state": "applied_to_synthetic_shadow_mirror",
                "operation": change.operation,
                "deletion_lineage_count": len(change.deletion_lineage_ids),
            },
        )

    @property
    def record_count(self) -> int:
        return len(self._records)


def run_controlled_mirror_batch(
    *,
    manifest: ControlledMirrorManifest,
    changes: Iterable[CanonicalChangeEnvelope],
    write_change: Callable[[CanonicalChangeEnvelope], MirrorDestinationResult],
    completed_at: object,
) -> ControlledMirrorBatchReceipt:
    items = tuple(changes)
    results = tuple(write_change(item) for item in items)
    return ControlledMirrorBatchReceipt.create(
        manifest=manifest,
        changes=items,
        results=results,
        completed_at=completed_at,
    )


@dataclass(frozen=True)
class ProjectionBuildManifest:
    """Stage G generation plan over registered claims/evidence and deletion state."""

    scope: ProductHostScope
    build_id: str
    destination_candidate_id: str
    source_generation_id: str
    source_snapshot_sha256: str
    graph_generation_id: str
    vector_generation_id: str
    workflow_generation_id: str
    deletion_watermark: str
    projection_planes: tuple[str, ...]
    created_at: str
    profile_state: str
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        build_id: object,
        destination_candidate_id: object,
        source_generation_id: object,
        source_snapshot_sha256: object,
        graph_generation_id: object,
        vector_generation_id: object,
        workflow_generation_id: object,
        deletion_watermark: object,
        projection_planes: Iterable[object],
        created_at: object,
        profile_state: object = "nonproduction_projection_generation",
    ) -> "ProjectionBuildManifest":
        scope.validate()
        planes = normalize_identifier_sequence(
            tuple(projection_planes), "projection_planes"
        )
        if set(planes) != set(PROJECTION_PLANES):
            raise CognitiveKernelContractError(
                "Stage G prototype requires graph, vector, and workflow planes"
            )
        state = require_identifier(profile_state, "profile_state")
        if state != "nonproduction_projection_generation":
            raise CognitiveKernelContractError(
                "Stage G profile state must identify nonproduction projection generation"
            )
        record = cls(
            scope=scope,
            build_id=require_identifier(build_id, "build_id"),
            destination_candidate_id=require_identifier(
                destination_candidate_id, "destination_candidate_id"
            ),
            source_generation_id=require_identifier(
                source_generation_id, "source_generation_id"
            ),
            source_snapshot_sha256=require_sha256(
                source_snapshot_sha256, "source_snapshot_sha256"
            ),
            graph_generation_id=require_identifier(
                graph_generation_id, "graph_generation_id"
            ),
            vector_generation_id=require_identifier(
                vector_generation_id, "vector_generation_id"
            ),
            workflow_generation_id=require_identifier(
                workflow_generation_id, "workflow_generation_id"
            ),
            deletion_watermark=require_identifier(
                deletion_watermark, "deletion_watermark"
            ),
            projection_planes=planes,
            created_at=normalize_timestamp(created_at, "created_at"),
            profile_state=state,
            manifest_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "manifest_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "build_id": self.build_id,
            "destination_candidate_id": self.destination_candidate_id,
            "source_generation_id": self.source_generation_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "graph_generation_id": self.graph_generation_id,
            "vector_generation_id": self.vector_generation_id,
            "workflow_generation_id": self.workflow_generation_id,
            "deletion_watermark": self.deletion_watermark,
            "projection_planes": list(self.projection_planes),
            "created_at": self.created_at,
            "profile_state": self.profile_state,
        }
        if include_digest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


@dataclass(frozen=True)
class ProjectionBuildReceipt:
    manifest_sha256: str
    source_record_count: int
    graph_node_count: int
    graph_edge_count: int
    vector_record_count: int
    workflow_activity_count: int
    deletion_exclusion_count: int
    repair_action_count: int
    graph_generation_sha256: str
    vector_generation_sha256: str
    workflow_generation_sha256: str
    deletion_watermark: str
    completed_at: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest: ProjectionBuildManifest,
        source_record_count: object,
        graph_node_count: object,
        graph_edge_count: object,
        vector_record_count: object,
        workflow_activity_count: object,
        deletion_exclusion_count: object,
        repair_action_count: object,
        graph_generation_sha256: object,
        vector_generation_sha256: object,
        workflow_generation_sha256: object,
        completed_at: object,
    ) -> "ProjectionBuildReceipt":
        record = cls(
            manifest_sha256=manifest.manifest_sha256,
            source_record_count=_require_non_negative_int(
                source_record_count, "source_record_count"
            ),
            graph_node_count=_require_non_negative_int(
                graph_node_count, "graph_node_count"
            ),
            graph_edge_count=_require_non_negative_int(
                graph_edge_count, "graph_edge_count"
            ),
            vector_record_count=_require_non_negative_int(
                vector_record_count, "vector_record_count"
            ),
            workflow_activity_count=_require_non_negative_int(
                workflow_activity_count, "workflow_activity_count"
            ),
            deletion_exclusion_count=_require_non_negative_int(
                deletion_exclusion_count, "deletion_exclusion_count"
            ),
            repair_action_count=_require_non_negative_int(
                repair_action_count, "repair_action_count"
            ),
            graph_generation_sha256=require_sha256(
                graph_generation_sha256, "graph_generation_sha256"
            ),
            vector_generation_sha256=require_sha256(
                vector_generation_sha256, "vector_generation_sha256"
            ),
            workflow_generation_sha256=require_sha256(
                workflow_generation_sha256, "workflow_generation_sha256"
            ),
            deletion_watermark=manifest.deletion_watermark,
            completed_at=normalize_timestamp(completed_at, "completed_at"),
            receipt_sha256="0" * 64,
        )
        digest = canonical_sha256(record.metadata_record(include_digest=False))
        return cls(**{**record.__dict__, "receipt_sha256": digest})

    def metadata_record(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_sha256": self.manifest_sha256,
            "source_record_count": self.source_record_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "vector_record_count": self.vector_record_count,
            "workflow_activity_count": self.workflow_activity_count,
            "deletion_exclusion_count": self.deletion_exclusion_count,
            "repair_action_count": self.repair_action_count,
            "graph_generation_sha256": self.graph_generation_sha256,
            "vector_generation_sha256": self.vector_generation_sha256,
            "workflow_generation_sha256": self.workflow_generation_sha256,
            "deletion_watermark": self.deletion_watermark,
            "completed_at": self.completed_at,
        }
        if include_digest:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


def build_synthetic_stage_f_manifest(
    *, scope: ProductHostScope, created_at: object
) -> ControlledMirrorManifest:
    return ControlledMirrorManifest.create(
        scope=scope,
        manifest_id="phase2-shadow-stage-f-synthetic",
        canonical_writer_id="phase2.released.writer",
        canonical_authority_generation="phase2.released.generation",
        destination_candidate_id="m2.reversible.polyglot.candidate",
        outbox_stream_id="phase2.synthetic.outbox",
        mapping_version="1.0.0",
        workload_class="synthetic",
        created_at=created_at,
    )


def build_synthetic_stage_g_manifest(
    *, scope: ProductHostScope, created_at: object
) -> ProjectionBuildManifest:
    return ProjectionBuildManifest.create(
        scope=scope,
        build_id="phase2-shadow-stage-g-synthetic",
        destination_candidate_id="m2.reversible.polyglot.candidate",
        source_generation_id="stage-f.synthetic.mirror.generation",
        source_snapshot_sha256=canonical_sha256(
            {"source_generation": "stage-f.synthetic.mirror.generation"}
        ),
        graph_generation_id="graph.synthetic.g1",
        vector_generation_id="vector.synthetic.g1",
        workflow_generation_id="workflow.synthetic.g1",
        deletion_watermark="deletion.synthetic.0001",
        projection_planes=("graph", "vector", "workflow"),
        created_at=created_at,
    )
