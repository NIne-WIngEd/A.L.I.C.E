"""Persistent Stage F+G reference integration and backend-candidate registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable

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
from .shadow_migration_stage_f_g import (
    CanonicalChangeEnvelope,
    MirrorDestinationResult,
    ProjectionBuildManifest,
    ProjectionBuildReceipt,
)

SHADOW_MIGRATION_STAGE_F_G_PERSISTENT_SCHEMA_VERSION = "1.0.0"
PERSISTENT_WORKLOAD_CLASSES = frozenset({"synthetic", "owner_authorized"})
PERSISTENT_CANDIDATE_STATES = frozenset(
    {"reference_operational", "research_candidate"}
)
PERSISTENT_PROJECTION_WRITE_OUTCOMES = frozenset(
    {"applied", "duplicate", "quarantined"}
)


def _require_choice(value: object, field: str, allowed: frozenset[str]) -> str:
    normalized = require_identifier(value, field)
    if normalized not in allowed:
        raise CognitiveKernelContractError(f"{field} is not an allowed value")
    return normalized


def _require_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CognitiveKernelContractError(
            f"{field} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True)
class PersistentBackendCandidate:
    """One replaceable persistent-backend research candidate."""

    candidate_id: str
    engine_family: str
    roles: tuple[str, ...]
    deployment_profiles: tuple[str, ...]
    client_package: str | None
    candidate_state: str
    architecture_state: str
    candidate_sha256: str

    @classmethod
    def create(
        cls,
        *,
        candidate_id: object,
        engine_family: object,
        roles: Iterable[object],
        deployment_profiles: Iterable[object],
        client_package: object | None,
        candidate_state: object,
        architecture_state: object = "candidate_not_destination_ceiling",
    ) -> "PersistentBackendCandidate":
        normalized_roles = normalize_identifier_sequence(
            tuple(roles), "roles"
        )
        normalized_profiles = normalize_identifier_sequence(
            tuple(deployment_profiles), "deployment_profiles"
        )
        if not normalized_roles:
            raise CognitiveKernelContractError(
                "persistent backend candidate requires at least one role"
            )
        if not normalized_profiles:
            raise CognitiveKernelContractError(
                "persistent backend candidate requires deployment profiles"
            )

        package = (
            require_identifier(client_package, "client_package")
            if client_package is not None
            else None
        )
        state = _require_choice(
            candidate_state,
            "candidate_state",
            PERSISTENT_CANDIDATE_STATES,
        )
        architecture = require_identifier(
            architecture_state, "architecture_state"
        )
        if architecture != "candidate_not_destination_ceiling":
            raise CognitiveKernelContractError(
                "persistent candidate must remain explicitly non-ceiling"
            )

        record = cls(
            candidate_id=require_identifier(
                candidate_id, "candidate_id"
            ),
            engine_family=require_identifier(
                engine_family, "engine_family"
            ),
            roles=normalized_roles,
            deployment_profiles=normalized_profiles,
            client_package=package,
            candidate_state=state,
            architecture_state=architecture,
            candidate_sha256="0" * 64,
        )
        digest = canonical_sha256(
            record.metadata_record(include_digest=False)
        )
        return cls(**{**record.__dict__, "candidate_sha256": digest})

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "engine_family": self.engine_family,
            "roles": list(self.roles),
            "deployment_profiles": list(self.deployment_profiles),
            "client_package": self.client_package,
            "candidate_state": self.candidate_state,
            "architecture_state": self.architecture_state,
        }
        if include_digest:
            payload["candidate_sha256"] = self.candidate_sha256
        return payload

    def validate(self) -> None:
        recreated = PersistentBackendCandidate.create(
            candidate_id=self.candidate_id,
            engine_family=self.engine_family,
            roles=self.roles,
            deployment_profiles=self.deployment_profiles,
            client_package=self.client_package,
            candidate_state=self.candidate_state,
            architecture_state=self.architecture_state,
        )
        if recreated.candidate_sha256 != require_sha256(
            self.candidate_sha256, "candidate_sha256"
        ):
            raise CognitiveKernelContractError(
                "persistent backend candidate digest mismatch"
            )


def build_persistent_backend_candidate_registry(
) -> tuple[PersistentBackendCandidate, ...]:
    """Return current evaluated candidates without making a destination choice."""
    return (
        PersistentBackendCandidate.create(
            candidate_id="alice.reference.sqlite",
            engine_family="sqlite_reference",
            roles=(
                "mirror_receipt_store",
                "projection_receipt_store",
                "durability_oracle",
            ),
            deployment_profiles=("edge", "single_workstation"),
            client_package=None,
            candidate_state="reference_operational",
        ),
        PersistentBackendCandidate.create(
            candidate_id="alice.candidate.kurrentdb",
            engine_family="kurrentdb_event_store",
            roles=(
                "event_stream",
                "outbox_subscription",
                "replay_checkpoint",
            ),
            deployment_profiles=(
                "single_workstation",
                "private_cluster",
                "hybrid_cloud",
                "distributed_multi_region",
            ),
            client_package="kurrentdbclient",
            candidate_state="research_candidate",
        ),
        PersistentBackendCandidate.create(
            candidate_id="alice.candidate.neo4j",
            engine_family="neo4j_graph",
            roles=(
                "cognitive_graph",
                "graph_projection",
                "graph_reasoning",
            ),
            deployment_profiles=(
                "single_workstation",
                "private_cluster",
                "hybrid_cloud",
                "distributed_multi_region",
            ),
            client_package="neo4j",
            candidate_state="research_candidate",
        ),
        PersistentBackendCandidate.create(
            candidate_id="alice.candidate.qdrant",
            engine_family="qdrant_vector",
            roles=(
                "vector_projection",
                "multimodal_retrieval",
                "vector_generation",
            ),
            deployment_profiles=(
                "single_workstation",
                "private_cluster",
                "hybrid_cloud",
                "distributed_multi_region",
            ),
            client_package="qdrant_client",
            candidate_state="research_candidate",
        ),
        PersistentBackendCandidate.create(
            candidate_id="alice.candidate.temporal",
            engine_family="temporal_workflow",
            roles=(
                "durable_workflow",
                "projection_orchestration",
                "recovery_coordination",
            ),
            deployment_profiles=(
                "single_workstation",
                "private_cluster",
                "hybrid_cloud",
                "distributed_multi_region",
            ),
            client_package="temporalio",
            candidate_state="research_candidate",
        ),
    )


def persistent_backend_candidate_registry_sha256() -> str:
    return canonical_sha256(
        [
            item.metadata_record()
            for item in build_persistent_backend_candidate_registry()
        ]
    )


@dataclass(frozen=True)
class PersistentIntegrationManifest:
    scope: ProductHostScope
    manifest_id: str
    reference_adapter_id: str
    candidate_registry_sha256: str
    workload_class: str
    authorization_reference_id: str | None
    canonical_writer_state: str
    profile_state: str
    created_at: str
    manifest_sha256: str

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        manifest_id: object,
        reference_adapter_id: object,
        candidate_registry_sha256: object,
        workload_class: object,
        created_at: object,
        authorization_reference_id: object | None = None,
        canonical_writer_state: object = (
            "phase2_released_writer_remains_current"
        ),
        profile_state: object = (
            "nonproduction_persistent_integration_reference"
        ),
    ) -> "PersistentIntegrationManifest":
        scope.validate()
        workload = _require_choice(
            workload_class,
            "workload_class",
            PERSISTENT_WORKLOAD_CLASSES,
        )
        authorization = (
            require_identifier(
                authorization_reference_id,
                "authorization_reference_id",
            )
            if authorization_reference_id is not None
            else None
        )
        if workload == "owner_authorized" and authorization is None:
            raise CognitiveKernelContractError(
                "owner_authorized persistent workload requires authorization"
            )
        if workload == "synthetic" and authorization is not None:
            raise CognitiveKernelContractError(
                "synthetic persistent workload may not claim authorization"
            )

        writer_state = require_identifier(
            canonical_writer_state, "canonical_writer_state"
        )
        if writer_state != "phase2_released_writer_remains_current":
            raise CognitiveKernelContractError(
                "persistent integration reference does not transition authority"
            )
        profile = require_identifier(profile_state, "profile_state")
        if profile != "nonproduction_persistent_integration_reference":
            raise CognitiveKernelContractError(
                "persistent integration profile state is invalid"
            )

        record = cls(
            scope=scope,
            manifest_id=require_identifier(
                manifest_id, "manifest_id"
            ),
            reference_adapter_id=require_identifier(
                reference_adapter_id, "reference_adapter_id"
            ),
            candidate_registry_sha256=require_sha256(
                candidate_registry_sha256,
                "candidate_registry_sha256",
            ),
            workload_class=workload,
            authorization_reference_id=authorization,
            canonical_writer_state=writer_state,
            profile_state=profile,
            created_at=normalize_timestamp(created_at, "created_at"),
            manifest_sha256="0" * 64,
        )
        digest = canonical_sha256(
            record.metadata_record(include_digest=False)
        )
        return cls(**{**record.__dict__, "manifest_sha256": digest})

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope": self.scope.metadata_record(),
            "manifest_id": self.manifest_id,
            "reference_adapter_id": self.reference_adapter_id,
            "candidate_registry_sha256": self.candidate_registry_sha256,
            "workload_class": self.workload_class,
            "authorization_reference_id": self.authorization_reference_id,
            "canonical_writer_state": self.canonical_writer_state,
            "profile_state": self.profile_state,
            "created_at": self.created_at,
        }
        if include_digest:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


@dataclass(frozen=True)
class PersistentProjectionWriteReceipt:
    manifest_sha256: str
    projection_receipt_sha256: str
    outcome: str
    persistent_state_sha256: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        manifest_sha256: object,
        projection_receipt_sha256: object,
        outcome: object,
        persistent_state_sha256: object,
    ) -> "PersistentProjectionWriteReceipt":
        normalized_outcome = _require_choice(
            outcome,
            "outcome",
            PERSISTENT_PROJECTION_WRITE_OUTCOMES,
        )
        record = cls(
            manifest_sha256=require_sha256(
                manifest_sha256, "manifest_sha256"
            ),
            projection_receipt_sha256=require_sha256(
                projection_receipt_sha256,
                "projection_receipt_sha256",
            ),
            outcome=normalized_outcome,
            persistent_state_sha256=require_sha256(
                persistent_state_sha256,
                "persistent_state_sha256",
            ),
            receipt_sha256="0" * 64,
        )
        digest = canonical_sha256(
            record.metadata_record(include_digest=False)
        )
        return cls(**{**record.__dict__, "receipt_sha256": digest})

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_sha256": self.manifest_sha256,
            "projection_receipt_sha256": (
                self.projection_receipt_sha256
            ),
            "outcome": self.outcome,
            "persistent_state_sha256": (
                self.persistent_state_sha256
            ),
        }
        if include_digest:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class PersistentReferenceIntegrityReceipt:
    adapter_id: str
    schema_version: str
    journal_mode: str
    integrity_state: str
    mirror_record_count: int
    projection_generation_count: int
    persistent_state_sha256: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        adapter_id: object,
        schema_version: object,
        journal_mode: object,
        integrity_state: object,
        mirror_record_count: object,
        projection_generation_count: object,
        persistent_state_sha256: object,
    ) -> "PersistentReferenceIntegrityReceipt":
        record = cls(
            adapter_id=require_identifier(adapter_id, "adapter_id"),
            schema_version=require_schema_version(schema_version),
            journal_mode=require_identifier(
                journal_mode, "journal_mode"
            ),
            integrity_state=require_identifier(
                integrity_state, "integrity_state"
            ),
            mirror_record_count=_require_non_negative_int(
                mirror_record_count, "mirror_record_count"
            ),
            projection_generation_count=_require_non_negative_int(
                projection_generation_count,
                "projection_generation_count",
            ),
            persistent_state_sha256=require_sha256(
                persistent_state_sha256,
                "persistent_state_sha256",
            ),
            receipt_sha256="0" * 64,
        )
        digest = canonical_sha256(
            record.metadata_record(include_digest=False)
        )
        return cls(**{**record.__dict__, "receipt_sha256": digest})

    def metadata_record(
        self, *, include_digest: bool = True
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "adapter_id": self.adapter_id,
            "schema_version": self.schema_version,
            "journal_mode": self.journal_mode,
            "integrity_state": self.integrity_state,
            "mirror_record_count": self.mirror_record_count,
            "projection_generation_count": (
                self.projection_generation_count
            ),
            "persistent_state_sha256": (
                self.persistent_state_sha256
            ),
        }
        if include_digest:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


class SQLitePersistentStageFGReferenceAdapter:
    """Durable compatibility oracle; never a permanent destination choice."""

    adapter_id = "alice.reference.sqlite.stage_f_g_persistent"

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path)
        if str(path) == ":memory:":
            raise CognitiveKernelContractError(
                "persistent reference adapter requires a filesystem path"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = path
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._initialize_schema()

    def _configure(self) -> None:
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        journal = self._connection.execute(
            "PRAGMA journal_mode = WAL"
        ).fetchone()
        if journal is None or str(journal[0]).lower() != "wal":
            raise CognitiveKernelContractError(
                "persistent reference adapter could not enable WAL"
            )

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alice_stage_fg_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alice_stage_fg_mirror (
                    idempotency_key TEXT PRIMARY KEY,
                    manifest_id TEXT NOT NULL,
                    change_id TEXT NOT NULL,
                    outbox_sequence INTEGER NOT NULL,
                    operation TEXT NOT NULL,
                    canonical_record_sha256 TEXT NOT NULL,
                    evidence_lineage_json TEXT NOT NULL,
                    deletion_lineage_json TEXT NOT NULL,
                    envelope_sha256 TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    UNIQUE(manifest_id, outbox_sequence)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alice_stage_fg_projection (
                    manifest_sha256 TEXT PRIMARY KEY,
                    build_id TEXT NOT NULL,
                    source_snapshot_sha256 TEXT NOT NULL,
                    graph_generation_sha256 TEXT NOT NULL,
                    vector_generation_sha256 TEXT NOT NULL,
                    workflow_generation_sha256 TEXT NOT NULL,
                    deletion_watermark TEXT NOT NULL,
                    projection_receipt_sha256 TEXT NOT NULL,
                    UNIQUE(build_id)
                )
                """
            )
            self._connection.execute(
                """
                INSERT INTO alice_stage_fg_meta(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (SHADOW_MIGRATION_STAGE_F_G_PERSISTENT_SCHEMA_VERSION,),
            )
            self._connection.execute(
                """
                INSERT INTO alice_stage_fg_meta(key, value)
                VALUES('adapter_role', 'compatibility_reference')
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLitePersistentStageFGReferenceAdapter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _validate_change(change: CanonicalChangeEnvelope) -> None:
        expected_envelope = canonical_sha256(
            change.metadata_record(include_digest=False)
        )
        if expected_envelope != require_sha256(
            change.envelope_sha256, "envelope_sha256"
        ):
            raise CognitiveKernelContractError(
                "canonical change envelope digest mismatch"
            )

    @staticmethod
    def _validate_projection(
        manifest: ProjectionBuildManifest,
        receipt: ProjectionBuildReceipt,
    ) -> None:
        manifest_digest = canonical_sha256(
            manifest.metadata_record(include_digest=False)
        )
        if manifest_digest != require_sha256(
            manifest.manifest_sha256, "manifest_sha256"
        ):
            raise CognitiveKernelContractError(
                "projection manifest digest mismatch"
            )
        if receipt.manifest_sha256 != manifest.manifest_sha256:
            raise CognitiveKernelContractError(
                "projection receipt is bound to another manifest"
            )
        receipt_digest = canonical_sha256(
            receipt.metadata_record(include_digest=False)
        )
        if receipt_digest != require_sha256(
            receipt.receipt_sha256, "receipt_sha256"
        ):
            raise CognitiveKernelContractError(
                "projection receipt digest mismatch"
            )

    def write(
        self, change: CanonicalChangeEnvelope
    ) -> MirrorDestinationResult:
        self._validate_change(change)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                """
                SELECT canonical_record_sha256
                FROM alice_stage_fg_mirror
                WHERE idempotency_key = ?
                """,
                (change.idempotency_key,),
            ).fetchone()
            if existing is not None:
                self._connection.commit()
                return MirrorDestinationResult.create(
                    idempotency_key=change.idempotency_key,
                    outcome="duplicate",
                    destination_record_sha256=existing[
                        "canonical_record_sha256"
                    ],
                    detail={
                        "state": "persistent_idempotent_replay",
                        "adapter_id": self.adapter_id,
                    },
                )

            sequence_conflict = self._connection.execute(
                """
                SELECT idempotency_key
                FROM alice_stage_fg_mirror
                WHERE manifest_id = ? AND outbox_sequence = ?
                """,
                (change.manifest_id, change.outbox_sequence),
            ).fetchone()
            if sequence_conflict is not None:
                self._connection.rollback()
                return MirrorDestinationResult.create(
                    idempotency_key=change.idempotency_key,
                    outcome="quarantined",
                    destination_record_sha256=None,
                    detail={
                        "state": "outbox_sequence_conflict",
                        "adapter_id": self.adapter_id,
                    },
                )

            applied = MirrorDestinationResult.create(
                idempotency_key=change.idempotency_key,
                outcome="applied",
                destination_record_sha256=(
                    change.canonical_record_sha256
                ),
                detail={
                    "state": "persistent_reference_applied",
                    "adapter_id": self.adapter_id,
                    "operation": change.operation,
                    "deletion_lineage_count": len(
                        change.deletion_lineage_ids
                    ),
                },
            )
            self._connection.execute(
                """
                INSERT INTO alice_stage_fg_mirror(
                    idempotency_key,
                    manifest_id,
                    change_id,
                    outbox_sequence,
                    operation,
                    canonical_record_sha256,
                    evidence_lineage_json,
                    deletion_lineage_json,
                    envelope_sha256,
                    result_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change.idempotency_key,
                    change.manifest_id,
                    change.change_id,
                    change.outbox_sequence,
                    change.operation,
                    change.canonical_record_sha256,
                    json.dumps(
                        list(change.evidence_lineage_ids),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        list(change.deletion_lineage_ids),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    change.envelope_sha256,
                    applied.result_sha256,
                ),
            )
            self._connection.commit()
            return applied
        except Exception:
            self._connection.rollback()
            raise

    def persist_projection(
        self,
        manifest: ProjectionBuildManifest,
        receipt: ProjectionBuildReceipt,
    ) -> PersistentProjectionWriteReceipt:
        self._validate_projection(manifest, receipt)

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                """
                SELECT projection_receipt_sha256
                FROM alice_stage_fg_projection
                WHERE manifest_sha256 = ?
                """,
                (manifest.manifest_sha256,),
            ).fetchone()
            if existing is not None:
                self._connection.commit()
                return PersistentProjectionWriteReceipt.create(
                    manifest_sha256=manifest.manifest_sha256,
                    projection_receipt_sha256=existing[
                        "projection_receipt_sha256"
                    ],
                    outcome="duplicate",
                    persistent_state_sha256=self.state_sha256(),
                )

            build_conflict = self._connection.execute(
                """
                SELECT manifest_sha256
                FROM alice_stage_fg_projection
                WHERE build_id = ?
                """,
                (manifest.build_id,),
            ).fetchone()
            if build_conflict is not None:
                self._connection.rollback()
                return PersistentProjectionWriteReceipt.create(
                    manifest_sha256=manifest.manifest_sha256,
                    projection_receipt_sha256=receipt.receipt_sha256,
                    outcome="quarantined",
                    persistent_state_sha256=self.state_sha256(),
                )

            self._connection.execute(
                """
                INSERT INTO alice_stage_fg_projection(
                    manifest_sha256,
                    build_id,
                    source_snapshot_sha256,
                    graph_generation_sha256,
                    vector_generation_sha256,
                    workflow_generation_sha256,
                    deletion_watermark,
                    projection_receipt_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.manifest_sha256,
                    manifest.build_id,
                    manifest.source_snapshot_sha256,
                    receipt.graph_generation_sha256,
                    receipt.vector_generation_sha256,
                    receipt.workflow_generation_sha256,
                    receipt.deletion_watermark,
                    receipt.receipt_sha256,
                ),
            )
            self._connection.commit()
            return PersistentProjectionWriteReceipt.create(
                manifest_sha256=manifest.manifest_sha256,
                projection_receipt_sha256=receipt.receipt_sha256,
                outcome="applied",
                persistent_state_sha256=self.state_sha256(),
            )
        except Exception:
            self._connection.rollback()
            raise

    @property
    def mirror_record_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM alice_stage_fg_mirror"
        ).fetchone()
        return int(row[0])

    @property
    def projection_generation_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM alice_stage_fg_projection"
        ).fetchone()
        return int(row[0])

    def state_sha256(self) -> str:
        mirror = [
            dict(row)
            for row in self._connection.execute(
                """
                SELECT
                    idempotency_key,
                    manifest_id,
                    change_id,
                    outbox_sequence,
                    operation,
                    canonical_record_sha256,
                    evidence_lineage_json,
                    deletion_lineage_json,
                    envelope_sha256,
                    result_sha256
                FROM alice_stage_fg_mirror
                ORDER BY manifest_id, outbox_sequence, idempotency_key
                """
            ).fetchall()
        ]
        projection = [
            dict(row)
            for row in self._connection.execute(
                """
                SELECT
                    manifest_sha256,
                    build_id,
                    source_snapshot_sha256,
                    graph_generation_sha256,
                    vector_generation_sha256,
                    workflow_generation_sha256,
                    deletion_watermark,
                    projection_receipt_sha256
                FROM alice_stage_fg_projection
                ORDER BY build_id, manifest_sha256
                """
            ).fetchall()
        ]
        return canonical_sha256(
            {
                "schema_version": (
                    SHADOW_MIGRATION_STAGE_F_G_PERSISTENT_SCHEMA_VERSION
                ),
                "mirror": mirror,
                "projection": projection,
            }
        )

    def integrity_receipt(
        self,
    ) -> PersistentReferenceIntegrityReceipt:
        integrity = self._connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()
        integrity_state = (
            str(integrity[0]) if integrity is not None else "unknown"
        )
        journal = self._connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()
        journal_mode = (
            str(journal[0]).lower() if journal is not None else "unknown"
        )
        return PersistentReferenceIntegrityReceipt.create(
            adapter_id=self.adapter_id,
            schema_version=(
                SHADOW_MIGRATION_STAGE_F_G_PERSISTENT_SCHEMA_VERSION
            ),
            journal_mode=journal_mode,
            integrity_state=integrity_state,
            mirror_record_count=self.mirror_record_count,
            projection_generation_count=(
                self.projection_generation_count
            ),
            persistent_state_sha256=self.state_sha256(),
        )

    def checkpoint_wal(self) -> tuple[int, int, int]:
        row = self._connection.execute(
            "PRAGMA wal_checkpoint(FULL)"
        ).fetchone()
        if row is None:
            raise CognitiveKernelContractError(
                "WAL checkpoint returned no result"
            )
        return (int(row[0]), int(row[1]), int(row[2]))


def build_synthetic_persistent_integration_manifest(
    *,
    scope: ProductHostScope,
    created_at: object,
) -> PersistentIntegrationManifest:
    return PersistentIntegrationManifest.create(
        scope=scope,
        manifest_id="phase2-shadow-stage-f-g-persistent-synthetic",
        reference_adapter_id=(
            SQLitePersistentStageFGReferenceAdapter.adapter_id
        ),
        candidate_registry_sha256=(
            persistent_backend_candidate_registry_sha256()
        ),
        workload_class="synthetic",
        created_at=created_at,
    )
