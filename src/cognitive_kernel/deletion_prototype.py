"""Memory M2.6 reversible cross-plane deletion prototype."""

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
from .deletion_contracts import (
    DELETION_CONTRACT_SCHEMA_VERSION,
    DELETION_PLANE_KINDS,
    DELETION_PLANE_STATES,
    DELETION_PROPAGATION_STATES,
    DeletionPlaneReceipt,
    DeletionPropagationReceipt,
    RestoreFilterDecision,
)
from .memory_contracts import MemoryUnitEnvelope

DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
DELETION_PROPAGATION_PROTOTYPE_STATE = "reversible_nonproduction"


class DeletionPropagationPrototypeError(RuntimeError):
    """Base error for the M2.6 deletion-propagation profile."""


class UnsafeDeletionPropagationPathError(
    DeletionPropagationPrototypeError
):
    """Raised when deletion state would enter the public repository."""


class DeletionPropagationIsolationError(
    DeletionPropagationPrototypeError
):
    """Raised when deletion state belongs to another scope."""


class DeletionPropagationIntegrityError(
    DeletionPropagationPrototypeError
):
    """Raised when persisted deletion state fails integrity checks."""


class DeletionPropagationConflictError(
    DeletionPropagationPrototypeError
):
    """Raised for idempotency or optimistic-generation conflicts."""


class DeletionPropagationTransactionError(
    DeletionPropagationPrototypeError
):
    """Raised when one deletion mutation cannot commit."""


@dataclass(frozen=True)
class DeletionPropagationProfile:
    """Profile-selected behavior for the reversible M2.6 rehearsal."""

    scope: ProductHostScope
    authority_namespace_id: str
    profile_id: str
    required_plane_kinds: tuple[str, ...]
    production_influence: bool = False
    canonical_claim_authority: bool = False
    destructive_live_deletion: bool = False

    @classmethod
    def create(
        cls,
        *,
        scope: ProductHostScope,
        authority_namespace_id: object,
        profile_id: object,
        required_plane_kinds: Iterable[object],
        production_influence: object = False,
        canonical_claim_authority: object = False,
        destructive_live_deletion: object = False,
    ) -> "DeletionPropagationProfile":
        if not isinstance(scope, ProductHostScope):
            raise CognitiveKernelContractError(
                "scope must be ProductHostScope"
            )
        scope.validate()
        planes = tuple(
            sorted(
                require_identifier(item, "required_plane_kinds")
                for item in required_plane_kinds
            )
        )
        if not planes:
            raise CognitiveKernelContractError(
                "required_plane_kinds may not be empty"
            )
        if len(set(planes)) != len(planes):
            raise CognitiveKernelContractError(
                "required_plane_kinds may not contain duplicates"
            )
        unknown = sorted(set(planes) - set(DELETION_PLANE_KINDS))
        if unknown:
            raise CognitiveKernelContractError(
                f"unratified deletion plane kinds: {unknown}"
            )
        for value, field in (
            (production_influence, "production_influence"),
            (canonical_claim_authority, "canonical_claim_authority"),
            (destructive_live_deletion, "destructive_live_deletion"),
        ):
            if not isinstance(value, bool):
                raise CognitiveKernelContractError(
                    f"{field} must be boolean"
                )
        if production_influence:
            raise CognitiveKernelContractError(
                "M2.6 reversible profile may not influence production"
            )
        if canonical_claim_authority:
            raise CognitiveKernelContractError(
                "M2.6 profile may not write canonical Claim Authority"
            )
        if destructive_live_deletion:
            raise CognitiveKernelContractError(
                "M2.6 profile may not execute destructive live deletion"
            )
        value = cls(
            scope=scope,
            authority_namespace_id=require_identifier(
                authority_namespace_id,
                "authority_namespace_id",
            ),
            profile_id=require_identifier(profile_id, "profile_id"),
            required_plane_kinds=planes,
            production_influence=False,
            canonical_claim_authority=False,
            destructive_live_deletion=False,
        )
        value.validate()
        return value

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": (
                DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION
            ),
            "prototype_state": (
                DELETION_PROPAGATION_PROTOTYPE_STATE
            ),
            "scope": self.scope.metadata_record(),
            "authority_namespace_id": self.authority_namespace_id,
            "profile_id": self.profile_id,
            "required_plane_kinds": list(self.required_plane_kinds),
            "production_influence": self.production_influence,
            "canonical_claim_authority": (
                self.canonical_claim_authority
            ),
            "destructive_live_deletion": (
                self.destructive_live_deletion
            ),
        }

    def profile_sha256(self) -> str:
        return canonical_sha256(self.metadata_record())

    def validate(self) -> None:
        self.scope.validate()
        if not self.required_plane_kinds:
            raise CognitiveKernelContractError(
                "required_plane_kinds may not be empty"
            )
        if self.production_influence:
            raise CognitiveKernelContractError(
                "M2.6 reversible profile may not influence production"
            )
        if self.canonical_claim_authority:
            raise CognitiveKernelContractError(
                "M2.6 profile may not write canonical Claim Authority"
            )
        if self.destructive_live_deletion:
            raise CognitiveKernelContractError(
                "M2.6 profile may not execute destructive live deletion"
            )


@dataclass(frozen=True)
class DeletionPropagationOperationReceipt:
    operation_id: str
    operation_kind: str
    request_id: str
    generation: int
    result_record_id: str
    result_sha256: str
    full_operation_content: dict[str, object]
    operation_sha256: str

    def metadata_record(self) -> dict[str, object]:
        return {
            "schema_version": (
                DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION
            ),
            "operation_id": self.operation_id,
            "operation_kind": self.operation_kind,
            "request_id": self.request_id,
            "generation": self.generation,
            "result_record_id": self.result_record_id,
            "result_sha256": self.result_sha256,
            "full_operation_content_digest": canonical_sha256(
                self.full_operation_content
            ),
            "operation_sha256": self.operation_sha256,
        }


@dataclass(frozen=True)
class DeletionPropagationIntegrityReport:
    checked_requests: int
    checked_plane_receipts: int
    checked_restore_filters: int
    checked_rehearsals: int
    checked_operations: int
    problems: tuple[str, ...]

    @property
    def healthy(self) -> bool:
        return not self.problems


def validate_deletion_propagation_path(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name in {"", ".", ".."}:
        raise UnsafeDeletionPropagationPathError(
            "deletion database path is invalid"
        )
    if repository_root is not None:
        repository = repository_root.expanduser().resolve()
        try:
            resolved.relative_to(repository)
        except ValueError:
            pass
        else:
            raise UnsafeDeletionPropagationPathError(
                "deletion database must remain outside public Git"
            )
    return resolved


def _scope_from_record(record: Mapping[str, object]) -> ProductHostScope:
    return ProductHostScope.create(
        product_id=record["product_id"],
        host_instance_id=record["host_instance_id"],
        schema_version=record["schema_version"],
        encryption_domain=record["encryption_domain"],
    )


def _envelope_from_record(
    record: Mapping[str, object],
) -> MemoryUnitEnvelope:
    return MemoryUnitEnvelope(
        scope=_scope_from_record(record["scope"]),
        record_id=str(record["record_id"]),
        record_type=str(record["record_type"]),
        authority_namespace_id=str(
            record["authority_namespace_id"]
        ),
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


def _plane_from_record(
    record: Mapping[str, object],
) -> DeletionPlaneReceipt:
    return DeletionPlaneReceipt(
        plane_receipt_id=str(record["plane_receipt_id"]),
        request_id=str(record["request_id"]),
        plane_kind=str(record["plane_kind"]),
        component_id=str(record["component_id"]),
        deletion_mode=str(record["deletion_mode"]),
        state=str(record["state"]),
        requested_at=str(record["requested_at"]),
        completed_at=(
            str(record["completed_at"])
            if record["completed_at"] is not None
            else None
        ),
        target_count=int(record["target_count"]),
        deleted_count=int(record["deleted_count"]),
        blocked_count=int(record["blocked_count"]),
        evidence_record_ids=tuple(
            str(item) for item in record["evidence_record_ids"]
        ),
        error_code=(
            str(record["error_code"])
            if record["error_code"] is not None
            else None
        ),
        result_content_digest=str(
            record["result_content_digest"]
        ),
        plane_receipt_sha256=str(
            record["plane_receipt_sha256"]
        ),
    )


def _restore_from_record(
    record: Mapping[str, object],
) -> RestoreFilterDecision:
    return RestoreFilterDecision(
        decision_id=str(record["decision_id"]),
        request_id=str(record["request_id"]),
        target_record_id=str(record["target_record_id"]),
        source_snapshot_id=str(record["source_snapshot_id"]),
        action=str(record["action"]),
        reason_code=str(record["reason_code"]),
        evaluated_at=str(record["evaluated_at"]),
        replacement_record_id=(
            str(record["replacement_record_id"])
            if record["replacement_record_id"] is not None
            else None
        ),
        source_content_digest=str(
            record["source_content_digest"]
        ),
        decision_sha256=str(record["decision_sha256"]),
    )


def _receipt_from_record(
    record: Mapping[str, object],
) -> DeletionPropagationReceipt:
    return DeletionPropagationReceipt(
        envelope=_envelope_from_record(record["envelope"]),
        receipt_id=str(record["receipt_id"]),
        request_id=str(record["request_id"]),
        deletion_mode=str(record["deletion_mode"]),
        propagation_state=str(record["propagation_state"]),
        target_record_ids=tuple(
            str(item) for item in record["target_record_ids"]
        ),
        reason_code=str(record["reason_code"]),
        authority_decision_id=str(
            record["authority_decision_id"]
        ),
        requested_by=str(record["requested_by"]),
        requested_at=str(record["requested_at"]),
        effective_at=(
            str(record["effective_at"])
            if record["effective_at"] is not None
            else None
        ),
        plane_receipts=tuple(
            _plane_from_record(item)
            for item in record["plane_receipts"]
        ),
        restore_filter_decision_ids=tuple(
            str(item)
            for item in record["restore_filter_decision_ids"]
        ),
        rollback_state=str(record["rollback_state"]),
        retirement_state=str(record["retirement_state"]),
        generation=int(record["generation"]),
        previous_receipt_id=(
            str(record["previous_receipt_id"])
            if record["previous_receipt_id"] is not None
            else None
        ),
        receipt_content_digest=str(
            record["receipt_content_digest"]
        ),
        receipt_sha256=str(record["receipt_sha256"]),
    )


def _identifier_tuple(
    values: Iterable[object],
    field: str,
) -> tuple[str, ...]:
    normalized = tuple(
        require_identifier(value, field)
        for value in values
    )
    if not normalized:
        raise CognitiveKernelContractError(
            f"{field} may not be empty"
        )
    if len(set(normalized)) != len(normalized):
        raise CognitiveKernelContractError(
            f"{field} may not contain duplicates"
        )
    return tuple(sorted(normalized))


def _make_id(prefix: str, material: object) -> str:
    return f"{prefix}-{canonical_sha256(material)[:24]}"


class DeletionPropagationPrototypeStore:
    """Persistent, isolated, nonproduction M2.6 deletion rehearsal."""

    def __init__(
        self,
        *,
        path: Path,
        profile: DeletionPropagationProfile,
        repository_root: Path | None = None,
    ) -> None:
        self.path = validate_deletion_propagation_path(
            path,
            repository_root=repository_root,
        )
        profile.validate()
        self.profile = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()
        self._bind_profile()

    def __enter__(self) -> "DeletionPropagationPrototypeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def remove_database(self) -> None:
        self.close()
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            candidate.unlink(missing_ok=True)

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                generation INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plane_receipts (
                plane_receipt_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                plane_key TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                UNIQUE(request_id, plane_key),
                UNIQUE(request_id, sequence_no),
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE TABLE IF NOT EXISTS restore_filters (
                decision_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                target_key TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                decision_sha256 TEXT NOT NULL,
                UNIQUE(request_id, target_key),
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE TABLE IF NOT EXISTS rehearsals (
                rehearsal_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                rehearsal_kind TEXT NOT NULL,
                rehearsal_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                rehearsal_sha256 TEXT NOT NULL,
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE TABLE IF NOT EXISTS propagation_receipts (
                receipt_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                receipt_json TEXT NOT NULL,
                full_content_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                UNIQUE(request_id, generation),
                FOREIGN KEY(request_id) REFERENCES requests(request_id)
            );
            CREATE TABLE IF NOT EXISTS current_receipts (
                request_id TEXT PRIMARY KEY,
                receipt_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                FOREIGN KEY(request_id) REFERENCES requests(request_id),
                FOREIGN KEY(receipt_id)
                    REFERENCES propagation_receipts(receipt_id)
            );
            CREATE TABLE IF NOT EXISTS operations (
                idempotency_namespace TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                operation_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                PRIMARY KEY(idempotency_namespace, idempotency_key)
            );
            """
        )
        self._connection.commit()

    def _bind_profile(self) -> None:
        profile_json = canonical_json_bytes(
            self.profile.metadata_record()
        ).decode("utf-8")
        profile_sha256 = self.profile.profile_sha256()
        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'profile_json'"
        ).fetchone()
        if row is None:
            self._connection.executemany(
                "INSERT INTO metadata(key, value) VALUES(?, ?)",
                (
                    ("profile_json", profile_json),
                    ("profile_sha256", profile_sha256),
                    (
                        "contract_schema_version",
                        DELETION_CONTRACT_SCHEMA_VERSION,
                    ),
                    (
                        "prototype_schema_version",
                        DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION,
                    ),
                ),
            )
            self._connection.commit()
            return
        if row["value"] != profile_json:
            raise DeletionPropagationIsolationError(
                "deletion database belongs to another profile"
            )

    def _existing_operation(
        self,
        *,
        idempotency_namespace: str,
        idempotency_key: str,
        operation_digest: str,
    ) -> DeletionPropagationOperationReceipt | None:
        row = self._connection.execute(
            """
            SELECT operation_digest, receipt_json
            FROM operations
            WHERE idempotency_namespace = ?
              AND idempotency_key = ?
            """,
            (idempotency_namespace, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["operation_digest"] != operation_digest:
            raise DeletionPropagationConflictError(
                "idempotency key was reused with different content"
            )
        record = json.loads(row["receipt_json"])
        return DeletionPropagationOperationReceipt(
            operation_id=record["operation_id"],
            operation_kind=record["operation_kind"],
            request_id=record["request_id"],
            generation=int(record["generation"]),
            result_record_id=record["result_record_id"],
            result_sha256=record["result_sha256"],
            full_operation_content=record[
                "full_operation_content"
            ],
            operation_sha256=record["operation_sha256"],
        )

    def _store_operation(
        self,
        *,
        cursor: sqlite3.Cursor,
        receipt: DeletionPropagationOperationReceipt,
        idempotency_namespace: str,
        idempotency_key: str,
        operation_digest: str,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO operations(
                idempotency_namespace,
                idempotency_key,
                operation_digest,
                receipt_json
            ) VALUES(?, ?, ?, ?)
            """,
            (
                idempotency_namespace,
                idempotency_key,
                operation_digest,
                canonical_json_bytes(
                    {
                        **receipt.metadata_record(),
                        "full_operation_content": (
                            receipt.full_operation_content
                        ),
                    }
                ).decode("utf-8"),
            ),
        )

    def _operation_receipt(
        self,
        *,
        operation_kind: str,
        request_id: str,
        generation: int,
        result_record_id: str,
        result_sha256: str,
        full_operation_content: Mapping[str, object],
    ) -> DeletionPropagationOperationReceipt:
        material = {
            "operation_kind": operation_kind,
            "request_id": request_id,
            "generation": generation,
            "result_record_id": result_record_id,
            "result_sha256": result_sha256,
            "full_operation_content": dict(full_operation_content),
        }
        operation_id = _make_id("deletion-operation", material)
        operation_sha256 = canonical_sha256(
            {
                "operation_id": operation_id,
                **material,
            }
        )
        return DeletionPropagationOperationReceipt(
            operation_id=operation_id,
            operation_kind=operation_kind,
            request_id=request_id,
            generation=generation,
            result_record_id=result_record_id,
            result_sha256=result_sha256,
            full_operation_content=dict(full_operation_content),
            operation_sha256=operation_sha256,
        )

    def begin_request(
        self,
        *,
        request_id: object,
        target_record_ids: Iterable[object],
        deletion_mode: object,
        reason_code: object,
        authority_decision_id: object,
        requested_by: object,
        requested_at: object,
        full_request_content: Mapping[str, object],
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DeletionPropagationOperationReceipt:
        request = {
            "schema_version": (
                DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION
            ),
            "request_id": require_identifier(
                request_id, "request_id"
            ),
            "target_record_ids": list(
                _identifier_tuple(
                    target_record_ids, "target_record_ids"
                )
            ),
            "deletion_mode": require_identifier(
                deletion_mode, "deletion_mode"
            ),
            "reason_code": require_identifier(
                reason_code, "reason_code"
            ),
            "authority_decision_id": require_identifier(
                authority_decision_id,
                "authority_decision_id",
            ),
            "requested_by": require_identifier(
                requested_by, "requested_by"
            ),
            "requested_at": normalize_timestamp(
                requested_at, "requested_at"
            ),
            "required_plane_kinds": list(
                self.profile.required_plane_kinds
            ),
            "state": "requested",
        }
        namespace = require_identifier(
            idempotency_namespace,
            "idempotency_namespace",
        )
        key = require_identifier(
            idempotency_key, "idempotency_key"
        )
        full_content = dict(full_request_content)
        operation_digest = canonical_sha256(
            {
                "operation": "begin_request",
                "request": request,
                "full_request_content": full_content,
            }
        )
        existing = self._existing_operation(
            idempotency_namespace=namespace,
            idempotency_key=key,
            operation_digest=operation_digest,
        )
        if existing is not None:
            return existing
        request_sha256 = canonical_sha256(
            {
                "request": request,
                "full_request_content": full_content,
            }
        )
        receipt = self._operation_receipt(
            operation_kind="begin_request",
            request_id=request["request_id"],
            generation=1,
            result_record_id=request["request_id"],
            result_sha256=request_sha256,
            full_operation_content=full_content,
        )
        try:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                INSERT INTO requests(
                    request_id,
                    request_json,
                    full_content_json,
                    request_sha256,
                    generation
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    request["request_id"],
                    canonical_json_bytes(request).decode("utf-8"),
                    canonical_json_bytes(full_content).decode("utf-8"),
                    request_sha256,
                    1,
                ),
            )
            self._store_operation(
                cursor=cursor,
                receipt=receipt,
                idempotency_namespace=namespace,
                idempotency_key=key,
                operation_digest=operation_digest,
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DeletionPropagationConflictError(
                "request already exists"
            ) from exc
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, DeletionPropagationPrototypeError):
                raise
            raise DeletionPropagationTransactionError(
                "begin_request failed"
            ) from exc
        return receipt

    def request_content(
        self,
        request_id: object,
    ) -> dict[str, object]:
        normalized = require_identifier(request_id, "request_id")
        row = self._connection.execute(
            """
            SELECT request_json, full_content_json
            FROM requests WHERE request_id = ?
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return {
            "request": json.loads(row["request_json"]),
            "full_content": json.loads(row["full_content_json"]),
        }

    def record_plane_result(
        self,
        *,
        request_id: object,
        plane_kind: object,
        component_id: object,
        deletion_mode: object,
        state: object,
        completed_at: object | None,
        target_count: object,
        deleted_count: object,
        blocked_count: object,
        evidence_record_ids: Iterable[object],
        error_code: object | None,
        full_result_content: Mapping[str, object],
        expected_generation: object,
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DeletionPropagationOperationReceipt:
        normalized_request = require_identifier(
            request_id, "request_id"
        )
        normalized_plane = require_identifier(
            plane_kind, "plane_kind"
        )
        normalized_component = require_identifier(
            component_id, "component_id"
        )
        if normalized_plane not in self.profile.required_plane_kinds:
            raise CognitiveKernelContractError(
                "plane_kind is not selected by this profile"
            )
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 1
        ):
            raise CognitiveKernelContractError(
                "expected_generation must be positive"
            )
        plane_key = f"{normalized_plane}:{normalized_component}"
        full_content = dict(full_result_content)
        material = {
            "request_id": normalized_request,
            "plane_kind": normalized_plane,
            "component_id": normalized_component,
            "deletion_mode": require_identifier(
                deletion_mode, "deletion_mode"
            ),
            "state": require_identifier(state, "state"),
            "completed_at": (
                normalize_timestamp(completed_at, "completed_at")
                if completed_at is not None
                else None
            ),
            "target_count": target_count,
            "deleted_count": deleted_count,
            "blocked_count": blocked_count,
            "evidence_record_ids": list(evidence_record_ids),
            "error_code": error_code,
            "full_result_content": full_content,
            "generation": expected_generation + 1,
        }
        namespace = require_identifier(
            idempotency_namespace, "idempotency_namespace"
        )
        key = require_identifier(
            idempotency_key, "idempotency_key"
        )
        operation_digest = canonical_sha256(
            {"operation": "record_plane_result", **material}
        )
        existing = self._existing_operation(
            idempotency_namespace=namespace,
            idempotency_key=key,
            operation_digest=operation_digest,
        )
        if existing is not None:
            return existing
        request_row = self._connection.execute(
            """
            SELECT request_json, full_content_json, generation
            FROM requests WHERE request_id = ?
            """,
            (normalized_request,),
        ).fetchone()
        if request_row is None:
            raise KeyError(normalized_request)
        if int(request_row["generation"]) != expected_generation:
            raise DeletionPropagationConflictError(
                "expected request generation does not match"
            )
        request_record = json.loads(request_row["request_json"])
        result_digest = canonical_sha256(full_content)
        plane_receipt_id = _make_id(
            "deletion-plane",
            {
                "request_id": normalized_request,
                "plane_key": plane_key,
                "generation": expected_generation + 1,
                "result_digest": result_digest,
            },
        )
        plane_receipt = DeletionPlaneReceipt.create(
            plane_receipt_id=plane_receipt_id,
            request_id=normalized_request,
            plane_kind=normalized_plane,
            component_id=normalized_component,
            deletion_mode=material["deletion_mode"],
            state=material["state"],
            requested_at=request_record["requested_at"],
            completed_at=material["completed_at"],
            target_count=target_count,
            deleted_count=deleted_count,
            blocked_count=blocked_count,
            evidence_record_ids=material["evidence_record_ids"],
            error_code=error_code,
            result_content_digest=result_digest,
        )
        receipt = self._operation_receipt(
            operation_kind="record_plane_result",
            request_id=normalized_request,
            generation=expected_generation + 1,
            result_record_id=plane_receipt.plane_receipt_id,
            result_sha256=plane_receipt.plane_receipt_sha256,
            full_operation_content=full_content,
        )
        try:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            current = cursor.execute(
                "SELECT generation FROM requests WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()
            if (
                current is None
                or int(current["generation"]) != expected_generation
            ):
                raise DeletionPropagationConflictError(
                    "request generation changed during plane append"
                )
            sequence_no = int(
                cursor.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM plane_receipts
                    WHERE request_id = ?
                    """,
                    (normalized_request,),
                ).fetchone()["count"]
            ) + 1
            cursor.execute(
                """
                INSERT INTO plane_receipts(
                    plane_receipt_id,
                    request_id,
                    plane_key,
                    receipt_json,
                    full_content_json,
                    receipt_sha256,
                    sequence_no
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plane_receipt.plane_receipt_id,
                    normalized_request,
                    plane_key,
                    canonical_json_bytes(
                        plane_receipt.metadata_record()
                    ).decode("utf-8"),
                    canonical_json_bytes(full_content).decode("utf-8"),
                    plane_receipt.plane_receipt_sha256,
                    sequence_no,
                ),
            )
            request_record["state"] = "propagating"
            updated_request_sha256 = canonical_sha256(
                {
                    "request": request_record,
                    "full_request_content": json.loads(
                        request_row["full_content_json"]
                    ),
                }
            )
            cursor.execute(
                """
                UPDATE requests
                SET request_json = ?, request_sha256 = ?, generation = ?
                WHERE request_id = ?
                """,
                (
                    canonical_json_bytes(request_record).decode(
                        "utf-8"
                    ),
                    updated_request_sha256,
                    expected_generation + 1,
                    normalized_request,
                ),
            )
            self._store_operation(
                cursor=cursor,
                receipt=receipt,
                idempotency_namespace=namespace,
                idempotency_key=key,
                operation_digest=operation_digest,
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DeletionPropagationConflictError(
                "plane result already exists"
            ) from exc
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, DeletionPropagationPrototypeError):
                raise
            raise DeletionPropagationTransactionError(
                "record_plane_result failed"
            ) from exc
        return receipt

    def plane_receipts(
        self,
        request_id: object,
    ) -> tuple[DeletionPlaneReceipt, ...]:
        normalized = require_identifier(request_id, "request_id")
        rows = self._connection.execute(
            """
            SELECT receipt_json
            FROM plane_receipts
            WHERE request_id = ?
            ORDER BY sequence_no
            """,
            (normalized,),
        ).fetchall()
        return tuple(
            _plane_from_record(json.loads(row["receipt_json"]))
            for row in rows
        )

    def evaluate_restore_filter(
        self,
        *,
        request_id: object,
        target_record_id: object,
        source_snapshot_id: object,
        action: object,
        reason_code: object,
        evaluated_at: object,
        source_content: Mapping[str, object],
        replacement_record_id: object | None,
        full_decision_content: Mapping[str, object],
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DeletionPropagationOperationReceipt:
        normalized_request = require_identifier(
            request_id, "request_id"
        )
        target = require_identifier(
            target_record_id, "target_record_id"
        )
        if self._connection.execute(
            "SELECT 1 FROM requests WHERE request_id = ?",
            (normalized_request,),
        ).fetchone() is None:
            raise KeyError(normalized_request)
        full_content = dict(full_decision_content)
        source_digest = canonical_sha256(dict(source_content))
        material = {
            "request_id": normalized_request,
            "target_record_id": target,
            "source_snapshot_id": require_identifier(
                source_snapshot_id, "source_snapshot_id"
            ),
            "action": require_identifier(action, "action"),
            "reason_code": require_identifier(
                reason_code, "reason_code"
            ),
            "evaluated_at": normalize_timestamp(
                evaluated_at, "evaluated_at"
            ),
            "source_content_digest": source_digest,
            "replacement_record_id": replacement_record_id,
            "full_decision_content": full_content,
        }
        namespace = require_identifier(
            idempotency_namespace, "idempotency_namespace"
        )
        key = require_identifier(
            idempotency_key, "idempotency_key"
        )
        operation_digest = canonical_sha256(
            {"operation": "evaluate_restore_filter", **material}
        )
        existing = self._existing_operation(
            idempotency_namespace=namespace,
            idempotency_key=key,
            operation_digest=operation_digest,
        )
        if existing is not None:
            return existing
        decision_id = _make_id("restore-filter", material)
        decision = RestoreFilterDecision.create(
            decision_id=decision_id,
            request_id=normalized_request,
            target_record_id=target,
            source_snapshot_id=material["source_snapshot_id"],
            action=material["action"],
            reason_code=material["reason_code"],
            evaluated_at=material["evaluated_at"],
            replacement_record_id=replacement_record_id,
            source_content_digest=source_digest,
        )
        generation = int(
            self._connection.execute(
                "SELECT generation FROM requests WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()["generation"]
        )
        receipt = self._operation_receipt(
            operation_kind="evaluate_restore_filter",
            request_id=normalized_request,
            generation=generation,
            result_record_id=decision.decision_id,
            result_sha256=decision.decision_sha256,
            full_operation_content=full_content,
        )
        try:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                INSERT INTO restore_filters(
                    decision_id,
                    request_id,
                    target_key,
                    decision_json,
                    full_content_json,
                    decision_sha256
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    normalized_request,
                    target,
                    canonical_json_bytes(
                        decision.metadata_record()
                    ).decode("utf-8"),
                    canonical_json_bytes(full_content).decode("utf-8"),
                    decision.decision_sha256,
                ),
            )
            self._store_operation(
                cursor=cursor,
                receipt=receipt,
                idempotency_namespace=namespace,
                idempotency_key=key,
                operation_digest=operation_digest,
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DeletionPropagationConflictError(
                "restore-filter decision already exists"
            ) from exc
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, DeletionPropagationPrototypeError):
                raise
            raise DeletionPropagationTransactionError(
                "evaluate_restore_filter failed"
            ) from exc
        return receipt

    def restore_filter_decisions(
        self,
        request_id: object,
    ) -> tuple[RestoreFilterDecision, ...]:
        normalized = require_identifier(request_id, "request_id")
        rows = self._connection.execute(
            """
            SELECT decision_json
            FROM restore_filters
            WHERE request_id = ?
            ORDER BY target_key
            """,
            (normalized,),
        ).fetchall()
        return tuple(
            _restore_from_record(json.loads(row["decision_json"]))
            for row in rows
        )

    def record_rehearsal(
        self,
        *,
        request_id: object,
        rehearsal_kind: object,
        outcome: object,
        evaluated_at: object,
        affected_artifact_ids: Iterable[object],
        measurements: Mapping[str, object],
        full_rehearsal_content: Mapping[str, object],
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DeletionPropagationOperationReceipt:
        normalized_request = require_identifier(
            request_id, "request_id"
        )
        if self._connection.execute(
            "SELECT 1 FROM requests WHERE request_id = ?",
            (normalized_request,),
        ).fetchone() is None:
            raise KeyError(normalized_request)
        full_content = dict(full_rehearsal_content)
        rehearsal = {
            "schema_version": (
                DELETION_PROPAGATION_PROTOTYPE_SCHEMA_VERSION
            ),
            "request_id": normalized_request,
            "rehearsal_kind": require_identifier(
                rehearsal_kind, "rehearsal_kind"
            ),
            "outcome": require_identifier(outcome, "outcome"),
            "evaluated_at": normalize_timestamp(
                evaluated_at, "evaluated_at"
            ),
            "affected_artifact_ids": list(
                _identifier_tuple(
                    affected_artifact_ids,
                    "affected_artifact_ids",
                )
            ),
            "measurements": dict(measurements),
        }
        rehearsal_id = _make_id("deletion-rehearsal", rehearsal)
        rehearsal["rehearsal_id"] = rehearsal_id
        rehearsal_sha256 = canonical_sha256(
            {
                "rehearsal": rehearsal,
                "full_content": full_content,
            }
        )
        namespace = require_identifier(
            idempotency_namespace, "idempotency_namespace"
        )
        key = require_identifier(
            idempotency_key, "idempotency_key"
        )
        operation_digest = canonical_sha256(
            {
                "operation": "record_rehearsal",
                "rehearsal": rehearsal,
                "full_content": full_content,
            }
        )
        existing = self._existing_operation(
            idempotency_namespace=namespace,
            idempotency_key=key,
            operation_digest=operation_digest,
        )
        if existing is not None:
            return existing
        generation = int(
            self._connection.execute(
                "SELECT generation FROM requests WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()["generation"]
        )
        receipt = self._operation_receipt(
            operation_kind="record_rehearsal",
            request_id=normalized_request,
            generation=generation,
            result_record_id=rehearsal_id,
            result_sha256=rehearsal_sha256,
            full_operation_content=full_content,
        )
        try:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                """
                INSERT INTO rehearsals(
                    rehearsal_id,
                    request_id,
                    rehearsal_kind,
                    rehearsal_json,
                    full_content_json,
                    rehearsal_sha256
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    rehearsal_id,
                    normalized_request,
                    rehearsal["rehearsal_kind"],
                    canonical_json_bytes(rehearsal).decode("utf-8"),
                    canonical_json_bytes(full_content).decode("utf-8"),
                    rehearsal_sha256,
                ),
            )
            self._store_operation(
                cursor=cursor,
                receipt=receipt,
                idempotency_namespace=namespace,
                idempotency_key=key,
                operation_digest=operation_digest,
            )
            self._connection.commit()
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DeletionPropagationConflictError(
                "rehearsal already exists"
            ) from exc
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, DeletionPropagationPrototypeError):
                raise
            raise DeletionPropagationTransactionError(
                "record_rehearsal failed"
            ) from exc
        return receipt

    def rehearsals(
        self,
        request_id: object,
    ) -> tuple[dict[str, object], ...]:
        normalized = require_identifier(request_id, "request_id")
        rows = self._connection.execute(
            """
            SELECT rehearsal_json, full_content_json
            FROM rehearsals
            WHERE request_id = ?
            ORDER BY rehearsal_id
            """,
            (normalized,),
        ).fetchall()
        return tuple(
            {
                "rehearsal": json.loads(row["rehearsal_json"]),
                "full_content": json.loads(
                    row["full_content_json"]
                ),
            }
            for row in rows
        )

    def finalize(
        self,
        *,
        request_id: object,
        propagation_state: object,
        effective_at: object | None,
        rollback_state: object,
        retirement_state: object,
        full_receipt_content: Mapping[str, object],
        expected_generation: object,
        completed_at: object,
        idempotency_namespace: object,
        idempotency_key: object,
    ) -> DeletionPropagationOperationReceipt:
        normalized_request = require_identifier(
            request_id, "request_id"
        )
        normalized_state = require_identifier(
            propagation_state, "propagation_state"
        )
        if normalized_state not in DELETION_PROPAGATION_STATES:
            raise CognitiveKernelContractError(
                "propagation_state is not ratified"
            )
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation < 1
        ):
            raise CognitiveKernelContractError(
                "expected_generation must be positive"
            )
        full_content = dict(full_receipt_content)
        namespace = require_identifier(
            idempotency_namespace, "idempotency_namespace"
        )
        key = require_identifier(
            idempotency_key, "idempotency_key"
        )
        operation_digest = canonical_sha256(
            {
                "operation": "finalize",
                "request_id": normalized_request,
                "propagation_state": normalized_state,
                "effective_at": effective_at,
                "rollback_state": rollback_state,
                "retirement_state": retirement_state,
                "full_receipt_content": full_content,
                "expected_generation": expected_generation,
                "completed_at": completed_at,
            }
        )
        existing = self._existing_operation(
            idempotency_namespace=namespace,
            idempotency_key=key,
            operation_digest=operation_digest,
        )
        if existing is not None:
            return existing
        row = self._connection.execute(
            """
            SELECT request_json, full_content_json, generation
            FROM requests WHERE request_id = ?
            """,
            (normalized_request,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized_request)
        if int(row["generation"]) != expected_generation:
            raise DeletionPropagationConflictError(
                "expected request generation does not match"
            )
        request = json.loads(row["request_json"])
        planes = self.plane_receipts(normalized_request)
        observed_plane_kinds = {item.plane_kind for item in planes}
        missing_planes = sorted(
            set(self.profile.required_plane_kinds)
            - observed_plane_kinds
        )
        if missing_planes:
            raise DeletionPropagationConflictError(
                f"required planes are missing: {missing_planes}"
            )
        decisions = self.restore_filter_decisions(
            normalized_request
        )
        request_next_generation = expected_generation + 1
        previous = self.current_receipt(normalized_request)
        receipt_generation = (
            1 if previous is None else previous.generation + 1
        )
        receipt_id = _make_id(
            "deletion-propagation",
            {
                "request_id": normalized_request,
                "generation": receipt_generation,
                "planes": [
                    item.plane_receipt_sha256 for item in planes
                ],
                "decisions": [
                    item.decision_sha256 for item in decisions
                ],
                "full_content": full_content,
            },
        )
        content_digest = canonical_sha256(full_content)
        timestamp = normalize_timestamp(
            completed_at, "completed_at"
        )
        envelope = MemoryUnitEnvelope.create(
            scope=self.profile.scope,
            record_id=receipt_id,
            record_type="deletion_propagation_receipt",
            authority_namespace_id=(
                self.profile.authority_namespace_id
            ),
            host_or_cluster_id=(
                self.profile.scope.host_instance_id
            ),
            authority_role="operational_workflow_state",
            deployment_profile=self.profile.profile_id,
            created_at=timestamp,
            valid_from=timestamp,
            valid_to=None,
            transaction_time=timestamp,
            logical_clock=request_next_generation,
            causal_parents=(
                (previous.receipt_id,)
                if previous is not None
                else ()
            ),
            source_records=tuple(
                item.plane_receipt_id for item in planes
            ),
            generation=receipt_generation,
            state=normalized_state,
            data_classification="owner_private",
            retention_class="owner_hold",
            deletion_state="deletion_rehearsal",
            provenance_digest=canonical_sha256(
                {
                    "request_id": normalized_request,
                    "authority_decision_id": request[
                        "authority_decision_id"
                    ],
                }
            ),
            content_digest=content_digest,
            writer="deletion-propagation-prototype",
            workflow_or_request_id=normalized_request,
            idempotency_namespace="memory-m2-6",
            idempotency_key=receipt_id,
            supersedes=(
                (previous.receipt_id,)
                if previous is not None
                else ()
            ),
            superseded_by=(),
            rollback_reference=None,
        )
        propagation_receipt = DeletionPropagationReceipt.create(
            envelope=envelope,
            receipt_id=receipt_id,
            request_id=normalized_request,
            deletion_mode=request["deletion_mode"],
            propagation_state=normalized_state,
            target_record_ids=request["target_record_ids"],
            reason_code=request["reason_code"],
            authority_decision_id=request[
                "authority_decision_id"
            ],
            requested_by=request["requested_by"],
            requested_at=request["requested_at"],
            effective_at=effective_at,
            plane_receipts=planes,
            restore_filter_decision_ids=tuple(
                item.decision_id for item in decisions
            ),
            rollback_state=rollback_state,
            retirement_state=retirement_state,
            generation=receipt_generation,
            previous_receipt_id=(
                previous.receipt_id
                if previous is not None
                else None
            ),
            receipt_content_digest=content_digest,
        )
        operation = self._operation_receipt(
            operation_kind="finalize",
            request_id=normalized_request,
            generation=request_next_generation,
            result_record_id=receipt_id,
            result_sha256=propagation_receipt.receipt_sha256,
            full_operation_content=full_content,
        )
        try:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            current = cursor.execute(
                "SELECT generation FROM requests WHERE request_id = ?",
                (normalized_request,),
            ).fetchone()
            if (
                current is None
                or int(current["generation"]) != expected_generation
            ):
                raise DeletionPropagationConflictError(
                    "request generation changed during finalize"
                )
            cursor.execute(
                """
                INSERT INTO propagation_receipts(
                    receipt_id,
                    request_id,
                    generation,
                    receipt_json,
                    full_content_json,
                    receipt_sha256
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    normalized_request,
                    receipt_generation,
                    canonical_json_bytes(
                        propagation_receipt.metadata_record()
                    ).decode("utf-8"),
                    canonical_json_bytes(full_content).decode("utf-8"),
                    propagation_receipt.receipt_sha256,
                ),
            )
            cursor.execute(
                """
                INSERT INTO current_receipts(
                    request_id,
                    receipt_id,
                    generation,
                    receipt_sha256
                ) VALUES(?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    receipt_id = excluded.receipt_id,
                    generation = excluded.generation,
                    receipt_sha256 = excluded.receipt_sha256
                """,
                (
                    normalized_request,
                    receipt_id,
                    receipt_generation,
                    propagation_receipt.receipt_sha256,
                ),
            )
            request["state"] = normalized_state
            updated_request_sha256 = canonical_sha256(
                {
                    "request": request,
                    "full_request_content": json.loads(
                        row["full_content_json"]
                    ),
                }
            )
            cursor.execute(
                """
                UPDATE requests
                SET request_json = ?, request_sha256 = ?, generation = ?
                WHERE request_id = ?
                """,
                (
                    canonical_json_bytes(request).decode("utf-8"),
                    updated_request_sha256,
                    request_next_generation,
                    normalized_request,
                ),
            )
            self._store_operation(
                cursor=cursor,
                receipt=operation,
                idempotency_namespace=namespace,
                idempotency_key=key,
                operation_digest=operation_digest,
            )
            self._connection.commit()
        except Exception as exc:
            self._connection.rollback()
            if isinstance(exc, DeletionPropagationPrototypeError):
                raise
            if isinstance(exc, sqlite3.IntegrityError):
                raise DeletionPropagationConflictError(
                    "propagation receipt already exists"
                ) from exc
            raise DeletionPropagationTransactionError(
                "finalize failed"
            ) from exc
        return operation

    def current_receipt(
        self,
        request_id: object,
    ) -> DeletionPropagationReceipt | None:
        normalized = require_identifier(request_id, "request_id")
        row = self._connection.execute(
            """
            SELECT p.receipt_json
            FROM current_receipts c
            JOIN propagation_receipts p
              ON p.receipt_id = c.receipt_id
            WHERE c.request_id = ?
            """,
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        receipt = _receipt_from_record(
            json.loads(row["receipt_json"])
        )
        receipt.validate()
        return receipt

    def receipt_history(
        self,
        request_id: object,
    ) -> tuple[DeletionPropagationReceipt, ...]:
        normalized = require_identifier(request_id, "request_id")
        rows = self._connection.execute(
            """
            SELECT receipt_json
            FROM propagation_receipts
            WHERE request_id = ?
            ORDER BY generation
            """,
            (normalized,),
        ).fetchall()
        return tuple(
            _receipt_from_record(json.loads(row["receipt_json"]))
            for row in rows
        )

    def verify_integrity(
        self,
    ) -> DeletionPropagationIntegrityReport:
        problems: list[str] = []
        request_rows = self._connection.execute(
            """
            SELECT request_id, request_json, full_content_json,
                   request_sha256
            FROM requests ORDER BY request_id
            """
        ).fetchall()
        for row in request_rows:
            expected = canonical_sha256(
                {
                    "request": json.loads(row["request_json"]),
                    "full_request_content": json.loads(
                        row["full_content_json"]
                    ),
                }
            )
            if expected != row["request_sha256"]:
                problems.append(
                    f"request:{row['request_id']}:digest"
                )
        plane_rows = self._connection.execute(
            """
            SELECT plane_receipt_id, receipt_json,
                   full_content_json, receipt_sha256
            FROM plane_receipts ORDER BY plane_receipt_id
            """
        ).fetchall()
        for row in plane_rows:
            try:
                value = _plane_from_record(
                    json.loads(row["receipt_json"])
                )
                value.validate()
                if value.plane_receipt_sha256 != row["receipt_sha256"]:
                    problems.append(
                        f"plane:{row['plane_receipt_id']}:digest"
                    )
                if canonical_sha256(
                    json.loads(row["full_content_json"])
                ) != value.result_content_digest:
                    problems.append(
                        f"plane:{row['plane_receipt_id']}:content"
                    )
            except Exception:
                problems.append(
                    f"plane:{row['plane_receipt_id']}:invalid"
                )
        restore_rows = self._connection.execute(
            """
            SELECT decision_id, decision_json, decision_sha256
            FROM restore_filters ORDER BY decision_id
            """
        ).fetchall()
        for row in restore_rows:
            try:
                value = _restore_from_record(
                    json.loads(row["decision_json"])
                )
                value.validate()
                if value.decision_sha256 != row["decision_sha256"]:
                    problems.append(
                        f"restore:{row['decision_id']}:digest"
                    )
            except Exception:
                problems.append(
                    f"restore:{row['decision_id']}:invalid"
                )
        rehearsal_rows = self._connection.execute(
            """
            SELECT rehearsal_id, rehearsal_json,
                   full_content_json, rehearsal_sha256
            FROM rehearsals ORDER BY rehearsal_id
            """
        ).fetchall()
        for row in rehearsal_rows:
            expected = canonical_sha256(
                {
                    "rehearsal": json.loads(row["rehearsal_json"]),
                    "full_content": json.loads(
                        row["full_content_json"]
                    ),
                }
            )
            if expected != row["rehearsal_sha256"]:
                problems.append(
                    f"rehearsal:{row['rehearsal_id']}:digest"
                )
        operation_rows = self._connection.execute(
            """
            SELECT idempotency_namespace, idempotency_key,
                   operation_digest, receipt_json
            FROM operations
            ORDER BY idempotency_namespace, idempotency_key
            """
        ).fetchall()
        for row in operation_rows:
            record = json.loads(row["receipt_json"])
            expected_sha = canonical_sha256(
                {
                    "operation_id": record["operation_id"],
                    "operation_kind": record["operation_kind"],
                    "request_id": record["request_id"],
                    "generation": record["generation"],
                    "result_record_id": record["result_record_id"],
                    "result_sha256": record["result_sha256"],
                    "full_operation_content": record[
                        "full_operation_content"
                    ],
                }
            )
            if expected_sha != record["operation_sha256"]:
                problems.append(
                    "operation:"
                    f"{row['idempotency_namespace']}:"
                    f"{row['idempotency_key']}:digest"
                )
        receipt_rows = self._connection.execute(
            """
            SELECT receipt_id, receipt_json, full_content_json,
                   receipt_sha256
            FROM propagation_receipts ORDER BY receipt_id
            """
        ).fetchall()
        for row in receipt_rows:
            try:
                value = _receipt_from_record(
                    json.loads(row["receipt_json"])
                )
                value.validate()
                if value.receipt_sha256 != row["receipt_sha256"]:
                    problems.append(
                        f"receipt:{row['receipt_id']}:digest"
                    )
                if canonical_sha256(
                    json.loads(row["full_content_json"])
                ) != value.receipt_content_digest:
                    problems.append(
                        f"receipt:{row['receipt_id']}:content"
                    )
            except Exception:
                problems.append(
                    f"receipt:{row['receipt_id']}:invalid"
                )
        return DeletionPropagationIntegrityReport(
            checked_requests=len(request_rows),
            checked_plane_receipts=len(plane_rows),
            checked_restore_filters=len(restore_rows),
            checked_rehearsals=len(rehearsal_rows),
            checked_operations=len(operation_rows),
            problems=tuple(problems),
        )


def open_deletion_propagation_prototype(
    *,
    path: Path,
    profile: DeletionPropagationProfile,
    repository_root: Path | None = None,
) -> DeletionPropagationPrototypeStore:
    return DeletionPropagationPrototypeStore(
        path=path,
        profile=profile,
        repository_root=repository_root,
    )
