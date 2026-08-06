"""Memory M2.1 reversible embedded Claim Authority persistence prototype.

This module is an operational research prototype. It persists canonical claim
content and append-only versions outside the public repository. It does not
register itself as production authority or enable automatic memory formation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
    require_sha256,
)
from .claim_contracts import (
    CLAIM_CONFLICT_STATES,
    CLAIM_DELETION_STATES,
    CLAIM_VALIDITY_STATES,
    CanonicalTaggedValue,
    ClaimIdentity,
    ClaimQualifier,
    ClaimVersion,
    CurrentClaimProjection,
    normalize_claim_qualifiers,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

CLAIM_AUTHORITY_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
CLAIM_AUTHORITY_PROTOTYPE_STATE = "reversible_nonproduction"


class ClaimAuthorityPrototypeError(RuntimeError):
    """Base error for the reversible Claim Authority prototype."""


class UnsafeClaimAuthorityPrototypePathError(ClaimAuthorityPrototypeError):
    """Raised when a prototype database resolves inside public Git."""


class ClaimAuthorityPrototypeIsolationError(ClaimAuthorityPrototypeError):
    """Raised when product, host, encryption, or authority scope differs."""


class ClaimAuthorityPrototypeConflictError(ClaimAuthorityPrototypeError):
    """Raised for expected-version, idempotency, or digest conflicts."""


class ClaimAuthorityPrototypeIntegrityError(ClaimAuthorityPrototypeError):
    """Raised when persisted prototype records fail integrity checks."""


class ClaimAuthorityPrototypeTransactionError(ClaimAuthorityPrototypeError):
    """Raised for invalid or nested prototype write transactions."""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_claim_authority_prototype_path(
    database_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    candidate = Path(database_path).expanduser().resolve(strict=False)
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else default_repository_root().resolve(strict=True)
    )
    if _is_within(candidate, root):
        raise UnsafeClaimAuthorityPrototypePathError(
            "Refusing to create or open a Claim Authority prototype "
            f"inside the public repository: {candidate}"
        )
    return candidate


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = None if mode_row is None else str(mode_row[0]).lower()
    if mode != "wal":
        raise ClaimAuthorityPrototypeError(
            "Claim Authority prototype requires SQLite WAL mode"
        )
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=5000")


def _scope_record(scope: ProductHostScope) -> dict[str, str]:
    scope.validate()
    return {
        "product_id": scope.product_id,
        "host_instance_id": scope.host_instance_id,
        "encryption_domain": scope.encryption_domain,
    }


def _scope_digest(
    scope: ProductHostScope,
    authority_namespace_id: str,
) -> str:
    return canonical_sha256(
        {
            "scope": _scope_record(scope),
            "authority_namespace_id": authority_namespace_id,
            "prototype_schema_version": CLAIM_AUTHORITY_PROTOTYPE_SCHEMA_VERSION,
        }
    )


def _canonical_json(value: dict[str, object]) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _load_json_object(raw: object, *, field: str) -> dict[str, object]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ClaimAuthorityPrototypeIntegrityError(
            f"stored {field} JSON is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ClaimAuthorityPrototypeIntegrityError(
            f"stored {field} must be a JSON object"
        )
    return value


def _verify_record_digest(
    record: dict[str, object],
    *,
    digest_field: str,
    field: str,
) -> str:
    value = dict(record)
    try:
        digest = require_sha256(value.pop(digest_field, None), digest_field)
    except CognitiveKernelContractError as exc:
        raise ClaimAuthorityPrototypeIntegrityError(
            f"stored {field} digest field is invalid"
        ) from exc
    if canonical_sha256(value) != digest:
        raise ClaimAuthorityPrototypeIntegrityError(
            f"stored {field} digest mismatch"
        )
    return digest


def _same_scope(first: ProductHostScope, second: ProductHostScope) -> bool:
    return first.metadata_record() == second.metadata_record()


@dataclass(frozen=True)
class ClaimAuthorityAppendRequest:
    """One explicit, caller-authorized prototype append request."""

    identity: ClaimIdentity
    version_envelope: MemoryUnitEnvelope
    projection_envelope: MemoryUnitEnvelope
    value: CanonicalTaggedValue
    qualifiers: tuple[ClaimQualifier, ...]
    authority_class: str
    confidence: float | None
    adjudication_state: str
    evidence_relation_ids: tuple[str, ...]
    conflict_set_id: str | None
    correction_of: tuple[str, ...]
    request_digest: str
    expected_current_claim_version_id: str | None
    validity_state: str
    conflict_state: str
    deletion_state: str

    @classmethod
    def create(
        cls,
        *,
        identity: ClaimIdentity,
        version_envelope: MemoryUnitEnvelope,
        projection_envelope: MemoryUnitEnvelope,
        value: CanonicalTaggedValue,
        qualifiers: Iterable[ClaimQualifier] = (),
        authority_class: object,
        confidence: float | None,
        adjudication_state: object,
        evidence_relation_ids: Iterable[object] = (),
        conflict_set_id: object | None = None,
        correction_of: Iterable[object] = (),
        request_digest: object,
        expected_current_claim_version_id: object | None = None,
        validity_state: object = "current",
        conflict_state: object = "none",
        deletion_state: object = "active",
    ) -> "ClaimAuthorityAppendRequest":
        request = cls(
            identity=identity,
            version_envelope=version_envelope,
            projection_envelope=projection_envelope,
            value=value,
            qualifiers=normalize_claim_qualifiers(qualifiers),
            authority_class=require_identifier(
                authority_class, "authority_class"
            ),
            confidence=confidence,
            adjudication_state=require_identifier(
                adjudication_state, "adjudication_state"
            ),
            evidence_relation_ids=tuple(
                sorted(
                    require_identifier(item, "evidence_relation_ids")
                    for item in evidence_relation_ids
                )
            ),
            conflict_set_id=(
                require_identifier(conflict_set_id, "conflict_set_id")
                if conflict_set_id is not None
                else None
            ),
            correction_of=tuple(
                sorted(
                    require_identifier(item, "correction_of")
                    for item in correction_of
                )
            ),
            request_digest=require_sha256(
                request_digest, "request_digest"
            ),
            expected_current_claim_version_id=(
                require_identifier(
                    expected_current_claim_version_id,
                    "expected_current_claim_version_id",
                )
                if expected_current_claim_version_id is not None
                else None
            ),
            validity_state=require_identifier(
                validity_state, "validity_state"
            ),
            conflict_state=require_identifier(
                conflict_state, "conflict_state"
            ),
            deletion_state=require_identifier(
                deletion_state, "deletion_state"
            ),
        )
        request.validate()
        return request

    def validate(self) -> None:
        self.identity.validate()
        self.version_envelope.validate()
        self.projection_envelope.validate()
        self.value.validate()
        normalize_claim_qualifiers(self.qualifiers)
        require_sha256(self.request_digest, "request_digest")
        if self.version_envelope.record_type != "claim_version":
            raise CognitiveKernelContractError(
                "version envelope must use record_type claim_version"
            )
        if self.version_envelope.authority_role != "claim_authority":
            raise CognitiveKernelContractError(
                "version envelope must use claim_authority"
            )
        if self.projection_envelope.record_type != "current_claim_projection":
            raise CognitiveKernelContractError(
                "projection envelope must use current_claim_projection"
            )
        if self.projection_envelope.authority_role != "registered_projection":
            raise CognitiveKernelContractError(
                "projection envelope must use registered_projection"
            )
        scopes = (
            self.identity.envelope.scope,
            self.version_envelope.scope,
            self.projection_envelope.scope,
        )
        if not all(_same_scope(scopes[0], scope) for scope in scopes[1:]):
            raise CognitiveKernelContractError(
                "append request crosses product-host-encryption scope"
            )
        namespaces = {
            self.identity.envelope.authority_namespace_id,
            self.version_envelope.authority_namespace_id,
            self.projection_envelope.authority_namespace_id,
        }
        if len(namespaces) != 1:
            raise CognitiveKernelContractError(
                "append request crosses authority namespaces"
            )
        claim_id = self.identity.claim_id
        version_id = self.version_envelope.record_id
        if claim_id not in self.version_envelope.source_records:
            raise CognitiveKernelContractError(
                "version envelope must bind the claim identity"
            )
        if claim_id not in self.projection_envelope.source_records:
            raise CognitiveKernelContractError(
                "projection envelope must bind the claim identity"
            )
        if version_id not in self.projection_envelope.source_records:
            raise CognitiveKernelContractError(
                "projection envelope must bind the claim version"
            )
        if self.validity_state not in CLAIM_VALIDITY_STATES:
            raise CognitiveKernelContractError("validity_state is not ratified")
        if self.conflict_state not in CLAIM_CONFLICT_STATES:
            raise CognitiveKernelContractError("conflict_state is not ratified")
        if self.deletion_state not in CLAIM_DELETION_STATES:
            raise CognitiveKernelContractError("deletion_state is not ratified")
        if self.deletion_state != self.projection_envelope.deletion_state:
            raise CognitiveKernelContractError(
                "projection deletion state differs from request"
            )

    def idempotency_tuple(self) -> tuple[str, str, str]:
        self.validate()
        return (
            self.version_envelope.idempotency_namespace,
            self.version_envelope.idempotency_key,
            self.request_digest,
        )


@dataclass(frozen=True)
class ClaimAuthorityAppendReceipt:
    """Inspectable receipt for one prototype append or idempotent replay."""

    authority_id: str
    claim_id: str
    claim_version_id: str
    store_sequence: int
    version_sequence: int
    projection_id: str
    projection_generation: int
    request_digest: str
    committed_at: str
    idempotent_replay: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        authority_id: str,
        claim_id: str,
        claim_version_id: str,
        store_sequence: int,
        version_sequence: int,
        projection_id: str,
        projection_generation: int,
        request_digest: str,
        committed_at: object,
        idempotent_replay: bool,
    ) -> "ClaimAuthorityAppendReceipt":
        draft = cls(
            authority_id=require_identifier(authority_id, "authority_id"),
            claim_id=require_identifier(claim_id, "claim_id"),
            claim_version_id=require_identifier(
                claim_version_id, "claim_version_id"
            ),
            store_sequence=store_sequence,
            version_sequence=version_sequence,
            projection_id=require_identifier(projection_id, "projection_id"),
            projection_generation=projection_generation,
            request_digest=require_sha256(request_digest, "request_digest"),
            committed_at=normalize_timestamp(committed_at, "committed_at"),
            idempotent_replay=bool(idempotent_replay),
            receipt_sha256="0" * 64,
        )
        if store_sequence < 1 or version_sequence < 1:
            raise CognitiveKernelContractError(
                "append receipt sequences must be positive"
            )
        if projection_generation < 1:
            raise CognitiveKernelContractError(
                "projection generation must be positive"
            )
        result = cls(
            **{
                **draft.__dict__,
                "receipt_sha256": canonical_sha256(draft.material_record()),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "authority_id": self.authority_id,
            "claim_id": self.claim_id,
            "claim_version_id": self.claim_version_id,
            "store_sequence": self.store_sequence,
            "version_sequence": self.version_sequence,
            "projection_id": self.projection_id,
            "projection_generation": self.projection_generation,
            "request_digest": self.request_digest,
            "committed_at": self.committed_at,
            "idempotent_replay": self.idempotent_replay,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["receipt_sha256"] = self.receipt_sha256
        return record

    def validate(self) -> None:
        require_sha256(self.receipt_sha256, "receipt_sha256")
        if canonical_sha256(self.material_record()) != self.receipt_sha256:
            raise CognitiveKernelContractError("append receipt digest mismatch")

    @classmethod
    def from_record(
        cls,
        record: dict[str, object],
        *,
        idempotent_replay: bool,
    ) -> "ClaimAuthorityAppendReceipt":
        result = cls.create(
            authority_id=record["authority_id"],
            claim_id=record["claim_id"],
            claim_version_id=record["claim_version_id"],
            store_sequence=int(record["store_sequence"]),
            version_sequence=int(record["version_sequence"]),
            projection_id=record["projection_id"],
            projection_generation=int(record["projection_generation"]),
            request_digest=record["request_digest"],
            committed_at=record["committed_at"],
            idempotent_replay=idempotent_replay,
        )
        return result


@dataclass(frozen=True)
class ClaimAuthorityIntegrityReport:
    authority_id: str
    identity_count: int
    version_count: int
    current_count: int
    last_store_sequence: int
    valid: bool


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS claim_authority_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    prototype_state TEXT NOT NULL,
    authority_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    host_instance_id TEXT NOT NULL,
    encryption_domain TEXT NOT NULL,
    authority_namespace_id TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS claim_authority_counter (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_store_sequence INTEGER NOT NULL CHECK (next_store_sequence > 0)
);
CREATE TABLE IF NOT EXISTS claim_identities (
    claim_id TEXT PRIMARY KEY,
    semantic_digest TEXT NOT NULL,
    identity_sha256 TEXT NOT NULL UNIQUE,
    identity_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS claim_identities_semantic_digest_idx
ON claim_identities(semantic_digest);
CREATE TABLE IF NOT EXISTS claim_versions (
    store_sequence INTEGER PRIMARY KEY CHECK (store_sequence > 0),
    claim_version_id TEXT NOT NULL UNIQUE,
    claim_id TEXT NOT NULL,
    version_sequence INTEGER NOT NULL CHECK (version_sequence > 0),
    request_digest TEXT NOT NULL,
    version_sha256 TEXT NOT NULL UNIQUE,
    version_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    UNIQUE (claim_id, version_sequence),
    FOREIGN KEY (claim_id) REFERENCES claim_identities(claim_id)
);
CREATE TABLE IF NOT EXISTS current_claims (
    claim_id TEXT PRIMARY KEY,
    current_claim_version_id TEXT NOT NULL,
    projection_generation INTEGER NOT NULL CHECK (projection_generation > 0),
    source_position INTEGER NOT NULL CHECK (source_position > 0),
    projection_sha256 TEXT NOT NULL,
    projection_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (claim_id) REFERENCES claim_identities(claim_id),
    FOREIGN KEY (current_claim_version_id)
        REFERENCES claim_versions(claim_version_id)
);
CREATE TABLE IF NOT EXISTS claim_idempotency (
    idempotency_namespace TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    PRIMARY KEY (idempotency_namespace, idempotency_key)
);
CREATE TRIGGER IF NOT EXISTS claim_identities_no_update
BEFORE UPDATE ON claim_identities
BEGIN
    SELECT RAISE(ABORT, 'claim identities are append-only');
END;
CREATE TRIGGER IF NOT EXISTS claim_identities_no_delete
BEFORE DELETE ON claim_identities
BEGIN
    SELECT RAISE(ABORT, 'claim identities are append-only');
END;
CREATE TRIGGER IF NOT EXISTS claim_versions_no_update
BEFORE UPDATE ON claim_versions
BEGIN
    SELECT RAISE(ABORT, 'claim versions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS claim_versions_no_delete
BEFORE DELETE ON claim_versions
BEGIN
    SELECT RAISE(ABORT, 'claim versions are append-only');
END;
CREATE TRIGGER IF NOT EXISTS claim_authority_metadata_no_update
BEFORE UPDATE ON claim_authority_metadata
BEGIN
    SELECT RAISE(ABORT, 'claim authority metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS claim_authority_metadata_no_delete
BEFORE DELETE ON claim_authority_metadata
BEGIN
    SELECT RAISE(ABORT, 'claim authority metadata is immutable');
END;
"""


class ClaimAuthorityPrototypeStore:
    """One scoped, reversible embedded Claim Authority prototype."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        database_path: Path,
        scope: ProductHostScope,
        authority_namespace_id: str,
        authority_id: str,
        scope_digest: str,
    ) -> None:
        self._connection = connection
        self.database_path = database_path
        self.scope = scope
        self.authority_namespace_id = authority_namespace_id
        self.authority_id = authority_id
        self.scope_digest = scope_digest
        self.prototype_state = CLAIM_AUTHORITY_PROTOTYPE_STATE

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        if self._connection.in_transaction:
            raise ClaimAuthorityPrototypeTransactionError(
                "nested Claim Authority prototype transactions are unsupported"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ClaimAuthorityPrototypeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _assert_request_scope(self, request: ClaimAuthorityAppendRequest) -> None:
        request.validate()
        if not _same_scope(request.identity.envelope.scope, self.scope):
            raise ClaimAuthorityPrototypeIsolationError(
                "claim request scope does not match prototype store"
            )
        if (
            request.identity.envelope.authority_namespace_id
            != self.authority_namespace_id
        ):
            raise ClaimAuthorityPrototypeIsolationError(
                "claim request authority namespace does not match store"
            )

    def append(
        self,
        request: ClaimAuthorityAppendRequest,
        *,
        committed_at: object,
    ) -> ClaimAuthorityAppendReceipt:
        self._assert_request_scope(request)
        normalized_time = normalize_timestamp(committed_at, "committed_at")
        namespace, key, digest = request.idempotency_tuple()

        with self._write_transaction() as connection:
            prior = connection.execute(
                "SELECT request_digest, receipt_json FROM claim_idempotency "
                "WHERE idempotency_namespace = ? AND idempotency_key = ?",
                (namespace, key),
            ).fetchone()
            if prior is not None:
                prior_digest = str(prior["request_digest"])
                if prior_digest != digest:
                    raise ClaimAuthorityPrototypeConflictError(
                        "idempotency key was reused with a different request digest"
                    )
                prior_record = _load_json_object(
                    prior["receipt_json"], field="append receipt"
                )
                return ClaimAuthorityAppendReceipt.from_record(
                    prior_record,
                    idempotent_replay=True,
                )

            identity = request.identity
            by_claim = connection.execute(
                "SELECT identity_json FROM claim_identities WHERE claim_id = ?",
                (identity.claim_id,),
            ).fetchone()
            by_digest = connection.execute(
                "SELECT claim_id, identity_json FROM claim_identities "
                "WHERE semantic_digest = ?",
                (identity.semantic_digest,),
            ).fetchall()
            identity_json = _canonical_json(identity.metadata_record())
            if by_claim is not None and str(by_claim["identity_json"]) != identity_json:
                raise ClaimAuthorityPrototypeConflictError(
                    "claim_id already exists with different canonical identity"
                )
            for row in by_digest:
                stored = _load_json_object(
                    row["identity_json"], field="claim identity"
                )
                _verify_record_digest(
                    stored,
                    digest_field="identity_sha256",
                    field="claim identity",
                )
                if str(row["claim_id"]) != identity.claim_id:
                    stored_semantic = dict(stored)
                    stored_semantic.pop("identity_sha256", None)
                    if stored_semantic != identity.material_record():
                        raise ClaimAuthorityPrototypeConflictError(
                            "semantic digest collision requires full equality"
                        )
            if by_claim is None:
                connection.execute(
                    "INSERT INTO claim_identities ("
                    "claim_id, semantic_digest, identity_sha256, identity_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        identity.claim_id,
                        identity.semantic_digest,
                        identity.identity_sha256,
                        identity_json,
                        identity.envelope.created_at,
                    ),
                )

            current = connection.execute(
                "SELECT current_claim_version_id, projection_generation "
                "FROM current_claims WHERE claim_id = ?",
                (identity.claim_id,),
            ).fetchone()
            actual_current = (
                None
                if current is None
                else str(current["current_claim_version_id"])
            )
            if actual_current != request.expected_current_claim_version_id:
                raise ClaimAuthorityPrototypeConflictError(
                    "expected current claim version does not match"
                )

            counter = connection.execute(
                "SELECT next_store_sequence FROM claim_authority_counter "
                "WHERE singleton = 1"
            ).fetchone()
            if counter is None:
                raise ClaimAuthorityPrototypeIntegrityError(
                    "claim authority counter is missing"
                )
            store_sequence = int(counter["next_store_sequence"])
            version_row = connection.execute(
                "SELECT MAX(version_sequence) AS maximum FROM claim_versions "
                "WHERE claim_id = ?",
                (identity.claim_id,),
            ).fetchone()
            maximum = None if version_row is None else version_row["maximum"]
            version_sequence = 1 if maximum is None else int(maximum) + 1
            projection_generation = (
                1
                if current is None
                else int(current["projection_generation"]) + 1
            )

            version = ClaimVersion.create(
                envelope=request.version_envelope,
                claim_version_id=request.version_envelope.record_id,
                claim_id=identity.claim_id,
                version_sequence=version_sequence,
                store_sequence=store_sequence,
                event_stream_position=None,
                value=request.value,
                qualifiers=request.qualifiers,
                authority_class=request.authority_class,
                confidence=request.confidence,
                adjudication_state=request.adjudication_state,
                evidence_relation_ids=request.evidence_relation_ids,
                conflict_set_id=request.conflict_set_id,
                correction_of=request.correction_of,
                request_digest=request.request_digest,
            )
            projection = CurrentClaimProjection.create(
                envelope=request.projection_envelope,
                projection_id=request.projection_envelope.record_id,
                claim_id=identity.claim_id,
                current_claim_version_id=version.claim_version_id,
                authority_generation=1,
                projection_generation=projection_generation,
                adjudication_state=version.adjudication_state,
                validity_state=request.validity_state,
                conflict_state=request.conflict_state,
                deletion_state=request.deletion_state,
                source_position=store_sequence,
            )
            projection.assert_projects(identity, version)

            connection.execute(
                "INSERT INTO claim_versions ("
                "store_sequence, claim_version_id, claim_id, version_sequence, "
                "request_digest, version_sha256, version_json, committed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    store_sequence,
                    version.claim_version_id,
                    version.claim_id,
                    version.version_sequence,
                    version.request_digest,
                    version.version_sha256,
                    _canonical_json(version.metadata_record()),
                    normalized_time,
                ),
            )
            connection.execute(
                "INSERT INTO current_claims ("
                "claim_id, current_claim_version_id, projection_generation, "
                "source_position, projection_sha256, projection_json, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(claim_id) DO UPDATE SET "
                "current_claim_version_id=excluded.current_claim_version_id, "
                "projection_generation=excluded.projection_generation, "
                "source_position=excluded.source_position, "
                "projection_sha256=excluded.projection_sha256, "
                "projection_json=excluded.projection_json, "
                "updated_at=excluded.updated_at",
                (
                    identity.claim_id,
                    version.claim_version_id,
                    projection.projection_generation,
                    projection.source_position,
                    projection.projection_sha256,
                    _canonical_json(projection.metadata_record()),
                    normalized_time,
                ),
            )
            connection.execute(
                "UPDATE claim_authority_counter SET next_store_sequence = ? "
                "WHERE singleton = 1",
                (store_sequence + 1,),
            )
            receipt = ClaimAuthorityAppendReceipt.create(
                authority_id=self.authority_id,
                claim_id=identity.claim_id,
                claim_version_id=version.claim_version_id,
                store_sequence=store_sequence,
                version_sequence=version_sequence,
                projection_id=projection.projection_id,
                projection_generation=projection_generation,
                request_digest=digest,
                committed_at=normalized_time,
                idempotent_replay=False,
            )
            connection.execute(
                "INSERT INTO claim_idempotency ("
                "idempotency_namespace, idempotency_key, request_digest, receipt_json"
                ") VALUES (?, ?, ?, ?)",
                (
                    namespace,
                    key,
                    digest,
                    _canonical_json(receipt.metadata_record()),
                ),
            )
            return receipt

    def load_current(self, claim_id: object) -> dict[str, object]:
        normalized = require_identifier(claim_id, "claim_id")
        row = self._connection.execute(
            "SELECT projection_json FROM current_claims WHERE claim_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        record = _load_json_object(
            row["projection_json"], field="current claim projection"
        )
        _verify_record_digest(
            record,
            digest_field="projection_sha256",
            field="current claim projection",
        )
        return record

    def history(
        self,
        claim_id: object,
        *,
        limit: int = 100,
    ) -> tuple[dict[str, object], ...]:
        normalized = require_identifier(claim_id, "claim_id")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ClaimAuthorityPrototypeError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise ClaimAuthorityPrototypeError("limit must be between 1 and 1000")
        rows = self._connection.execute(
            "SELECT version_json FROM claim_versions WHERE claim_id = ? "
            "ORDER BY version_sequence ASC LIMIT ?",
            (normalized, limit),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            record = _load_json_object(row["version_json"], field="claim version")
            _verify_record_digest(
                record,
                digest_field="version_sha256",
                field="claim version",
            )
            result.append(record)
        return tuple(result)

    def verify_integrity(self) -> ClaimAuthorityIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM claim_authority_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise ClaimAuthorityPrototypeIntegrityError(
                "claim authority metadata is missing"
            )
        if str(metadata["authority_id"]) != self.authority_id:
            raise ClaimAuthorityPrototypeIntegrityError("authority identity changed")
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise ClaimAuthorityPrototypeIntegrityError("authority scope changed")

        identity_rows = self._connection.execute(
            "SELECT * FROM claim_identities ORDER BY claim_id"
        ).fetchall()
        for row in identity_rows:
            record = _load_json_object(row["identity_json"], field="claim identity")
            digest = _verify_record_digest(
                record,
                digest_field="identity_sha256",
                field="claim identity",
            )
            if digest != str(row["identity_sha256"]):
                raise ClaimAuthorityPrototypeIntegrityError(
                    "identity digest column mismatch"
                )

        version_rows = self._connection.execute(
            "SELECT * FROM claim_versions ORDER BY store_sequence"
        ).fetchall()
        expected_store_sequence = 1
        per_claim: dict[str, int] = {}
        for row in version_rows:
            if int(row["store_sequence"]) != expected_store_sequence:
                raise ClaimAuthorityPrototypeIntegrityError(
                    "store sequence is not contiguous"
                )
            expected_store_sequence += 1
            claim_id = str(row["claim_id"])
            per_claim[claim_id] = per_claim.get(claim_id, 0) + 1
            if int(row["version_sequence"]) != per_claim[claim_id]:
                raise ClaimAuthorityPrototypeIntegrityError(
                    "claim version sequence is not contiguous"
                )
            record = _load_json_object(row["version_json"], field="claim version")
            digest = _verify_record_digest(
                record,
                digest_field="version_sha256",
                field="claim version",
            )
            if digest != str(row["version_sha256"]):
                raise ClaimAuthorityPrototypeIntegrityError(
                    "version digest column mismatch"
                )

        current_rows = self._connection.execute(
            "SELECT * FROM current_claims ORDER BY claim_id"
        ).fetchall()
        for row in current_rows:
            record = _load_json_object(
                row["projection_json"], field="current claim projection"
            )
            digest = _verify_record_digest(
                record,
                digest_field="projection_sha256",
                field="current claim projection",
            )
            if digest != str(row["projection_sha256"]):
                raise ClaimAuthorityPrototypeIntegrityError(
                    "projection digest column mismatch"
                )
            version = self._connection.execute(
                "SELECT store_sequence FROM claim_versions "
                "WHERE claim_version_id = ?",
                (str(row["current_claim_version_id"]),),
            ).fetchone()
            if version is None:
                raise ClaimAuthorityPrototypeIntegrityError(
                    "current projection points to a missing version"
                )
            if int(row["source_position"]) < int(version["store_sequence"]):
                raise ClaimAuthorityPrototypeIntegrityError(
                    "current projection precedes its version"
                )

        return ClaimAuthorityIntegrityReport(
            authority_id=self.authority_id,
            identity_count=len(identity_rows),
            version_count=len(version_rows),
            current_count=len(current_rows),
            last_store_sequence=len(version_rows),
            valid=True,
        )


def _initialize_or_validate(
    connection: sqlite3.Connection,
    *,
    scope: ProductHostScope,
    authority_namespace_id: str,
    created_at: object | None,
) -> tuple[str, str]:
    namespace = require_identifier(
        authority_namespace_id, "authority_namespace_id"
    )
    digest = _scope_digest(scope, namespace)
    authority_id = f"claim-authority-prototype-{digest[:32]}"
    row = connection.execute(
        "SELECT * FROM claim_authority_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        if created_at is None:
            raise ClaimAuthorityPrototypeError(
                "created_at is required for a new prototype store"
            )
        normalized = normalize_timestamp(created_at, "created_at")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO claim_authority_metadata ("
                "singleton, schema_version, prototype_state, authority_id, "
                "product_id, host_instance_id, encryption_domain, "
                "authority_namespace_id, scope_digest, created_at"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    CLAIM_AUTHORITY_PROTOTYPE_SCHEMA_VERSION,
                    CLAIM_AUTHORITY_PROTOTYPE_STATE,
                    authority_id,
                    scope.product_id,
                    scope.host_instance_id,
                    scope.encryption_domain,
                    namespace,
                    digest,
                    normalized,
                ),
            )
            connection.execute(
                "INSERT INTO claim_authority_counter "
                "(singleton, next_store_sequence) VALUES (1, 1)"
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return authority_id, digest

    actual_scope = {
        "product_id": str(row["product_id"]),
        "host_instance_id": str(row["host_instance_id"]),
        "encryption_domain": str(row["encryption_domain"]),
    }
    if actual_scope != _scope_record(scope):
        raise ClaimAuthorityPrototypeIsolationError(
            "prototype store product-host-encryption scope differs"
        )
    if str(row["authority_namespace_id"]) != namespace:
        raise ClaimAuthorityPrototypeIsolationError(
            "prototype store authority namespace differs"
        )
    if str(row["scope_digest"]) != digest:
        raise ClaimAuthorityPrototypeIntegrityError(
            "prototype store scope digest differs"
        )
    if str(row["prototype_state"]) != CLAIM_AUTHORITY_PROTOTYPE_STATE:
        raise ClaimAuthorityPrototypeIntegrityError(
            "prototype state differs from reversible_nonproduction"
        )
    return authority_id, digest


def open_claim_authority_prototype(
    database_path: str | Path,
    *,
    scope: ProductHostScope,
    authority_namespace_id: str,
    created_at: object | None = None,
    repository_root: str | Path | None = None,
) -> ClaimAuthorityPrototypeStore:
    path = validate_claim_authority_prototype_path(
        database_path,
        repository_root=repository_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        _configure_connection(connection)
        connection.executescript(_SCHEMA_SQL)
        authority_id, digest = _initialize_or_validate(
            connection,
            scope=scope,
            authority_namespace_id=authority_namespace_id,
            created_at=created_at,
        )
    except Exception:
        connection.close()
        raise
    return ClaimAuthorityPrototypeStore(
        connection=connection,
        database_path=path,
        scope=scope,
        authority_namespace_id=require_identifier(
            authority_namespace_id, "authority_namespace_id"
        ),
        authority_id=authority_id,
        scope_digest=digest,
    )
