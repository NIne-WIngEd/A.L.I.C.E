"""Memory M2.2 reversible evidence-to-candidate shadow adjudication prototype.

The prototype persists full canonical candidate content, evidence relations,
rejections, quarantines, conflicts, and shadow decisions outside public Git.
It never writes canonical Claim Authority state or enables production influence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator

from .adjudication_contracts import (
    ADJUDICATION_OUTCOMES,
    AdjudicationRecord,
    ClaimCandidate,
    ClaimConflictRecord,
    ClaimEvidenceRelation,
)
from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_confidence,
    require_identifier,
    require_sha256,
)
from .contracts import ProductHostScope
from .memory_contracts import MemoryUnitEnvelope

SHADOW_ADJUDICATION_PROTOTYPE_SCHEMA_VERSION = "1.0.0"
SHADOW_ADJUDICATION_PROTOTYPE_STATE = "reversible_nonproduction"


class ShadowAdjudicationPrototypeError(RuntimeError):
    """Base error for the reversible shadow-adjudication prototype."""


class UnsafeShadowAdjudicationPathError(ShadowAdjudicationPrototypeError):
    """Raised when a prototype database resolves inside public Git."""


class ShadowAdjudicationIsolationError(ShadowAdjudicationPrototypeError):
    """Raised when product, host, encryption, or authority scope differs."""


class ShadowAdjudicationConflictError(ShadowAdjudicationPrototypeError):
    """Raised for idempotency, expected-current, or immutable-data conflicts."""


class ShadowAdjudicationIntegrityError(ShadowAdjudicationPrototypeError):
    """Raised when persisted shadow records fail integrity checks."""


class ShadowAdjudicationTransactionError(ShadowAdjudicationPrototypeError):
    """Raised for invalid or nested prototype write transactions."""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_shadow_adjudication_path(
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
        raise UnsafeShadowAdjudicationPathError(
            "Refusing to create or open a shadow-adjudication prototype "
            f"inside the public repository: {candidate}"
        )
    return candidate


def _same_scope(first: ProductHostScope, second: ProductHostScope) -> bool:
    return first.metadata_record() == second.metadata_record()


def _canonical_json(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _load_json_object(value: str, *, field: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ShadowAdjudicationIntegrityError(
            f"stored {field} is not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ShadowAdjudicationIntegrityError(
            f"stored {field} is not a JSON object"
        )
    return parsed


def _verify_record_digest(
    record: dict[str, object],
    *,
    digest_field: str,
    field: str,
) -> str:
    material = dict(record)
    try:
        digest = require_sha256(material.pop(digest_field), digest_field)
    except (KeyError, CognitiveKernelContractError) as exc:
        raise ShadowAdjudicationIntegrityError(
            f"stored {field} digest field is invalid"
        ) from exc
    if canonical_sha256(material) != digest:
        raise ShadowAdjudicationIntegrityError(
            f"stored {field} digest mismatch"
        )
    return digest


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = None if mode_row is None else str(mode_row[0]).lower()
    if mode != "wal":
        raise ShadowAdjudicationPrototypeError(
            "shadow-adjudication prototype requires WAL journal mode"
        )
    connection.execute("PRAGMA synchronous=FULL")


@dataclass(frozen=True)
class ShadowAdjudicationProfile:
    """Deterministic evaluation profile for one reversible shadow run."""

    profile_id: str
    profile_version: str
    minimum_support_relations: int
    minimum_confidence: float
    derivation_counts_as_support: bool
    profile_sha256: str

    @classmethod
    def create(
        cls,
        *,
        profile_id: object = "memory.m2.shadow_adjudication",
        profile_version: object = "1.0.0",
        minimum_support_relations: int = 1,
        minimum_confidence: float = 0.65,
        derivation_counts_as_support: bool = True,
    ) -> "ShadowAdjudicationProfile":
        if (
            isinstance(minimum_support_relations, bool)
            or not isinstance(minimum_support_relations, int)
            or minimum_support_relations < 1
        ):
            raise CognitiveKernelContractError(
                "minimum_support_relations must be a positive integer"
            )
        draft = cls(
            profile_id=require_identifier(profile_id, "profile_id"),
            profile_version=require_identifier(
                profile_version, "profile_version"
            ),
            minimum_support_relations=minimum_support_relations,
            minimum_confidence=require_confidence(
                minimum_confidence, "minimum_confidence"
            ),
            derivation_counts_as_support=bool(
                derivation_counts_as_support
            ),
            profile_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "profile_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "minimum_support_relations": self.minimum_support_relations,
            "minimum_confidence": self.minimum_confidence,
            "derivation_counts_as_support": (
                self.derivation_counts_as_support
            ),
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["profile_sha256"] = self.profile_sha256
        return record

    def validate(self) -> None:
        require_sha256(self.profile_sha256, "profile_sha256")
        if canonical_sha256(self.material_record()) != self.profile_sha256:
            raise CognitiveKernelContractError(
                "shadow adjudication profile digest mismatch"
            )


@dataclass(frozen=True)
class ShadowAdjudicationSubmission:
    """One explicit candidate, evidence set, and shadow-decision envelope."""

    candidate: ClaimCandidate
    evidence_relations: tuple[ClaimEvidenceRelation, ...]
    adjudication_envelope: MemoryUnitEnvelope
    conflict_envelope: MemoryUnitEnvelope | None
    request_digest: str
    expected_current_adjudication_id: str | None

    @classmethod
    def create(
        cls,
        *,
        candidate: ClaimCandidate,
        evidence_relations: Iterable[ClaimEvidenceRelation],
        adjudication_envelope: MemoryUnitEnvelope,
        conflict_envelope: MemoryUnitEnvelope | None,
        request_digest: object,
        expected_current_adjudication_id: object | None = None,
    ) -> "ShadowAdjudicationSubmission":
        relations = tuple(
            sorted(evidence_relations, key=lambda item: item.relation_id)
        )
        result = cls(
            candidate=candidate,
            evidence_relations=relations,
            adjudication_envelope=adjudication_envelope,
            conflict_envelope=conflict_envelope,
            request_digest=require_sha256(
                request_digest, "request_digest"
            ),
            expected_current_adjudication_id=(
                require_identifier(
                    expected_current_adjudication_id,
                    "expected_current_adjudication_id",
                )
                if expected_current_adjudication_id is not None
                else None
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        self.candidate.validate()
        if self.request_digest != self.candidate.request_digest:
            raise CognitiveKernelContractError(
                "submission request digest differs from candidate"
            )
        relation_ids: list[str] = []
        for relation in self.evidence_relations:
            relation.validate()
            if relation.target_record_id != self.candidate.candidate_id:
                raise CognitiveKernelContractError(
                    "evidence relation targets another candidate"
                )
            if relation.target_record_type != "claim_candidate":
                raise CognitiveKernelContractError(
                    "shadow evidence relation must target claim_candidate"
                )
            if not _same_scope(
                relation.envelope.scope, self.candidate.envelope.scope
            ):
                raise CognitiveKernelContractError(
                    "submission evidence crosses product-host scope"
                )
            if (
                relation.envelope.authority_namespace_id
                != self.candidate.envelope.authority_namespace_id
            ):
                raise CognitiveKernelContractError(
                    "submission evidence crosses authority namespace"
                )
            relation_ids.append(relation.relation_id)
        if len(relation_ids) != len(set(relation_ids)):
            raise CognitiveKernelContractError(
                "submission evidence relation IDs must be unique"
            )
        if tuple(sorted(relation_ids)) != self.candidate.evidence_relation_ids:
            raise CognitiveKernelContractError(
                "candidate evidence_relation_ids differ from submission"
            )
        self.adjudication_envelope.validate()
        if self.adjudication_envelope.record_type != "adjudication_record":
            raise CognitiveKernelContractError(
                "adjudication envelope has wrong record type"
            )
        if self.adjudication_envelope.authority_role != "evaluation_artifact":
            raise CognitiveKernelContractError(
                "shadow adjudication must use evaluation_artifact"
            )
        if not _same_scope(
            self.adjudication_envelope.scope,
            self.candidate.envelope.scope,
        ):
            raise CognitiveKernelContractError(
                "adjudication envelope crosses product-host scope"
            )
        if self.conflict_envelope is not None:
            self.conflict_envelope.validate()
            if self.conflict_envelope.record_type != "claim_conflict_record":
                raise CognitiveKernelContractError(
                    "conflict envelope has wrong record type"
                )
            if self.conflict_envelope.authority_role not in {
                "candidate",
                "evaluation_artifact",
            }:
                raise CognitiveKernelContractError(
                    "shadow conflict envelope has invalid authority role"
                )
            if not _same_scope(
                self.conflict_envelope.scope,
                self.candidate.envelope.scope,
            ):
                raise CognitiveKernelContractError(
                    "conflict envelope crosses product-host scope"
                )

    def idempotency_tuple(self) -> tuple[str, str, str]:
        self.validate()
        return (
            self.candidate.envelope.idempotency_namespace,
            self.candidate.envelope.idempotency_key,
            self.request_digest,
        )


@dataclass(frozen=True)
class ShadowAdjudicationReceipt:
    """Receipt proving one decision remained shadow-only."""

    prototype_id: str
    candidate_id: str
    adjudication_id: str
    decision_sequence: int
    decision_generation: int
    outcome: str
    candidate_state: str
    conflict_id: str | None
    canonical_claim_written: bool
    request_digest: str
    decided_at: str
    idempotent_replay: bool
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        prototype_id: object,
        candidate_id: object,
        adjudication_id: object,
        decision_sequence: int,
        decision_generation: int,
        outcome: object,
        candidate_state: object,
        conflict_id: object | None,
        request_digest: object,
        decided_at: object,
        idempotent_replay: bool,
    ) -> "ShadowAdjudicationReceipt":
        if (
            isinstance(decision_sequence, bool)
            or decision_sequence < 1
            or isinstance(decision_generation, bool)
            or decision_generation < 1
        ):
            raise CognitiveKernelContractError(
                "decision sequence and generation must be positive"
            )
        normalized_outcome = require_identifier(outcome, "outcome")
        if normalized_outcome not in ADJUDICATION_OUTCOMES:
            raise CognitiveKernelContractError(
                "receipt outcome is not ratified"
            )
        draft = cls(
            prototype_id=require_identifier(prototype_id, "prototype_id"),
            candidate_id=require_identifier(candidate_id, "candidate_id"),
            adjudication_id=require_identifier(
                adjudication_id, "adjudication_id"
            ),
            decision_sequence=decision_sequence,
            decision_generation=decision_generation,
            outcome=normalized_outcome,
            candidate_state=require_identifier(
                candidate_state, "candidate_state"
            ),
            conflict_id=(
                require_identifier(conflict_id, "conflict_id")
                if conflict_id is not None
                else None
            ),
            canonical_claim_written=False,
            request_digest=require_sha256(
                request_digest, "request_digest"
            ),
            decided_at=normalize_timestamp(decided_at, "decided_at"),
            idempotent_replay=bool(idempotent_replay),
            receipt_sha256="0" * 64,
        )
        result = cls(
            **{
                **draft.__dict__,
                "receipt_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        result.validate()
        return result

    @classmethod
    def from_record(
        cls,
        record: dict[str, object],
        *,
        idempotent_replay: bool,
    ) -> "ShadowAdjudicationReceipt":
        result = cls(
            prototype_id=str(record["prototype_id"]),
            candidate_id=str(record["candidate_id"]),
            adjudication_id=str(record["adjudication_id"]),
            decision_sequence=int(record["decision_sequence"]),
            decision_generation=int(record["decision_generation"]),
            outcome=str(record["outcome"]),
            candidate_state=str(record["candidate_state"]),
            conflict_id=(
                None
                if record.get("conflict_id") is None
                else str(record["conflict_id"])
            ),
            canonical_claim_written=bool(
                record["canonical_claim_written"]
            ),
            request_digest=str(record["request_digest"]),
            decided_at=str(record["decided_at"]),
            idempotent_replay=bool(idempotent_replay),
            receipt_sha256=str(record["receipt_sha256"]),
        )
        result.validate()
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "prototype_id": self.prototype_id,
            "candidate_id": self.candidate_id,
            "adjudication_id": self.adjudication_id,
            "decision_sequence": self.decision_sequence,
            "decision_generation": self.decision_generation,
            "outcome": self.outcome,
            "candidate_state": self.candidate_state,
            "conflict_id": self.conflict_id,
            "canonical_claim_written": self.canonical_claim_written,
            "request_digest": self.request_digest,
            "decided_at": self.decided_at,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["receipt_sha256"] = self.receipt_sha256
        return record

    def validate(self) -> None:
        if self.canonical_claim_written:
            raise CognitiveKernelContractError(
                "shadow receipt cannot assert a canonical claim write"
            )
        require_sha256(self.receipt_sha256, "receipt_sha256")
        if canonical_sha256(self.material_record()) != self.receipt_sha256:
            raise CognitiveKernelContractError(
                "shadow adjudication receipt digest mismatch"
            )


@dataclass(frozen=True)
class ShadowAdjudicationIntegrityReport:
    prototype_id: str
    candidate_count: int
    evidence_relation_count: int
    conflict_count: int
    adjudication_count: int
    current_state_count: int
    integrity_state: str
    report_sha256: str

    @classmethod
    def create(
        cls,
        *,
        prototype_id: object,
        candidate_count: int,
        evidence_relation_count: int,
        conflict_count: int,
        adjudication_count: int,
        current_state_count: int,
    ) -> "ShadowAdjudicationIntegrityReport":
        draft = cls(
            prototype_id=require_identifier(prototype_id, "prototype_id"),
            candidate_count=candidate_count,
            evidence_relation_count=evidence_relation_count,
            conflict_count=conflict_count,
            adjudication_count=adjudication_count,
            current_state_count=current_state_count,
            integrity_state="verified",
            report_sha256="0" * 64,
        )
        counts = (
            candidate_count,
            evidence_relation_count,
            conflict_count,
            adjudication_count,
            current_state_count,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise CognitiveKernelContractError(
                "integrity report counts must be non-negative integers"
            )
        result = cls(
            **{
                **draft.__dict__,
                "report_sha256": canonical_sha256(
                    draft.material_record()
                ),
            }
        )
        return result

    def material_record(self) -> dict[str, object]:
        return {
            "prototype_id": self.prototype_id,
            "candidate_count": self.candidate_count,
            "evidence_relation_count": self.evidence_relation_count,
            "conflict_count": self.conflict_count,
            "adjudication_count": self.adjudication_count,
            "current_state_count": self.current_state_count,
            "integrity_state": self.integrity_state,
        }

    def metadata_record(self) -> dict[str, object]:
        record = self.material_record()
        record["report_sha256"] = self.report_sha256
        return record


_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    prototype_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    authority_namespace_id TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_counter (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    next_decision_sequence INTEGER NOT NULL CHECK (next_decision_sequence >= 1)
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence_relations (
    relation_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    evidence_record_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    relation_sha256 TEXT NOT NULL,
    relation_json TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE TABLE IF NOT EXISTS conflict_records (
    conflict_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    conflict_sha256 TEXT NOT NULL,
    conflict_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE TABLE IF NOT EXISTS adjudication_records (
    decision_sequence INTEGER PRIMARY KEY,
    adjudication_id TEXT UNIQUE NOT NULL,
    candidate_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    adjudication_sha256 TEXT NOT NULL,
    adjudication_json TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE TABLE IF NOT EXISTS candidate_current (
    candidate_id TEXT PRIMARY KEY,
    current_adjudication_id TEXT NOT NULL,
    decision_generation INTEGER NOT NULL,
    decision_sequence INTEGER NOT NULL,
    candidate_state TEXT NOT NULL,
    outcome TEXT NOT NULL,
    conflict_id TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(current_adjudication_id) REFERENCES adjudication_records(adjudication_id)
);
CREATE TABLE IF NOT EXISTS shadow_idempotency (
    idempotency_namespace TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    PRIMARY KEY(idempotency_namespace, idempotency_key)
);
CREATE TRIGGER IF NOT EXISTS candidates_no_update
BEFORE UPDATE ON candidates BEGIN
    SELECT RAISE(ABORT, 'candidates are append-only');
END;
CREATE TRIGGER IF NOT EXISTS candidates_no_delete
BEFORE DELETE ON candidates BEGIN
    SELECT RAISE(ABORT, 'candidates are append-only');
END;
CREATE TRIGGER IF NOT EXISTS evidence_relations_no_update
BEFORE UPDATE ON evidence_relations BEGIN
    SELECT RAISE(ABORT, 'evidence relations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS evidence_relations_no_delete
BEFORE DELETE ON evidence_relations BEGIN
    SELECT RAISE(ABORT, 'evidence relations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS conflict_records_no_update
BEFORE UPDATE ON conflict_records BEGIN
    SELECT RAISE(ABORT, 'conflict records are append-only');
END;
CREATE TRIGGER IF NOT EXISTS conflict_records_no_delete
BEFORE DELETE ON conflict_records BEGIN
    SELECT RAISE(ABORT, 'conflict records are append-only');
END;
CREATE TRIGGER IF NOT EXISTS adjudication_records_no_update
BEFORE UPDATE ON adjudication_records BEGIN
    SELECT RAISE(ABORT, 'adjudication records are append-only');
END;
CREATE TRIGGER IF NOT EXISTS adjudication_records_no_delete
BEFORE DELETE ON adjudication_records BEGIN
    SELECT RAISE(ABORT, 'adjudication records are append-only');
END;
CREATE TRIGGER IF NOT EXISTS shadow_metadata_no_update
BEFORE UPDATE ON shadow_metadata BEGIN
    SELECT RAISE(ABORT, 'shadow metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS shadow_metadata_no_delete
BEFORE DELETE ON shadow_metadata BEGIN
    SELECT RAISE(ABORT, 'shadow metadata is immutable');
END;
"""


class ShadowAdjudicationPrototypeStore:
    """One scoped, persistent, reversible shadow-adjudication prototype."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        database_path: Path,
        scope: ProductHostScope,
        authority_namespace_id: str,
        prototype_id: str,
        profile: ShadowAdjudicationProfile,
        scope_digest: str,
    ) -> None:
        self._connection = connection
        self.database_path = database_path
        self.scope = scope
        self.authority_namespace_id = authority_namespace_id
        self.prototype_id = prototype_id
        self.profile = profile
        self.scope_digest = scope_digest
        self.prototype_state = SHADOW_ADJUDICATION_PROTOTYPE_STATE

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ShadowAdjudicationPrototypeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        if self._connection.in_transaction:
            raise ShadowAdjudicationTransactionError(
                "nested shadow-adjudication transactions are unsupported"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _assert_submission_scope(
        self, submission: ShadowAdjudicationSubmission
    ) -> None:
        submission.validate()
        if not _same_scope(submission.candidate.envelope.scope, self.scope):
            raise ShadowAdjudicationIsolationError(
                "candidate scope does not match shadow store"
            )
        if (
            submission.candidate.envelope.authority_namespace_id
            != self.authority_namespace_id
        ):
            raise ShadowAdjudicationIsolationError(
                "candidate authority namespace does not match shadow store"
            )

    def _evaluate(
        self,
        submission: ShadowAdjudicationSubmission,
    ) -> tuple[str, str, tuple[str, ...], bool]:
        relation_types = [
            relation.relation_type
            for relation in submission.evidence_relations
        ]
        support_count = relation_types.count("support")
        if self.profile.derivation_counts_as_support:
            support_count += relation_types.count("derivation")
        contradiction_count = relation_types.count("contradiction")
        deletion_count = relation_types.count("deletion_cause")

        if deletion_count:
            return (
                "reject",
                "rejected",
                ("deletion_cause_present",),
                False,
            )
        if support_count < self.profile.minimum_support_relations:
            return (
                "reject",
                "rejected",
                ("insufficient_support",),
                False,
            )
        if contradiction_count:
            return (
                "quarantine",
                "quarantined",
                ("support_contradiction_conflict",),
                True,
            )
        confidence = submission.candidate.confidence
        if confidence is None or confidence < self.profile.minimum_confidence:
            return (
                "quarantine",
                "quarantined",
                ("confidence_below_shadow_threshold",),
                False,
            )
        proposed = submission.candidate.proposed_action
        if proposed not in {
            "add",
            "revise",
            "supersede",
            "dispute",
            "merge",
            "split",
        }:
            return (
                "quarantine",
                "quarantined",
                ("action_requires_nonautomatic_review",),
                False,
            )
        return (
            proposed,
            "eligible",
            ("shadow_eligibility_satisfied",),
            False,
        )

    def submit_and_adjudicate(
        self,
        submission: ShadowAdjudicationSubmission,
        *,
        decided_at: object,
    ) -> ShadowAdjudicationReceipt:
        self._assert_submission_scope(submission)
        normalized_time = normalize_timestamp(decided_at, "decided_at")
        namespace, key, digest = submission.idempotency_tuple()

        with self._write_transaction() as connection:
            prior = connection.execute(
                "SELECT request_digest, receipt_json FROM shadow_idempotency "
                "WHERE idempotency_namespace = ? AND idempotency_key = ?",
                (namespace, key),
            ).fetchone()
            if prior is not None:
                if str(prior["request_digest"]) != digest:
                    raise ShadowAdjudicationConflictError(
                        "idempotency key was reused with another request digest"
                    )
                prior_record = _load_json_object(
                    str(prior["receipt_json"]), field="shadow receipt"
                )
                return ShadowAdjudicationReceipt.from_record(
                    prior_record,
                    idempotent_replay=True,
                )

            candidate = submission.candidate
            candidate_json = _canonical_json(candidate.metadata_record())
            stored_candidate = connection.execute(
                "SELECT candidate_json FROM candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
            if stored_candidate is None:
                connection.execute(
                    "INSERT INTO candidates (candidate_id, claim_id, "
                    "candidate_sha256, candidate_json, submitted_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        candidate.candidate_id,
                        candidate.identity.claim_id,
                        candidate.candidate_sha256,
                        candidate_json,
                        candidate.envelope.created_at,
                    ),
                )
            elif str(stored_candidate["candidate_json"]) != candidate_json:
                raise ShadowAdjudicationConflictError(
                    "candidate_id already exists with different content"
                )

            for relation in submission.evidence_relations:
                relation_json = _canonical_json(relation.metadata_record())
                stored_relation = connection.execute(
                    "SELECT relation_json FROM evidence_relations "
                    "WHERE relation_id = ?",
                    (relation.relation_id,),
                ).fetchone()
                if stored_relation is None:
                    connection.execute(
                        "INSERT INTO evidence_relations (relation_id, "
                        "candidate_id, evidence_record_id, relation_type, "
                        "relation_sha256, relation_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            relation.relation_id,
                            candidate.candidate_id,
                            relation.evidence_record_id,
                            relation.relation_type,
                            relation.relation_sha256,
                            relation_json,
                        ),
                    )
                elif str(stored_relation["relation_json"]) != relation_json:
                    raise ShadowAdjudicationConflictError(
                        "relation_id already exists with different content"
                    )

            current = connection.execute(
                "SELECT current_adjudication_id, decision_generation "
                "FROM candidate_current WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
            actual_current = (
                None
                if current is None
                else str(current["current_adjudication_id"])
            )
            if actual_current != submission.expected_current_adjudication_id:
                raise ShadowAdjudicationConflictError(
                    "expected current adjudication does not match"
                )

            outcome, candidate_state, rationale_codes, needs_conflict = (
                self._evaluate(submission)
            )
            conflict: ClaimConflictRecord | None = None
            if needs_conflict:
                if submission.conflict_envelope is None:
                    raise ShadowAdjudicationConflictError(
                        "conflict outcome requires a conflict envelope"
                    )
                conflicting_relations = tuple(
                    relation
                    for relation in submission.evidence_relations
                    if relation.relation_type in {
                        "support",
                        "derivation",
                        "contradiction",
                    }
                )
                member_ids = tuple(
                    sorted(
                        {
                            relation.evidence_record_id
                            for relation in conflicting_relations
                        }
                    )
                )
                if len(member_ids) < 2:
                    raise ShadowAdjudicationConflictError(
                        "conflict requires at least two distinct evidence records"
                    )
                conflict = ClaimConflictRecord.create(
                    envelope=submission.conflict_envelope,
                    conflict_id=submission.conflict_envelope.record_id,
                    claim_id=candidate.identity.claim_id,
                    member_record_ids=member_ids,
                    evidence_relation_ids=(
                        relation.relation_id
                        for relation in conflicting_relations
                    ),
                    conflict_type="support_contradiction",
                    resolution_state="quarantined",
                    detected_by=self.prototype_id,
                    detection_rule_id="support-contradiction-v1",
                    resolution_adjudication_id=None,
                    rollback_reference="delete-shadow-prototype-database",
                )
                connection.execute(
                    "INSERT INTO conflict_records (conflict_id, candidate_id, "
                    "conflict_sha256, conflict_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        conflict.conflict_id,
                        candidate.candidate_id,
                        conflict.conflict_sha256,
                        _canonical_json(conflict.metadata_record()),
                        normalized_time,
                    ),
                )
            elif submission.conflict_envelope is not None:
                # A supplied spare envelope is harmless but never persisted.
                submission.conflict_envelope.validate()

            counter = connection.execute(
                "SELECT next_decision_sequence FROM shadow_counter "
                "WHERE singleton = 1"
            ).fetchone()
            if counter is None:
                raise ShadowAdjudicationIntegrityError(
                    "shadow decision counter is missing"
                )
            decision_sequence = int(counter["next_decision_sequence"])
            decision_generation = (
                1
                if current is None
                else int(current["decision_generation"]) + 1
            )
            adjudication = AdjudicationRecord.create(
                envelope=submission.adjudication_envelope,
                adjudication_id=(
                    submission.adjudication_envelope.record_id
                ),
                candidate_id=candidate.candidate_id,
                claim_id=candidate.identity.claim_id,
                authority_class="algorithmic",
                authority_actor_id=self.prototype_id,
                policy_profile=self.profile.profile_id,
                rule_id="deterministic-shadow-adjudication",
                rule_version=self.profile.profile_version,
                evidence_relation_ids=(
                    relation.relation_id
                    for relation in submission.evidence_relations
                ),
                alternatives=tuple(
                    sorted({outcome, "quarantine", "reject"})
                ),
                confidence=candidate.confidence,
                outcome=outcome,
                execution_mode="shadow",
                canonical_effect=False,
                conflict_record_id=(
                    None if conflict is None else conflict.conflict_id
                ),
                rationale_codes=rationale_codes,
                rollback_reference="delete-shadow-prototype-database",
            )
            adjudication.assert_adjudicates(candidate)
            connection.execute(
                "INSERT INTO adjudication_records (decision_sequence, "
                "adjudication_id, candidate_id, outcome, adjudication_sha256, "
                "adjudication_json, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_sequence,
                    adjudication.adjudication_id,
                    candidate.candidate_id,
                    adjudication.outcome,
                    adjudication.adjudication_sha256,
                    _canonical_json(adjudication.metadata_record()),
                    normalized_time,
                ),
            )
            connection.execute(
                "INSERT INTO candidate_current (candidate_id, "
                "current_adjudication_id, decision_generation, "
                "decision_sequence, candidate_state, outcome, conflict_id, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(candidate_id) DO UPDATE SET "
                "current_adjudication_id=excluded.current_adjudication_id, "
                "decision_generation=excluded.decision_generation, "
                "decision_sequence=excluded.decision_sequence, "
                "candidate_state=excluded.candidate_state, "
                "outcome=excluded.outcome, "
                "conflict_id=excluded.conflict_id, "
                "updated_at=excluded.updated_at",
                (
                    candidate.candidate_id,
                    adjudication.adjudication_id,
                    decision_generation,
                    decision_sequence,
                    candidate_state,
                    outcome,
                    None if conflict is None else conflict.conflict_id,
                    normalized_time,
                ),
            )
            connection.execute(
                "UPDATE shadow_counter SET next_decision_sequence = ? "
                "WHERE singleton = 1",
                (decision_sequence + 1,),
            )
            receipt = ShadowAdjudicationReceipt.create(
                prototype_id=self.prototype_id,
                candidate_id=candidate.candidate_id,
                adjudication_id=adjudication.adjudication_id,
                decision_sequence=decision_sequence,
                decision_generation=decision_generation,
                outcome=outcome,
                candidate_state=candidate_state,
                conflict_id=(
                    None if conflict is None else conflict.conflict_id
                ),
                request_digest=digest,
                decided_at=normalized_time,
                idempotent_replay=False,
            )
            connection.execute(
                "INSERT INTO shadow_idempotency (idempotency_namespace, "
                "idempotency_key, request_digest, receipt_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    namespace,
                    key,
                    digest,
                    _canonical_json(receipt.metadata_record()),
                ),
            )
            return receipt

    def load_candidate(self, candidate_id: object) -> dict[str, object]:
        normalized = require_identifier(candidate_id, "candidate_id")
        row = self._connection.execute(
            "SELECT candidate_json FROM candidates WHERE candidate_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        record = _load_json_object(
            str(row["candidate_json"]), field="candidate"
        )
        _verify_record_digest(
            record,
            digest_field="candidate_sha256",
            field="candidate",
        )
        return record

    def load_current_state(
        self, candidate_id: object
    ) -> dict[str, object]:
        normalized = require_identifier(candidate_id, "candidate_id")
        row = self._connection.execute(
            "SELECT * FROM candidate_current WHERE candidate_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return {
            "candidate_id": normalized,
            "current_adjudication_id": str(
                row["current_adjudication_id"]
            ),
            "decision_generation": int(row["decision_generation"]),
            "decision_sequence": int(row["decision_sequence"]),
            "candidate_state": str(row["candidate_state"]),
            "outcome": str(row["outcome"]),
            "conflict_id": (
                None
                if row["conflict_id"] is None
                else str(row["conflict_id"])
            ),
            "canonical_claim_written": False,
            "updated_at": str(row["updated_at"]),
        }

    def decision_history(
        self,
        candidate_id: object,
        *,
        limit: int = 100,
    ) -> tuple[dict[str, object], ...]:
        normalized = require_identifier(candidate_id, "candidate_id")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > 1000
        ):
            raise ShadowAdjudicationPrototypeError(
                "limit must be an integer between 1 and 1000"
            )
        rows = self._connection.execute(
            "SELECT adjudication_json FROM adjudication_records "
            "WHERE candidate_id = ? ORDER BY decision_sequence ASC LIMIT ?",
            (normalized, limit),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            record = _load_json_object(
                str(row["adjudication_json"]), field="adjudication"
            )
            _verify_record_digest(
                record,
                digest_field="adjudication_sha256",
                field="adjudication",
            )
            result.append(record)
        return tuple(result)

    def verify_integrity(self) -> ShadowAdjudicationIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM shadow_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise ShadowAdjudicationIntegrityError(
                "shadow metadata is missing"
            )
        if str(metadata["prototype_id"]) != self.prototype_id:
            raise ShadowAdjudicationIntegrityError(
                "shadow prototype identity changed"
            )
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise ShadowAdjudicationIntegrityError(
                "shadow prototype scope changed"
            )

        candidates = self._connection.execute(
            "SELECT * FROM candidates ORDER BY candidate_id"
        ).fetchall()
        for row in candidates:
            record = _load_json_object(
                str(row["candidate_json"]), field="candidate"
            )
            digest = _verify_record_digest(
                record,
                digest_field="candidate_sha256",
                field="candidate",
            )
            if digest != str(row["candidate_sha256"]):
                raise ShadowAdjudicationIntegrityError(
                    "candidate digest column mismatch"
                )

        relations = self._connection.execute(
            "SELECT * FROM evidence_relations ORDER BY relation_id"
        ).fetchall()
        for row in relations:
            record = _load_json_object(
                str(row["relation_json"]), field="evidence relation"
            )
            digest = _verify_record_digest(
                record,
                digest_field="relation_sha256",
                field="evidence relation",
            )
            if digest != str(row["relation_sha256"]):
                raise ShadowAdjudicationIntegrityError(
                    "evidence relation digest column mismatch"
                )

        conflicts = self._connection.execute(
            "SELECT * FROM conflict_records ORDER BY conflict_id"
        ).fetchall()
        for row in conflicts:
            record = _load_json_object(
                str(row["conflict_json"]), field="conflict"
            )
            digest = _verify_record_digest(
                record,
                digest_field="conflict_sha256",
                field="conflict",
            )
            if digest != str(row["conflict_sha256"]):
                raise ShadowAdjudicationIntegrityError(
                    "conflict digest column mismatch"
                )

        adjudications = self._connection.execute(
            "SELECT * FROM adjudication_records ORDER BY decision_sequence"
        ).fetchall()
        expected = 1
        for row in adjudications:
            if int(row["decision_sequence"]) != expected:
                raise ShadowAdjudicationIntegrityError(
                    "decision sequence is not contiguous"
                )
            expected += 1
            record = _load_json_object(
                str(row["adjudication_json"]), field="adjudication"
            )
            digest = _verify_record_digest(
                record,
                digest_field="adjudication_sha256",
                field="adjudication",
            )
            if digest != str(row["adjudication_sha256"]):
                raise ShadowAdjudicationIntegrityError(
                    "adjudication digest column mismatch"
                )
            if bool(record.get("canonical_effect")):
                raise ShadowAdjudicationIntegrityError(
                    "shadow adjudication asserted canonical effect"
                )

        current = self._connection.execute(
            "SELECT * FROM candidate_current ORDER BY candidate_id"
        ).fetchall()
        for row in current:
            decision = self._connection.execute(
                "SELECT decision_sequence FROM adjudication_records "
                "WHERE adjudication_id = ?",
                (str(row["current_adjudication_id"]),),
            ).fetchone()
            if decision is None or int(decision["decision_sequence"]) != int(
                row["decision_sequence"]
            ):
                raise ShadowAdjudicationIntegrityError(
                    "current state does not reference its decision sequence"
                )

        return ShadowAdjudicationIntegrityReport.create(
            prototype_id=self.prototype_id,
            candidate_count=len(candidates),
            evidence_relation_count=len(relations),
            conflict_count=len(conflicts),
            adjudication_count=len(adjudications),
            current_state_count=len(current),
        )


def _initialize_or_validate(
    connection: sqlite3.Connection,
    *,
    prototype_id: str,
    scope: ProductHostScope,
    authority_namespace_id: str,
    profile: ShadowAdjudicationProfile,
    created_at: str,
) -> str:
    scope.validate()
    profile.validate()
    scope_digest = canonical_sha256(
        {
            "scope": scope.metadata_record(),
            "authority_namespace_id": authority_namespace_id,
        }
    )
    connection.executescript(_SCHEMA)
    row = connection.execute(
        "SELECT * FROM shadow_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO shadow_metadata (singleton, prototype_id, "
            "schema_version, scope_digest, authority_namespace_id, "
            "profile_json, created_at) VALUES (1, ?, ?, ?, ?, ?, ?)",
            (
                prototype_id,
                SHADOW_ADJUDICATION_PROTOTYPE_SCHEMA_VERSION,
                scope_digest,
                authority_namespace_id,
                _canonical_json(profile.metadata_record()),
                created_at,
            ),
        )
        connection.execute(
            "INSERT INTO shadow_counter (singleton, next_decision_sequence) "
            "VALUES (1, 1)"
        )
        connection.commit()
        return scope_digest
    if str(row["prototype_id"]) != prototype_id:
        raise ShadowAdjudicationIsolationError(
            "prototype database belongs to another prototype_id"
        )
    if str(row["scope_digest"]) != scope_digest:
        raise ShadowAdjudicationIsolationError(
            "prototype database belongs to another product-host scope"
        )
    if str(row["authority_namespace_id"]) != authority_namespace_id:
        raise ShadowAdjudicationIsolationError(
            "prototype database belongs to another authority namespace"
        )
    stored_profile = _load_json_object(
        str(row["profile_json"]), field="shadow profile"
    )
    _verify_record_digest(
        stored_profile,
        digest_field="profile_sha256",
        field="shadow profile",
    )
    if stored_profile != profile.metadata_record():
        raise ShadowAdjudicationIsolationError(
            "prototype database uses another shadow profile"
        )
    return scope_digest


def open_shadow_adjudication_prototype(
    database_path: str | Path,
    *,
    scope: ProductHostScope,
    authority_namespace_id: object,
    prototype_id: object = "shadow-adjudication-prototype",
    profile: ShadowAdjudicationProfile | None = None,
    created_at: object,
    repository_root: str | Path | None = None,
) -> ShadowAdjudicationPrototypeStore:
    path = validate_shadow_adjudication_path(
        database_path,
        repository_root=repository_root,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_namespace = require_identifier(
        authority_namespace_id, "authority_namespace_id"
    )
    normalized_prototype = require_identifier(
        prototype_id, "prototype_id"
    )
    normalized_time = normalize_timestamp(created_at, "created_at")
    selected_profile = profile or ShadowAdjudicationProfile.create()
    connection = sqlite3.connect(str(path), isolation_level=None)
    try:
        _configure_connection(connection)
        scope_digest = _initialize_or_validate(
            connection,
            prototype_id=normalized_prototype,
            scope=scope,
            authority_namespace_id=normalized_namespace,
            profile=selected_profile,
            created_at=normalized_time,
        )
    except Exception:
        connection.close()
        raise
    return ShadowAdjudicationPrototypeStore(
        connection=connection,
        database_path=path,
        scope=scope,
        authority_namespace_id=normalized_namespace,
        prototype_id=normalized_prototype,
        profile=selected_profile,
        scope_digest=scope_digest,
    )
