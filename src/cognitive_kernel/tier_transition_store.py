"""Governed source-preserving execution of approved physical tier transitions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
)
from .contracts import ProductHostScope
from .lifecycle import LifecycleDecision, RetentionBlockerRecord
from .lifecycle_store import LifecycleJournalStore
from .payload_store import RawBufferStore
from .tier_transition import (
    EXECUTABLE_SOURCE_TIERS,
    EXECUTABLE_TARGET_TIERS,
    TIER_TRANSITION_DECISION_TYPES,
    TIER_TRANSITION_SCHEMA_VERSION,
    TierPayloadReference,
    TierTransitionInspectionRecord,
    TierTransitionIntegrityReport,
    TierTransitionIntent,
    TierTransitionReceipt,
    tier_transition_scope_digest,
)


class TierTransitionStoreError(RuntimeError):
    """Base error for governed tier-transition execution."""


class UnsafeTierTransitionPathError(TierTransitionStoreError):
    """Raised when tier storage resolves inside the public repository."""


class TierTransitionIsolationError(TierTransitionStoreError):
    """Raised when stores or records cross product-host encryption scope."""


class TierTransitionAuthorizationError(TierTransitionStoreError):
    """Raised when a lifecycle decision is not executable."""


class TierTransitionBlockedError(TierTransitionStoreError):
    """Raised when an unresolved retention blocker prevents execution."""


class DuplicateTierTransitionError(TierTransitionStoreError):
    """Raised when transition identities conflict."""


class TierTransitionIntegrityError(TierTransitionStoreError):
    """Raised when transition metadata or opaque bytes fail integrity checks."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tier_transition_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    store_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    host_instance_id TEXT NOT NULL,
    encryption_domain TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tier_transition_intents (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    transition_id TEXT NOT NULL UNIQUE,
    lifecycle_decision_id TEXT NOT NULL UNIQUE,
    lifecycle_decision_sha256 TEXT NOT NULL,
    subject_reference TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    target_tier TEXT NOT NULL,
    source_reference_id TEXT NOT NULL,
    target_reference_id TEXT NOT NULL UNIQUE,
    byte_length INTEGER NOT NULL CHECK (byte_length > 0),
    target_relative_path TEXT NOT NULL,
    prepared_at TEXT NOT NULL,
    intent_json TEXT NOT NULL,
    intent_sha256 TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS tier_transition_publications (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    publication_id TEXT NOT NULL UNIQUE,
    transition_id TEXT NOT NULL UNIQUE,
    target_reference_id TEXT NOT NULL UNIQUE,
    published_at TEXT NOT NULL,
    source_preserved INTEGER NOT NULL CHECK (source_preserved = 1),
    physical_object_created INTEGER NOT NULL CHECK (
        physical_object_created IN (0, 1)
    ),
    recovered_from_prepared_intent INTEGER NOT NULL CHECK (
        recovered_from_prepared_intent IN (0, 1)
    ),
    receipt_json TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL UNIQUE,
    FOREIGN KEY (transition_id)
        REFERENCES tier_transition_intents(transition_id)
);
CREATE INDEX IF NOT EXISTS tier_transition_intents_subject
ON tier_transition_intents(subject_reference, content_digest);
CREATE INDEX IF NOT EXISTS tier_transition_intents_target
ON tier_transition_intents(target_tier, content_digest);
CREATE TRIGGER IF NOT EXISTS tier_transition_metadata_no_update
BEFORE UPDATE ON tier_transition_metadata
BEGIN
    SELECT RAISE(ABORT, 'tier-transition metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS tier_transition_metadata_no_delete
BEFORE DELETE ON tier_transition_metadata
BEGIN
    SELECT RAISE(ABORT, 'tier-transition metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS tier_transition_intents_no_update
BEFORE UPDATE ON tier_transition_intents
BEGIN
    SELECT RAISE(ABORT, 'tier-transition intents are append-only');
END;
CREATE TRIGGER IF NOT EXISTS tier_transition_intents_no_delete
BEFORE DELETE ON tier_transition_intents
BEGIN
    SELECT RAISE(ABORT, 'tier-transition intents are append-only');
END;
CREATE TRIGGER IF NOT EXISTS tier_transition_publications_no_update
BEFORE UPDATE ON tier_transition_publications
BEGIN
    SELECT RAISE(ABORT, 'tier-transition publications are append-only');
END;
CREATE TRIGGER IF NOT EXISTS tier_transition_publications_no_delete
BEFORE DELETE ON tier_transition_publications
BEGIN
    SELECT RAISE(ABORT, 'tier-transition publications are append-only');
END;
"""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_tier_transition_root(
    store_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    candidate = Path(store_root).expanduser().resolve(strict=False)
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else default_repository_root().resolve(strict=True)
    )
    if _is_within(candidate, root):
        raise UnsafeTierTransitionPathError(
            "Refusing to create or open tier-transition storage inside "
            f"the public repository: {candidate}"
        )
    if candidate.exists() and not candidate.is_dir():
        raise UnsafeTierTransitionPathError(
            f"tier-transition root must be a directory: {candidate}"
        )
    return candidate


def _configure(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    if mode is None or str(mode[0]).lower() != "wal":
        raise TierTransitionStoreError(
            "tier-transition metadata requires SQLite WAL"
        )
    connection.execute("PRAGMA synchronous=FULL")
    level = connection.execute("PRAGMA synchronous").fetchone()
    if level is None or int(level[0]) != 2:
        raise TierTransitionStoreError(
            "tier-transition metadata requires SQLite synchronous=FULL"
        )
    connection.execute("PRAGMA busy_timeout=5000")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _same_scope(left: ProductHostScope, right: ProductHostScope) -> bool:
    left.validate()
    right.validate()
    return left.storage_scope() == right.storage_scope()


class TierTransitionStore:
    """One product-host-encryption scoped non-destructive tier store."""

    def __init__(
        self,
        *,
        root: Path,
        connection: sqlite3.Connection,
        scope: ProductHostScope,
        store_id: str,
        scope_digest: str,
    ) -> None:
        self.root = root
        self._connection = connection
        self.scope = scope
        self.store_id = store_id
        self.scope_digest = scope_digest
        self.tiers_root = root / "tiers"

    def __enter__(self) -> "TierTransitionStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _relative_object_path(self, tier: str, content_digest: str) -> Path:
        if tier not in EXECUTABLE_TARGET_TIERS:
            raise TierTransitionStoreError("target tier is not executable")
        return Path(
            "tiers",
            tier,
            self.scope_digest[:2],
            content_digest[:2],
            f"{self.scope_digest}-{content_digest}.payload",
        )

    def _object_path(self, tier: str, content_digest: str) -> Path:
        candidate = (
            self.root / self._relative_object_path(tier, content_digest)
        ).resolve(strict=False)
        if not _is_within(candidate, self.root.resolve(strict=True)):
            raise TierTransitionIntegrityError(
                "tier object path escaped the transition root"
            )
        return candidate

    def _verify_object(
        self,
        *,
        path: Path,
        content_digest: str,
        byte_length: int,
    ) -> None:
        if not path.is_file():
            raise TierTransitionIntegrityError(
                f"tier object is missing: {content_digest}"
            )
        payload = path.read_bytes()
        if len(payload) != byte_length:
            raise TierTransitionIntegrityError(
                f"tier object length changed: {content_digest}"
            )
        if _digest_bytes(payload) != content_digest:
            raise TierTransitionIntegrityError(
                f"tier object digest changed: {content_digest}"
            )

    def _publish_object(
        self,
        payload: bytes,
        *,
        target_tier: str,
        content_digest: str,
    ) -> bool:
        target = self._object_path(target_tier, content_digest)
        if target.exists():
            self._verify_object(
                path=target,
                content_digest=content_digest,
                byte_length=len(payload),
            )
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{content_digest}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary_payload = temporary.read_bytes()
            if len(temporary_payload) != len(payload):
                raise TierTransitionIntegrityError(
                    "temporary tier object length mismatch"
                )
            if _digest_bytes(temporary_payload) != content_digest:
                raise TierTransitionIntegrityError(
                    "temporary tier object digest mismatch"
                )
            os.replace(temporary, target)
            try:
                directory_descriptor = os.open(target.parent, os.O_RDONLY)
            except OSError:
                directory_descriptor = None
            if directory_descriptor is not None:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        self._verify_object(
            path=target,
            content_digest=content_digest,
            byte_length=len(payload),
        )
        return True

    def _load_intent_row(self, transition_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM tier_transition_intents WHERE transition_id = ?",
            (transition_id,),
        ).fetchone()

    def _load_intent(self, row: sqlite3.Row) -> TierTransitionIntent:
        try:
            value = json.loads(str(row["intent_json"]))
        except json.JSONDecodeError as exc:
            raise TierTransitionIntegrityError(
                "stored tier-transition intent JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise TierTransitionIntegrityError(
                "stored tier-transition intent must be an object"
            )
        try:
            intent = TierTransitionIntent.from_record(value)
        except CognitiveKernelContractError as exc:
            raise TierTransitionIntegrityError(
                "stored tier-transition intent failed validation"
            ) from exc
        if not _same_scope(intent.scope, self.scope):
            raise TierTransitionIsolationError(
                "stored tier-transition intent scope changed"
            )
        return intent

    def _load_receipt_row(self, transition_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT * FROM tier_transition_publications "
            "WHERE transition_id = ?",
            (transition_id,),
        ).fetchone()

    def _load_receipt(self, row: sqlite3.Row) -> TierTransitionReceipt:
        try:
            value = json.loads(str(row["receipt_json"]))
        except json.JSONDecodeError as exc:
            raise TierTransitionIntegrityError(
                "stored tier-transition receipt JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise TierTransitionIntegrityError(
                "stored tier-transition receipt must be an object"
            )
        try:
            receipt = TierTransitionReceipt.from_record(value)
        except CognitiveKernelContractError as exc:
            raise TierTransitionIntegrityError(
                "stored tier-transition receipt failed validation"
            ) from exc
        if not _same_scope(receipt.target_reference.scope, self.scope):
            raise TierTransitionIsolationError(
                "stored tier-transition receipt scope changed"
            )
        return receipt

    def _journal_values(
        self, journal: LifecycleJournalStore
    ) -> tuple[LifecycleDecision | RetentionBlockerRecord, ...]:
        values: list[LifecycleDecision | RetentionBlockerRecord] = []
        after = 0
        while True:
            records = journal.inspect(after_sequence=after, limit=1000)
            if not records:
                break
            for record in records:
                value = journal.load_record(record.record_id)
                if not isinstance(
                    value, (LifecycleDecision, RetentionBlockerRecord)
                ):
                    raise TierTransitionIntegrityError(
                        "lifecycle journal returned an unsupported record"
                    )
                values.append(value)
            after = records[-1].sequence
        return tuple(values)

    def _validate_decision(
        self,
        journal: LifecycleJournalStore,
        decision_id: object,
    ) -> LifecycleDecision:
        if not _same_scope(journal.scope, self.scope):
            raise TierTransitionIsolationError(
                "lifecycle journal scope does not match tier store"
            )
        journal.verify_integrity()
        normalized = require_identifier(decision_id, "decision_id")
        try:
            decision = journal.load_record(normalized)
        except KeyError as exc:
            raise TierTransitionAuthorizationError(
                "lifecycle decision does not exist"
            ) from exc
        if not isinstance(decision, LifecycleDecision):
            raise TierTransitionAuthorizationError(
                "requested lifecycle record is not a decision"
            )
        decision.validate()
        if not _same_scope(decision.scope, self.scope):
            raise TierTransitionIsolationError(
                "lifecycle decision scope does not match tier store"
            )
        if decision.decision_type not in TIER_TRANSITION_DECISION_TYPES:
            raise TierTransitionAuthorizationError(
                "lifecycle decision type is not executable in P5.1d"
            )
        if decision.outcome != "approved":
            raise TierTransitionAuthorizationError(
                "lifecycle decision is not approved"
            )
        if decision.current_tier not in EXECUTABLE_SOURCE_TIERS:
            raise TierTransitionAuthorizationError(
                "lifecycle source tier is not executable"
            )
        if decision.proposed_tier not in EXECUTABLE_TARGET_TIERS:
            raise TierTransitionAuthorizationError(
                "P5.1d does not execute deletion, ledger, or raw-buffer targets"
            )
        values = self._journal_values(journal)
        latest_blockers: dict[str, RetentionBlockerRecord] = {}
        for value in values:
            if isinstance(value, LifecycleDecision):
                if value.parent_decision_id == decision.decision_id:
                    raise TierTransitionAuthorizationError(
                        "lifecycle decision has been superseded"
                    )
                continue
            if (
                value.subject_reference == decision.subject_reference
                and value.content_digest == decision.content_digest
            ):
                latest_blockers[value.blocker_id] = value
        unresolved = tuple(
            value
            for value in latest_blockers.values()
            if value.state == "open"
        )
        if unresolved:
            kinds = ",".join(
                sorted({value.blocker_type for value in unresolved})
            )
            raise TierTransitionBlockedError(
                "lifecycle transition is blocked by unresolved retention "
                f"records: {kinds}"
            )
        return decision

    def _source_payload(
        self,
        *,
        decision: LifecycleDecision,
        source_reference_id: object,
        raw_buffer_store: RawBufferStore | None,
    ) -> tuple[bytes, str, str, str, int]:
        source_id = require_identifier(
            source_reference_id, "source_reference_id"
        )
        if decision.current_tier == "raw_buffer":
            if raw_buffer_store is None:
                raise TierTransitionStoreError(
                    "raw_buffer transitions require a raw-buffer source store"
                )
            if not _same_scope(raw_buffer_store.scope, self.scope):
                raise TierTransitionIsolationError(
                    "raw-buffer store scope does not match tier store"
                )
            raw_buffer_store.verify_integrity()
            try:
                reference = raw_buffer_store.get_reference(source_id)
            except KeyError as exc:
                raise TierTransitionStoreError(
                    "raw-buffer source reference does not exist"
                ) from exc
            if reference.storage_tier != "raw_buffer":
                raise TierTransitionStoreError(
                    "raw-buffer source reference changed tier"
                )
            payload = raw_buffer_store.load_opaque_payload(source_id)
        else:
            if raw_buffer_store is not None:
                raise TierTransitionStoreError(
                    "managed-tier transitions may not use a raw-buffer source"
                )
            reference = self.get_reference(source_id)
            if reference.storage_tier != decision.current_tier:
                raise TierTransitionStoreError(
                    "source reference tier does not match lifecycle decision"
                )
            payload = self.load_opaque_payload(source_id)
        subject = (
            reference.subject_reference
            if isinstance(reference, TierPayloadReference)
            else reference.logical_record_id
        )
        if subject != decision.subject_reference:
            raise TierTransitionStoreError(
                "source reference subject does not match lifecycle decision"
            )
        if reference.content_digest != decision.content_digest:
            raise TierTransitionStoreError(
                "source reference digest does not match lifecycle decision"
            )
        if reference.retention_class != decision.retention_class:
            raise TierTransitionStoreError(
                "source retention class does not match lifecycle decision"
            )
        if reference.host_sealed is not True:
            raise TierTransitionStoreError(
                "source payload is not host-sealed opaque bytes"
            )
        if len(payload) != reference.byte_length:
            raise TierTransitionIntegrityError(
                "source payload length changed during transition"
            )
        if _digest_bytes(payload) != decision.content_digest:
            raise TierTransitionIntegrityError(
                "source payload digest changed during transition"
            )
        return (
            payload,
            reference.media_type,
            reference.sensitivity_class,
            reference.retention_class,
            reference.byte_length,
        )

    def _insert_intent(self, intent: TierTransitionIntent) -> None:
        intent.validate()
        relative = self._relative_object_path(
            intent.target_tier, intent.content_digest
        ).as_posix()
        intent_json = canonical_json_bytes(intent.record()).decode("utf-8")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            sequence_row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 "
                "FROM tier_transition_intents"
            ).fetchone()
            sequence = 1 if sequence_row is None else int(sequence_row[0])
            self._connection.execute(
                "INSERT INTO tier_transition_intents ("
                "sequence, transition_id, lifecycle_decision_id, "
                "lifecycle_decision_sha256, subject_reference, "
                "content_digest, source_tier, target_tier, "
                "source_reference_id, target_reference_id, byte_length, "
                "target_relative_path, prepared_at, intent_json, intent_sha256"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sequence,
                    intent.transition_id,
                    intent.lifecycle_decision_id,
                    intent.lifecycle_decision_sha256,
                    intent.subject_reference,
                    intent.content_digest,
                    intent.source_tier,
                    intent.target_tier,
                    intent.source_reference_id,
                    intent.target_reference_id,
                    intent.byte_length,
                    relative,
                    intent.prepared_at,
                    intent_json,
                    intent.intent_sha256,
                ),
            )
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            raise DuplicateTierTransitionError(
                "tier-transition intent already exists"
            ) from exc
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _insert_publication(
        self,
        *,
        intent: TierTransitionIntent,
        published_at: object,
        physical_object_created: bool,
        recovered_from_prepared_intent: bool,
    ) -> TierTransitionReceipt:
        normalized_time = normalize_timestamp(published_at, "published_at")
        reference = TierPayloadReference.create(
            transition_id=intent.transition_id,
            lifecycle_decision_id=intent.lifecycle_decision_id,
            subject_reference=intent.subject_reference,
            source_reference_id=intent.source_reference_id,
            scope=self.scope,
            content_digest=intent.content_digest,
            byte_length=intent.byte_length,
            media_type=intent.media_type,
            sensitivity_class=intent.sensitivity_class,
            retention_class=intent.retention_class,
            storage_tier=intent.target_tier,
            published_at=normalized_time,
            host_sealed=True,
        )
        if reference.reference_id != intent.target_reference_id:
            raise TierTransitionIntegrityError(
                "published target reference does not match prepared intent"
            )
        receipt = TierTransitionReceipt.create(
            transition_id=intent.transition_id,
            lifecycle_decision_id=intent.lifecycle_decision_id,
            source_tier=intent.source_tier,
            target_tier=intent.target_tier,
            source_reference_id=intent.source_reference_id,
            target_reference=reference,
            source_preserved=True,
            physical_object_created=physical_object_created,
            recovered_from_prepared_intent=recovered_from_prepared_intent,
        )
        receipt_json = canonical_json_bytes(receipt.record()).decode("utf-8")
        publication_id = (
            "tier-publication-" + canonical_sha256(receipt.record())[:32]
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            sequence_row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 "
                "FROM tier_transition_publications"
            ).fetchone()
            sequence = 1 if sequence_row is None else int(sequence_row[0])
            self._connection.execute(
                "INSERT INTO tier_transition_publications ("
                "sequence, publication_id, transition_id, "
                "target_reference_id, published_at, source_preserved, "
                "physical_object_created, recovered_from_prepared_intent, "
                "receipt_json, receipt_sha256"
                ") VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (
                    sequence,
                    publication_id,
                    intent.transition_id,
                    reference.reference_id,
                    normalized_time,
                    int(physical_object_created),
                    int(recovered_from_prepared_intent),
                    receipt_json,
                    receipt.receipt_sha256,
                ),
            )
        except sqlite3.IntegrityError as exc:
            self._connection.rollback()
            existing = self._load_receipt_row(intent.transition_id)
            if existing is not None:
                return self._load_receipt(existing)
            raise DuplicateTierTransitionError(
                "tier-transition publication already exists"
            ) from exc
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()
        return receipt

    def execute(
        self,
        *,
        lifecycle_journal: LifecycleJournalStore,
        decision_id: object,
        source_reference_id: object,
        executed_at: object,
        raw_buffer_store: RawBufferStore | None = None,
    ) -> TierTransitionReceipt:
        decision = self._validate_decision(lifecycle_journal, decision_id)
        normalized_source = require_identifier(
            source_reference_id, "source_reference_id"
        )
        payload, media_type, sensitivity, retention, byte_length = (
            self._source_payload(
                decision=decision,
                source_reference_id=normalized_source,
                raw_buffer_store=raw_buffer_store,
            )
        )
        normalized_time = normalize_timestamp(executed_at, "executed_at")
        candidate = TierTransitionIntent.create(
            lifecycle_decision_id=decision.decision_id,
            lifecycle_decision_sha256=decision.decision_sha256,
            scope=self.scope,
            subject_reference=decision.subject_reference,
            content_digest=decision.content_digest,
            source_tier=decision.current_tier,
            target_tier=decision.proposed_tier,
            source_reference_id=normalized_source,
            byte_length=byte_length,
            media_type=media_type,
            sensitivity_class=sensitivity,
            retention_class=retention,
            prepared_at=normalized_time,
        )
        existing_intent_row = self._connection.execute(
            "SELECT * FROM tier_transition_intents "
            "WHERE lifecycle_decision_id = ?",
            (decision.decision_id,),
        ).fetchone()
        pending_before = existing_intent_row is not None
        if existing_intent_row is None:
            self._insert_intent(candidate)
            intent = candidate
        else:
            intent = self._load_intent(existing_intent_row)
            comparable = {
                key: value
                for key, value in intent.record().items()
                if key != "prepared_at" and key != "intent_sha256"
            }
            candidate_comparable = {
                key: value
                for key, value in candidate.record().items()
                if key != "prepared_at" and key != "intent_sha256"
            }
            if comparable != candidate_comparable:
                raise DuplicateTierTransitionError(
                    "existing prepared intent conflicts with requested execution"
                )
        existing_receipt_row = self._load_receipt_row(intent.transition_id)
        if existing_receipt_row is not None:
            return self._load_receipt(existing_receipt_row)
        target = self._object_path(intent.target_tier, intent.content_digest)
        target_existed_before = target.exists()
        created = self._publish_object(
            payload,
            target_tier=intent.target_tier,
            content_digest=intent.content_digest,
        )
        return self._insert_publication(
            intent=intent,
            published_at=normalized_time,
            physical_object_created=created,
            recovered_from_prepared_intent=(
                pending_before and target_existed_before
            ),
        )

    def recover_pending(
        self,
        *,
        lifecycle_journal: LifecycleJournalStore,
        recovered_at: object,
        raw_buffer_store: RawBufferStore | None = None,
    ) -> tuple[TierTransitionReceipt, ...]:
        rows = self._connection.execute(
            "SELECT i.* FROM tier_transition_intents i "
            "LEFT JOIN tier_transition_publications p "
            "ON p.transition_id = i.transition_id "
            "WHERE p.transition_id IS NULL ORDER BY i.sequence ASC"
        ).fetchall()
        receipts: list[TierTransitionReceipt] = []
        for row in rows:
            intent = self._load_intent(row)
            receipts.append(
                self.execute(
                    lifecycle_journal=lifecycle_journal,
                    decision_id=intent.lifecycle_decision_id,
                    source_reference_id=intent.source_reference_id,
                    executed_at=recovered_at,
                    raw_buffer_store=(
                        raw_buffer_store
                        if intent.source_tier == "raw_buffer"
                        else None
                    ),
                )
            )
        return tuple(receipts)

    def get_reference(self, reference_id: object) -> TierPayloadReference:
        normalized = require_identifier(reference_id, "reference_id")
        row = self._connection.execute(
            "SELECT * FROM tier_transition_publications "
            "WHERE target_reference_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return self._load_receipt(row).target_reference

    def load_opaque_payload(self, reference_id: object) -> bytes:
        reference = self.get_reference(reference_id)
        path = self._object_path(
            reference.storage_tier, reference.content_digest
        )
        self._verify_object(
            path=path,
            content_digest=reference.content_digest,
            byte_length=reference.byte_length,
        )
        return path.read_bytes()

    def inspect(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[TierTransitionInspectionRecord, ...]:
        if isinstance(after_sequence, bool) or not isinstance(
            after_sequence, int
        ) or after_sequence < 0:
            raise TierTransitionStoreError(
                "after_sequence must be a non-negative integer"
            )
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TierTransitionStoreError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise TierTransitionStoreError(
                "limit must be between 1 and 1000"
            )
        rows = self._connection.execute(
            "SELECT i.sequence, i.transition_id, i.lifecycle_decision_id, "
            "i.subject_reference, i.content_digest, i.source_tier, "
            "i.target_tier, i.source_reference_id, i.target_reference_id, "
            "i.prepared_at, p.published_at, p.source_preserved "
            "FROM tier_transition_intents i "
            "LEFT JOIN tier_transition_publications p "
            "ON p.transition_id = i.transition_id "
            "WHERE i.sequence > ? ORDER BY i.sequence ASC LIMIT ?",
            (after_sequence, limit),
        ).fetchall()
        return tuple(
            TierTransitionInspectionRecord(
                sequence=int(row["sequence"]),
                transition_id=str(row["transition_id"]),
                lifecycle_decision_id=str(row["lifecycle_decision_id"]),
                subject_reference=str(row["subject_reference"]),
                content_digest=str(row["content_digest"]),
                source_tier=str(row["source_tier"]),
                target_tier=str(row["target_tier"]),
                source_reference_id=str(row["source_reference_id"]),
                target_reference_id=str(row["target_reference_id"]),
                state=(
                    "published"
                    if row["published_at"] is not None
                    else "prepared"
                ),
                prepared_at=str(row["prepared_at"]),
                published_at=(
                    None
                    if row["published_at"] is None
                    else str(row["published_at"])
                ),
                source_preserved=(
                    None
                    if row["source_preserved"] is None
                    else bool(row["source_preserved"])
                ),
            )
            for row in rows
        )

    def verify_integrity(self) -> TierTransitionIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM tier_transition_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise TierTransitionIntegrityError(
                "tier-transition metadata is missing"
            )
        if str(metadata["store_id"]) != self.store_id:
            raise TierTransitionIntegrityError(
                "tier-transition store identity changed"
            )
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise TierTransitionIntegrityError(
                "tier-transition scope digest changed"
            )
        intent_rows = self._connection.execute(
            "SELECT * FROM tier_transition_intents ORDER BY sequence ASC"
        ).fetchall()
        publication_rows = self._connection.execute(
            "SELECT * FROM tier_transition_publications ORDER BY sequence ASC"
        ).fetchall()
        intents: dict[str, TierTransitionIntent] = {}
        expected_paths: set[Path] = set()
        for index, row in enumerate(intent_rows, start=1):
            if int(row["sequence"]) != index:
                raise TierTransitionIntegrityError(
                    "tier-transition intent sequence is not contiguous"
                )
            intent = self._load_intent(row)
            expected = {
                "transition_id": intent.transition_id,
                "lifecycle_decision_id": intent.lifecycle_decision_id,
                "lifecycle_decision_sha256": (
                    intent.lifecycle_decision_sha256
                ),
                "subject_reference": intent.subject_reference,
                "content_digest": intent.content_digest,
                "source_tier": intent.source_tier,
                "target_tier": intent.target_tier,
                "source_reference_id": intent.source_reference_id,
                "target_reference_id": intent.target_reference_id,
                "byte_length": intent.byte_length,
                "prepared_at": intent.prepared_at,
                "intent_sha256": intent.intent_sha256,
            }
            for column, value in expected.items():
                actual = row[column]
                actual = int(actual) if column == "byte_length" else str(actual)
                if actual != value:
                    raise TierTransitionIntegrityError(
                        f"tier-transition intent {column} column changed"
                    )
            relative = self._relative_object_path(
                intent.target_tier, intent.content_digest
            ).as_posix()
            if str(row["target_relative_path"]) != relative:
                raise TierTransitionIntegrityError(
                    "tier-transition target path metadata changed"
                )
            intents[intent.transition_id] = intent
        receipts: dict[str, TierTransitionReceipt] = {}
        for index, row in enumerate(publication_rows, start=1):
            if int(row["sequence"]) != index:
                raise TierTransitionIntegrityError(
                    "tier-transition publication sequence is not contiguous"
                )
            receipt = self._load_receipt(row)
            intent = intents.get(receipt.transition_id)
            if intent is None:
                raise TierTransitionIntegrityError(
                    "tier-transition publication lacks an intent"
                )
            if (
                receipt.lifecycle_decision_id
                != intent.lifecycle_decision_id
                or receipt.source_tier != intent.source_tier
                or receipt.target_tier != intent.target_tier
                or receipt.source_reference_id != intent.source_reference_id
                or receipt.target_reference.reference_id
                != intent.target_reference_id
                or receipt.target_reference.content_digest
                != intent.content_digest
            ):
                raise TierTransitionIntegrityError(
                    "tier-transition publication changed prepared lineage"
                )
            if str(row["transition_id"]) != receipt.transition_id:
                raise TierTransitionIntegrityError(
                    "tier-transition publication identity changed"
                )
            if str(row["target_reference_id"]) != (
                receipt.target_reference.reference_id
            ):
                raise TierTransitionIntegrityError(
                    "tier-transition target reference column changed"
                )
            if str(row["published_at"]) != (
                receipt.target_reference.published_at
            ):
                raise TierTransitionIntegrityError(
                    "tier-transition published_at column changed"
                )
            if int(row["source_preserved"]) != 1:
                raise TierTransitionIntegrityError(
                    "tier-transition source-preservation state changed"
                )
            if str(row["receipt_sha256"]) != receipt.receipt_sha256:
                raise TierTransitionIntegrityError(
                    "tier-transition receipt digest column changed"
                )
            path = self._object_path(
                receipt.target_tier,
                receipt.target_reference.content_digest,
            )
            self._verify_object(
                path=path,
                content_digest=receipt.target_reference.content_digest,
                byte_length=receipt.target_reference.byte_length,
            )
            expected_paths.add(path.resolve())
            receipts[receipt.transition_id] = receipt
        for intent in intents.values():
            path = self._object_path(intent.target_tier, intent.content_digest)
            if intent.transition_id not in receipts and path.exists():
                self._verify_object(
                    path=path,
                    content_digest=intent.content_digest,
                    byte_length=intent.byte_length,
                )
                expected_paths.add(path.resolve())
        actual_paths = {
            path.resolve()
            for path in self.tiers_root.rglob("*.payload")
            if path.is_file()
        }
        if actual_paths != expected_paths:
            raise TierTransitionIntegrityError(
                "tier-transition physical object manifest changed"
            )
        pending_count = len(intents) - len(receipts)
        head = canonical_sha256(
            {
                "store_id": self.store_id,
                "intents": [
                    intents[key].record() for key in sorted(intents)
                ],
                "receipts": [
                    receipts[key].record() for key in sorted(receipts)
                ],
                "physical_paths": sorted(
                    path.relative_to(self.root).as_posix()
                    for path in expected_paths
                ),
            }
        )
        return TierTransitionIntegrityReport(
            store_id=self.store_id,
            intent_count=len(intents),
            published_count=len(receipts),
            pending_count=pending_count,
            physical_object_count=len(expected_paths),
            head_sha256=head,
            valid=True,
        )


def _initialize_or_validate(
    connection: sqlite3.Connection,
    *,
    scope: ProductHostScope,
    created_at: object | None,
) -> tuple[str, str]:
    digest = tier_transition_scope_digest(scope)
    store_id = f"tier-transition-store-{digest[:32]}"
    row = connection.execute(
        "SELECT * FROM tier_transition_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        if created_at is None:
            raise TierTransitionStoreError(
                "created_at is required when initializing tier-transition storage"
            )
        normalized = normalize_timestamp(created_at, "created_at")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO tier_transition_metadata ("
                "singleton, schema_version, store_id, product_id, "
                "host_instance_id, encryption_domain, scope_digest, created_at"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    TIER_TRANSITION_SCHEMA_VERSION,
                    store_id,
                    scope.product_id,
                    scope.host_instance_id,
                    scope.encryption_domain,
                    digest,
                    normalized,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return store_id, digest
    expected = {
        "schema_version": TIER_TRANSITION_SCHEMA_VERSION,
        "store_id": store_id,
        "product_id": scope.product_id,
        "host_instance_id": scope.host_instance_id,
        "encryption_domain": scope.encryption_domain,
        "scope_digest": digest,
    }
    for key, value in expected.items():
        if str(row[key]) != value:
            raise TierTransitionIsolationError(
                f"tier-transition metadata does not match scope: {key}"
            )
    if created_at is not None:
        if str(row["created_at"]) != normalize_timestamp(
            created_at, "created_at"
        ):
            raise TierTransitionStoreError(
                "created_at does not match existing tier-transition store"
            )
    return store_id, digest


def _remove_orphan_temporary_files(root: Path) -> None:
    for path in root.rglob("*.tmp"):
        if path.is_file() and path.name.startswith("."):
            path.unlink(missing_ok=True)


def open_tier_transition_store(
    store_root: str | Path,
    *,
    scope: ProductHostScope,
    repository_root: str | Path | None = None,
    created_at: object | None = None,
) -> TierTransitionStore:
    scope.validate()
    root = validate_tier_transition_root(
        store_root,
        repository_root=repository_root,
    )
    root.mkdir(parents=True, exist_ok=True)
    tiers_root = root / "tiers"
    tiers_root.mkdir(parents=True, exist_ok=True)
    for tier in sorted(EXECUTABLE_TARGET_TIERS):
        (tiers_root / tier).mkdir(parents=True, exist_ok=True)
    _remove_orphan_temporary_files(root)
    database_path = root / "tier-transition.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        _configure(connection)
        connection.executescript(_SCHEMA_SQL)
        connection.commit()
        store_id, scope_digest = _initialize_or_validate(
            connection,
            scope=scope,
            created_at=created_at,
        )
        store = TierTransitionStore(
            root=root,
            connection=connection,
            scope=scope,
            store_id=store_id,
            scope_digest=scope_digest,
        )
        store.verify_integrity()
        return store
    except Exception:
        connection.close()
        raise
