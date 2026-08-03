"""Crash-safe SQLite persistence for retention-lifecycle journal records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
    require_identifier,
)
from .contracts import ProductHostScope
from .lifecycle import (
    LIFECYCLE_SCHEMA_VERSION,
    LifecycleDecision,
    LifecycleJournalEntryReceipt,
    LifecycleJournalIntegrityReport,
    LifecycleJournalRecord,
    LifecycleJournalTransactionReceipt,
    LifecycleJournalValue,
    RetentionBlockerRecord,
    lifecycle_scope_digest,
    lifecycle_scope_record,
    lifecycle_value_from_metadata,
)


class LifecycleJournalError(RuntimeError):
    """Base error for lifecycle-journal operations."""


class UnsafeLifecycleJournalPathError(LifecycleJournalError):
    """Raised when a journal path resolves inside the public repository."""


class LifecycleJournalConfigurationError(LifecycleJournalError):
    """Raised when SQLite cannot satisfy required durability settings."""


class LifecycleJournalIsolationError(LifecycleJournalError):
    """Raised when product, host, or encryption scope does not match."""


class DuplicateLifecycleRecordError(LifecycleJournalError):
    """Raised when a lifecycle record already exists."""


class LifecycleJournalIntegrityError(LifecycleJournalError):
    """Raised when journal rows, lineage, or hash chaining are invalid."""


class LifecycleJournalTransactionError(LifecycleJournalError):
    """Raised when an invalid or nested write transaction is attempted."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lifecycle_journal_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    journal_schema_version TEXT NOT NULL,
    journal_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    host_instance_id TEXT NOT NULL,
    encryption_domain TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    genesis_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lifecycle_journal_entries (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    record_id TEXT NOT NULL UNIQUE,
    record_kind TEXT NOT NULL,
    subject_reference TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    decision_type TEXT,
    current_tier TEXT,
    proposed_tier TEXT,
    retention_class TEXT,
    outcome TEXT,
    blocker_type TEXT,
    blocker_state TEXT,
    record_sha256 TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL,
    previous_entry_sha256 TEXT NOT NULL,
    entry_sha256 TEXT NOT NULL UNIQUE,
    committed_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS lifecycle_journal_entries_no_update
BEFORE UPDATE ON lifecycle_journal_entries
BEGIN
    SELECT RAISE(ABORT, 'lifecycle journal entries are append-only');
END;
CREATE TRIGGER IF NOT EXISTS lifecycle_journal_entries_no_delete
BEFORE DELETE ON lifecycle_journal_entries
BEGIN
    SELECT RAISE(ABORT, 'lifecycle journal entries are append-only');
END;
CREATE TRIGGER IF NOT EXISTS lifecycle_journal_metadata_no_update
BEFORE UPDATE ON lifecycle_journal_metadata
BEGIN
    SELECT RAISE(ABORT, 'lifecycle journal metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS lifecycle_journal_metadata_no_delete
BEFORE DELETE ON lifecycle_journal_metadata
BEGIN
    SELECT RAISE(ABORT, 'lifecycle journal metadata is immutable');
END;
"""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_lifecycle_journal_path(
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
        raise UnsafeLifecycleJournalPathError(
            "Refusing to create or open the lifecycle journal inside "
            f"the public repository: {candidate}"
        )
    return candidate


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = None if mode_row is None else str(mode_row[0]).lower()
    if mode != "wal":
        raise LifecycleJournalConfigurationError(
            "lifecycle journal requires SQLite WAL journal mode"
        )
    connection.execute("PRAGMA synchronous=FULL")
    synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
    synchronous = (
        None if synchronous_row is None else int(synchronous_row[0])
    )
    if synchronous != 2:
        raise LifecycleJournalConfigurationError(
            "lifecycle journal requires SQLite synchronous=FULL"
        )
    connection.execute("PRAGMA busy_timeout=5000")


def _scope_material(scope: ProductHostScope) -> dict[str, str]:
    scope.validate()
    return lifecycle_scope_record(
        product_id=scope.product_id,
        host_instance_id=scope.host_instance_id,
        encryption_domain=scope.encryption_domain,
    )


def _journal_identity(scope: ProductHostScope) -> tuple[str, str]:
    material = _scope_material(scope)
    digest = lifecycle_scope_digest(**material)
    return f"lifecycle-journal-{digest[:32]}", digest


def _record_columns(record: LifecycleJournalValue) -> dict[str, object | None]:
    if isinstance(record, LifecycleDecision):
        return {
            "record_kind": "decision",
            "subject_reference": record.subject_reference,
            "content_digest": record.content_digest,
            "decision_type": record.decision_type,
            "current_tier": record.current_tier,
            "proposed_tier": record.proposed_tier,
            "retention_class": record.retention_class,
            "outcome": record.outcome,
            "blocker_type": None,
            "blocker_state": None,
        }
    return {
        "record_kind": "blocker",
        "subject_reference": record.subject_reference,
        "content_digest": record.content_digest,
        "decision_type": None,
        "current_tier": None,
        "proposed_tier": None,
        "retention_class": None,
        "outcome": None,
        "blocker_type": record.blocker_type,
        "blocker_state": record.state,
    }


class LifecycleJournalStore:
    """One product-host-encryption scoped, append-only lifecycle journal."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        database_path: Path,
        scope: ProductHostScope,
        journal_id: str,
        scope_digest: str,
        genesis_sha256: str,
    ) -> None:
        self._connection = connection
        self.database_path = database_path
        self.scope = scope
        self.journal_id = journal_id
        self.scope_digest = scope_digest
        self.genesis_sha256 = genesis_sha256

    def _assert_record_scope(self, record: LifecycleJournalValue) -> None:
        record.validate()
        expected = _scope_material(self.scope)
        actual = _scope_material(record.scope)
        if actual != expected:
            raise LifecycleJournalIsolationError(
                "lifecycle record product-host encryption scope does not "
                "match journal"
            )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        if self._connection.in_transaction:
            raise LifecycleJournalTransactionError(
                "nested lifecycle journal transactions are not supported"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _load_value_from_row(self, row: sqlite3.Row) -> LifecycleJournalValue:
        try:
            value = json.loads(str(row["record_json"]))
        except json.JSONDecodeError as exc:
            raise LifecycleJournalIntegrityError(
                "stored lifecycle record JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise LifecycleJournalIntegrityError(
                "stored lifecycle record must be a JSON object"
            )
        try:
            record = lifecycle_value_from_metadata(
                str(row["record_kind"]), value
            )
        except CognitiveKernelContractError as exc:
            raise LifecycleJournalIntegrityError(
                "stored lifecycle record failed validation"
            ) from exc
        self._assert_record_scope(record)
        return record

    def _validate_lineage(
        self,
        connection: sqlite3.Connection,
        record: LifecycleJournalValue,
    ) -> None:
        if isinstance(record, LifecycleDecision):
            if record.parent_decision_id is None:
                return
            row = connection.execute(
                "SELECT * FROM lifecycle_journal_entries "
                "WHERE record_id = ?",
                (record.parent_decision_id,),
            ).fetchone()
            if row is None:
                raise LifecycleJournalTransactionError(
                    "lifecycle decision parent does not exist"
                )
            parent = self._load_value_from_row(row)
            if not isinstance(parent, LifecycleDecision):
                raise LifecycleJournalTransactionError(
                    "lifecycle decision parent is not a decision"
                )
            if (
                parent.subject_reference != record.subject_reference
                or parent.content_digest != record.content_digest
            ):
                raise LifecycleJournalTransactionError(
                    "lifecycle decision parent changed the subject"
                )
            if (
                record.decision_type == "override"
                and record.current_tier != parent.proposed_tier
            ):
                raise LifecycleJournalTransactionError(
                    "override current tier does not continue prior decision"
                )
            return

        if record.state == "open":
            blocker_rows = connection.execute(
                "SELECT * FROM lifecycle_journal_entries "
                "WHERE record_kind = 'blocker'"
            ).fetchall()
            for blocker_row in blocker_rows:
                prior = self._load_value_from_row(blocker_row)
                if (
                    isinstance(prior, RetentionBlockerRecord)
                    and prior.blocker_id == record.blocker_id
                ):
                    raise DuplicateLifecycleRecordError(
                        "retention blocker is already open or resolved"
                    )
            return

        row = connection.execute(
            "SELECT * FROM lifecycle_journal_entries WHERE record_id = ?",
            (record.parent_record_id,),
        ).fetchone()
        if row is None:
            raise LifecycleJournalTransactionError(
                "retention blocker resolution parent does not exist"
            )
        parent = self._load_value_from_row(row)
        if not isinstance(parent, RetentionBlockerRecord):
            raise LifecycleJournalTransactionError(
                "retention blocker parent is not a blocker record"
            )
        if parent.state != "open" or parent.blocker_id != record.blocker_id:
            raise LifecycleJournalTransactionError(
                "retention blocker resolution lineage is invalid"
            )
        if (
            parent.subject_reference != record.subject_reference
            or parent.content_digest != record.content_digest
            or parent.blocker_type != record.blocker_type
            or parent.opened_at != record.opened_at
        ):
            raise LifecycleJournalTransactionError(
                "retention blocker resolution changed protected lineage"
            )
        resolution_rows = connection.execute(
            "SELECT * FROM lifecycle_journal_entries "
            "WHERE record_kind = 'blocker' "
            "AND blocker_state = 'resolved'"
        ).fetchall()
        for resolution_row in resolution_rows:
            existing = self._load_value_from_row(resolution_row)
            if (
                isinstance(existing, RetentionBlockerRecord)
                and existing.blocker_id == record.blocker_id
            ):
                raise DuplicateLifecycleRecordError(
                    "retention blocker is already resolved"
                )

    def append_record(
        self,
        record: LifecycleJournalValue,
        *,
        committed_at: object,
    ) -> LifecycleJournalTransactionReceipt:
        return self.append_records((record,), committed_at=committed_at)

    def append_records(
        self,
        records: Iterable[LifecycleJournalValue],
        *,
        committed_at: object,
    ) -> LifecycleJournalTransactionReceipt:
        normalized_time = normalize_timestamp(committed_at, "committed_at")
        values = tuple(records)
        if not values:
            raise LifecycleJournalTransactionError(
                "lifecycle journal transaction requires records"
            )
        seen_ids: set[str] = set()
        seen_digests: set[str] = set()
        for record in values:
            if not isinstance(record, (LifecycleDecision, RetentionBlockerRecord)):
                raise LifecycleJournalTransactionError(
                    "lifecycle journal accepts only lifecycle records"
                )
            self._assert_record_scope(record)
            if record.record_id in seen_ids or record.record_sha256 in seen_digests:
                raise DuplicateLifecycleRecordError(
                    "lifecycle journal transaction contains duplicate records"
                )
            seen_ids.add(record.record_id)
            seen_digests.add(record.record_sha256)

        receipts: list[LifecycleJournalEntryReceipt] = []
        with self._write_transaction() as connection:
            head = connection.execute(
                "SELECT sequence, entry_sha256 "
                "FROM lifecycle_journal_entries "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 0 if head is None else int(head["sequence"])
            previous = (
                self.genesis_sha256
                if head is None
                else str(head["entry_sha256"])
            )
            for record in values:
                duplicate = connection.execute(
                    "SELECT record_id FROM lifecycle_journal_entries "
                    "WHERE record_id = ? OR record_sha256 = ? LIMIT 1",
                    (record.record_id, record.record_sha256),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateLifecycleRecordError(
                        f"lifecycle record already exists: {record.record_id}"
                    )
                self._validate_lineage(connection, record)
                sequence += 1
                receipt = LifecycleJournalEntryReceipt.create(
                    sequence=sequence,
                    record_id=record.record_id,
                    record_sha256=record.record_sha256,
                    previous_entry_sha256=previous,
                    committed_at=normalized_time,
                )
                columns = _record_columns(record)
                record_json = canonical_json_bytes(
                    record.metadata_record()
                ).decode("utf-8")
                try:
                    connection.execute(
                        "INSERT INTO lifecycle_journal_entries ("
                        "sequence, record_id, record_kind, subject_reference, "
                        "content_digest, decision_type, current_tier, "
                        "proposed_tier, retention_class, outcome, blocker_type, "
                        "blocker_state, record_sha256, record_json, "
                        "previous_entry_sha256, entry_sha256, committed_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            receipt.sequence,
                            record.record_id,
                            columns["record_kind"],
                            columns["subject_reference"],
                            columns["content_digest"],
                            columns["decision_type"],
                            columns["current_tier"],
                            columns["proposed_tier"],
                            columns["retention_class"],
                            columns["outcome"],
                            columns["blocker_type"],
                            columns["blocker_state"],
                            record.record_sha256,
                            record_json,
                            receipt.previous_entry_sha256,
                            receipt.entry_sha256,
                            receipt.committed_at,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateLifecycleRecordError(
                        f"lifecycle record already exists: {record.record_id}"
                    ) from exc
                receipts.append(receipt)
                previous = receipt.entry_sha256
            transaction = LifecycleJournalTransactionReceipt.create(
                journal_id=self.journal_id,
                committed_at=normalized_time,
                entries=tuple(receipts),
            )
        return transaction

    def load_record(self, record_id: object) -> LifecycleJournalValue:
        normalized = require_identifier(record_id, "record_id")
        row = self._connection.execute(
            "SELECT * FROM lifecycle_journal_entries WHERE record_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        return self._load_value_from_row(row)

    def inspect(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[LifecycleJournalRecord, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise LifecycleJournalError("after_sequence must be an integer")
        if after_sequence < 0:
            raise LifecycleJournalError("after_sequence may not be negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise LifecycleJournalError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise LifecycleJournalError("limit must be between 1 and 1000")
        rows = self._connection.execute(
            "SELECT sequence, record_id, record_kind, subject_reference, "
            "content_digest, decision_type, current_tier, proposed_tier, "
            "retention_class, outcome, blocker_type, blocker_state, "
            "record_sha256, previous_entry_sha256, entry_sha256, committed_at "
            "FROM lifecycle_journal_entries WHERE sequence > ? "
            "ORDER BY sequence ASC LIMIT ?",
            (after_sequence, limit),
        ).fetchall()
        return tuple(
            LifecycleJournalRecord(
                sequence=int(row["sequence"]),
                record_id=str(row["record_id"]),
                record_kind=str(row["record_kind"]),
                subject_reference=str(row["subject_reference"]),
                content_digest=str(row["content_digest"]),
                decision_type=(
                    None
                    if row["decision_type"] is None
                    else str(row["decision_type"])
                ),
                current_tier=(
                    None
                    if row["current_tier"] is None
                    else str(row["current_tier"])
                ),
                proposed_tier=(
                    None
                    if row["proposed_tier"] is None
                    else str(row["proposed_tier"])
                ),
                retention_class=(
                    None
                    if row["retention_class"] is None
                    else str(row["retention_class"])
                ),
                outcome=(
                    None if row["outcome"] is None else str(row["outcome"])
                ),
                blocker_type=(
                    None
                    if row["blocker_type"] is None
                    else str(row["blocker_type"])
                ),
                blocker_state=(
                    None
                    if row["blocker_state"] is None
                    else str(row["blocker_state"])
                ),
                record_sha256=str(row["record_sha256"]),
                previous_entry_sha256=str(row["previous_entry_sha256"]),
                entry_sha256=str(row["entry_sha256"]),
                committed_at=str(row["committed_at"]),
            )
            for row in rows
        )

    def verify_integrity(self) -> LifecycleJournalIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM lifecycle_journal_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise LifecycleJournalIntegrityError(
                "lifecycle journal metadata is missing"
            )
        if str(metadata["journal_id"]) != self.journal_id:
            raise LifecycleJournalIntegrityError("journal identity changed")
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise LifecycleJournalIntegrityError(
                "journal scope digest changed"
            )
        if str(metadata["genesis_sha256"]) != self.genesis_sha256:
            raise LifecycleJournalIntegrityError(
                "journal genesis digest changed"
            )
        rows = self._connection.execute(
            "SELECT * FROM lifecycle_journal_entries ORDER BY sequence ASC"
        ).fetchall()
        previous = self.genesis_sha256
        seen: dict[str, LifecycleJournalValue] = {}
        for index, row in enumerate(rows, start=1):
            if int(row["sequence"]) != index:
                raise LifecycleJournalIntegrityError(
                    "lifecycle journal sequence is not contiguous"
                )
            if str(row["previous_entry_sha256"]) != previous:
                raise LifecycleJournalIntegrityError(
                    "lifecycle journal hash chain is broken"
                )
            record = self._load_value_from_row(row)
            columns = _record_columns(record)
            expected_columns = {
                "record_id": record.record_id,
                "record_kind": columns["record_kind"],
                "subject_reference": columns["subject_reference"],
                "content_digest": columns["content_digest"],
                "decision_type": columns["decision_type"],
                "current_tier": columns["current_tier"],
                "proposed_tier": columns["proposed_tier"],
                "retention_class": columns["retention_class"],
                "outcome": columns["outcome"],
                "blocker_type": columns["blocker_type"],
                "blocker_state": columns["blocker_state"],
                "record_sha256": record.record_sha256,
            }
            for column, expected in expected_columns.items():
                actual = row[column]
                if expected is None:
                    if actual is not None:
                        raise LifecycleJournalIntegrityError(
                            f"lifecycle journal {column} column changed"
                        )
                elif str(actual) != str(expected):
                    raise LifecycleJournalIntegrityError(
                        f"lifecycle journal {column} column changed"
                    )
            if isinstance(record, LifecycleDecision):
                if record.parent_decision_id is not None:
                    parent = seen.get(record.parent_decision_id)
                    if not isinstance(parent, LifecycleDecision):
                        raise LifecycleJournalIntegrityError(
                            "lifecycle decision lineage is invalid"
                        )
                    if (
                        parent.subject_reference != record.subject_reference
                        or parent.content_digest != record.content_digest
                    ):
                        raise LifecycleJournalIntegrityError(
                            "lifecycle decision lineage changed subject"
                        )
                    if (
                        record.decision_type == "override"
                        and record.current_tier != parent.proposed_tier
                    ):
                        raise LifecycleJournalIntegrityError(
                            "override lineage does not continue prior tier"
                        )
            elif record.state == "resolved":
                parent = seen.get(record.parent_record_id or "")
                if not isinstance(parent, RetentionBlockerRecord):
                    raise LifecycleJournalIntegrityError(
                        "retention blocker lineage is invalid"
                    )
                if parent.state != "open" or parent.blocker_id != record.blocker_id:
                    raise LifecycleJournalIntegrityError(
                        "retention blocker lineage changed"
                    )
                if (
                    parent.subject_reference != record.subject_reference
                    or parent.content_digest != record.content_digest
                    or parent.blocker_type != record.blocker_type
                    or parent.opened_at != record.opened_at
                ):
                    raise LifecycleJournalIntegrityError(
                        "retention blocker protected lineage changed"
                    )
            receipt = LifecycleJournalEntryReceipt.create(
                sequence=index,
                record_id=record.record_id,
                record_sha256=record.record_sha256,
                previous_entry_sha256=previous,
                committed_at=str(row["committed_at"]),
            )
            if str(row["entry_sha256"]) != receipt.entry_sha256:
                raise LifecycleJournalIntegrityError(
                    "lifecycle journal entry digest mismatch"
                )
            seen[record.record_id] = record
            previous = receipt.entry_sha256
        count = len(rows)
        return LifecycleJournalIntegrityReport.create(
            journal_id=self.journal_id,
            entry_count=count,
            first_sequence=1 if count else None,
            last_sequence=count if count else None,
            head_entry_sha256=previous,
            valid=True,
        )


def _initialize_or_validate(
    connection: sqlite3.Connection,
    *,
    scope: ProductHostScope,
    created_at: object | None,
) -> tuple[str, str, str]:
    journal_id, scope_digest = _journal_identity(scope)
    row = connection.execute(
        "SELECT * FROM lifecycle_journal_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        if created_at is None:
            raise LifecycleJournalConfigurationError(
                "created_at is required when initializing a new journal"
            )
        normalized_created_at = normalize_timestamp(created_at, "created_at")
        genesis = canonical_sha256(
            {
                "journal_schema_version": LIFECYCLE_SCHEMA_VERSION,
                "journal_id": journal_id,
                "scope": _scope_material(scope),
                "created_at": normalized_created_at,
            }
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO lifecycle_journal_metadata ("
                "singleton, journal_schema_version, journal_id, product_id, "
                "host_instance_id, encryption_domain, scope_digest, "
                "created_at, genesis_sha256"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    LIFECYCLE_SCHEMA_VERSION,
                    journal_id,
                    scope.product_id,
                    scope.host_instance_id,
                    scope.encryption_domain,
                    scope_digest,
                    normalized_created_at,
                    genesis,
                ),
            )
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        return journal_id, scope_digest, genesis

    expected = _scope_material(scope)
    actual = {
        "product_id": str(row["product_id"]),
        "host_instance_id": str(row["host_instance_id"]),
        "encryption_domain": str(row["encryption_domain"]),
    }
    if actual != expected:
        raise LifecycleJournalIsolationError(
            "lifecycle journal database is bound to another scope"
        )
    if str(row["journal_schema_version"]) != LIFECYCLE_SCHEMA_VERSION:
        raise LifecycleJournalConfigurationError(
            "lifecycle journal schema version changed"
        )
    if str(row["journal_id"]) != journal_id:
        raise LifecycleJournalIntegrityError("journal identity is invalid")
    if str(row["scope_digest"]) != scope_digest:
        raise LifecycleJournalIntegrityError(
            "journal scope digest is invalid"
        )
    if created_at is not None:
        normalized_created_at = normalize_timestamp(created_at, "created_at")
        if str(row["created_at"]) != normalized_created_at:
            raise LifecycleJournalConfigurationError(
                "created_at does not match existing journal"
            )
    genesis = str(row["genesis_sha256"])
    expected_genesis = canonical_sha256(
        {
            "journal_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "journal_id": journal_id,
            "scope": expected,
            "created_at": str(row["created_at"]),
        }
    )
    if genesis != expected_genesis:
        raise LifecycleJournalIntegrityError(
            "journal genesis digest is invalid"
        )
    return journal_id, scope_digest, genesis


@contextmanager
def open_lifecycle_journal(
    database_path: str | Path,
    *,
    scope: ProductHostScope,
    repository_root: str | Path | None = None,
    created_at: object | None = None,
) -> Iterator[LifecycleJournalStore]:
    scope.validate()
    database = validate_lifecycle_journal_path(
        database_path,
        repository_root=repository_root,
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        _configure_connection(connection)
        connection.executescript(_SCHEMA_SQL)
        version_row = connection.execute("PRAGMA user_version").fetchone()
        user_version = 0 if version_row is None else int(version_row[0])
        metadata_exists = connection.execute(
            "SELECT 1 FROM lifecycle_journal_metadata WHERE singleton = 1"
        ).fetchone() is not None
        if user_version == 0 and not metadata_exists:
            connection.execute("PRAGMA user_version=1")
        elif user_version != 1:
            raise LifecycleJournalConfigurationError(
                "lifecycle journal SQLite user_version must be 1"
            )
        journal_id, scope_digest, genesis = _initialize_or_validate(
            connection,
            scope=scope,
            created_at=created_at,
        )
        store = LifecycleJournalStore(
            connection=connection,
            database_path=database,
            scope=scope,
            journal_id=journal_id,
            scope_digest=scope_digest,
            genesis_sha256=genesis,
        )
        yield store
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()
