"""Crash-safe SQLite persistence for metadata-only Experience Events."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .canonical import (
    CognitiveKernelContractError,
    canonical_json_bytes,
    canonical_sha256,
    normalize_timestamp,
)
from .contracts import ProductHostScope
from .experience import ExperienceEvent
from .ledger import (
    LEDGER_SCHEMA_VERSION,
    ExperienceLedgerEntryReceipt,
    ExperienceLedgerIntegrityReport,
    ExperienceLedgerRecord,
    ExperienceLedgerTransactionReceipt,
    ledger_scope_digest,
    ledger_scope_record,
)


class ExperienceLedgerError(RuntimeError):
    """Base error for compact experience-ledger operations."""


class UnsafeExperienceLedgerPathError(ExperienceLedgerError):
    """Raised when a live ledger path resolves inside the public repository."""


class ExperienceLedgerConfigurationError(ExperienceLedgerError):
    """Raised when SQLite cannot satisfy the required durability settings."""


class ExperienceLedgerIsolationError(ExperienceLedgerError):
    """Raised when product, host, or encryption scope does not match."""


class DuplicateExperienceEventError(ExperienceLedgerError):
    """Raised when an event identity already exists in the ledger."""


class ExperienceLedgerIntegrityError(ExperienceLedgerError):
    """Raised when ledger rows, sequence, or hash chaining are invalid."""


class ExperienceLedgerTransactionError(ExperienceLedgerError):
    """Raised when an invalid or nested write transaction is attempted."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experience_ledger_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    ledger_schema_version TEXT NOT NULL,
    ledger_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    host_instance_id TEXT NOT NULL,
    encryption_domain TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    genesis_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experience_ledger_entries (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    retention_class TEXT NOT NULL,
    storage_tier TEXT NOT NULL,
    event_sha256 TEXT NOT NULL UNIQUE,
    event_json TEXT NOT NULL,
    previous_entry_sha256 TEXT NOT NULL,
    entry_sha256 TEXT NOT NULL UNIQUE,
    committed_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS experience_ledger_entries_no_update
BEFORE UPDATE ON experience_ledger_entries
BEGIN
    SELECT RAISE(ABORT, 'experience ledger entries are append-only');
END;
CREATE TRIGGER IF NOT EXISTS experience_ledger_entries_no_delete
BEFORE DELETE ON experience_ledger_entries
BEGIN
    SELECT RAISE(ABORT, 'experience ledger entries are append-only');
END;
CREATE TRIGGER IF NOT EXISTS experience_ledger_metadata_no_update
BEFORE UPDATE ON experience_ledger_metadata
BEGIN
    SELECT RAISE(ABORT, 'experience ledger metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS experience_ledger_metadata_no_delete
BEFORE DELETE ON experience_ledger_metadata
BEGIN
    SELECT RAISE(ABORT, 'experience ledger metadata is immutable');
END;
"""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_experience_ledger_path(
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
        raise UnsafeExperienceLedgerPathError(
            "Refusing to create or open the experience ledger inside "
            f"the public repository: {candidate}"
        )
    return candidate


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    mode = None if mode_row is None else str(mode_row[0]).lower()
    if mode != "wal":
        raise ExperienceLedgerConfigurationError(
            "experience ledger requires SQLite WAL journal mode"
        )
    connection.execute("PRAGMA synchronous=FULL")
    synchronous_row = connection.execute("PRAGMA synchronous").fetchone()
    synchronous = None if synchronous_row is None else int(synchronous_row[0])
    if synchronous != 2:
        raise ExperienceLedgerConfigurationError(
            "experience ledger requires SQLite synchronous=FULL"
        )
    connection.execute("PRAGMA busy_timeout=5000")


def _scope_material(scope: ProductHostScope) -> dict[str, str]:
    scope.validate()
    return ledger_scope_record(
        product_id=scope.product_id,
        host_instance_id=scope.host_instance_id,
        encryption_domain=scope.encryption_domain,
    )


def _ledger_identity(scope: ProductHostScope) -> tuple[str, str]:
    material = _scope_material(scope)
    digest = ledger_scope_digest(**material)
    return f"experience-ledger-{digest[:32]}", digest


class ExperienceLedgerStore:
    """One product-host-encryption scoped, logically append-only ledger."""

    def __init__(
        self,
        *,
        connection: sqlite3.Connection,
        database_path: Path,
        scope: ProductHostScope,
        ledger_id: str,
        scope_digest: str,
        genesis_sha256: str,
    ) -> None:
        self._connection = connection
        self.database_path = database_path
        self.scope = scope
        self.ledger_id = ledger_id
        self.scope_digest = scope_digest
        self.genesis_sha256 = genesis_sha256

    def _assert_event_scope(self, event: ExperienceEvent) -> None:
        event.validate()
        expected = _scope_material(self.scope)
        actual = _scope_material(event.scope)
        if actual != expected:
            raise ExperienceLedgerIsolationError(
                "experience event product-host encryption scope does not match ledger"
            )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        if self._connection.in_transaction:
            raise ExperienceLedgerTransactionError(
                "nested experience ledger transactions are not supported"
            )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def append_event(
        self,
        event: ExperienceEvent,
        *,
        committed_at: object,
    ) -> ExperienceLedgerTransactionReceipt:
        return self.append_events((event,), committed_at=committed_at)

    def append_events(
        self,
        events: Iterable[ExperienceEvent],
        *,
        committed_at: object,
    ) -> ExperienceLedgerTransactionReceipt:
        normalized_time = normalize_timestamp(committed_at, "committed_at")
        values = tuple(events)
        if not values:
            raise ExperienceLedgerTransactionError(
                "experience ledger transaction requires events"
            )
        seen_event_ids: set[str] = set()
        seen_event_digests: set[str] = set()
        for event in values:
            if not isinstance(event, ExperienceEvent):
                raise ExperienceLedgerTransactionError(
                    "experience ledger accepts only ExperienceEvent values"
                )
            self._assert_event_scope(event)
            if event.event_id in seen_event_ids or event.event_sha256 in seen_event_digests:
                raise DuplicateExperienceEventError(
                    "experience ledger transaction contains duplicate events"
                )
            seen_event_ids.add(event.event_id)
            seen_event_digests.add(event.event_sha256)
        receipts: list[ExperienceLedgerEntryReceipt] = []
        with self._write_transaction() as connection:
            head = connection.execute(
                "SELECT sequence, entry_sha256 "
                "FROM experience_ledger_entries "
                "ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 0 if head is None else int(head["sequence"])
            previous = (
                self.genesis_sha256
                if head is None
                else str(head["entry_sha256"])
            )
            for event in values:
                duplicate = connection.execute(
                    "SELECT event_id FROM experience_ledger_entries "
                    "WHERE event_id = ? OR event_sha256 = ? LIMIT 1",
                    (event.event_id, event.event_sha256),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateExperienceEventError(
                        f"experience event already exists: {event.event_id}"
                    )
                sequence += 1
                receipt = ExperienceLedgerEntryReceipt.create(
                    sequence=sequence,
                    event_id=event.event_id,
                    event_sha256=event.event_sha256,
                    previous_entry_sha256=previous,
                    committed_at=normalized_time,
                )
                event_json = canonical_json_bytes(
                    event.metadata_record()
                ).decode("utf-8")
                try:
                    connection.execute(
                        "INSERT INTO experience_ledger_entries ("
                        "sequence, event_id, event_type, occurred_at, "
                        "content_digest, retention_class, storage_tier, "
                        "event_sha256, event_json, previous_entry_sha256, "
                        "entry_sha256, committed_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            receipt.sequence,
                            event.event_id,
                            event.event_type,
                            event.occurred_at,
                            event.content_digest,
                            event.retention_class,
                            event.storage_tier,
                            event.event_sha256,
                            event_json,
                            receipt.previous_entry_sha256,
                            receipt.entry_sha256,
                            receipt.committed_at,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateExperienceEventError(
                        f"experience event already exists: {event.event_id}"
                    ) from exc
                receipts.append(receipt)
                previous = receipt.entry_sha256
            transaction_receipt = ExperienceLedgerTransactionReceipt.create(
                ledger_id=self.ledger_id,
                committed_at=normalized_time,
                entries=tuple(receipts),
            )
        return transaction_receipt

    def inspect(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[ExperienceLedgerRecord, ...]:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise ExperienceLedgerError("after_sequence must be an integer")
        if after_sequence < 0:
            raise ExperienceLedgerError("after_sequence may not be negative")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ExperienceLedgerError("limit must be an integer")
        if limit < 1 or limit > 1000:
            raise ExperienceLedgerError("limit must be between 1 and 1000")
        rows = self._connection.execute(
            "SELECT sequence, event_id, event_type, occurred_at, "
            "content_digest, retention_class, storage_tier, event_sha256, "
            "previous_entry_sha256, entry_sha256, committed_at "
            "FROM experience_ledger_entries WHERE sequence > ? "
            "ORDER BY sequence ASC LIMIT ?",
            (after_sequence, limit),
        ).fetchall()
        return tuple(
            ExperienceLedgerRecord(
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                event_type=str(row["event_type"]),
                occurred_at=str(row["occurred_at"]),
                content_digest=str(row["content_digest"]),
                retention_class=str(row["retention_class"]),
                storage_tier=str(row["storage_tier"]),
                event_sha256=str(row["event_sha256"]),
                previous_entry_sha256=str(row["previous_entry_sha256"]),
                entry_sha256=str(row["entry_sha256"]),
                committed_at=str(row["committed_at"]),
            )
            for row in rows
        )

    def load_event(self, event_id: object) -> ExperienceEvent:
        from .canonical import require_identifier

        normalized = require_identifier(event_id, "event_id")
        row = self._connection.execute(
            "SELECT event_json FROM experience_ledger_entries "
            "WHERE event_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(normalized)
        try:
            value = json.loads(str(row["event_json"]))
        except json.JSONDecodeError as exc:
            raise ExperienceLedgerIntegrityError(
                "stored experience event JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise ExperienceLedgerIntegrityError(
                "stored experience event must be a JSON object"
            )
        try:
            event = ExperienceEvent.from_metadata_record(value)
        except CognitiveKernelContractError as exc:
            raise ExperienceLedgerIntegrityError(
                "stored experience event failed validation"
            ) from exc
        self._assert_event_scope(event)
        return event

    def verify_integrity(self) -> ExperienceLedgerIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM experience_ledger_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise ExperienceLedgerIntegrityError(
                "experience ledger metadata is missing"
            )
        if str(metadata["ledger_id"]) != self.ledger_id:
            raise ExperienceLedgerIntegrityError("ledger identity changed")
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise ExperienceLedgerIntegrityError("ledger scope digest changed")
        if str(metadata["genesis_sha256"]) != self.genesis_sha256:
            raise ExperienceLedgerIntegrityError("ledger genesis digest changed")
        rows = self._connection.execute(
            "SELECT * FROM experience_ledger_entries ORDER BY sequence ASC"
        ).fetchall()
        previous = self.genesis_sha256
        for index, row in enumerate(rows, start=1):
            if int(row["sequence"]) != index:
                raise ExperienceLedgerIntegrityError(
                    "experience ledger sequence is not contiguous"
                )
            if str(row["previous_entry_sha256"]) != previous:
                raise ExperienceLedgerIntegrityError(
                    "experience ledger hash chain is broken"
                )
            try:
                value = json.loads(str(row["event_json"]))
            except json.JSONDecodeError as exc:
                raise ExperienceLedgerIntegrityError(
                    "stored experience event JSON is invalid"
                ) from exc
            if not isinstance(value, dict):
                raise ExperienceLedgerIntegrityError(
                    "stored experience event must be a JSON object"
                )
            try:
                event = ExperienceEvent.from_metadata_record(value)
            except CognitiveKernelContractError as exc:
                raise ExperienceLedgerIntegrityError(
                    "stored experience event failed validation"
                ) from exc
            self._assert_event_scope(event)
            column_pairs = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "occurred_at": event.occurred_at,
                "content_digest": event.content_digest,
                "retention_class": event.retention_class,
                "storage_tier": event.storage_tier,
                "event_sha256": event.event_sha256,
            }
            for column, expected in column_pairs.items():
                if str(row[column]) != expected:
                    raise ExperienceLedgerIntegrityError(
                        f"experience ledger {column} column does not match event"
                    )
            expected_receipt = ExperienceLedgerEntryReceipt.create(
                sequence=index,
                event_id=event.event_id,
                event_sha256=event.event_sha256,
                previous_entry_sha256=previous,
                committed_at=str(row["committed_at"]),
            )
            if str(row["entry_sha256"]) != expected_receipt.entry_sha256:
                raise ExperienceLedgerIntegrityError(
                    "experience ledger entry digest mismatch"
                )
            previous = expected_receipt.entry_sha256
        count = len(rows)
        return ExperienceLedgerIntegrityReport.create(
            ledger_id=self.ledger_id,
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
    ledger_id, scope_digest = _ledger_identity(scope)
    row = connection.execute(
        "SELECT * FROM experience_ledger_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        if created_at is None:
            raise ExperienceLedgerConfigurationError(
                "created_at is required when initializing a new ledger"
            )
        normalized_created_at = normalize_timestamp(created_at, "created_at")
        genesis = canonical_sha256(
            {
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "ledger_id": ledger_id,
                "scope": _scope_material(scope),
                "created_at": normalized_created_at,
            }
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO experience_ledger_metadata ("
                "singleton, ledger_schema_version, ledger_id, product_id, "
                "host_instance_id, encryption_domain, scope_digest, "
                "created_at, genesis_sha256"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    LEDGER_SCHEMA_VERSION,
                    ledger_id,
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
        return ledger_id, scope_digest, genesis
    expected = _scope_material(scope)
    actual = {
        "product_id": str(row["product_id"]),
        "host_instance_id": str(row["host_instance_id"]),
        "encryption_domain": str(row["encryption_domain"]),
    }
    if actual != expected:
        raise ExperienceLedgerIsolationError(
            "experience ledger database is bound to another scope"
        )
    if str(row["ledger_schema_version"]) != LEDGER_SCHEMA_VERSION:
        raise ExperienceLedgerConfigurationError(
            "experience ledger schema version changed"
        )
    if str(row["ledger_id"]) != ledger_id:
        raise ExperienceLedgerIntegrityError("ledger identity is invalid")
    if str(row["scope_digest"]) != scope_digest:
        raise ExperienceLedgerIntegrityError("ledger scope digest is invalid")
    if created_at is not None:
        normalized_created_at = normalize_timestamp(created_at, "created_at")
        if str(row["created_at"]) != normalized_created_at:
            raise ExperienceLedgerConfigurationError(
                "created_at does not match existing ledger"
            )
    genesis = str(row["genesis_sha256"])
    expected_genesis = canonical_sha256(
        {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "ledger_id": ledger_id,
            "scope": expected,
            "created_at": str(row["created_at"]),
        }
    )
    if genesis != expected_genesis:
        raise ExperienceLedgerIntegrityError("ledger genesis digest is invalid")
    return ledger_id, scope_digest, genesis


@contextmanager
def open_experience_ledger(
    database_path: str | Path,
    *,
    scope: ProductHostScope,
    repository_root: str | Path | None = None,
    created_at: object | None = None,
) -> Iterator[ExperienceLedgerStore]:
    scope.validate()
    database = validate_experience_ledger_path(
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
            "SELECT 1 FROM experience_ledger_metadata WHERE singleton = 1"
        ).fetchone() is not None
        if user_version == 0 and not metadata_exists:
            connection.execute("PRAGMA user_version=1")
        elif user_version != 1:
            raise ExperienceLedgerConfigurationError(
                "experience ledger SQLite user_version must be 1"
            )
        ledger_id, scope_digest, genesis = _initialize_or_validate(
            connection,
            scope=scope,
            created_at=created_at,
        )
        store = ExperienceLedgerStore(
            connection=connection,
            database_path=database,
            scope=scope,
            ledger_id=ledger_id,
            scope_digest=scope_digest,
            genesis_sha256=genesis,
        )
        yield store
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()
