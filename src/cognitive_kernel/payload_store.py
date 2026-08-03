"""Host-scoped content-addressed storage for opaque raw-buffer payloads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .canonical import canonical_json_bytes, canonical_sha256, normalize_timestamp
from .contracts import ProductHostScope
from .raw_buffer import (
    RAW_BUFFER_SCHEMA_VERSION,
    PayloadStoreAccounting,
    PayloadStoreIntegrityReport,
    RawBufferCaptureReceipt,
    RawBufferReference,
)


class RawBufferStoreError(RuntimeError):
    """Base error for raw-buffer content-store operations."""


class UnsafeRawBufferPathError(RawBufferStoreError):
    """Raised when a raw-buffer root resolves inside the public repository."""


class RawBufferIsolationError(RawBufferStoreError):
    """Raised when a store is opened under the wrong product-host scope."""


class DuplicateRawBufferReferenceError(RawBufferStoreError):
    """Raised when an identical logical reference already exists."""


class RawBufferIntegrityError(RawBufferStoreError):
    """Raised when metadata or payload bytes fail integrity checks."""


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_buffer_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    store_id TEXT NOT NULL UNIQUE,
    product_id TEXT NOT NULL,
    host_instance_id TEXT NOT NULL,
    encryption_domain TEXT NOT NULL,
    scope_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_buffer_references (
    sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
    reference_id TEXT NOT NULL UNIQUE,
    logical_record_id TEXT NOT NULL,
    payload_object_id TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    byte_length INTEGER NOT NULL CHECK (byte_length > 0),
    reference_json TEXT NOT NULL,
    reference_sha256 TEXT NOT NULL UNIQUE,
    object_relative_path TEXT NOT NULL,
    captured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS raw_buffer_references_content_digest
ON raw_buffer_references(content_digest);
CREATE TRIGGER IF NOT EXISTS raw_buffer_references_no_update
BEFORE UPDATE ON raw_buffer_references
BEGIN
    SELECT RAISE(ABORT, 'raw-buffer references are append-only');
END;
CREATE TRIGGER IF NOT EXISTS raw_buffer_references_no_delete
BEFORE DELETE ON raw_buffer_references
BEGIN
    SELECT RAISE(ABORT, 'raw-buffer references are append-only');
END;
CREATE TRIGGER IF NOT EXISTS raw_buffer_metadata_no_update
BEFORE UPDATE ON raw_buffer_metadata
BEGIN
    SELECT RAISE(ABORT, 'raw-buffer metadata is immutable');
END;
CREATE TRIGGER IF NOT EXISTS raw_buffer_metadata_no_delete
BEFORE DELETE ON raw_buffer_metadata
BEGIN
    SELECT RAISE(ABORT, 'raw-buffer metadata is immutable');
END;
"""


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def validate_raw_buffer_root(
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
        raise UnsafeRawBufferPathError(
            "Refusing to create or open raw-buffer storage inside "
            f"the public repository: {candidate}"
        )
    if candidate.exists() and not candidate.is_dir():
        raise UnsafeRawBufferPathError(
            f"raw-buffer root must be a directory: {candidate}"
        )
    return candidate


def _scope_digest(scope: ProductHostScope) -> str:
    scope.validate()
    return canonical_sha256(
        {
            "product_id": scope.product_id,
            "host_instance_id": scope.host_instance_id,
            "encryption_domain": scope.encryption_domain,
        }
    )


def _configure(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
    if mode is None or str(mode[0]).lower() != "wal":
        raise RawBufferStoreError("raw-buffer metadata requires SQLite WAL")
    connection.execute("PRAGMA synchronous=FULL")
    level = connection.execute("PRAGMA synchronous").fetchone()
    if level is None or int(level[0]) != 2:
        raise RawBufferStoreError(
            "raw-buffer metadata requires SQLite synchronous=FULL"
        )
    connection.execute("PRAGMA busy_timeout=5000")


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class RawBufferStore:
    """One product-host-encryption scoped opaque payload store."""

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
        self.objects_root = root / "objects"

    def __enter__(self) -> "RawBufferStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _relative_object_path(self, content_digest: str) -> Path:
        return Path(
            "objects",
            self.scope_digest[:2],
            content_digest[:2],
            f"{self.scope_digest}-{content_digest}.payload",
        )

    def _object_path(self, content_digest: str) -> Path:
        return self.root / self._relative_object_path(content_digest)

    def _verify_object(
        self,
        *,
        path: Path,
        content_digest: str,
        byte_length: int,
    ) -> None:
        if not path.is_file():
            raise RawBufferIntegrityError(
                f"raw-buffer object is missing: {content_digest}"
            )
        payload = path.read_bytes()
        if len(payload) != byte_length:
            raise RawBufferIntegrityError(
                f"raw-buffer object length changed: {content_digest}"
            )
        if _digest_bytes(payload) != content_digest:
            raise RawBufferIntegrityError(
                f"raw-buffer object digest changed: {content_digest}"
            )

    def _publish_object(self, payload: bytes, content_digest: str) -> bool:
        target = self._object_path(content_digest)
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
            if _digest_bytes(temporary.read_bytes()) != content_digest:
                raise RawBufferIntegrityError(
                    "temporary raw-buffer object digest mismatch"
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
        return True

    def capture(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        logical_record_id: object,
        media_type: object,
        sensitivity_class: object,
        retention_class: object,
        captured_at: object,
        host_sealed: object = True,
    ) -> RawBufferCaptureReceipt:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise RawBufferStoreError("payload must be bytes-like")
        opaque = bytes(payload)
        if not opaque:
            raise RawBufferStoreError("payload must not be empty")
        digest = _digest_bytes(opaque)
        reference = RawBufferReference.create(
            logical_record_id=logical_record_id,
            scope=self.scope,
            content_digest=digest,
            byte_length=len(opaque),
            media_type=media_type,
            sensitivity_class=sensitivity_class,
            retention_class=retention_class,
            captured_at=captured_at,
            host_sealed=host_sealed,
        )
        duplicate = self._connection.execute(
            "SELECT reference_id FROM raw_buffer_references "
            "WHERE reference_id = ? OR reference_sha256 = ? LIMIT 1",
            (reference.reference_id, reference.reference_sha256),
        ).fetchone()
        if duplicate is not None:
            raise DuplicateRawBufferReferenceError(
                f"raw-buffer reference already exists: {reference.reference_id}"
            )
        created = self._publish_object(opaque, digest)
        relative = self._relative_object_path(digest).as_posix()
        reference_json = canonical_json_bytes(reference.record()).decode("utf-8")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            head = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM raw_buffer_references"
            ).fetchone()
            sequence = 1 if head is None else int(head[0]) + 1
            self._connection.execute(
                "INSERT INTO raw_buffer_references ("
                "sequence, reference_id, logical_record_id, payload_object_id, "
                "content_digest, byte_length, reference_json, reference_sha256, "
                "object_relative_path, captured_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sequence,
                    reference.reference_id,
                    reference.logical_record_id,
                    reference.payload_object_id,
                    reference.content_digest,
                    reference.byte_length,
                    reference_json,
                    reference.reference_sha256,
                    relative,
                    reference.captured_at,
                ),
            )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            if created:
                still_referenced = self._connection.execute(
                    "SELECT 1 FROM raw_buffer_references "
                    "WHERE content_digest = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                if still_referenced is None:
                    self._object_path(digest).unlink(missing_ok=True)
            raise
        return RawBufferCaptureReceipt.create(
            reference=reference,
            physical_object_created=created,
            deduplicated=not created,
        )

    def get_reference(self, reference_id: str) -> RawBufferReference:
        row = self._connection.execute(
            "SELECT reference_json FROM raw_buffer_references "
            "WHERE reference_id = ?",
            (reference_id,),
        ).fetchone()
        if row is None:
            raise KeyError(reference_id)
        value = json.loads(str(row["reference_json"]))
        if not isinstance(value, dict):
            raise RawBufferIntegrityError("stored reference JSON is invalid")
        return RawBufferReference.from_record(value)

    def load_opaque_payload(self, reference_id: str) -> bytes:
        reference = self.get_reference(reference_id)
        path = self._object_path(reference.content_digest)
        self._verify_object(
            path=path,
            content_digest=reference.content_digest,
            byte_length=reference.byte_length,
        )
        return path.read_bytes()

    def inspect(self) -> tuple[RawBufferReference, ...]:
        rows = self._connection.execute(
            "SELECT reference_json FROM raw_buffer_references "
            "ORDER BY sequence ASC"
        ).fetchall()
        values: list[RawBufferReference] = []
        for row in rows:
            raw = json.loads(str(row["reference_json"]))
            if not isinstance(raw, dict):
                raise RawBufferIntegrityError("stored reference JSON is invalid")
            values.append(RawBufferReference.from_record(raw))
        return tuple(values)

    def accounting(self) -> PayloadStoreAccounting:
        rows = self._connection.execute(
            "SELECT content_digest, byte_length FROM raw_buffer_references"
        ).fetchall()
        logical_bytes = sum(int(row["byte_length"]) for row in rows)
        unique: dict[str, int] = {}
        for row in rows:
            unique[str(row["content_digest"])] = int(row["byte_length"])
        physical_bytes = sum(unique.values())
        return PayloadStoreAccounting(
            logical_reference_count=len(rows),
            physical_object_count=len(unique),
            logical_bytes=logical_bytes,
            physical_bytes=physical_bytes,
            deduplicated_bytes=logical_bytes - physical_bytes,
        )

    def verify_integrity(self) -> PayloadStoreIntegrityReport:
        metadata = self._connection.execute(
            "SELECT * FROM raw_buffer_metadata WHERE singleton = 1"
        ).fetchone()
        if metadata is None:
            raise RawBufferIntegrityError("raw-buffer metadata is missing")
        if str(metadata["store_id"]) != self.store_id:
            raise RawBufferIntegrityError("raw-buffer store identity changed")
        if str(metadata["scope_digest"]) != self.scope_digest:
            raise RawBufferIntegrityError("raw-buffer scope digest changed")
        rows = self._connection.execute(
            "SELECT * FROM raw_buffer_references ORDER BY sequence ASC"
        ).fetchall()
        references = self.inspect()
        if len(rows) != len(references):
            raise RawBufferIntegrityError(
                "raw-buffer reference count changed during inspection"
            )
        expected_paths: set[Path] = set()
        for sequence, (row, reference) in enumerate(
            zip(rows, references), start=1
        ):
            if int(row["sequence"]) != sequence:
                raise RawBufferIntegrityError(
                    "raw-buffer reference sequence is not contiguous"
                )
            if reference.scope.storage_scope() != self.scope.storage_scope():
                raise RawBufferIntegrityError("stored reference scope changed")
            expected_columns = {
                "reference_id": reference.reference_id,
                "logical_record_id": reference.logical_record_id,
                "payload_object_id": reference.payload_object_id,
                "content_digest": reference.content_digest,
                "byte_length": reference.byte_length,
                "reference_sha256": reference.reference_sha256,
                "captured_at": reference.captured_at,
            }
            for column, expected in expected_columns.items():
                actual = row[column]
                if column == "byte_length":
                    actual = int(actual)
                else:
                    actual = str(actual)
                if actual != expected:
                    raise RawBufferIntegrityError(
                        f"raw-buffer {column} column does not match reference"
                    )
            expected_relative = self._relative_object_path(
                reference.content_digest
            ).as_posix()
            if str(row["object_relative_path"]) != expected_relative:
                raise RawBufferIntegrityError(
                    "raw-buffer object path metadata changed"
                )
            path = self._object_path(reference.content_digest)
            self._verify_object(
                path=path,
                content_digest=reference.content_digest,
                byte_length=reference.byte_length,
            )
            expected_paths.add(path.resolve())
        actual_paths = {
            path.resolve()
            for path in self.objects_root.rglob("*.payload")
            if path.is_file()
        }
        if actual_paths != expected_paths:
            raise RawBufferIntegrityError(
                "raw-buffer physical object manifest does not match references"
            )
        accounting = self.accounting()
        head = canonical_sha256(
            {
                "store_id": self.store_id,
                "references": [item.record() for item in references],
                "accounting": accounting.record(),
            }
        )
        return PayloadStoreIntegrityReport(
            store_id=self.store_id,
            logical_reference_count=accounting.logical_reference_count,
            physical_object_count=accounting.physical_object_count,
            head_sha256=head,
            valid=True,
        )


def _initialize_or_validate(
    connection: sqlite3.Connection,
    *,
    scope: ProductHostScope,
    created_at: object | None,
) -> tuple[str, str]:
    digest = _scope_digest(scope)
    store_id = f"raw-buffer-{digest[:32]}"
    row = connection.execute(
        "SELECT * FROM raw_buffer_metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        if created_at is None:
            raise RawBufferStoreError(
                "created_at is required when initializing a raw-buffer store"
            )
        normalized = normalize_timestamp(created_at, "created_at")
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO raw_buffer_metadata ("
                "singleton, schema_version, store_id, product_id, "
                "host_instance_id, encryption_domain, scope_digest, created_at"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    RAW_BUFFER_SCHEMA_VERSION,
                    store_id,
                    scope.product_id,
                    scope.host_instance_id,
                    scope.encryption_domain,
                    digest,
                    normalized,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return store_id, digest
    expected = {
        "schema_version": RAW_BUFFER_SCHEMA_VERSION,
        "store_id": store_id,
        "product_id": scope.product_id,
        "host_instance_id": scope.host_instance_id,
        "encryption_domain": scope.encryption_domain,
        "scope_digest": digest,
    }
    for key, value in expected.items():
        if str(row[key]) != value:
            raise RawBufferIsolationError(
                f"raw-buffer metadata does not match requested scope: {key}"
            )
    return store_id, digest


def open_raw_buffer_store(
    store_root: str | Path,
    *,
    scope: ProductHostScope,
    repository_root: str | Path | None = None,
    created_at: object | None = None,
) -> RawBufferStore:
    scope.validate()
    root = validate_raw_buffer_root(
        store_root,
        repository_root=repository_root,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "objects").mkdir(parents=True, exist_ok=True)
    database_path = root / "raw-buffer-metadata.sqlite3"
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
        store = RawBufferStore(
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
