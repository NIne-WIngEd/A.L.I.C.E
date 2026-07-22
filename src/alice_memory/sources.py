"""Validated provenance-source contracts for Phase 2 memory creation.

Every authoritative Memory Core record must carry at least one source record.
Source insertion is transaction-neutral: callers own the surrounding Memory
Core transaction so memory, provenance, relations, and events can commit or
roll back atomically.
"""

from __future__ import annotations

import sqlite3
import string
import uuid
from dataclasses import dataclass

from .schema import SOURCE_TYPES, SUPPORT_RELATIONS


_SOURCE_NAMESPACE = uuid.UUID(
    "0bcbe12b-9f52-4e89-94ef-919363f64392"
)
_HEX = set(string.hexdigits)


class MemorySourceError(RuntimeError):
    """Base error for Memory Core source validation and persistence."""


class MemorySourceValidationError(MemorySourceError):
    """Raised when a source specification violates provenance invariants."""


@dataclass(frozen=True)
class MemorySourceSpec:
    """Private-safe provenance metadata supplied with a memory write.

    ``source_ref`` must be an immutable identifier or stable locator, never a
    copy of private plaintext. Hashes are optional for source types that do not
    naturally have content artifacts.
    """

    source_type: str
    source_ref: str
    support_relation: str
    source_content_sha256: str | None = None
    source_text_sha256: str | None = None
    chunk_id: str | None = None
    file_id: str | None = None
    source_date: str | None = None


@dataclass(frozen=True)
class MemorySourceRecord:
    """Persisted private-safe source-link representation."""

    memory_source_id: str
    memory_id: str
    source_type: str
    source_ref: str
    source_content_sha256: str | None
    source_text_sha256: str | None
    chunk_id: str | None
    file_id: str | None
    source_date: str | None
    support_relation: str
    created_at: str


def _validate_optional_sha256(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise MemorySourceValidationError(
            f"{field_name} must be a 64-character hexadecimal SHA-256 digest."
        )


def validate_memory_source(source: MemorySourceSpec) -> None:
    """Validate one provenance source without accessing plaintext."""
    if source.source_type not in SOURCE_TYPES:
        raise MemorySourceValidationError(
            f"Unsupported memory source type: {source.source_type!r}"
        )
    if not source.source_ref.strip():
        raise MemorySourceValidationError(
            "Memory source_ref cannot be empty."
        )
    if source.support_relation not in SUPPORT_RELATIONS:
        raise MemorySourceValidationError(
            "Unsupported memory support relation: "
            f"{source.support_relation!r}"
        )

    _validate_optional_sha256(
        source.source_content_sha256,
        field_name="source_content_sha256",
    )
    _validate_optional_sha256(
        source.source_text_sha256,
        field_name="source_text_sha256",
    )

    if source.source_type == "phase1_chunk":
        if not source.chunk_id:
            raise MemorySourceValidationError(
                "phase1_chunk provenance requires chunk_id."
            )
        if source.source_content_sha256 is None:
            raise MemorySourceValidationError(
                "phase1_chunk provenance requires source_content_sha256."
            )
        if source.source_text_sha256 is None:
            raise MemorySourceValidationError(
                "phase1_chunk provenance requires source_text_sha256."
            )


def validate_memory_sources(
    sources: tuple[MemorySourceSpec, ...],
) -> None:
    """Require at least one unique valid source for an authoritative memory."""
    if not sources:
        raise MemorySourceValidationError(
            "Authoritative Memory Core records require at least one "
            "provenance source."
        )

    seen: set[tuple[object, ...]] = set()
    for source in sources:
        validate_memory_source(source)
        identity = (
            source.source_type,
            source.source_ref,
            source.chunk_id,
            source.file_id,
            source.support_relation,
        )
        if identity in seen:
            raise MemorySourceValidationError(
                "Duplicate provenance source specifications are not allowed."
            )
        seen.add(identity)


def memory_source_id(
    *,
    memory_id: str,
    source: MemorySourceSpec,
) -> str:
    canonical = "|".join(
        (
            memory_id,
            source.source_type,
            source.source_ref,
            source.chunk_id or "",
            source.file_id or "",
            source.support_relation,
        )
    )
    return str(
        uuid.uuid5(
            _SOURCE_NAMESPACE,
            canonical,
        )
    )


def _row_to_source_record(
    row: sqlite3.Row,
) -> MemorySourceRecord:
    return MemorySourceRecord(
        memory_source_id=str(row["memory_source_id"]),
        memory_id=str(row["memory_id"]),
        source_type=str(row["source_type"]),
        source_ref=str(row["source_ref"]),
        source_content_sha256=(
            None
            if row["source_content_sha256"] is None
            else str(row["source_content_sha256"])
        ),
        source_text_sha256=(
            None
            if row["source_text_sha256"] is None
            else str(row["source_text_sha256"])
        ),
        chunk_id=(
            None
            if row["chunk_id"] is None
            else str(row["chunk_id"])
        ),
        file_id=(
            None
            if row["file_id"] is None
            else str(row["file_id"])
        ),
        source_date=(
            None
            if row["source_date"] is None
            else str(row["source_date"])
        ),
        support_relation=str(row["support_relation"]),
        created_at=str(row["created_at"]),
    )


def insert_memory_sources_in_transaction(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    sources: tuple[MemorySourceSpec, ...],
    created_at: str,
    allow_existing: bool = False,
) -> tuple[MemorySourceRecord, ...]:
    """Insert source links without committing the caller's transaction."""
    validate_memory_sources(sources)

    persisted: list[MemorySourceRecord] = []
    for source in sources:
        source_id = memory_source_id(
            memory_id=memory_id,
            source=source,
        )
        verb = "INSERT OR IGNORE" if allow_existing else "INSERT"
        connection.execute(
            f"""
            {verb} INTO memory_sources (
                memory_source_id,
                memory_id,
                source_type,
                source_ref,
                source_content_sha256,
                source_text_sha256,
                chunk_id,
                file_id,
                source_date,
                support_relation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                memory_id,
                source.source_type,
                source.source_ref,
                source.source_content_sha256,
                source.source_text_sha256,
                source.chunk_id,
                source.file_id,
                source.source_date,
                source.support_relation,
                created_at,
            ),
        )

        row = connection.execute(
            """
            SELECT
                memory_source_id,
                memory_id,
                source_type,
                source_ref,
                source_content_sha256,
                source_text_sha256,
                chunk_id,
                file_id,
                source_date,
                support_relation,
                created_at
            FROM memory_sources
            WHERE memory_id = ?
              AND source_type = ?
              AND source_ref = ?
              AND (
                  (chunk_id IS NULL AND ? IS NULL)
                  OR chunk_id = ?
              )
              AND (
                  (file_id IS NULL AND ? IS NULL)
                  OR file_id = ?
              )
              AND support_relation = ?
            """,
            (
                memory_id,
                source.source_type,
                source.source_ref,
                source.chunk_id,
                source.chunk_id,
                source.file_id,
                source.file_id,
                source.support_relation,
            ),
        ).fetchone()
        if row is None:
            raise MemorySourceError(
                "Memory source insertion did not persist a source record."
            )
        persisted.append(
            _row_to_source_record(row)
        )

    return tuple(persisted)
