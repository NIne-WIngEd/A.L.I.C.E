"""Phase 2 provenance attachment for verified Phase 1 evidence."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .phase1_bridge import Phase1ChunkEvidence
from .sources import (
    MemorySourceRecord,
    MemorySourceSpec,
    insert_memory_sources_in_transaction,
)


class MemoryProvenanceError(RuntimeError):
    """Base error for Memory Core provenance operations."""


class MemoryNotFoundError(MemoryProvenanceError):
    """Raised when provenance targets a nonexistent Phase 2 memory."""


@dataclass(frozen=True)
class AttachedMemorySource:
    """Private-safe Phase 2 source-link representation."""

    memory_source_id: str
    memory_id: str
    source_type: str
    source_ref: str
    source_content_sha256: str | None
    source_text_sha256: str | None
    chunk_id: str | None
    file_id: str | None
    support_relation: str
    created_at: str


def _memory_exists(
    connection: sqlite3.Connection,
    memory_id: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM memories
        WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    return row is not None


def phase1_chunk_source_specs(
    evidence: Phase1ChunkEvidence,
    *,
    support_relation: str,
) -> tuple[MemorySourceSpec, ...]:
    """Convert verified Phase 1 evidence to private-safe creation sources."""
    provenance_file_ids: tuple[str | None, ...]
    if evidence.provenance_paths:
        provenance_file_ids = tuple(
            item.file_id
            for item in evidence.provenance_paths
        )
    else:
        provenance_file_ids = (None,)

    specs: list[MemorySourceSpec] = []
    for file_id in provenance_file_ids:
        source_ref = evidence.source_ref
        if file_id is not None:
            source_ref = f"{source_ref}:file:{file_id}"

        specs.append(
            MemorySourceSpec(
                source_type="phase1_chunk",
                source_ref=source_ref,
                source_content_sha256=evidence.source_content_sha256,
                source_text_sha256=evidence.source_text_sha256,
                chunk_id=evidence.chunk_id,
                file_id=file_id,
                support_relation=support_relation,
            )
        )

    return tuple(specs)


def _attached(
    record: MemorySourceRecord,
) -> AttachedMemorySource:
    return AttachedMemorySource(
        memory_source_id=record.memory_source_id,
        memory_id=record.memory_id,
        source_type=record.source_type,
        source_ref=record.source_ref,
        source_content_sha256=record.source_content_sha256,
        source_text_sha256=record.source_text_sha256,
        chunk_id=record.chunk_id,
        file_id=record.file_id,
        support_relation=record.support_relation,
        created_at=record.created_at,
    )


def attach_phase1_chunk_evidence(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    evidence: Phase1ChunkEvidence,
    support_relation: str,
    created_at: str,
) -> tuple[AttachedMemorySource, ...]:
    """Attach verified Phase 1 chunk provenance without copying plaintext.

    New creation should pass ``phase1_chunk_source_specs(...)`` through
    ``MemoryCreateRequest.sources`` so memory and provenance are atomic.
    This function remains for attaching verified evidence to existing records
    and intentionally does not commit.
    """
    if not _memory_exists(connection, memory_id):
        raise MemoryNotFoundError(
            f"Cannot attach provenance to missing memory: {memory_id}"
        )

    records = insert_memory_sources_in_transaction(
        connection,
        memory_id=memory_id,
        sources=phase1_chunk_source_specs(
            evidence,
            support_relation=support_relation,
        ),
        created_at=created_at,
        allow_existing=True,
    )
    return tuple(
        _attached(record)
        for record in records
    )


def list_memory_sources(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[AttachedMemorySource, ...]:
    """Return private-safe source links for one memory."""
    rows = connection.execute(
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
            support_relation,
            created_at
        FROM memory_sources
        WHERE memory_id = ?
        ORDER BY memory_source_id
        """,
        (memory_id,),
    ).fetchall()

    return tuple(
        AttachedMemorySource(
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
            support_relation=str(row["support_relation"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    )
