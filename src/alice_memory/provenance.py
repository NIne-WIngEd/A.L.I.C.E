"""Phase 2 provenance attachment for verified Phase 1 evidence."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass

from .phase1_bridge import Phase1ChunkEvidence

_PROVENANCE_NAMESPACE = uuid.UUID(
    "e726be32-b6e2-4a41-9c58-4078d79e2168"
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
    source_content_sha256: str
    source_text_sha256: str
    chunk_id: str
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


def _source_id(
    *,
    memory_id: str,
    source_ref: str,
    chunk_id: str,
    file_id: str | None,
    support_relation: str,
) -> str:
    canonical = "|".join(
        (
            memory_id,
            source_ref,
            chunk_id,
            file_id or "",
            support_relation,
        )
    )
    return str(
        uuid.uuid5(
            _PROVENANCE_NAMESPACE,
            canonical,
        )
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

    One source link is created per Phase 1 provenance file ID. If a chunk has
    no provenance paths, a chunk-level source link is created with file_id=None.

    The function does not commit. Callers should use the Memory Core transaction
    context so memory creation and provenance attachment can be atomic.
    """
    if not _memory_exists(connection, memory_id):
        raise MemoryNotFoundError(
            f"Cannot attach provenance to missing memory: {memory_id}"
        )

    provenance_file_ids: tuple[str | None, ...]
    if evidence.provenance_paths:
        provenance_file_ids = tuple(
            item.file_id
            for item in evidence.provenance_paths
        )
    else:
        provenance_file_ids = (None,)

    attached: list[AttachedMemorySource] = []

    for file_id in provenance_file_ids:
        source_ref = evidence.source_ref
        if file_id is not None:
            source_ref = f"{source_ref}:file:{file_id}"

        memory_source_id = _source_id(
            memory_id=memory_id,
            source_ref=source_ref,
            chunk_id=evidence.chunk_id,
            file_id=file_id,
            support_relation=support_relation,
        )

        connection.execute(
            """
            INSERT OR IGNORE INTO memory_sources (
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
                memory_source_id,
                memory_id,
                "phase1_chunk",
                source_ref,
                evidence.source_content_sha256,
                evidence.source_text_sha256,
                evidence.chunk_id,
                file_id,
                None,
                support_relation,
                created_at,
            ),
        )

        attached.append(
            AttachedMemorySource(
                memory_source_id=memory_source_id,
                memory_id=memory_id,
                source_type="phase1_chunk",
                source_ref=source_ref,
                source_content_sha256=evidence.source_content_sha256,
                source_text_sha256=evidence.source_text_sha256,
                chunk_id=evidence.chunk_id,
                file_id=file_id,
                support_relation=support_relation,
                created_at=created_at,
            )
        )

    return tuple(attached)


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
            source_content_sha256=str(row["source_content_sha256"]),
            source_text_sha256=str(row["source_text_sha256"]),
            chunk_id=str(row["chunk_id"]),
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
