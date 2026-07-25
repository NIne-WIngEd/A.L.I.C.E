"""Deletion-integrity helpers for A.L.I.C.E. P2.8d.

A completed memory deletion must not leave authoritative dependents, promoted
candidate plaintext, candidate provenance, candidate audit rows, or unmanaged
active cache/summary tables behind. Only sanitized deletion evidence may
remain.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass

DELETION_INTEGRITY_VERSION = "p2.8d-v1"


class MemoryDeletionIntegrityError(RuntimeError):
    """Raised when deletion-integrity validation fails closed."""


@dataclass(frozen=True)
class PromotedCandidateLineage:
    """Snapshot of promoted candidate state tied to one authoritative memory."""

    memory_id: str
    candidate_ids: tuple[str, ...]
    candidate_id_sha256: tuple[str, ...]
    source_count: int
    event_count: int

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)


_UNMANAGED_ACTIVE_DERIVATIVE_TABLE_NAMES = frozenset(
    {
        "active_memory_summaries",
        "active_memory_summary",
        "memory_cache",
        "memory_caches",
        "memory_summaries",
        "memory_summary",
    }
)

_FORBIDDEN_AUDIT_KEYS = frozenset(
    {
        "ciphertext",
        "content",
        "key_id",
        "nonce",
        "plaintext",
        "reason",
        "source_content_sha256",
        "source_ref",
        "source_text_sha256",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require_no_unmanaged_active_derivative_tables(
    connection: sqlite3.Connection,
) -> None:
    """Fail closed if an active cache or summary table lacks a purge handler."""
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()
    unsupported = sorted(
        str(row["name"])
        for row in rows
        if str(row["name"]).casefold()
        in _UNMANAGED_ACTIVE_DERIVATIVE_TABLE_NAMES
    )
    if unsupported:
        raise MemoryDeletionIntegrityError(
            "Memory deletion cannot complete while unmanaged active cache or "
            "summary tables exist: " + ", ".join(unsupported)
        )


def collect_promoted_candidate_lineage(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    content_sha256: str,
) -> PromotedCandidateLineage:
    """Load and validate promoted candidates that retain deleted plaintext."""
    rows = connection.execute(
        """
        SELECT candidate_id, candidate_state, content_sha256
        FROM memory_candidates
        WHERE promoted_memory_id = ?
        ORDER BY candidate_id
        """,
        (memory_id,),
    ).fetchall()

    candidate_ids: list[str] = []
    candidate_id_sha256: list[str] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if str(row["candidate_state"]) != "promoted":
            raise MemoryDeletionIntegrityError(
                "A memory-linked candidate is not in promoted state."
            )
        if str(row["content_sha256"]) != content_sha256:
            raise MemoryDeletionIntegrityError(
                "A promoted candidate digest does not match its memory."
            )
        candidate_ids.append(candidate_id)
        candidate_id_sha256.append(
            hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        )

    if not candidate_ids:
        return PromotedCandidateLineage(
            memory_id=memory_id,
            candidate_ids=(),
            candidate_id_sha256=(),
            source_count=0,
            event_count=0,
        )

    placeholders = ",".join("?" for _ in candidate_ids)
    source_count = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM memory_candidate_sources
            WHERE candidate_id IN ({placeholders})
            """,
            tuple(candidate_ids),
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM memory_candidate_events
            WHERE candidate_id IN ({placeholders})
            """,
            tuple(candidate_ids),
        ).fetchone()[0]
    )
    return PromotedCandidateLineage(
        memory_id=memory_id,
        candidate_ids=tuple(candidate_ids),
        candidate_id_sha256=tuple(candidate_id_sha256),
        source_count=source_count,
        event_count=event_count,
    )


def promoted_candidate_lineage_audit(
    lineage: PromotedCandidateLineage,
) -> dict[str, object]:
    """Return sanitized candidate-lineage evidence for deletion audit."""
    return {
        "candidate_count": lineage.candidate_count,
        "candidate_id_sha256": list(lineage.candidate_id_sha256),
        "event_count": lineage.event_count,
        "integrity_version": DELETION_INTEGRITY_VERSION,
        "purged": True,
        "source_count": lineage.source_count,
    }


def purge_promoted_candidate_lineage(
    connection: sqlite3.Connection,
    *,
    lineage: PromotedCandidateLineage,
) -> None:
    """Remove promoted candidate plaintext, provenance, and candidate events."""
    if not lineage.candidate_ids:
        return

    placeholders = ",".join("?" for _ in lineage.candidate_ids)
    event_cursor = connection.execute(
        f"""
        DELETE FROM memory_candidate_events
        WHERE candidate_id IN ({placeholders})
        """,
        lineage.candidate_ids,
    )
    if event_cursor.rowcount != lineage.event_count:
        raise MemoryDeletionIntegrityError(
            "Promoted candidate event state changed before deletion commit."
        )

    candidate_cursor = connection.execute(
        f"""
        DELETE FROM memory_candidates
        WHERE candidate_id IN ({placeholders})
          AND promoted_memory_id = ?
          AND candidate_state = 'promoted'
        """,
        lineage.candidate_ids + (lineage.memory_id,),
    )
    if candidate_cursor.rowcount != lineage.candidate_count:
        raise MemoryDeletionIntegrityError(
            "Promoted candidate state changed before deletion commit."
        )

    verify_promoted_candidate_lineage_removed(
        connection,
        lineage=lineage,
    )


def verify_promoted_candidate_lineage_removed(
    connection: sqlite3.Connection,
    *,
    lineage: PromotedCandidateLineage,
) -> None:
    """Verify no candidate plaintext, provenance, or linked events remain."""
    if not lineage.candidate_ids:
        return
    placeholders = ",".join("?" for _ in lineage.candidate_ids)
    remaining = {
        "candidates": connection.execute(
            f"""
            SELECT COUNT(*) FROM memory_candidates
            WHERE candidate_id IN ({placeholders})
            """,
            lineage.candidate_ids,
        ).fetchone()[0],
        "candidate_sources": connection.execute(
            f"""
            SELECT COUNT(*) FROM memory_candidate_sources
            WHERE candidate_id IN ({placeholders})
            """,
            lineage.candidate_ids,
        ).fetchone()[0],
        "candidate_events": connection.execute(
            f"""
            SELECT COUNT(*) FROM memory_candidate_events
            WHERE candidate_id IN ({placeholders})
            """,
            lineage.candidate_ids,
        ).fetchone()[0],
    }
    nonzero = {
        name: int(count)
        for name, count in remaining.items()
        if int(count) != 0
    }
    if nonzero:
        raise MemoryDeletionIntegrityError(
            "Promoted candidate lineage remained after memory deletion: "
            + ", ".join(
                f"{name}={count}" for name, count in sorted(nonzero.items())
            )
        )


def _walk_audit_value(value: object) -> None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key).casefold()
            if key in _FORBIDDEN_AUDIT_KEYS:
                raise MemoryDeletionIntegrityError(
                    f"Deletion audit contains forbidden field: {raw_key}"
                )
            _walk_audit_value(nested)
    elif isinstance(value, list):
        for nested in value:
            _walk_audit_value(nested)


def validate_sanitized_deletion_details(details: dict[str, object]) -> None:
    """Validate that deletion audit details contain no protected payload data."""
    _walk_audit_value(details)
    lineage = details.get("promoted_candidate_lineage")
    if not isinstance(lineage, dict):
        raise MemoryDeletionIntegrityError(
            "Deletion audit is missing promoted candidate lineage evidence."
        )
    if lineage.get("integrity_version") != DELETION_INTEGRITY_VERSION:
        raise MemoryDeletionIntegrityError(
            "Deletion audit has an unsupported integrity version."
        )
    if lineage.get("purged") is not True:
        raise MemoryDeletionIntegrityError(
            "Deletion audit does not prove candidate-lineage purge."
        )

    counts = {}
    for name in ("candidate_count", "event_count", "source_count"):
        value = lineage.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise MemoryDeletionIntegrityError(
                f"Deletion audit contains invalid {name}."
            )
        counts[name] = value

    digests = lineage.get("candidate_id_sha256")
    if not isinstance(digests, list) or len(digests) != counts["candidate_count"]:
        raise MemoryDeletionIntegrityError(
            "Deletion audit candidate digest count is inconsistent."
        )
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in digests):
        raise MemoryDeletionIntegrityError(
            "Deletion audit contains an invalid candidate identifier digest."
        )


def parse_and_validate_deletion_details(raw: object) -> dict[str, object]:
    """Parse one deletion event payload and enforce sanitization."""
    try:
        details = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise MemoryDeletionIntegrityError(
            "Deletion event contains invalid JSON."
        ) from exc
    if not isinstance(details, dict):
        raise MemoryDeletionIntegrityError(
            "Deletion event details must be a JSON object."
        )
    validate_sanitized_deletion_details(details)
    return details
