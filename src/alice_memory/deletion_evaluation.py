"""Deterministic deletion-security benchmark for A.L.I.C.E. P2.8e.

The benchmark verifies the final deletion invariant against authoritative state,
sanitized audit evidence, protected payload storage, candidate lineage, and fresh
lexical and semantic indexes. It does not delete records itself; callers must
complete deletion and rebuild indexes before running the gate.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .deletion import load_memory_deletion
from .deletion_indexes import MemoryDeletionIndexVerificationError
from .lexical_index import memory_lexical_index_path, verify_memory_lexical_index
from .semantic_index import verify_memory_semantic_index
from .sensitive_deletion import (
    SENSITIVE_MEMORY_DELETION_SCOPE,
    load_sensitive_memory_deletion,
)

DELETION_BENCHMARK_VERSION = "p2.8e-v1"


class MemoryDeletionBenchmarkError(RuntimeError):
    """Raised when any final deletion benchmark gate fails closed."""


@dataclass(frozen=True)
class MemoryDeletionBenchmarkReport:
    benchmark_version: str
    passed: bool
    deleted_memory_ids: tuple[str, ...]
    retained_memory_ids: tuple[str, ...]
    gate_names: tuple[str, ...]
    authoritative_digest: str
    indexed_record_count: int


def _normalized_ids(values, *, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise MemoryDeletionBenchmarkError(f"{field_name} cannot be empty.")
    if any(not value for value in result):
        raise MemoryDeletionBenchmarkError(
            f"{field_name} contains an empty memory identifier."
        )
    if len(set(result)) != len(result):
        raise MemoryDeletionBenchmarkError(
            f"{field_name} contains duplicate memory identifiers."
        )
    return tuple(sorted(result))


def _lexical_ids(index_path: Path) -> tuple[set[str], set[str]]:
    try:
        with closing(
            sqlite3.connect(
                f"{index_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        ) as index:
            metadata = {
                str(row[0])
                for row in index.execute(
                    "SELECT memory_id FROM indexed_memories"
                ).fetchall()
            }
            plaintext = {
                str(row[0])
                for row in index.execute(
                    "SELECT memory_id FROM memory_fts"
                ).fetchall()
            }
    except sqlite3.Error as exc:
        raise MemoryDeletionBenchmarkError(
            "Lexical benchmark index is missing or unreadable."
        ) from exc
    return metadata, plaintext


def _semantic_ids(index_path: Path) -> set[str]:
    map_path = index_path / "memory-map.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in map_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MemoryDeletionBenchmarkError(
            "Semantic benchmark memory map is missing or invalid."
        ) from exc
    if any(not isinstance(row, dict) for row in rows):
        raise MemoryDeletionBenchmarkError(
            "Semantic benchmark memory map contains a non-object row."
        )
    memory_ids = [str(row.get("memory_id", "")) for row in rows]
    if any(not memory_id for memory_id in memory_ids):
        raise MemoryDeletionBenchmarkError(
            "Semantic benchmark memory map contains an empty memory ID."
        )
    if len(set(memory_ids)) != len(memory_ids):
        raise MemoryDeletionBenchmarkError(
            "Semantic benchmark memory map contains duplicate memory IDs."
        )
    return set(memory_ids)


def run_memory_deletion_benchmark(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    deleted_memory_ids,
    retained_memory_ids,
    repository_root: str | Path | None = None,
) -> MemoryDeletionBenchmarkReport:
    """Run final deletion gates against rebuilt authoritative indexes."""
    deleted = _normalized_ids(
        deleted_memory_ids,
        field_name="deleted_memory_ids",
    )
    retained = _normalized_ids(
        retained_memory_ids,
        field_name="retained_memory_ids",
    )
    if set(deleted).intersection(retained):
        raise MemoryDeletionBenchmarkError(
            "Deleted and retained benchmark sets must be disjoint."
        )

    lexical_path = memory_lexical_index_path(
        vault_root,
        repository_root=repository_root,
    )
    try:
        lexical_manifest = verify_memory_lexical_index(
            connection,
            lexical_path,
        )
        semantic_manifest, semantic_path = verify_memory_semantic_index(
            connection,
            vault_root,
            repository_root=repository_root,
        )
    except Exception as exc:
        raise MemoryDeletionBenchmarkError(
            "Deletion benchmark requires fresh verified retrieval indexes."
        ) from exc

    if (
        lexical_manifest.authoritative_digest
        != semantic_manifest.authoritative_digest
        or lexical_manifest.record_count != semantic_manifest.record_count
    ):
        raise MemoryDeletionBenchmarkError(
            "Lexical and semantic benchmark indexes disagree."
        )

    lexical_metadata, lexical_plaintext = _lexical_ids(lexical_path)
    semantic_ids = _semantic_ids(semantic_path)
    if lexical_metadata != lexical_plaintext or lexical_metadata != semantic_ids:
        raise MemoryDeletionBenchmarkError(
            "Rebuilt retrieval indexes contain inconsistent memory IDs."
        )

    for memory_id in deleted:
        deletion = load_memory_deletion(connection, memory_id=memory_id)
        if deletion.tombstone.deletion_scope == SENSITIVE_MEMORY_DELETION_SCOPE:
            load_sensitive_memory_deletion(connection, memory_id=memory_id)
        if connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone() is not None:
            raise MemoryDeletionBenchmarkError(
                "Deleted memory remains in the authoritative store."
            )
        if connection.execute(
            "SELECT 1 FROM memory_sensitive_payloads WHERE memory_id = ?",
            (memory_id,),
        ).fetchone() is not None:
            raise MemoryDeletionBenchmarkError(
                "Deleted sensitive payload remains in protected storage."
            )
        if connection.execute(
            "SELECT 1 FROM memory_candidates WHERE promoted_memory_id = ?",
            (memory_id,),
        ).fetchone() is not None:
            raise MemoryDeletionBenchmarkError(
                "Deleted memory retains promoted candidate lineage."
            )
        if memory_id in lexical_metadata or memory_id in semantic_ids:
            raise MemoryDeletionIndexVerificationError(
                "Deleted memory remains in a rebuilt retrieval index."
            )

    for memory_id in retained:
        row = connection.execute(
            """
            SELECT data_classification, deletion_state, retention_state
            FROM memories
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        if row is None:
            raise MemoryDeletionBenchmarkError(
                "Retained benchmark memory is missing from authoritative state."
            )
        if (
            str(row["deletion_state"]) != "active"
            or str(row["retention_state"]) == "archived"
            or str(row["data_classification"]) == "HIGHLY_SENSITIVE"
        ):
            raise MemoryDeletionBenchmarkError(
                "Retained benchmark memory is not eligible for ordinary retrieval."
            )
        if memory_id not in lexical_metadata or memory_id not in semantic_ids:
            raise MemoryDeletionBenchmarkError(
                "Retained benchmark memory is missing from rebuilt indexes."
            )

    return MemoryDeletionBenchmarkReport(
        benchmark_version=DELETION_BENCHMARK_VERSION,
        passed=True,
        deleted_memory_ids=deleted,
        retained_memory_ids=retained,
        gate_names=(
            "completed_deletion_integrity",
            "authoritative_absence",
            "protected_payload_absence",
            "candidate_lineage_absence",
            "sanitized_audit_integrity",
            "rebuilt_index_absence",
            "retained_memory_presence",
        ),
        authoritative_digest=lexical_manifest.authoritative_digest,
        indexed_record_count=lexical_manifest.record_count,
    )
