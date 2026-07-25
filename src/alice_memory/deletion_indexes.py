"""Derived-index purge and rebuild guarantees for A.L.I.C.E. P2.8b.

The authoritative Memory Core is the source of truth. A completed deletion
therefore makes every older retrieval index stale immediately. This module
adds the filesystem cleanup and deterministic rebuild workflow needed to prove
that deleted memories remain absent after lexical and semantic indexes are
destroyed and recreated from authoritative state.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deletion import (
    MemoryDeletionAuthorization,
    MemoryDeletionResult,
    MemoryDeletionStateError,
    MemoryTombstoneNotFoundError,
    delete_memory,
    load_memory_deletion,
)
from .lexical_index import (
    MemoryLexicalIndexManifest,
    build_memory_lexical_index,
    memory_lexical_index_path,
    verify_memory_lexical_index,
)
from .semantic_index import (
    MemorySemanticIndexManifest,
    build_memory_semantic_index,
    memory_semantic_index_root,
    verify_memory_semantic_index,
)


class MemoryDeletionIndexError(RuntimeError):
    """Base error for deleted-memory retrieval-index maintenance."""


class MemoryDeletionIndexStateError(MemoryDeletionIndexError):
    """Raised when index cleanup lacks a valid completed deletion."""


class MemoryDeletionIndexVerificationError(MemoryDeletionIndexError):
    """Raised when rebuilt indexes still reference a deleted memory."""


@dataclass(frozen=True)
class MemoryDeletionIndexPurgeResult:
    """Metadata-safe result of destroying current derived retrieval indexes."""

    deleted_memory_id: str
    lexical_root: Path
    semantic_root: Path
    lexical_root_removed: bool
    semantic_root_removed: bool


@dataclass(frozen=True)
class MemoryDeletionIndexVerification:
    """Proof that fresh indexes match authoritative state after deletion."""

    deleted_memory_id: str
    lexical_manifest: MemoryLexicalIndexManifest
    semantic_manifest: MemorySemanticIndexManifest
    lexical_index_path: Path
    semantic_index_path: Path


@dataclass(frozen=True)
class MemoryDeletionIndexRebuildResult:
    """Combined purge, rebuild, and absence-verification result."""

    purge: MemoryDeletionIndexPurgeResult
    verification: MemoryDeletionIndexVerification


@dataclass(frozen=True)
class MemoryDeletionWithIndexPurgeResult:
    """Completed authoritative deletion plus derived-index destruction."""

    deletion: MemoryDeletionResult
    purge: MemoryDeletionIndexPurgeResult


def _completed_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryDeletionResult:
    try:
        return load_memory_deletion(
            connection,
            memory_id=memory_id,
        )
    except (MemoryTombstoneNotFoundError, MemoryDeletionStateError) as exc:
        raise MemoryDeletionIndexStateError(
            "Derived-index deletion cleanup requires an intact completed "
            "memory deletion and tombstone."
        ) from exc


def _require_under_vault(path: Path, *, vault_root: str | Path) -> Path:
    vault = Path(vault_root).expanduser().resolve(strict=True)
    resolved = path.expanduser().resolve(strict=False)
    if resolved == vault or vault not in resolved.parents:
        raise MemoryDeletionIndexStateError(
            "Refusing to purge a derived-index path outside the private vault."
        )
    return resolved


def _remove_path(path: Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def purge_deleted_memory_indexes(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    memory_id: str,
    repository_root: str | Path | None = None,
) -> MemoryDeletionIndexPurgeResult:
    """Destroy all current lexical and semantic index artifacts after deletion.

    The entire derived index roots are removed rather than attempting an
    in-place row/vector edit. This is conservative and prevents immutable old
    semantic index generations from retaining deleted-memory mappings.
    """
    deletion = _completed_deletion(
        connection,
        memory_id=memory_id,
    )
    if deletion.tombstone.deleted_memory_id != memory_id:
        raise MemoryDeletionIndexStateError(
            "Completed deletion does not match the requested memory."
        )

    lexical_path = memory_lexical_index_path(
        vault_root,
        repository_root=repository_root,
    )
    lexical_root = _require_under_vault(
        lexical_path.parent,
        vault_root=vault_root,
    )
    semantic_root = _require_under_vault(
        memory_semantic_index_root(
            vault_root,
            repository_root=repository_root,
        ),
        vault_root=vault_root,
    )

    lexical_removed = _remove_path(lexical_root)
    semantic_removed = _remove_path(semantic_root)

    if lexical_root.exists() or semantic_root.exists():
        raise MemoryDeletionIndexStateError(
            "Derived retrieval indexes were not fully removed."
        )

    return MemoryDeletionIndexPurgeResult(
        deleted_memory_id=memory_id,
        lexical_root=lexical_root,
        semantic_root=semantic_root,
        lexical_root_removed=lexical_removed,
        semantic_root_removed=semantic_removed,
    )


def _lexical_contains_memory(
    index_path: Path,
    *,
    memory_id: str,
) -> bool:
    with closing(
        sqlite3.connect(
            f"{index_path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
    ) as index:
        metadata_row = index.execute(
            "SELECT 1 FROM indexed_memories WHERE memory_id = ? LIMIT 1",
            (memory_id,),
        ).fetchone()
        fts_row = index.execute(
            "SELECT 1 FROM memory_fts WHERE memory_id = ? LIMIT 1",
            (memory_id,),
        ).fetchone()
    return metadata_row is not None or fts_row is not None


def _semantic_contains_memory(
    index_path: Path,
    *,
    memory_id: str,
) -> bool:
    map_path = index_path / "memory-map.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in map_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MemoryDeletionIndexVerificationError(
            "Semantic memory map is missing or invalid."
        ) from exc
    return any(
        str(row.get("memory_id", "")) == memory_id
        for row in rows
        if isinstance(row, dict)
    )


def verify_deleted_memory_absent_from_indexes(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    memory_id: str,
    repository_root: str | Path | None = None,
) -> MemoryDeletionIndexVerification:
    """Verify fresh lexical and semantic indexes contain no deleted-memory ID."""
    _completed_deletion(
        connection,
        memory_id=memory_id,
    )

    lexical_path = memory_lexical_index_path(
        vault_root,
        repository_root=repository_root,
    )
    lexical_manifest = verify_memory_lexical_index(
        connection,
        lexical_path,
    )
    semantic_manifest, semantic_path = verify_memory_semantic_index(
        connection,
        vault_root,
        repository_root=repository_root,
    )

    if (
        lexical_manifest.authoritative_digest
        != semantic_manifest.authoritative_digest
        or lexical_manifest.record_count != semantic_manifest.record_count
    ):
        raise MemoryDeletionIndexVerificationError(
            "Lexical and semantic indexes do not represent the same "
            "authoritative memory state."
        )

    if _lexical_contains_memory(lexical_path, memory_id=memory_id):
        raise MemoryDeletionIndexVerificationError(
            "Rebuilt lexical index still references the deleted memory."
        )
    if _semantic_contains_memory(semantic_path, memory_id=memory_id):
        raise MemoryDeletionIndexVerificationError(
            "Rebuilt semantic index still references the deleted memory."
        )

    return MemoryDeletionIndexVerification(
        deleted_memory_id=memory_id,
        lexical_manifest=lexical_manifest,
        semantic_manifest=semantic_manifest,
        lexical_index_path=lexical_path,
        semantic_index_path=semantic_path,
    )


def rebuild_indexes_after_memory_deletion(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    memory_id: str,
    model: Any,
    built_at: str,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
) -> MemoryDeletionIndexRebuildResult:
    """Destroy all old indexes, rebuild them, and prove deletion persistence."""
    purge = purge_deleted_memory_indexes(
        connection,
        vault_root,
        memory_id=memory_id,
        repository_root=repository_root,
    )

    build_memory_lexical_index(
        connection,
        vault_root,
        repository_root=repository_root,
        built_at=built_at,
    )
    build_memory_semantic_index(
        connection,
        vault_root,
        model=model,
        policy_path=policy_path,
        repository_root=repository_root,
        built_at=built_at,
    )

    verification = verify_deleted_memory_absent_from_indexes(
        connection,
        vault_root,
        memory_id=memory_id,
        repository_root=repository_root,
    )
    return MemoryDeletionIndexRebuildResult(
        purge=purge,
        verification=verification,
    )


def delete_memory_with_index_purge(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    memory_id: str,
    authorization: MemoryDeletionAuthorization,
    deleted_at: str,
    repository_root: str | Path | None = None,
) -> MemoryDeletionWithIndexPurgeResult:
    """Delete authoritative memory, then destroy all derived retrieval indexes.

    If filesystem cleanup fails, the authoritative deletion remains committed.
    Existing indexes still fail closed because their authoritative digest is
    stale. The purge operation is idempotent and can be retried safely.
    """
    deletion = delete_memory(
        connection,
        memory_id=memory_id,
        authorization=authorization,
        deleted_at=deleted_at,
    )
    purge = purge_deleted_memory_indexes(
        connection,
        vault_root,
        memory_id=memory_id,
        repository_root=repository_root,
    )
    return MemoryDeletionWithIndexPurgeResult(
        deletion=deletion,
        purge=purge,
    )
