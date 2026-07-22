"""Private, derived, rebuildable lexical index for Phase 2 memory.

The authoritative SQLite Memory Core remains the source of truth. This module
creates a separate private FTS5 index under the A.L.I.C.E. vault and verifies
that it matches the current authoritative memory state before search.
"""

from __future__ import annotations

import hashlib
from contextlib import closing
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .retrieval_models import (
    MemoryLexicalIndexError,
    StaleMemoryLexicalIndexError,
)
from .store import (
    default_repository_root,
    validate_private_database_path,
)


MEMORY_LEXICAL_INDEX_RELATIVE_PATH = Path(
    "memory",
    "phase2",
    "indexes",
    "lexical",
    "memory-lexical.sqlite3",
)

_INDEX_VERSION = 1
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class MemoryLexicalIndexManifest:
    index_id: str
    index_version: int
    authoritative_digest: str
    record_count: int
    built_at: str


def memory_lexical_index_path(
    vault_root: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    """Return the canonical private lexical-index path."""
    vault = Path(vault_root).expanduser().resolve(strict=True)
    repository = (
        default_repository_root()
        if repository_root is None
        else Path(repository_root).expanduser().resolve(strict=True)
    )
    path = (
        vault
        / MEMORY_LEXICAL_INDEX_RELATIVE_PATH
    ).resolve(strict=False)
    return validate_private_database_path(
        path,
        repository_root=repository,
    )


def _eligible_rows(
    connection: sqlite3.Connection,
) -> list[sqlite3.Row]:
    """Rows eligible for derived lexical indexing.

    Pending-deletion records and HIGHLY_SENSITIVE records fail closed and are
    excluded. Archived and historical records remain indexable because they
    may be explicitly requested later; runtime retrieval filters them.
    """
    return connection.execute(
        """
        SELECT
            memory_id,
            content,
            content_sha256,
            memory_key,
            category,
            knowledge_status,
            confidence,
            data_classification,
            valid_from,
            valid_to,
            recorded_at,
            validity_state,
            retention_state,
            deletion_state,
            updated_at
        FROM memories
        WHERE deletion_state = 'active'
          AND EXISTS (
              SELECT 1
              FROM memory_sources AS source
              WHERE source.memory_id = memories.memory_id
          )
          AND data_classification IN (
              'PUBLIC',
              'INTERNAL',
              'PRIVATE'
          )
        ORDER BY memory_id
        """
    ).fetchall()


def authoritative_retrieval_digest(
    connection: sqlite3.Connection,
) -> tuple[str, int]:
    """Digest the authoritative records that may appear in the lexical index."""
    records = []
    for row in _eligible_rows(connection):
        records.append(
            {
                "memory_id": str(row["memory_id"]),
                "content_sha256": str(row["content_sha256"]),
                "memory_key": row["memory_key"],
                "category": str(row["category"]),
                "knowledge_status": str(row["knowledge_status"]),
                "confidence": float(row["confidence"]),
                "data_classification": str(row["data_classification"]),
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "recorded_at": str(row["recorded_at"]),
                "validity_state": str(row["validity_state"]),
                "retention_state": str(row["retention_state"]),
                "deletion_state": str(row["deletion_state"]),
                "updated_at": str(row["updated_at"]),
            }
        )

    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(records)


def _require_fts5(
    connection: sqlite3.Connection,
) -> None:
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE temp.__alice_fts5_check USING fts5(value)"
        )
        connection.execute(
            "DROP TABLE temp.__alice_fts5_check"
        )
    except sqlite3.OperationalError as exc:
        raise MemoryLexicalIndexError(
            "SQLite FTS5 support is required for Phase 2 lexical retrieval."
        ) from exc


def _index_id(
    *,
    authoritative_digest: str,
    record_count: int,
) -> str:
    payload = (
        f"memory-lexical-v{_INDEX_VERSION}|"
        f"{authoritative_digest}|{record_count}"
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:32]


def build_memory_lexical_index(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    built_at: str,
) -> MemoryLexicalIndexManifest:
    """Rebuild the private lexical index atomically from authoritative memory."""
    index_path = memory_lexical_index_path(
        vault_root,
        repository_root=repository_root,
    )
    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    authoritative_digest, record_count = (
        authoritative_retrieval_digest(connection)
    )
    index_id = _index_id(
        authoritative_digest=authoritative_digest,
        record_count=record_count,
    )

    temp_path = index_path.with_name(
        f"{index_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with closing(sqlite3.connect(temp_path)) as index:
            index.row_factory = sqlite3.Row
            _require_fts5(index)
            index.execute(
                """
                CREATE TABLE index_manifest (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    index_id TEXT NOT NULL,
                    index_version INTEGER NOT NULL,
                    authoritative_digest TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    built_at TEXT NOT NULL
                )
                """
            )
            index.execute(
                """
                CREATE TABLE indexed_memories (
                    memory_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    data_classification TEXT NOT NULL
                )
                """
            )
            index.execute(
                """
                CREATE VIRTUAL TABLE memory_fts USING fts5(
                    memory_id UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                )
                """
            )

            rows = _eligible_rows(connection)
            for row in rows:
                memory_id = str(row["memory_id"])
                index.execute(
                    """
                    INSERT INTO indexed_memories (
                        memory_id,
                        content_sha256,
                        data_classification
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        memory_id,
                        str(row["content_sha256"]),
                        str(row["data_classification"]),
                    ),
                )
                index.execute(
                    """
                    INSERT INTO memory_fts (
                        memory_id,
                        content
                    )
                    VALUES (?, ?)
                    """,
                    (
                        memory_id,
                        str(row["content"]),
                    ),
                )

            index.execute(
                """
                INSERT INTO index_manifest (
                    singleton,
                    index_id,
                    index_version,
                    authoritative_digest,
                    record_count,
                    built_at
                )
                VALUES (1, ?, ?, ?, ?, ?)
                """,
                (
                    index_id,
                    _INDEX_VERSION,
                    authoritative_digest,
                    record_count,
                    built_at,
                ),
            )
            index.commit()

        os.replace(
            temp_path,
            index_path,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return MemoryLexicalIndexManifest(
        index_id=index_id,
        index_version=_INDEX_VERSION,
        authoritative_digest=authoritative_digest,
        record_count=record_count,
        built_at=built_at,
    )


def load_memory_lexical_index_manifest(
    index_path: str | Path,
) -> MemoryLexicalIndexManifest:
    path = Path(index_path)
    if not path.exists():
        raise MemoryLexicalIndexError(
            f"Memory lexical index does not exist: {path}"
        )

    with closing(
        sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
    ) as index:
        index.row_factory = sqlite3.Row
        row = index.execute(
            """
            SELECT
                index_id,
                index_version,
                authoritative_digest,
                record_count,
                built_at
            FROM index_manifest
            WHERE singleton = 1
            """
        ).fetchone()

        if row is None:
            raise MemoryLexicalIndexError(
                "Memory lexical index manifest is missing."
            )

        return MemoryLexicalIndexManifest(
            index_id=str(row["index_id"]),
            index_version=int(row["index_version"]),
            authoritative_digest=str(row["authoritative_digest"]),
            record_count=int(row["record_count"]),
            built_at=str(row["built_at"]),
        )


def verify_memory_lexical_index(
    connection: sqlite3.Connection,
    index_path: str | Path,
) -> MemoryLexicalIndexManifest:
    """Fail closed if the derived index is stale or malformed."""
    manifest = load_memory_lexical_index_manifest(
        index_path
    )

    if manifest.index_version != _INDEX_VERSION:
        raise MemoryLexicalIndexError(
            "Unsupported memory lexical index version: "
            f"{manifest.index_version}"
        )

    digest, record_count = authoritative_retrieval_digest(
        connection
    )
    if (
        manifest.authoritative_digest != digest
        or manifest.record_count != record_count
    ):
        raise StaleMemoryLexicalIndexError(
            "Memory lexical index is stale relative to the authoritative "
            "Memory Core. Rebuild the index before retrieval."
        )

    with closing(
        sqlite3.connect(
            f"{Path(index_path).resolve().as_uri()}?mode=ro",
            uri=True,
        )
    ) as index:
        indexed_count = index.execute(
            "SELECT COUNT(*) FROM indexed_memories"
        ).fetchone()[0]
        fts_count = index.execute(
            "SELECT COUNT(*) FROM memory_fts"
        ).fetchone()[0]

    if (
        indexed_count != manifest.record_count
        or fts_count != manifest.record_count
    ):
        raise MemoryLexicalIndexError(
            "Memory lexical index row counts do not match its manifest."
        )

    return manifest


def lexical_query_expression(
    query: str,
) -> str:
    """Convert user text to a conservative FTS5 AND expression.

    P2.5 lexical retrieval intentionally requires every distinct query token
    to match. This avoids broad shared terms from surfacing unrelated memories.
    Higher-recall semantic or hybrid fallback belongs in a later retrieval
    layer and must preserve the same lifecycle and authorization filters.
    """
    tokens = [
        token
        for token in _TOKEN_PATTERN.findall(
            query.casefold()
        )
        if token
    ]
    if not tokens:
        raise MemoryLexicalIndexError(
            "Lexical query must contain at least one searchable token."
        )

    unique_tokens = list(dict.fromkeys(tokens))
    return " AND ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"'
        for token in unique_tokens
    )


def search_memory_lexical_candidates(
    index_path: str | Path,
    *,
    query: str,
    limit: int,
) -> list[tuple[str, float]]:
    """Return only candidate IDs and lexical scores from the derived index."""
    if limit <= 0:
        return []

    expression = lexical_query_expression(
        query
    )

    with closing(
        sqlite3.connect(
            f"{Path(index_path).resolve().as_uri()}?mode=ro",
            uri=True,
        )
    ) as index:
        rows = index.execute(
            """
            SELECT
                memory_id,
                bm25(memory_fts) AS rank
            FROM memory_fts
            WHERE memory_fts MATCH ?
            ORDER BY rank, memory_id
            LIMIT ?
            """,
            (
                expression,
                limit,
            ),
        ).fetchall()

    return [
        (
            str(memory_id),
            -float(rank),
        )
        for memory_id, rank in rows
    ]
