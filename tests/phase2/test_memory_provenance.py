"""P2.2 Phase 2 provenance-attachment tests."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from alice_memory.phase1_bridge import load_phase1_chunk_evidence
from alice_memory.provenance import (
    MemoryNotFoundError,
    attach_phase1_chunk_evidence,
    list_memory_sources,
)
from alice_memory.schema import SCHEMA_VERSION
from alice_memory.store import open_memory_store, transaction


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_phase1_catalog(root: Path) -> Path:
    root.mkdir(parents=True)
    text_dir = root / "text"
    text_dir.mkdir()

    catalog = root / "chunks.sqlite3"

    chunks = (
        ("chunk-1", "file-1", b"first evidence", "a", "b", "c"),
        ("chunk-2", "file-2", b"second evidence", "1", "2", "3"),
    )

    with sqlite3.connect(catalog) as connection:
        connection.executescript(
            """
            CREATE TABLE chunk_sets(
              chunk_set_id TEXT PRIMARY KEY, pilot_name TEXT, policy_id TEXT,
              policy_digest TEXT, pilot_manifest_sha256 TEXT,
              extraction_registry_digest TEXT, chunk_count INTEGER,
              source_count INTEGER, created_at TEXT
            );
            CREATE TABLE chunks(
              chunk_id TEXT PRIMARY KEY, chunk_set_id TEXT,
              source_content_sha256 TEXT, source_text_sha256 TEXT,
              normalized_source_text_sha256 TEXT, chunk_index INTEGER,
              start_char INTEGER, end_char INTEGER, char_count INTEGER,
              chunk_text_sha256 TEXT, family TEXT, parser_id TEXT,
              source_extraction_truncated INTEGER,
              provenance_path_count INTEGER, text_relative_path TEXT
            );
            CREATE TABLE chunk_provenance(
              chunk_id TEXT, file_id TEXT, original_relative_path TEXT,
              filename TEXT, role TEXT, family TEXT, source_bucket TEXT,
              year_hint TEXT, duplicate_control_group TEXT,
              known_contradiction_group TEXT,
              PRIMARY KEY(chunk_id,file_id)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO chunk_sets
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "set-1",
                "pilot-v1",
                "policy",
                "d" * 64,
                "e" * 64,
                "f" * 64,
                2,
                2,
                "2026-07-21T00:00:00Z",
            ),
        )

        for index, (
            chunk_id,
            file_id,
            text,
            source_content_char,
            source_text_char,
            normalized_char,
        ) in enumerate(chunks):
            (text_dir / f"{chunk_id}.txt").write_bytes(text)

            connection.execute(
                """
                INSERT INTO chunks
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    "set-1",
                    source_content_char * 64,
                    source_text_char * 64,
                    normalized_char * 64,
                    index,
                    0,
                    len(text),
                    len(text),
                    _sha256(text),
                    "txt",
                    "test-parser",
                    0,
                    1,
                    f"text/{chunk_id}.txt",
                ),
            )
            connection.execute(
                """
                INSERT INTO chunk_provenance
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    file_id,
                    f"private/{chunk_id}.txt",
                    f"{chunk_id}.txt",
                    "primary",
                    "txt",
                    "private-bucket",
                    "2026",
                    "",
                    "none",
                ),
            )

    return catalog


def _insert_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO memories (
            memory_id,
            schema_version,
            content,
            content_sha256,
            memory_key,
            category,
            knowledge_status,
            confidence,
            data_classification,
            valid_from,
            valid_to,
            time_precision,
            recorded_at,
            verified_at,
            rayan_confirmed,
            validity_state,
            retention_state,
            deletion_state,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory_id,
            SCHEMA_VERSION,
            "Test durable memory",
            "a" * 64,
            None,
            "project",
            "verified_fact",
            1.0,
            "PRIVATE",
            None,
            None,
            None,
            "2026-07-21T00:00:00Z",
            "2026-07-21T00:00:00Z",
            1,
            "current",
            "durable",
            "active",
            "2026-07-21T00:00:00Z",
            "2026-07-21T00:00:00Z",
        ),
    )


def test_attach_phase1_evidence_stores_only_private_safe_identifiers(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    catalog = _create_phase1_catalog(tmp_path / "phase1")
    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-1",
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            _insert_memory(
                connection,
                memory_id="memory-1",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=evidence,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )

        sources = list_memory_sources(
            connection,
            memory_id="memory-1",
        )

        assert len(sources) == 1
        source = sources[0]
        assert source.chunk_id == "chunk-1"
        assert source.file_id == "file-1"
        assert source.source_content_sha256 == "a" * 64
        assert source.source_text_sha256 == "b" * 64
        assert "private/" not in source.source_ref
        assert ".txt" not in source.source_ref


def test_multiple_phase1_chunks_can_support_one_memory(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    catalog = _create_phase1_catalog(tmp_path / "phase1")
    first = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-1",
    )
    second = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-2",
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            _insert_memory(
                connection,
                memory_id="memory-1",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=first,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=second,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )

        sources = list_memory_sources(
            connection,
            memory_id="memory-1",
        )

        assert {source.chunk_id for source in sources} == {
            "chunk-1",
            "chunk-2",
        }


def test_provenance_attachment_is_idempotent(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    catalog = _create_phase1_catalog(tmp_path / "phase1")
    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-1",
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            _insert_memory(
                connection,
                memory_id="memory-1",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=evidence,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=evidence,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_sources
            WHERE memory_id = ?
            """,
            ("memory-1",),
        ).fetchone()[0]

        assert count == 1


def test_missing_memory_rejects_provenance_attachment(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    catalog = _create_phase1_catalog(tmp_path / "phase1")
    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-1",
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with pytest.raises(MemoryNotFoundError):
            attach_phase1_chunk_evidence(
                connection,
                memory_id="missing-memory",
                evidence=evidence,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )


def test_phase1_catalog_row_count_is_unchanged_after_attachment(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()

    catalog = _create_phase1_catalog(tmp_path / "phase1")

    with sqlite3.connect(catalog) as connection:
        before = connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id="chunk-1",
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with transaction(connection):
            _insert_memory(
                connection,
                memory_id="memory-1",
            )
            attach_phase1_chunk_evidence(
                connection,
                memory_id="memory-1",
                evidence=evidence,
                support_relation="supports",
                created_at="2026-07-21T00:00:00Z",
            )

    with sqlite3.connect(catalog) as connection:
        after = connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]

    assert after == before
