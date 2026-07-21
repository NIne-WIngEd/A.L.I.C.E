"""P2.2 read-only Phase 1 evidence-bridge tests."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from alice_memory.phase1_bridge import (
    Phase1CatalogSchemaError,
    Phase1EvidenceMismatchError,
    Phase1EvidenceNotFoundError,
    load_phase1_chunk_evidence,
    open_phase1_catalog,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_phase1_catalog(
    root: Path,
    *,
    second_chunk: bool = False,
) -> tuple[Path, dict[str, str]]:
    root.mkdir(parents=True)
    text_dir = root / "text"
    text_dir.mkdir()

    first_text = b"verified phase one evidence"
    first_chunk_id = "chunk-1"
    first_chunk_hash = _sha256(first_text)
    first_source_content = "a" * 64
    first_source_text = "b" * 64
    first_normalized = "c" * 64

    (text_dir / f"{first_chunk_id}.txt").write_bytes(first_text)

    catalog = root / "chunks.sqlite3"

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
                2 if second_chunk else 1,
                2 if second_chunk else 1,
                "2026-07-21T00:00:00Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO chunks
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_chunk_id,
                "set-1",
                first_source_content,
                first_source_text,
                first_normalized,
                0,
                0,
                len(first_text),
                len(first_text),
                first_chunk_hash,
                "txt",
                "test-parser",
                0,
                1,
                f"text/{first_chunk_id}.txt",
            ),
        )
        connection.execute(
            """
            INSERT INTO chunk_provenance
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_chunk_id,
                "file-1",
                "private/path.txt",
                "path.txt",
                "primary",
                "txt",
                "private-bucket",
                "2026",
                "",
                "none",
            ),
        )

        if second_chunk:
            second_text = b"second verified phase one evidence"
            second_chunk_id = "chunk-2"
            second_hash = _sha256(second_text)
            (text_dir / f"{second_chunk_id}.txt").write_bytes(second_text)

            connection.execute(
                """
                INSERT INTO chunks
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    second_chunk_id,
                    "set-1",
                    "1" * 64,
                    "2" * 64,
                    "3" * 64,
                    1,
                    0,
                    len(second_text),
                    len(second_text),
                    second_hash,
                    "txt",
                    "test-parser",
                    0,
                    1,
                    f"text/{second_chunk_id}.txt",
                ),
            )
            connection.execute(
                """
                INSERT INTO chunk_provenance
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    second_chunk_id,
                    "file-2",
                    "private/second.txt",
                    "second.txt",
                    "primary",
                    "txt",
                    "private-bucket",
                    "2026",
                    "",
                    "none",
                ),
            )

    return catalog, {
        "chunk_id": first_chunk_id,
        "source_content_sha256": first_source_content,
        "source_text_sha256": first_source_text,
        "chunk_text_sha256": first_chunk_hash,
        "file_id": "file-1",
    }


def test_phase1_catalog_is_opened_read_only(
    tmp_path: Path,
) -> None:
    catalog, _ = _build_phase1_catalog(tmp_path / "chunk-set")

    with open_phase1_catalog(catalog) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                """
                INSERT INTO chunk_sets
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "forbidden",
                    "pilot-v1",
                    "policy",
                    "a",
                    "b",
                    "c",
                    0,
                    0,
                    "2026-07-21T00:00:00Z",
                ),
            )


def test_load_phase1_chunk_returns_identity_without_plaintext(
    tmp_path: Path,
) -> None:
    catalog, expected = _build_phase1_catalog(tmp_path / "chunk-set")

    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id=expected["chunk_id"],
    )

    assert evidence.chunk_id == expected["chunk_id"]
    assert (
        evidence.source_content_sha256
        == expected["source_content_sha256"]
    )
    assert evidence.source_text_sha256 == expected["source_text_sha256"]
    assert evidence.chunk_text_sha256 == expected["chunk_text_sha256"]
    assert evidence.provenance_paths[0].file_id == expected["file_id"]
    assert not hasattr(evidence, "text")
    assert not hasattr(evidence, "original_relative_path")
    assert not hasattr(evidence, "filename")


def test_expected_hashes_and_file_id_are_validated(
    tmp_path: Path,
) -> None:
    catalog, expected = _build_phase1_catalog(tmp_path / "chunk-set")

    evidence = load_phase1_chunk_evidence(
        catalog,
        chunk_id=expected["chunk_id"],
        expected_source_content_sha256=expected[
            "source_content_sha256"
        ],
        expected_source_text_sha256=expected["source_text_sha256"],
        expected_chunk_text_sha256=expected["chunk_text_sha256"],
        expected_file_id=expected["file_id"],
    )

    assert evidence.chunk_id == expected["chunk_id"]


def test_source_hash_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    catalog, expected = _build_phase1_catalog(tmp_path / "chunk-set")

    with pytest.raises(Phase1EvidenceMismatchError):
        load_phase1_chunk_evidence(
            catalog,
            chunk_id=expected["chunk_id"],
            expected_source_content_sha256="0" * 64,
        )


def test_expected_file_id_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    catalog, expected = _build_phase1_catalog(tmp_path / "chunk-set")

    with pytest.raises(Phase1EvidenceMismatchError):
        load_phase1_chunk_evidence(
            catalog,
            chunk_id=expected["chunk_id"],
            expected_file_id="wrong-file",
        )


def test_missing_chunk_is_rejected(
    tmp_path: Path,
) -> None:
    catalog, _ = _build_phase1_catalog(tmp_path / "chunk-set")

    with pytest.raises(Phase1EvidenceNotFoundError):
        load_phase1_chunk_evidence(
            catalog,
            chunk_id="missing-chunk",
        )


def test_chunk_text_tampering_is_detected(
    tmp_path: Path,
) -> None:
    catalog, expected = _build_phase1_catalog(tmp_path / "chunk-set")

    chunk_text = (
        catalog.parent
        / "text"
        / f"{expected['chunk_id']}.txt"
    )
    chunk_text.write_text("tampered", encoding="utf-8")

    from alice_memory.phase1_bridge import Phase1CatalogIntegrityError

    with pytest.raises(Phase1CatalogIntegrityError):
        load_phase1_chunk_evidence(
            catalog,
            chunk_id=expected["chunk_id"],
        )


def test_missing_required_catalog_table_is_rejected(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "bad.sqlite3"

    with sqlite3.connect(catalog) as connection:
        connection.execute(
            "CREATE TABLE chunk_sets(chunk_set_id TEXT PRIMARY KEY)"
        )

    with pytest.raises(Phase1CatalogSchemaError):
        with open_phase1_catalog(catalog):
            pass
