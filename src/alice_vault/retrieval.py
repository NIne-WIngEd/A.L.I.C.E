from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

POLICY_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "in", "into", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class RetrievalPolicy:
    policy_id: str
    tokenizer: str
    prefix_indexes: tuple[int, ...]
    query_mode: str
    minimum_token_length: int
    bm25_weights: dict[str, float]
    snippet_tokens: int
    default_limit: int
    candidate_multiplier: int
    max_chunks_per_source: int
    collapse_adjacent_chunks: bool
    include_truncated_by_default: bool
    lexical_benchmark_cases: int
    lexical_benchmark_min_term_length: int
    lexical_benchmark_max_document_frequency: int
    digest: str
    source_path: Path


@dataclass(frozen=True)
class SearchFilters:
    families: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    source_buckets: tuple[str, ...] = ()
    contradiction_labels: tuple[str, ...] = ()
    include_truncated: bool = True


def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "retrieval_policy.json"


def load_policy(path: Path | None = None) -> RetrievalPolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("retrieval_policy_schema_version", -1)) != POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported retrieval-policy schema version")
    policy = RetrievalPolicy(
        policy_id=str(data["policy_id"]),
        tokenizer=str(data["tokenizer"]),
        prefix_indexes=tuple(int(item) for item in data["prefix_indexes"]),
        query_mode=str(data["query_mode"]),
        minimum_token_length=int(data["minimum_token_length"]),
        bm25_weights={
            str(key): float(value)
            for key, value in dict(data["bm25_weights"]).items()
        },
        snippet_tokens=int(data["snippet_tokens"]),
        default_limit=int(data["default_limit"]),
        candidate_multiplier=int(data["candidate_multiplier"]),
        max_chunks_per_source=int(data["max_chunks_per_source"]),
        collapse_adjacent_chunks=bool(data["collapse_adjacent_chunks"]),
        include_truncated_by_default=bool(data["include_truncated_by_default"]),
        lexical_benchmark_cases=int(data["lexical_benchmark_cases"]),
        lexical_benchmark_min_term_length=int(
            data["lexical_benchmark_min_term_length"]
        ),
        lexical_benchmark_max_document_frequency=int(
            data["lexical_benchmark_max_document_frequency"]
        ),
        digest=sha256_bytes(canonical_json(data)),
        source_path=source,
    )
    if policy.query_mode != "and_then_or_fallback":
        raise ValueError("Unsupported query mode")
    if not policy.prefix_indexes or any(item < 1 for item in policy.prefix_indexes):
        raise ValueError("Invalid prefix indexes")
    if not 1 <= policy.snippet_tokens <= 64:
        raise ValueError("snippet_tokens must be between 1 and 64")
    if set(policy.bm25_weights) != {"chunk_id", "body", "filename", "source_path"}:
        raise ValueError("BM25 weights are incomplete")
    return policy


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            output.append(item)
    return output


def locate_chunk_set(
    *,
    vault_root: Path,
    pilot_name: str,
    chunk_set_id: str | None = None,
) -> Path:
    chunks_root = vault_root / "derived" / pilot_name / "chunks"
    if not chunks_root.is_dir():
        raise FileNotFoundError(f"Chunk root not found: {chunks_root}")
    if chunk_set_id:
        target = chunks_root / chunk_set_id
        if not (target / "chunk-manifest.json").is_file():
            raise FileNotFoundError(f"Chunk set not found: {chunk_set_id}")
        return target.resolve(strict=True)
    candidates = [
        path
        for path in chunks_root.iterdir()
        if path.is_dir() and (path / "chunk-manifest.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError("No complete chunk set was found")
    candidates.sort(
        key=lambda path: (
            (path / "chunk-manifest.json").stat().st_mtime_ns,
            path.name,
        ),
        reverse=True,
    )
    return candidates[0].resolve(strict=True)


def load_chunk_catalog(
    chunk_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = chunk_root / "chunk-manifest.json"
    records_path = chunk_root / "chunk-records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = _read_jsonl(records_path)
    if manifest.get("chunk_records_sha256") != sha256_file(records_path):
        raise ValueError("Chunk-records digest mismatch")
    if int(manifest.get("chunk_count", -1)) != len(records):
        raise ValueError("Chunk count mismatch")
    if len({record["chunk_id"] for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs")
    return manifest, records


def _require_fts5(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(value)")
        connection.execute("DROP TABLE temp.fts5_probe")
    except sqlite3.OperationalError as exc:
        raise RuntimeError("This Python SQLite build does not include FTS5") from exc


def index_id_for(
    *,
    chunk_set_id: str,
    chunk_manifest_fingerprint: str,
    policy_digest: str,
) -> str:
    material = (
        "alice-retrieval-index-v1\0"
        f"{chunk_set_id}\0{chunk_manifest_fingerprint}\0{policy_digest}"
    )
    return sha256_bytes(material.encode("utf-8"))[:32]


def _provenance_text(provenance: list[dict[str, Any]], key: str) -> str:
    return "\n".join(
        sorted(
            {
                str(item.get(key, "")).strip()
                for item in provenance
                if str(item.get(key, "")).strip()
            }
        )
    )


def _record_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["chunk_id"])):
        material = {
            "chunk_id": record["chunk_id"],
            "source_content_sha256": record["source_content_sha256"],
            "chunk_index": int(record["chunk_index"]),
            "chunk_text_sha256": record["chunk_text_sha256"],
            "family": record["family"],
            "source_extraction_truncated": bool(
                record["source_extraction_truncated"]
            ),
            "provenance": record.get("provenance", []),
        }
        digest.update(canonical_json(material))
        digest.update(b"\n")
    return digest.hexdigest()


def _create_database(
    database: Path,
    *,
    chunk_root: Path,
    chunk_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    policy: RetrievalPolicy,
    index_id: str,
) -> dict[str, int]:
    connection = sqlite3.connect(database)
    _require_fts5(connection)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    tokenizer = policy.tokenizer.replace("'", "''")
    prefixes = " ".join(str(item) for item in policy.prefix_indexes)
    connection.executescript(
        f"""
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) STRICT;

        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            source_content_sha256 TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            family TEXT NOT NULL,
            source_extraction_truncated INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            chunk_text_sha256 TEXT NOT NULL,
            body TEXT NOT NULL
        ) STRICT;

        CREATE TABLE provenance (
            chunk_rowid INTEGER NOT NULL
                REFERENCES chunks(rowid) ON DELETE CASCADE,
            file_id TEXT NOT NULL,
            original_relative_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            role TEXT NOT NULL,
            source_bucket TEXT NOT NULL,
            year_hint TEXT NOT NULL,
            duplicate_control_group TEXT NOT NULL,
            known_contradiction_group TEXT NOT NULL,
            PRIMARY KEY (chunk_rowid, file_id)
        ) STRICT;

        CREATE INDEX idx_chunks_source ON chunks(source_content_sha256);
        CREATE INDEX idx_chunks_family ON chunks(family);
        CREATE INDEX idx_chunks_truncated ON chunks(source_extraction_truncated);
        CREATE INDEX idx_provenance_bucket ON provenance(source_bucket);
        CREATE INDEX idx_provenance_year ON provenance(year_hint);
        CREATE INDEX idx_provenance_contradiction
            ON provenance(known_contradiction_group);

        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            chunk_id UNINDEXED,
            body,
            filename,
            source_path,
            tokenize='{tokenizer}',
            prefix='{prefixes}'
        );
        """
    )
    meta = {
        "retrieval_index_schema_version": str(INDEX_SCHEMA_VERSION),
        "index_id": index_id,
        "policy_id": policy.policy_id,
        "policy_digest": policy.digest,
        "chunk_set_id": str(chunk_manifest["chunk_set_id"]),
        "chunk_manifest_fingerprint": str(
            chunk_manifest["manifest_fingerprint"]
        ),
        "chunk_records_sha256": str(chunk_manifest["chunk_records_sha256"]),
        "created_at": utc_now(),
    }
    connection.executemany(
        "INSERT INTO index_meta(key, value) VALUES (?, ?)",
        sorted(meta.items()),
    )

    provenance_rows = 0
    source_hashes: set[str] = set()
    for rowid, record in enumerate(records, start=1):
        text_path = chunk_root / "text" / f"{record['chunk_id']}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing chunk text: {record['chunk_id']}")
        body = text_path.read_text(encoding="utf-8")
        if sha256_bytes(body.encode("utf-8")) != record["chunk_text_sha256"]:
            raise ValueError(f"Chunk text hash mismatch: {record['chunk_id']}")
        source_hashes.add(str(record["source_content_sha256"]))
        provenance = list(record.get("provenance", []))
        connection.execute(
            """
            INSERT INTO chunks(
                rowid, chunk_id, source_content_sha256, chunk_index,
                family, source_extraction_truncated, char_count,
                chunk_text_sha256, body
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rowid,
                record["chunk_id"],
                record["source_content_sha256"],
                int(record["chunk_index"]),
                record["family"],
                int(bool(record["source_extraction_truncated"])),
                int(record["char_count"]),
                record["chunk_text_sha256"],
                body,
            ),
        )
        connection.execute(
            """
            INSERT INTO chunk_fts(rowid, chunk_id, body, filename, source_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                rowid,
                record["chunk_id"],
                body,
                _provenance_text(provenance, "filename"),
                _provenance_text(provenance, "original_relative_path"),
            ),
        )
        for item in provenance:
            connection.execute(
                """
                INSERT INTO provenance(
                    chunk_rowid, file_id, original_relative_path, filename,
                    role, source_bucket, year_hint, duplicate_control_group,
                    known_contradiction_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    str(item["file_id"]),
                    str(item["original_relative_path"]),
                    str(item["filename"]),
                    str(item.get("role", "")),
                    str(item.get("source_bucket", "")),
                    str(item.get("year_hint", "")),
                    str(item.get("duplicate_control_group", "")),
                    str(item.get("known_contradiction_group", "")),
                ),
            )
            provenance_rows += 1
    connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    return {
        "chunk_count": len(records),
        "source_count": len(source_hashes),
        "provenance_row_count": provenance_rows,
    }


def build_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    chunk_set_id: str | None = None,
    policy_path: Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
        chunk_set_id=chunk_set_id,
    )
    chunk_manifest, records = load_chunk_catalog(chunk_root)
    index_id = index_id_for(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(chunk_manifest["manifest_fingerprint"]),
        policy_digest=policy.digest,
    )
    output_root = vault_root / "derived" / pilot_name / "retrieval" / index_id
    exports = vault_root / "manifests" / "exports"
    private_root = vault_root / "manifests" / "retrieval" / pilot_name
    temporary_root = vault_root / "temporary"
    for path in (exports, private_root, temporary_root):
        path.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    record_digest = _record_digest(records)

    if output_root.exists():
        manifest_path = output_root / "retrieval-manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("index_id") == index_id
                and existing.get("index_record_digest") == record_digest
                and existing.get("policy_digest") == policy.digest
            ):
                result = {
                    **existing,
                    "successful_index_build": False,
                    "resumed_existing_index": True,
                    "output_root": str(output_root),
                    "manifest_path": str(manifest_path),
                    "database_path": str(output_root / "retrieval.sqlite3"),
                }
                summary_path = exports / f"retrieval-index-summary-{run_id}.json"
                atomic_json(summary_path, result)
                result["summary_path"] = str(summary_path)
                return result
        if not replace:
            raise FileExistsError(
                "A conflicting retrieval index already exists; investigate "
                "before using --replace"
            )
        shutil.rmtree(output_root)

    started_at = utc_now()
    with tempfile.TemporaryDirectory(
        prefix=f"alice-retrieval-{index_id}-",
        dir=temporary_root,
    ) as temp:
        stage = Path(temp) / index_id
        stage.mkdir()
        database = stage / "retrieval.sqlite3"
        counts = _create_database(
            database,
            chunk_root=chunk_root,
            chunk_manifest=chunk_manifest,
            records=records,
            policy=policy,
            index_id=index_id,
        )
        manifest = {
            "retrieval_index_schema_version": INDEX_SCHEMA_VERSION,
            "run_id": run_id,
            "index_id": index_id,
            "pilot_name": pilot_name,
            "created_at": utc_now(),
            "started_at": started_at,
            "policy_id": policy.policy_id,
            "policy_digest": policy.digest,
            "chunk_set_id": str(chunk_manifest["chunk_set_id"]),
            "chunk_manifest_fingerprint": str(
                chunk_manifest["manifest_fingerprint"]
            ),
            "chunk_records_sha256": str(
                chunk_manifest["chunk_records_sha256"]
            ),
            "index_record_digest": record_digest,
            "chunk_count": counts["chunk_count"],
            "source_count": counts["source_count"],
            "provenance_row_count": counts["provenance_row_count"],
            "truncated_chunk_count": sum(
                bool(record["source_extraction_truncated"])
                for record in records
            ),
            "tokenizer": policy.tokenizer,
            "prefix_indexes": list(policy.prefix_indexes),
            "source_files_modified": False,
            "database_sha256": sha256_file(database),
        }
        manifest["manifest_fingerprint"] = sha256_bytes(
            canonical_json(
                {
                    key: value
                    for key, value in manifest.items()
                    if key
                    not in {
                        "run_id",
                        "created_at",
                        "started_at",
                        "database_sha256",
                        "manifest_fingerprint",
                    }
                }
            )
        )
        atomic_json(stage / "retrieval-manifest.json", manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output_root)

    private_manifest = private_root / f"retrieval-manifest-{run_id}.json"
    shutil.copy2(output_root / "retrieval-manifest.json", private_manifest)
    result = {
        **manifest,
        "successful_index_build": True,
        "resumed_existing_index": False,
        "output_root": str(output_root),
        "database_path": str(output_root / "retrieval.sqlite3"),
        "manifest_path": str(output_root / "retrieval-manifest.json"),
        "private_manifest_path": str(private_manifest),
    }
    summary_path = exports / f"retrieval-index-summary-{run_id}.json"
    atomic_json(summary_path, result)
    result["summary_path"] = str(summary_path)
    return result


def _database_path(
    *,
    vault_root: Path,
    pilot_name: str,
    policy: RetrievalPolicy,
    chunk_set_id: str | None = None,
) -> Path:
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
        chunk_set_id=chunk_set_id,
    )
    chunk_manifest, _ = load_chunk_catalog(chunk_root)
    index_id = index_id_for(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(chunk_manifest["manifest_fingerprint"]),
        policy_digest=policy.digest,
    )
    database = (
        vault_root
        / "derived"
        / pilot_name
        / "retrieval"
        / index_id
        / "retrieval.sqlite3"
    )
    if not database.is_file():
        raise FileNotFoundError("Retrieval index is missing; build it first")
    return database


def verify_index(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    chunk_set_id: str | None = None,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=pilot_name,
        chunk_set_id=chunk_set_id,
    )
    chunk_manifest, records = load_chunk_catalog(chunk_root)
    index_id = index_id_for(
        chunk_set_id=str(chunk_manifest["chunk_set_id"]),
        chunk_manifest_fingerprint=str(chunk_manifest["manifest_fingerprint"]),
        policy_digest=policy.digest,
    )
    output_root = vault_root / "derived" / pilot_name / "retrieval" / index_id
    manifest_path = output_root / "retrieval-manifest.json"
    database = output_root / "retrieval.sqlite3"
    errors: list[str] = []
    if not manifest_path.is_file():
        errors.append("Retrieval manifest is missing")
    if not database.is_file():
        errors.append("Retrieval database is missing")
    if errors:
        return {
            "verification_schema_version": 1,
            "pilot_name": pilot_name,
            "index_id": index_id,
            "error_count": len(errors),
            "errors": errors,
            "ready_for_evaluation": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("index_id") != index_id:
        errors.append("Index ID mismatch")
    if manifest.get("policy_digest") != policy.digest:
        errors.append("Retrieval-policy digest mismatch")
    if manifest.get("chunk_manifest_fingerprint") != chunk_manifest[
        "manifest_fingerprint"
    ]:
        errors.append("Chunk-manifest fingerprint mismatch")
    if manifest.get("index_record_digest") != _record_digest(records):
        errors.append("Index-record digest mismatch")
    if manifest.get("database_sha256") != sha256_file(database):
        errors.append("Database file digest mismatch")

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    chunk_count = fts_count = provenance_count = 0
    sqlite_ok = fts_ok = False
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        sqlite_ok = integrity == "ok"
        if not sqlite_ok:
            errors.append(f"SQLite integrity check failed: {integrity}")
        try:
            connection.execute(
                "INSERT INTO chunk_fts(chunk_fts) VALUES('integrity-check')"
            )
            fts_ok = True
        except sqlite3.DatabaseError as exc:
            errors.append(f"FTS5 integrity check failed: {exc}")
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
        provenance_count = connection.execute(
            "SELECT COUNT(*) FROM provenance"
        ).fetchone()[0]
        if chunk_count != len(records):
            errors.append("Chunk table count mismatch")
        if fts_count != len(records):
            errors.append("FTS table count mismatch")
        expected_provenance = sum(
            len(record.get("provenance", [])) for record in records
        )
        if provenance_count != expected_provenance:
            errors.append("Provenance row count mismatch")
        for record in records[: min(25, len(records))]:
            row = connection.execute(
                "SELECT body, chunk_text_sha256 FROM chunks WHERE chunk_id=?",
                (record["chunk_id"],),
            ).fetchone()
            if row is None or sha256_bytes(row["body"].encode("utf-8")) != row[
                "chunk_text_sha256"
            ]:
                errors.append(f"Sample body hash mismatch: {record['chunk_id']}")
                break
    finally:
        connection.close()
    return {
        "verification_schema_version": 1,
        "pilot_name": pilot_name,
        "index_id": index_id,
        "expected_chunks": len(records),
        "verified_chunk_rows": chunk_count,
        "verified_fts_rows": fts_count,
        "verified_provenance_rows": provenance_count,
        "sqlite_integrity_ok": sqlite_ok,
        "fts5_integrity_ok": fts_ok,
        "error_count": len(errors),
        "errors": errors,
        "policy_digest": policy.digest,
        "ready_for_evaluation": not errors,
    }


def _tokens(query: str, minimum_length: int) -> list[str]:
    raw = [
        token.casefold()
        for token in re.findall(r"[\w]+", query, flags=re.UNICODE)
    ]
    filtered = [
        token
        for token in raw
        if len(token) >= minimum_length and token not in STOPWORDS
    ]
    if not filtered:
        filtered = [token for token in raw if len(token) >= minimum_length]
    output: list[str] = []
    seen: set[str] = set()
    for token in filtered:
        if token not in seen:
            output.append(token)
            seen.add(token)
    return output[:20]


def query_plans(query: str, policy: RetrievalPolicy) -> list[tuple[str, str]]:
    tokens = _tokens(query, policy.minimum_token_length)
    if not tokens:
        raise ValueError("Query contains no searchable tokens")
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    plans = [("and", " AND ".join(quoted))]
    if len(quoted) > 1:
        plans.append(("or_fallback", " OR ".join(quoted)))
    return plans


def _filter_sql(filters: SearchFilters) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if filters.families:
        placeholders = ",".join("?" for _ in filters.families)
        clauses.append(f"c.family IN ({placeholders})")
        parameters.extend(filters.families)
    if not filters.include_truncated:
        clauses.append("c.source_extraction_truncated=0")
    provenance_conditions: list[str] = []
    for column, values in (
        ("year_hint", filters.years),
        ("source_bucket", filters.source_buckets),
        ("known_contradiction_group", filters.contradiction_labels),
    ):
        if values:
            placeholders = ",".join("?" for _ in values)
            provenance_conditions.append(f"p.{column} IN ({placeholders})")
            parameters.extend(values)
    if provenance_conditions:
        clauses.append(
            "EXISTS (SELECT 1 FROM provenance p WHERE p.chunk_rowid=c.rowid AND "
            + " AND ".join(provenance_conditions)
            + ")"
        )
    return clauses, parameters


def _provenance(
    connection: sqlite3.Connection,
    rowid: int,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT file_id, original_relative_path, filename, role,
               source_bucket, year_hint, duplicate_control_group,
               known_contradiction_group
        FROM provenance
        WHERE chunk_rowid=?
        ORDER BY original_relative_path, file_id
        """,
        (rowid,),
    ).fetchall()
    return [dict(row) for row in rows]


def search_index(
    *,
    vault_root: Path,
    query: str,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    filters: SearchFilters | None = None,
    limit: int | None = None,
    max_chunks_per_source: int | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    database = _database_path(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    filters = filters or SearchFilters(
        include_truncated=policy.include_truncated_by_default
    )
    limit = limit or policy.default_limit
    max_per_source = max_chunks_per_source or policy.max_chunks_per_source
    candidate_limit = max(limit * policy.candidate_multiplier, 50)
    clauses, parameters = _filter_sql(filters)
    weights = policy.bm25_weights
    sql = f"""
        SELECT c.rowid, c.chunk_id, c.source_content_sha256,
               c.chunk_index, c.family, c.source_extraction_truncated,
               c.char_count,
               bm25(chunk_fts,
                    {weights['chunk_id']}, {weights['body']},
                    {weights['filename']}, {weights['source_path']}) AS score,
               snippet(chunk_fts, 1, '[[', ']]', ' … ',
                       {policy.snippet_tokens}) AS snippet
        FROM chunk_fts
        JOIN chunks c ON c.rowid=chunk_fts.rowid
        WHERE chunk_fts MATCH ?
    """
    if clauses:
        sql += " AND " + " AND ".join(clauses)
    sql += " ORDER BY score ASC, c.chunk_id ASC LIMIT ?"

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows: list[sqlite3.Row] = []
    plan_used = ""
    fts_query = ""
    try:
        for plan_name, plan_query in query_plans(query, policy):
            rows = connection.execute(
                sql,
                [plan_query, *parameters, candidate_limit],
            ).fetchall()
            if rows:
                plan_used = plan_name
                fts_query = plan_query
                break
        output: list[dict[str, Any]] = []
        kept: dict[str, list[int]] = {}
        for row in rows:
            source = str(row["source_content_sha256"])
            indices = kept.setdefault(source, [])
            if len(indices) >= max_per_source:
                continue
            if policy.collapse_adjacent_chunks and any(
                abs(int(row["chunk_index"]) - previous) <= 1
                for previous in indices
            ):
                continue
            indices.append(int(row["chunk_index"]))
            output.append(
                {
                    "rank": len(output) + 1,
                    "chunk_id": row["chunk_id"],
                    "source_content_sha256": source,
                    "chunk_index": int(row["chunk_index"]),
                    "family": row["family"],
                    "source_extraction_truncated": bool(
                        row["source_extraction_truncated"]
                    ),
                    "char_count": int(row["char_count"]),
                    "score": float(row["score"]),
                    "snippet": row["snippet"],
                    "provenance": _provenance(connection, int(row["rowid"])),
                }
            )
            if len(output) >= limit:
                break
    finally:
        connection.close()
    return {
        "retrieval_result_schema_version": 1,
        "pilot_name": pilot_name,
        "query": query,
        "query_plan": plan_used,
        "fts_query": fts_query,
        "filters": {
            "families": list(filters.families),
            "years": list(filters.years),
            "source_buckets": list(filters.source_buckets),
            "contradiction_labels": list(filters.contradiction_labels),
            "include_truncated": filters.include_truncated,
        },
        "result_count": len(output),
        "results": output,
    }


def _safe_term(term: str, minimum_length: int) -> bool:
    return (
        len(term) >= minimum_length
        and term.casefold() not in STOPWORDS
        and bool(re.fullmatch(r"[^\W\d_]+", term, flags=re.UNICODE))
    )


def create_lexical_benchmark(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    case_count: int | None = None,
    seed: str = "alice-lexical-smoke-v1",
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    database = _database_path(
        vault_root=vault_root,
        pilot_name=pilot_name,
        policy=policy,
    )
    desired = case_count or policy.lexical_benchmark_cases
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        # The vocabulary table lives in the TEMP schema, while the indexed
        # FTS5 table lives in MAIN. SQLite requires the three-argument form
        # to name the source database explicitly in this case.
        connection.execute(
            "CREATE VIRTUAL TABLE temp.term_instances "
            "USING fts5vocab(main, 'chunk_fts', 'instance')"
        )
        rows = connection.execute(
            """
            SELECT v.term, c.source_content_sha256, c.chunk_id
            FROM term_instances v
            JOIN chunks c ON c.rowid=v.doc
            WHERE v.col='body'
            """
        ).fetchall()
    finally:
        connection.close()
    by_term: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        term = str(row["term"])
        if _safe_term(term, policy.lexical_benchmark_min_term_length):
            by_term[term].append(row)
    candidates: list[dict[str, Any]] = []
    for term, term_rows in by_term.items():
        sources = {str(row["source_content_sha256"]) for row in term_rows}
        if len(sources) > policy.lexical_benchmark_max_document_frequency:
            continue
        candidates.append(
            {
                "term": term,
                "sources": sorted(sources),
                "chunks": sorted({str(row["chunk_id"]) for row in term_rows}),
                "df": len(sources),
            }
        )
    candidates.sort(key=lambda item: (item["df"], -len(item["term"]), item["term"]))
    pool = candidates[: max(desired * 4, desired)]
    random.Random(seed).shuffle(pool)
    selected = pool[:desired]
    benchmark_id = str(uuid.uuid4())
    cases = [
        {
            "query_id": f"lex-{index:03d}",
            "question": item["term"],
            "expected_source_sha256": item["sources"],
            "expected_chunk_ids": item["chunks"],
            "filters": {},
            "notes": "Automatically generated lexical smoke case",
        }
        for index, item in enumerate(selected, start=1)
    ]
    benchmark = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_type": "lexical_smoke",
        "pilot_name": pilot_name,
        "case_count": len(cases),
        "seed": seed,
        "cases": cases,
    }
    private_root = vault_root / "manifests" / "retrieval" / pilot_name
    exports = vault_root / "manifests" / "exports"
    private_root.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    benchmark_path = private_root / f"lexical-benchmark-{benchmark_id}.json"
    atomic_json(benchmark_path, benchmark)
    summary = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_type": "lexical_smoke",
        "pilot_name": pilot_name,
        "requested_cases": desired,
        "created_cases": len(cases),
        "candidate_terms": len(candidates),
        "benchmark_path": str(benchmark_path),
    }
    summary_path = exports / f"retrieval-benchmark-summary-{benchmark_id}.json"
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _latest_benchmark(vault_root: Path, pilot_name: str) -> Path:
    root = vault_root / "manifests" / "retrieval" / pilot_name
    candidates = sorted(
        root.glob("lexical-benchmark-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No lexical benchmark exists")
    return candidates[0]


def evaluate(
    *,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
    benchmark_path: Path | None = None,
    policy_path: Path | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = (
        benchmark_path.expanduser().resolve(strict=True)
        if benchmark_path
        else _latest_benchmark(vault_root, pilot_name)
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = list(benchmark.get("cases", []))
    if not cases:
        raise ValueError("Benchmark contains no cases")
    max_k = max(k_values)
    hits = Counter()
    reciprocal_ranks: list[float] = []
    details: list[dict[str, Any]] = []
    for case in cases:
        expected = set(str(item) for item in case["expected_source_sha256"])
        result = search_index(
            vault_root=vault_root,
            pilot_name=pilot_name,
            policy_path=policy_path,
            query=str(case["question"]),
            limit=max_k,
            max_chunks_per_source=1,
        )
        returned = [
            str(item["source_content_sha256"])
            for item in result["results"]
        ]
        first_rank = 0
        for index, source in enumerate(returned, start=1):
            if source in expected:
                first_rank = index
                break
        for k in k_values:
            if expected.intersection(returned[:k]):
                hits[k] += 1
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        details.append(
            {
                "query_id": case["query_id"],
                "question": case["question"],
                "expected_source_sha256": sorted(expected),
                "returned_source_sha256": returned,
                "first_relevant_rank": first_rank,
                "query_plan": result["query_plan"],
            }
        )
    count = len(cases)
    run_id = str(uuid.uuid4())
    summary = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "run_id": run_id,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_type": benchmark["benchmark_type"],
        "pilot_name": pilot_name,
        "case_count": count,
        "hit_rate_at_k": {
            str(k): round(hits[k] / count, 6) for k in k_values
        },
        "mean_reciprocal_rank_at_10": round(
            sum(reciprocal_ranks) / count,
            6,
        ),
        "missed_cases": sum(1 for value in reciprocal_ranks if value == 0),
    }
    exports = vault_root / "manifests" / "exports"
    private_root = vault_root / "manifests" / "retrieval" / pilot_name
    exports.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    details_path = private_root / f"retrieval-evaluation-details-{run_id}.json"
    atomic_json(
        details_path,
        {**summary, "benchmark_path": str(benchmark_path), "cases": details},
    )
    summary["private_details_path"] = str(details_path)
    summary_path = exports / f"retrieval-evaluation-summary-{run_id}.json"
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
