from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import (
    ChunkingPolicy,
    chunk_text,
    load_chunking_policy,
    normalize_text,
    sha256_text,
    stable_chunk_id,
    stable_chunk_set_id,
)

SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_inputs(vault: Path, pilot: str) -> tuple[Path, dict[str, Any], Path]:
    snapshot = (vault / "raw" / pilot).resolve(strict=True)
    manifest_path = snapshot / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pilot_name") != pilot or not isinstance(manifest.get("items"), list):
        raise ValueError("Invalid pilot manifest")
    extraction = (vault / "derived" / pilot / "extracted").resolve(strict=True)
    return manifest_path, manifest, extraction


def provenance(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    fields = [
        "file_id",
        "original_relative_path",
        "filename",
        "role",
        "family",
        "source_bucket",
        "year_hint",
        "duplicate_control_group",
        "known_contradiction_group",
    ]
    return [
        {field: str(item.get(field, "")) for field in fields}
        for item in sorted(
            items,
            key=lambda item: (str(item.get("original_relative_path", "")), str(item.get("file_id", ""))),
        )
    ]


def compute(vault: Path, pilot: str, policy: ChunkingPolicy) -> dict[str, Any]:
    manifest_path, manifest, extraction = load_inputs(vault, pilot)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest["items"]:
        grouped[str(item["sha256"])].append(dict(item))

    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    registry_digests: set[str] = set()
    family_counts: Counter[str] = Counter()
    truncated_sources = 0
    truncated_chunks = 0
    duplicate_paths = 0
    contradiction_labels: set[str] = set()

    for source_sha in sorted(grouped):
        items = grouped[source_sha]
        text_path = extraction / "text" / f"{source_sha}.txt"
        metadata_path = extraction / "metadata" / f"{source_sha}.json"
        if not text_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing extraction for {source_sha}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "success" or metadata.get("source_sha256") != source_sha:
            raise ValueError(f"Invalid extraction metadata for {source_sha}")
        if metadata.get("text_sha256") != sha256_file(text_path):
            raise ValueError(f"Extraction text hash mismatch for {source_sha}")
        registry_digests.add(str(metadata["registry_digest"]))
        normalized, spans = chunk_text(text_path.read_text(encoding="utf-8"), policy)
        prov = provenance(items)
        duplicate_paths += max(0, len(prov) - 1)
        family = str(items[0]["family"]).lower()
        family_counts[family] += 1
        is_truncated = bool(metadata.get("truncated", False))
        truncated_sources += int(is_truncated)
        for item in prov:
            if item["known_contradiction_group"]:
                contradiction_labels.add(item["known_contradiction_group"])
        ids: list[str] = []
        for span in spans:
            chunk_id = stable_chunk_id(source_sha, policy.digest, span)
            ids.append(chunk_id)
            truncated_chunks += int(is_truncated)
            records.append(
                {
                    "chunk_catalog_schema_version": SCHEMA_VERSION,
                    "chunk_id": chunk_id,
                    "source_content_sha256": source_sha,
                    "source_text_sha256": str(metadata["text_sha256"]),
                    "normalized_source_text_sha256": sha256_text(normalized),
                    "chunk_index": span.index,
                    "start_char": span.start,
                    "end_char": span.end,
                    "char_count": len(span.text),
                    "chunk_text_sha256": span.text_sha256,
                    "family": family,
                    "parser_id": str(metadata["parser_id"]),
                    "extraction_registry_digest": str(metadata["registry_digest"]),
                    "source_extraction_truncated": is_truncated,
                    "source_extraction_warnings": list(metadata.get("warnings", [])),
                    "provenance_path_count": len(prov),
                    "provenance": prov,
                }
            )
        sources.append(
            {
                "source_content_sha256": source_sha,
                "source_text_sha256": str(metadata["text_sha256"]),
                "normalized_source_text_sha256": sha256_text(normalized),
                "normalized_char_count": len(normalized),
                "family": family,
                "parser_id": str(metadata["parser_id"]),
                "extraction_run_id": str(metadata.get("run_id", "")),
                "extraction_registry_digest": str(metadata["registry_digest"]),
                "source_extraction_truncated": is_truncated,
                "source_extraction_warnings": list(metadata.get("warnings", [])),
                "chunk_ids": ids,
                "provenance": prov,
            }
        )

    if len(registry_digests) != 1:
        raise ValueError("Multiple extraction registry digests found")
    registry_digest = next(iter(registry_digests))
    pilot_hash = sha256_file(manifest_path)
    set_id = stable_chunk_set_id(pilot_hash, registry_digest, policy.digest)
    records.sort(key=lambda item: (item["source_content_sha256"], item["chunk_index"]))
    sources.sort(key=lambda item: item["source_content_sha256"])
    return {
        "chunk_set_id": set_id,
        "pilot_manifest_sha256": pilot_hash,
        "extraction_registry_digest": registry_digest,
        "records": records,
        "sources": sources,
        "family_counts": dict(family_counts),
        "truncated_source_count": truncated_sources,
        "chunks_from_truncated_sources": truncated_chunks,
        "duplicate_provenance_paths": duplicate_paths,
        "contradiction_labels": sorted(contradiction_labels),
        "pilot_item_count": len(manifest["items"]),
        "source_count": len(grouped),
        "extraction_root": extraction,
    }


def fingerprint(manifest: dict[str, Any]) -> str:
    filtered = {k: v for k, v in manifest.items() if k not in {"created_at", "run_id", "manifest_fingerprint"}}
    return sha256_text(json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def make_sqlite(path: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(
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
    con.execute(
        "INSERT INTO chunk_sets VALUES(?,?,?,?,?,?,?,?,?)",
        (
            manifest["chunk_set_id"], manifest["pilot_name"], manifest["policy_id"],
            manifest["policy_digest"], manifest["pilot_manifest_sha256"],
            manifest["extraction_registry_digest"], manifest["chunk_count"],
            manifest["source_count"], manifest["created_at"],
        ),
    )
    for record in records:
        con.execute(
            "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record["chunk_id"], manifest["chunk_set_id"], record["source_content_sha256"],
                record["source_text_sha256"], record["normalized_source_text_sha256"],
                record["chunk_index"], record["start_char"], record["end_char"],
                record["char_count"], record["chunk_text_sha256"], record["family"],
                record["parser_id"], int(record["source_extraction_truncated"]),
                record["provenance_path_count"], f"text/{record['chunk_id']}.txt",
            ),
        )
        for prov in record["provenance"]:
            con.execute(
                "INSERT INTO chunk_provenance VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    record["chunk_id"], prov["file_id"], prov["original_relative_path"],
                    prov["filename"], prov["role"], prov["family"], prov["source_bucket"],
                    prov["year_hint"], prov["duplicate_control_group"], prov["known_contradiction_group"],
                ),
            )
    con.commit()
    con.close()


def build_pilot_chunks(vault_root: Path, pilot_name: str = "pilot-v1", policy_path: Path | None = None, replace: bool = False) -> dict[str, Any]:
    vault = vault_root.expanduser().resolve(strict=True)
    policy = load_chunking_policy(policy_path)
    data = compute(vault, pilot_name, policy)
    set_id = data["chunk_set_id"]
    output = vault / "derived" / pilot_name / "chunks" / set_id
    exports = vault / "manifests" / "exports"
    private_indexes = vault / "manifests" / "chunks" / pilot_name
    temp_root = vault / "temporary"
    for path in (exports, private_indexes, temp_root):
        path.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix=f"alice-chunks-{set_id}-", dir=temp_root) as temp:
        stage = Path(temp) / set_id
        text_dir = stage / "text"
        text_dir.mkdir(parents=True)
        normalized_cache: dict[str, str] = {}
        for source in data["sources"]:
            source_sha = source["source_content_sha256"]
            path = data["extraction_root"] / "text" / f"{source_sha}.txt"
            normalized_cache[source_sha] = normalize_text(path.read_text(encoding="utf-8"))
        total_chars = 0
        for record in data["records"]:
            text = normalized_cache[record["source_content_sha256"]][record["start_char"]:record["end_char"]]
            if sha256_text(text) != record["chunk_text_sha256"]:
                raise RuntimeError("Chunk hash mismatch before write")
            target = text_dir / f"{record['chunk_id']}.txt"
            target.write_text(text, encoding="utf-8")
            total_chars += len(text)
            try:
                os.chmod(target, stat.S_IREAD)
            except OSError:
                pass

        records_path = stage / "chunk-records.jsonl"
        records_path.write_text("".join(canonical_line(r) + "\n" for r in data["records"]), encoding="utf-8")
        source_map_path = stage / "source-map.json"
        source_map_path.write_text(json.dumps({"chunk_catalog_schema_version": SCHEMA_VERSION, "chunk_set_id": set_id, "sources": data["sources"]}, indent=2, ensure_ascii=False), encoding="utf-8")
        lengths = [r["char_count"] for r in data["records"]]
        manifest = {
            "chunk_catalog_schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "chunk_set_id": set_id,
            "pilot_name": pilot_name,
            "created_at": now(),
            "policy_id": policy.policy_id,
            "policy_digest": policy.digest,
            "algorithm_version": policy.algorithm_version,
            "normalization_version": policy.normalization_version,
            "pilot_manifest_sha256": data["pilot_manifest_sha256"],
            "extraction_registry_digest": data["extraction_registry_digest"],
            "pilot_item_count": data["pilot_item_count"],
            "source_count": data["source_count"],
            "chunk_count": len(data["records"]),
            "total_chunk_chars": total_chars,
            "average_chunk_chars": round(total_chars / len(lengths), 3),
            "minimum_chunk_chars": min(lengths),
            "maximum_chunk_chars": max(lengths),
            "family_counts": data["family_counts"],
            "truncated_source_count": data["truncated_source_count"],
            "chunks_from_truncated_sources": data["chunks_from_truncated_sources"],
            "duplicate_provenance_paths": data["duplicate_provenance_paths"],
            "contradiction_labels": data["contradiction_labels"],
            "chunk_records_sha256": sha256_file(records_path),
            "source_map_sha256": sha256_file(source_map_path),
            "source_files_modified": False,
        }
        manifest["manifest_fingerprint"] = fingerprint(manifest)
        atomic_json(stage / "chunk-manifest.json", manifest)
        make_sqlite(stage / "chunks.sqlite3", manifest, data["records"])

        if output.exists():
            old_path = output / "chunk-manifest.json"
            if old_path.is_file():
                old = json.loads(old_path.read_text(encoding="utf-8"))
                if old.get("manifest_fingerprint") == manifest["manifest_fingerprint"]:
                    summary = {**old, "successful_chunk_build": False, "resumed_existing_chunk_set": True, "output_root": str(output), "manifest_path": str(old_path)}
                    summary_path = exports / f"pilot-chunk-summary-{run_id}.json"
                    atomic_json(summary_path, summary)
                    summary["summary_path"] = str(summary_path)
                    return summary
            if not replace:
                raise FileExistsError("Conflicting chunk set exists; investigate before using --replace")
            shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(output)

    private_index = private_indexes / f"chunk-index-{run_id}.json"
    shutil.copy2(output / "chunk-manifest.json", private_index)
    summary = {**manifest, "successful_chunk_build": True, "resumed_existing_chunk_set": False, "output_root": str(output), "manifest_path": str(output / "chunk-manifest.json"), "private_index_path": str(private_index)}
    summary_path = exports / f"pilot-chunk-summary-{run_id}.json"
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def verify_pilot_chunks(vault_root: Path, pilot_name: str = "pilot-v1", policy_path: Path | None = None) -> dict[str, Any]:
    vault = vault_root.expanduser().resolve(strict=True)
    policy = load_chunking_policy(policy_path)
    data = compute(vault, pilot_name, policy)
    set_id = data["chunk_set_id"]
    output = vault / "derived" / pilot_name / "chunks" / set_id
    errors: list[str] = []
    required = [output / "chunk-manifest.json", output / "chunk-records.jsonl", output / "source-map.json", output / "chunks.sqlite3"]
    for path in required:
        if not path.is_file():
            errors.append(f"Missing catalog file: {path.name}")
    if errors:
        return {"verification_schema_version": 1, "pilot_name": pilot_name, "chunk_set_id": set_id, "error_count": len(errors), "errors": errors, "ready_for_indexing": False}

    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    actual = [json.loads(line) for line in required[1].read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = data["records"]
    checks = [
        (manifest.get("chunk_set_id") == set_id, "Chunk-set ID mismatch"),
        (manifest.get("policy_digest") == policy.digest, "Policy digest mismatch"),
        (manifest.get("pilot_manifest_sha256") == data["pilot_manifest_sha256"], "Pilot manifest digest mismatch"),
        (manifest.get("chunk_records_sha256") == sha256_file(required[1]), "Chunk-records hash mismatch"),
        (manifest.get("source_map_sha256") == sha256_file(required[2]), "Source-map hash mismatch"),
        (manifest.get("manifest_fingerprint") == fingerprint(manifest), "Manifest fingerprint mismatch"),
        (actual == expected, "Chunk records do not match deterministic rebuild"),
    ]
    errors.extend(message for passed, message in checks if not passed)
    normalized: dict[str, str] = {}
    verified = 0
    for record in actual:
        chunk_path = output / "text" / f"{record['chunk_id']}.txt"
        if not chunk_path.is_file():
            errors.append(f"Missing chunk text: {record['chunk_id']}")
            continue
        text = chunk_path.read_text(encoding="utf-8")
        if sha256_text(text) != record["chunk_text_sha256"]:
            errors.append(f"Chunk text hash mismatch: {record['chunk_id']}")
            continue
        source_sha = record["source_content_sha256"]
        if source_sha not in normalized:
            source_path = data["extraction_root"] / "text" / f"{source_sha}.txt"
            normalized[source_sha] = normalize_text(source_path.read_text(encoding="utf-8"))
        expected_slice = normalized[source_sha][record["start_char"]:record["end_char"]]
        if text != expected_slice:
            errors.append(f"Chunk offset mismatch: {record['chunk_id']}")
            continue
        verified += 1
    try:
        con = sqlite3.connect(required[3])
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        prov_count = con.execute("SELECT COUNT(*) FROM chunk_provenance").fetchone()[0]
        con.close()
        if chunk_count != len(actual):
            errors.append("SQLite chunk count mismatch")
        if prov_count != sum(r["provenance_path_count"] for r in actual):
            errors.append("SQLite provenance count mismatch")
    except sqlite3.DatabaseError as exc:
        errors.append(f"SQLite verification error: {exc}")
    return {
        "verification_schema_version": 1,
        "pilot_name": pilot_name,
        "chunk_set_id": set_id,
        "expected_source_contents": data["source_count"],
        "expected_chunks": len(expected),
        "verified_chunks": verified,
        "truncated_source_count": data["truncated_source_count"],
        "chunks_from_truncated_sources": data["chunks_from_truncated_sources"],
        "duplicate_provenance_paths": data["duplicate_provenance_paths"],
        "deterministic_rebuild_match": actual == expected,
        "error_count": len(errors),
        "errors": errors,
        "policy_digest": policy.digest,
        "ready_for_indexing": not errors,
    }
