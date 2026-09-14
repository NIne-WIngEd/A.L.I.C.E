#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_chunks(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    """Deterministically bound sampling granularity without changing document split lineage."""
    start = 0
    size = len(text)
    while start < size:
        end = min(size, start + max_chars)
        if end < size:
            cut = text.rfind("\n", start, end)
            if cut <= start + min_chars:
                cut = text.rfind(" ", start, end)
            if cut > start + min_chars:
                end = cut
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            yield chunk
        start = max(end, start + 1)
        while start < size and text[start].isspace():
            start += 1


def trim_to_budget(text: str, remaining: int, min_chars: int) -> str:
    if remaining < min_chars:
        return ""
    if len(text) <= remaining:
        return text
    cut = text.rfind("\n", 0, remaining + 1)
    if cut < min_chars:
        cut = text.rfind(" ", 0, remaining + 1)
    if cut < min_chars:
        cut = remaining
    return text[:cut].strip()


@dataclass
class ShardWriter:
    root: Path
    source_id: str
    target_bytes: int
    index: int = 0
    bytes_written: int = 0
    handle: Any = None
    current_path: Path | None = None

    def _open(self) -> None:
        self.current_path = self.root / f"{self.source_id}-{self.index:05d}.jsonl"
        self.handle = self.current_path.open("w", encoding="utf-8")
        self.bytes_written = 0

    def write(self, row: dict[str, Any]) -> None:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        payload_bytes = len(payload.encode("utf-8"))
        if self.handle is None:
            self._open()
        elif self.bytes_written and self.bytes_written + payload_bytes > self.target_bytes:
            self.handle.close()
            self.index += 1
            self._open()
        self.handle.write(payload)
        self.bytes_written += payload_bytes

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def verify_parent(parent_dir: Path) -> tuple[dict[str, Any], str]:
    receipt_path = parent_dir / "corpus_receipt.json"
    if not receipt_path.is_file():
        raise SystemExit(f"missing parent corpus receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "alice.eipm.n0.public-corpus-receipt.v0.2":
        raise SystemExit("unexpected parent corpus receipt schema")
    if receipt.get("private_identity_data") is not False:
        raise SystemExit("parent corpus must declare private_identity_data=false")
    if receipt.get("private_identity_gradient") is not False:
        raise SystemExit("parent corpus must declare private_identity_gradient=false")

    for source in receipt.get("sources", []):
        for shard in source.get("shards", []):
            rel = Path(str(shard["path"]))
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit(f"unsafe parent shard path: {rel}")
            path = (parent_dir / rel).resolve()
            if parent_dir.resolve() not in path.parents:
                raise SystemExit(f"parent shard escapes corpus directory: {rel}")
            if not path.is_file():
                raise SystemExit(f"missing parent shard: {rel}")
            if path.stat().st_size != int(shard["bytes"]):
                raise SystemExit(f"parent shard byte mismatch: {rel}")
            if sha256_file(path) != str(shard["sha256"]):
                raise SystemExit(f"parent shard hash mismatch: {rel}")
    return receipt, sha256_file(receipt_path)


def load_active_sources(config_path: Path) -> list[dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "alice.eipm.n0.public-corpus-sources.v0.2.1":
        raise SystemExit("offline derivation requires the v0.2.1 activated manifest")
    if config.get("status") != "activated_public_n0_v02":
        raise SystemExit("v0.2.1 source manifest is not activated")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 21:
        raise SystemExit("v0.2.1 source manifest must contain exactly 21 active sources")
    shares = sum(float(source["target_share"]) for source in sources)
    if abs(shares - 1.0) > 1e-9:
        raise SystemExit(f"v0.2.1 source shares must sum to 1.0, got {shares}")
    return sources


def iter_parent_rows(parent_dir: Path, source_receipt: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for shard in source_receipt["shards"]:
        path = parent_dir / shard["path"]
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a balanced N0 v0.2.1 tokenizer corpus entirely from the already acquired "
            "v0.1 materialization. No network access is used."
        )
    )
    parser.add_argument("--parent-corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total-chars", type=int, default=100_000_000)
    parser.add_argument("--max-document-chunk-chars", type=int, default=16_000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--shard-mb", type=int, default=64)
    args = parser.parse_args()

    if args.target_total_chars < 1:
        raise SystemExit("--target-total-chars must be positive")
    if args.max_document_chunk_chars < args.min_chars:
        raise SystemExit("max document chunk size must be >= min chars")

    parent_dir = Path(args.parent_corpus_dir).resolve()
    config_path = Path(args.source_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty derived corpus directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    parent_receipt, parent_receipt_sha = verify_parent(parent_dir)
    active_sources = load_active_sources(config_path)
    parent_by_id = {str(row["source_id"]): row for row in parent_receipt["sources"]}

    db = sqlite3.connect(output_dir / "exact_dedup.sqlite3")
    db.execute("CREATE TABLE seen (text_sha256 TEXT PRIMARY KEY)")
    db.commit()

    receipt: dict[str, Any] = {
        "schema": "alice.eipm.n0.derived-tokenizer-corpus-receipt.v0.2.1",
        "source_config_sha256": sha256_file(config_path),
        "parent_corpus_receipt_sha256": parent_receipt_sha,
        "parent_corpus_status": parent_receipt.get("status"),
        "derivation_mode": "offline_from_verified_materialized_rows",
        "network_access_required": False,
        "target_total_chars": args.target_total_chars,
        "max_document_chunk_chars": args.max_document_chunk_chars,
        "sources": [],
        "private_identity_data": False,
        "private_identity_gradient": False,
        "model_training_performed": False,
    }

    total_chars = 0
    total_rows = 0
    failures: list[str] = []

    for active in active_sources:
        source_id = str(active["source_id"])
        parent_source = parent_by_id.get(source_id)
        if parent_source is None:
            failures.append(f"{source_id}: absent from parent materialization")
            continue
        if str(parent_source.get("resolved_revision")) != str(active["revision"]):
            failures.append(f"{source_id}: revision mismatch with parent materialization")
            continue

        allowed = {str(value) for value in active["allowed_license_values"]}
        char_budget = round(args.target_total_chars * float(active["target_share"]))
        writer = ShardWriter(shard_dir, source_id, args.shard_mb * 1024 * 1024)
        source_chars = 0
        source_rows = 0
        source_docs = 0
        duplicates = 0
        license_rejected = 0

        try:
            for row in iter_parent_rows(parent_dir, parent_source):
                if source_chars >= char_budget:
                    break
                if str(row.get("source_id")) != source_id:
                    raise SystemExit(f"parent row source_id mismatch in {source_id}")
                if str(row.get("source_revision")) != str(active["revision"]):
                    raise SystemExit(f"parent row revision mismatch in {source_id}")
                license_value = str(row.get("license_expression") or "")
                if license_value not in allowed:
                    license_rejected += 1
                    continue
                text = str(row.get("text") or "").strip()
                if len(text) < args.min_chars:
                    continue
                source_docs += 1
                parent_sha = str(row.get("text_sha256") or sha256_text(text))
                parent_split = str(row.get("split") or "train")

                for chunk_index, raw_chunk in enumerate(
                    split_chunks(text, args.max_document_chunk_chars, args.min_chars)
                ):
                    remaining = char_budget - source_chars
                    chunk = trim_to_budget(raw_chunk, remaining, args.min_chars)
                    if not chunk:
                        break
                    chunk_sha = sha256_text(chunk)
                    inserted = db.execute(
                        "INSERT OR IGNORE INTO seen(text_sha256) VALUES (?)", (chunk_sha,)
                    ).rowcount
                    if inserted == 0:
                        duplicates += 1
                        continue
                    out = {
                        "text": chunk,
                        "text_sha256": chunk_sha,
                        "parent_text_sha256": parent_sha,
                        "parent_chunk_index": chunk_index,
                        "split": parent_split,
                        "source_id": source_id,
                        "source_category": active["category"],
                        "source_target_share": float(active["target_share"]),
                        "source_repo": active["repo_id"],
                        "source_revision": active["revision"],
                        "source_record_id": row.get("source_record_id", ""),
                        "source_provenance": row.get("source_provenance"),
                        "source_url": row.get("source_url"),
                        "license_expression": license_value,
                        "derived_from_parent_materialization": True,
                    }
                    writer.write(out)
                    source_rows += 1
                    source_chars += len(chunk)
                    total_rows += 1
                    total_chars += len(chunk)
                    if source_chars >= char_budget:
                        break
        finally:
            writer.close()
            db.commit()

        fill_ratio = source_chars / char_budget if char_budget else 0.0
        if fill_ratio < 0.999:
            failures.append(
                f"{source_id}: available parent material only filled {source_chars}/{char_budget} chars"
            )
        source_files = sorted(shard_dir.glob(f"{source_id}-*.jsonl"))
        source_receipt = {
            "source_id": source_id,
            "repo_id": active["repo_id"],
            "revision": active["revision"],
            "category": active["category"],
            "target_share": float(active["target_share"]),
            "char_budget": char_budget,
            "accepted_chars": source_chars,
            "accepted_rows": source_rows,
            "parent_documents_consumed": source_docs,
            "exact_chunk_duplicates_rejected": duplicates,
            "license_rejected": license_rejected,
            "fill_ratio": fill_ratio,
            "allowed_license_values": sorted(allowed),
            "shards": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in source_files
            ],
        }
        receipt["sources"].append(source_receipt)
        print(json.dumps(source_receipt, sort_keys=True))

    db.close()
    receipt["accepted_chars"] = total_chars
    receipt["accepted_rows"] = total_rows
    receipt["source_count"] = len(receipt["sources"])
    receipt["fill_failures"] = failures
    receipt["status"] = "PASS" if not failures else "FAIL"

    receipt_path = output_dir / "corpus_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "source_count": receipt["source_count"],
                "accepted_rows": total_rows,
                "accepted_chars": total_chars,
                "target_total_chars": args.target_total_chars,
                "parent_corpus_receipt_sha256": parent_receipt_sha,
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "network_access_required": False,
                "private_identity_data": False,
                "private_identity_gradient": False,
            },
            sort_keys=True,
        )
    )
    if failures:
        for failure in failures:
            print(f"DERIVATION_FAILURE {failure}")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
