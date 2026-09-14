#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(document_sha256: str) -> str:
    bucket = int(document_sha256[:8], 16) % 10_000
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "dev"
    return "train"


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    current: Any = row
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def partition_match(document_sha256: str, index: int, count: int) -> bool:
    return int(document_sha256[8:24], 16) % count == index


def chunk_text(text: str, max_chars: int, min_chars: int) -> Iterator[str]:
    """Deterministically split very large source records without crossing document lineage.

    Common Pile book/government rows can contain entire books or reports. Treating one
    huge row as one sampling unit caused a single document to overshoot a source budget
    by more than 100%. Chunks stay within the same document split/partition and are cut
    at whitespace when practical.
    """
    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars")
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            window = text[start:end]
            # Prefer a paragraph/newline boundary, then ordinary whitespace, but do
            # not create a tiny chunk merely to hit a boundary.
            floor = max(min_chars, max_chars // 2)
            cut = window.rfind("\n", floor)
            if cut < floor:
                cut = window.rfind(" ", floor)
            if cut >= floor:
                end = start + cut
        piece = text[start:end].strip()
        if len(piece) >= min_chars:
            yield piece
        start = max(end, start + 1)


def trim_to_budget(text: str, remaining: int, min_chars: int) -> str:
    if len(text) <= remaining:
        return text
    if remaining < min_chars:
        return ""
    candidate = text[:remaining]
    floor = max(min_chars, int(remaining * 0.75))
    cut = candidate.rfind("\n", floor)
    if cut < floor:
        cut = candidate.rfind(" ", floor)
    if cut >= floor:
        candidate = candidate[:cut]
    return candidate.strip()


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
        encoded_len = len(payload.encode("utf-8"))
        if self.handle is None:
            self._open()
        elif self.bytes_written and self.bytes_written + encoded_len > self.target_bytes:
            self.handle.close()
            self.index += 1
            self._open()
        self.handle.write(payload)
        self.bytes_written += encoded_len

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def load_source_rows(repo_id: str, revision: str, split: str) -> tuple[str, Iterator[dict[str, Any]]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before materializing N0 v0.2") from exc

    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = str(info.sha)
    if resolved_revision != revision:
        raise RuntimeError(
            f"exact revision mismatch for {repo_id}: requested={revision} resolved={resolved_revision}"
        )
    dataset = load_dataset(repo_id, split=split, streaming=True, revision=revision)
    return resolved_revision, iter(dataset)


def validate_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("status") != "activated_public_n0_v02":
        raise SystemExit("N0 v0.2 materialization requires the activated source manifest")
    requirements = config.get("global_requirements", {})
    if requirements.get("private_identity_data") is not False:
        raise SystemExit("activated N0 v0.2 manifest must declare private_identity_data=false")
    if requirements.get("private_identity_gradient") is not False:
        raise SystemExit("activated N0 v0.2 manifest must declare private_identity_gradient=false")

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SystemExit("activated N0 v0.2 manifest contains no sources")

    seen_ids: set[str] = set()
    shares = 0.0
    for source in sources:
        source_id = str(source.get("source_id", ""))
        if not source_id or source_id in seen_ids:
            raise SystemExit(f"invalid/duplicate source_id: {source_id!r}")
        seen_ids.add(source_id)
        revision = str(source.get("revision", ""))
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
            raise SystemExit(f"source {source_id} does not pin a 40-hex exact revision")
        allowed = source.get("allowed_license_values")
        if not isinstance(allowed, list) or not allowed:
            raise SystemExit(f"source {source_id} has no explicit license allowlist")
        if source.get("require_row_license") is not True:
            raise SystemExit(f"source {source_id} must require row-level license")
        share = float(source.get("target_share", 0.0))
        if not 0.0 < share <= 0.125:
            raise SystemExit(f"source {source_id} has invalid target_share={share}")
        shares += share
    if abs(shares - 1.0) > 1e-9:
        raise SystemExit(f"source target shares must sum to 1.0, got {shares}")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an exact-revision, source-balanced N0 v0.2 public corpus tranche."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-total-chars", type=int, required=True)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--minimum-fill-ratio", type=float, default=0.95)
    parser.add_argument("--shard-mb", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-document-chunk-chars", type=int, default=16_000)
    args = parser.parse_args()

    if args.target_total_chars < 1:
        raise SystemExit("--target-total-chars must be positive")
    if args.partition_count < 1 or not 0 <= args.partition_index < args.partition_count:
        raise SystemExit("invalid deterministic partition index/count")
    if not 0.0 < args.minimum_fill_ratio <= 1.0:
        raise SystemExit("--minimum-fill-ratio must be in (0,1]")
    if args.max_document_chunk_chars < args.min_chars:
        raise SystemExit("--max-document-chunk-chars must be >= --min-chars")

    config_path = Path(args.source_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = validate_manifest(config)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to append into non-empty v0.2 tranche directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(output_dir / "exact_dedup.sqlite3")
    db.execute("CREATE TABLE seen_documents (sha256 TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE seen_chunks (sha256 TEXT PRIMARY KEY)")
    db.commit()

    receipt: dict[str, Any] = {
        "schema": "alice.eipm.n0.public-corpus-receipt.v0.2.1",
        "source_config_sha256": sha256_file(config_path),
        "target_total_chars": args.target_total_chars,
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "minimum_fill_ratio": args.minimum_fill_ratio,
        "max_document_chunk_chars": args.max_document_chunk_chars,
        "sources": [],
        "private_identity_data": False,
        "private_identity_gradient": False,
        "license_gate": "row_license_must_match_source_specific_allowlist",
        "exact_revision_required": True,
        "source_balanced": True,
        "document_level_split_and_partition": True,
    }

    failures: list[str] = []
    total_accepted_chars = 0
    total_accepted_rows = 0

    for source in sources:
        source_id = str(source["source_id"])
        repo_id = str(source["repo_id"])
        revision = str(source["revision"])
        split = str(source.get("split", "train"))
        text_field = str(source.get("text_field", "text"))
        license_field = str(source.get("license_field", "metadata.license"))
        provenance_field = str(source.get("provenance_field", "metadata.provenance"))
        url_field = str(source.get("url_field", "metadata.url"))
        id_field = str(source.get("id_field", "id"))
        allowed_licenses = {str(value) for value in source["allowed_license_values"]}
        target_share = float(source["target_share"])
        char_budget = max(1, round(args.target_total_chars * target_share))

        resolved_revision, rows = load_source_rows(repo_id, revision, split)
        writer = ShardWriter(shard_dir, source_id, args.shard_mb * 1024 * 1024)
        counters = {
            "seen": 0,
            "accepted_documents": 0,
            "accepted": 0,
            "short_or_empty": 0,
            "license_rejected": 0,
            "partition_rejected": 0,
            "exact_document_duplicate": 0,
            "exact_chunk_duplicate": 0,
            "accepted_chars": 0,
        }

        try:
            for row in rows:
                if counters["accepted_chars"] >= char_budget:
                    break
                counters["seen"] += 1
                text = normalize_text(str(nested_get(row, text_field) or ""))
                if len(text) < args.min_chars:
                    counters["short_or_empty"] += 1
                    continue

                row_license = nested_get(row, license_field)
                if row_license in (None, "") or str(row_license) not in allowed_licenses:
                    counters["license_rejected"] += 1
                    continue

                document_sha = sha256_text(text)
                if not partition_match(document_sha, args.partition_index, args.partition_count):
                    counters["partition_rejected"] += 1
                    continue
                if db.execute(
                    "INSERT OR IGNORE INTO seen_documents(sha256) VALUES (?)", (document_sha,)
                ).rowcount == 0:
                    counters["exact_document_duplicate"] += 1
                    continue

                source_record_id = str(nested_get(row, id_field) or "")
                document_split = stable_split(document_sha)
                wrote_document = False
                for chunk_index, raw_chunk in enumerate(
                    chunk_text(text, args.max_document_chunk_chars, args.min_chars)
                ):
                    remaining = char_budget - counters["accepted_chars"]
                    chunk = trim_to_budget(raw_chunk, remaining, args.min_chars)
                    if not chunk:
                        break
                    chunk_sha = sha256_text(chunk)
                    if db.execute(
                        "INSERT OR IGNORE INTO seen_chunks(sha256) VALUES (?)", (chunk_sha,)
                    ).rowcount == 0:
                        counters["exact_chunk_duplicate"] += 1
                        continue

                    writer.write(
                        {
                            "text": chunk,
                            "text_sha256": chunk_sha,
                            "document_sha256": document_sha,
                            "document_chunk_index": chunk_index,
                            "split": document_split,
                            "source_id": source_id,
                            "source_category": source["category"],
                            "source_target_share": target_share,
                            "source_repo": repo_id,
                            "source_revision": resolved_revision,
                            "source_record_id": source_record_id,
                            "source_provenance": nested_get(row, provenance_field),
                            "source_url": nested_get(row, url_field),
                            "license_expression": str(row_license),
                        }
                    )
                    wrote_document = True
                    counters["accepted"] += 1
                    counters["accepted_chars"] += len(chunk)
                    total_accepted_rows += 1
                    total_accepted_chars += len(chunk)
                    if counters["accepted_chars"] >= char_budget:
                        break
                if wrote_document:
                    counters["accepted_documents"] += 1
                if counters["accepted"] and counters["accepted"] % 10_000 == 0:
                    db.commit()
                    print(json.dumps({"source_id": source_id, **counters}, sort_keys=True))
        finally:
            writer.close()
            db.commit()

        source_files = sorted(shard_dir.glob(f"{source_id}-*.jsonl"))
        fill_ratio = counters["accepted_chars"] / char_budget
        if fill_ratio < args.minimum_fill_ratio:
            failures.append(
                f"{source_id}: fill_ratio={fill_ratio:.4f} below minimum={args.minimum_fill_ratio:.4f}"
            )

        source_receipt = {
            "category": source["category"],
            "source_id": source_id,
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "split": split,
            "target_share": target_share,
            "char_budget": char_budget,
            "fill_ratio": fill_ratio,
            "license_field": license_field,
            "allowed_license_values": sorted(allowed_licenses),
            "require_row_license": True,
            "counters": counters,
            "shards": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in source_files
            ],
        }
        receipt["sources"].append(source_receipt)
        print(json.dumps(source_receipt, sort_keys=True))

    db.close()
    receipt["accepted_rows"] = total_accepted_rows
    receipt["accepted_chars"] = total_accepted_chars
    receipt["fill_failures"] = failures
    receipt["status"] = "PASS" if not failures else "FAIL"

    receipt_path = output_dir / "corpus_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "accepted_rows": total_accepted_rows,
        "accepted_chars": total_accepted_chars,
        "source_count": len(sources),
        "partition_index": args.partition_index,
        "partition_count": args.partition_count,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"FILL_FAILURE {failure}")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
