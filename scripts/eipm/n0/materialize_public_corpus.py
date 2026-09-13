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


def stable_split(text_sha256: str) -> str:
    bucket = int(text_sha256[:8], 16) % 10_000
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


@dataclass
class ShardWriter:
    root: Path
    source_id: str
    target_bytes: int
    index: int = 0
    bytes_written: int = 0
    rows_written: int = 0
    handle: Any = None
    current_path: Path | None = None

    def _open(self) -> None:
        self.current_path = self.root / f"{self.source_id}-{self.index:05d}.jsonl"
        self.handle = self.current_path.open("a", encoding="utf-8")
        self.bytes_written = self.current_path.stat().st_size if self.current_path.exists() else 0

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
        self.rows_written += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def load_source_rows(repo_id: str, revision: str, split: str) -> tuple[str, Iterator[dict[str, Any]]]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before materializing the public corpus") from exc

    api = HfApi()
    info = api.dataset_info(repo_id=repo_id, revision=revision)
    resolved_revision = info.sha
    dataset = load_dataset(repo_id, split=split, streaming=True, revision=resolved_revision)
    return resolved_revision, iter(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize license-gated public N0 text from explicit Hugging Face sources."
    )
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-mb", type=int, default=256)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-chars-per-source", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=80)
    args = parser.parse_args()

    config_path = Path(args.source_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(output_dir / "exact_dedup.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS seen (text_sha256 TEXT PRIMARY KEY)")
    db.commit()

    run_receipt: dict[str, Any] = {
        "schema": "alice.eipm.n0.public-corpus-receipt.v0.1",
        "source_config_sha256": sha256_file(config_path),
        "sources": [],
        "private_identity_data": False,
        "license_gate": "row_or_source_license_must_match_explicit_allowlist",
    }

    for source in config["sources"]:
        source_id = str(source["source_id"])
        repo_id = str(source["repo_id"])
        revision = str(source.get("revision", "main"))
        split = str(source.get("split", "train"))
        text_field = str(source.get("text_field", "text"))
        license_field = source.get("license_field", "metadata.license")
        allowed_licenses = {str(value) for value in source.get("allowed_license_values", [])}
        source_level_license = source.get("source_level_license")
        require_row_license = bool(source.get("require_row_license", True))

        if not allowed_licenses and not source_level_license:
            raise SystemExit(f"{source_id}: no license allowlist/source-level license declared")

        resolved_revision, rows = load_source_rows(repo_id, revision, split)
        writer = ShardWriter(
            root=shard_dir,
            source_id=source_id,
            target_bytes=args.shard_mb * 1024 * 1024,
        )
        counters = {
            "seen": 0,
            "accepted": 0,
            "short_or_empty": 0,
            "license_rejected": 0,
            "exact_duplicate": 0,
            "accepted_chars": 0,
        }

        try:
            for row in rows:
                counters["seen"] += 1
                raw_text = nested_get(row, text_field)
                text = normalize_text("" if raw_text is None else str(raw_text))
                if len(text) < args.min_chars:
                    counters["short_or_empty"] += 1
                    continue

                row_license = nested_get(row, str(license_field)) if license_field else None
                effective_license = str(row_license) if row_license not in (None, "") else source_level_license
                if require_row_license and row_license in (None, ""):
                    counters["license_rejected"] += 1
                    continue
                if allowed_licenses and effective_license not in allowed_licenses:
                    counters["license_rejected"] += 1
                    continue
                if effective_license is None:
                    counters["license_rejected"] += 1
                    continue

                digest = sha256_text(text)
                inserted = db.execute(
                    "INSERT OR IGNORE INTO seen(text_sha256) VALUES (?)", (digest,)
                ).rowcount
                if inserted == 0:
                    counters["exact_duplicate"] += 1
                    continue

                provenance = nested_get(row, str(source.get("provenance_field", "metadata.provenance")))
                original_url = nested_get(row, str(source.get("url_field", "metadata.url")))
                output_row = {
                    "text": text,
                    "text_sha256": digest,
                    "split": stable_split(digest),
                    "source_id": source_id,
                    "source_repo": repo_id,
                    "source_revision": resolved_revision,
                    "source_record_id": str(row.get(source.get("id_field", "id"), "")),
                    "source_provenance": provenance,
                    "source_url": original_url,
                    "license_expression": effective_license,
                }
                writer.write(output_row)
                counters["accepted"] += 1
                counters["accepted_chars"] += len(text)

                if counters["accepted"] % 10_000 == 0:
                    db.commit()
                    print(json.dumps({"source": source_id, **counters}, sort_keys=True))

                if args.max_docs_per_source and counters["accepted"] >= args.max_docs_per_source:
                    break
                if args.max_chars_per_source and counters["accepted_chars"] >= args.max_chars_per_source:
                    break
        finally:
            writer.close()
            db.commit()

        source_files = sorted(shard_dir.glob(f"{source_id}-*.jsonl"))
        source_receipt = {
            "source_id": source_id,
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "split": split,
            "license_field": license_field,
            "allowed_license_values": sorted(allowed_licenses),
            "source_level_license": source_level_license,
            "require_row_license": require_row_license,
            "counters": counters,
            "shards": [
                {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in source_files
            ],
        }
        run_receipt["sources"].append(source_receipt)
        print(json.dumps(source_receipt, sort_keys=True))

    db.close()
    receipt_path = output_dir / "corpus_receipt.json"
    receipt_path.write_text(json.dumps(run_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), "sha256": sha256_file(receipt_path)}))


if __name__ == "__main__":
    main()
