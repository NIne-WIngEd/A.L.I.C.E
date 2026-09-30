"""Stage and reconcile public MFM foundation *candidates*, without admitting them.

An upstream shard is processed in full. The size of a candidate output part is
only a transport boundary, never a row, character, token or corpus limit.
Exact licenses from N0 are an initial metadata filter, not a grant of rights.
Every surviving row remains blocked on source, privacy and contamination review.

The CPU staging worker can run where pinned upstream bytes are accessible;
reconciliation uses a disk-backed index across all workers and source families.
Neither command downloads a dataset, authenticates its origin, creates training
admission, generates teacher labels or starts an optimizer.
"""

from __future__ import annotations

import argparse
import base64
from datetime import date, datetime, time
from decimal import Decimal
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import unicodedata
from urllib.parse import urlsplit


STAGE_SCHEMA = "mfm-native-public-shard-candidates-v1"
GLOBAL_SCHEMA = "mfm-native-public-global-candidates-v1"
EXCLUSION_SCHEMA = "mfm-native-public-exclusions-v1"
RECEIPT_LIST_SCHEMA = "mfm-native-public-receipt-list-v1"
RESERVED = frozenset({"LongMemEval-V2", "LoCoMo", "CareCall"})


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _checked_digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256")
    return value


def _verified_file(path: Path, expected: str) -> str:
    expected = _checked_digest(expected)
    if not path.is_file() or _digest(path) != expected:
        raise ValueError(f"file missing or SHA-256 mismatch: {path}")
    return expected


def _bound_json(path: Path, expected: str) -> dict:
    _verified_file(path, expected)
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def _json_bytes(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _nested(row: dict, field: str):
    result = row
    for part in field.split("."):
        if not isinstance(result, dict):
            return None
        result = result.get(part)
    return result


def _nonempty(value) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value) and isinstance(value, (dict, list, int, float))


def _text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False) if _nonempty(value) else ""


def _canonical(value):
    """Stable JSON form for Parquet scalar types in held-row evidence."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__float__": value.hex()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_base64__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (datetime, date, time)):
        return {"__type__": type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(part) for part in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("upstream row has non-string Parquet map keys")
        return {key: _canonical(part) for key, part in sorted(value.items())}
    raise ValueError(f"unsupported upstream Parquet scalar {type(value).__name__}")


def _row_sha256(row: dict) -> str:
    return sha256(json.dumps(_canonical(row), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")).hexdigest()


def _safe_relative(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or
            name.startswith("/") or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("unsafe relative path")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("relative path escapes transport root")
    return result


def _normalize(value: str) -> str:
    return "\n".join(line.rstrip() for line in
                     unicodedata.normalize("NFC", value).replace("\r\n", "\n")
                     .replace("\r", "\n").split("\n")).strip()


def _source_url(value) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        return ""
    try:
        url = urlsplit(value)
        valid = url.scheme in {"https", "http"} and bool(url.hostname)
    except ValueError:
        valid = False
    if not valid or any(ch.isspace() for ch in value):
        return ""
    return value


def _rows(path: Path, format_name: str):
    if format_name in {"jsonl", "jsonl.gz"}:
        opener = gzip.open if format_name == "jsonl.gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("empty upstream JSONL row")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("upstream row must be a JSON object")
                yield row
    elif format_name == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet shards require pyarrow in the CPU staging environment") from exc
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise ValueError("format must be jsonl, jsonl.gz or parquet")


def _source(candidate_file: Path, candidate_sha: str, n0_file: Path,
            n0_sha: str, candidate_id: str) -> tuple[dict, dict]:
    candidates = _bound_json(candidate_file, candidate_sha)
    n0 = _bound_json(n0_file, n0_sha)
    if (candidates.get("schema") != "mfm-native-foundation-source-candidates-v1" or
            candidates.get("status") != "candidate_only_no_training_admission" or
            n0.get("schema") != "alice.eipm.n0.public-corpus-sources.v0.2.1" or
            n0.get("status") != "activated_public_n0_v02" or
            candidates.get("prior_n0_source_manifest_reference", {}).get("sha256_observed") != n0_sha):
        raise ValueError("candidate inventory or N0 exact-source evidence is not pinned")
    matched = [row for row in candidates["candidates"] if row["candidate_id"] == candidate_id]
    evidence = [row for row in n0["sources"] if row["source_id"] == candidate_id]
    if len(matched) != 1 or len(evidence) != 1:
        raise ValueError("source is not one of the N0-rechecked Common Pile candidates")
    candidate, source = matched[0], evidence[0]
    if (candidate.get("status") != "candidate_not_admitted" or
            source.get("repo_id") != candidate.get("dataset_repo_id") or
            source.get("revision") != candidate.get("candidate_revision_from_n0") or
            source.get("allowed_license_values") != candidate.get("previous_n0_license_values_reference_only") or
            source.get("require_row_license") is not True or
            source.get("split") != candidate.get("split_from_n0")):
        raise ValueError("source-specific revision or row license evidence differs")
    return candidate, source


class _PartWriter:
    def __init__(self, root: Path, part_bytes: int):
        if part_bytes < 1:
            raise ValueError("part_bytes must be positive; it is a transport boundary")
        self.root, self.part_bytes = root, part_bytes
        self.parts: list[dict] = []
        self.stream = None
        self.size = 0
        self.count = 0

    def _close(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            name = f"candidates-{len(self.parts):06d}.jsonl"
            path = self.root / name
            self.parts.append({"path": name, "sha256": _digest(path),
                               "bytes": path.stat().st_size, "rows": self.count})
            self.stream, self.size, self.count = None, 0, 0

    def write(self, row: dict):
        payload = _json_bytes(row)
        if self.stream is not None and self.size and self.size + len(payload) > self.part_bytes:
            self._close()
        if self.stream is None:
            self.stream = (self.root / f"candidates-{len(self.parts):06d}.jsonl").open("xb")
        self.stream.write(payload)
        self.size += len(payload)
        self.count += 1

    def close(self):
        self._close()


def stage_shard(*, candidate_file: Path, candidate_sha: str, n0_file: Path,
                n0_sha: str, candidate_id: str, raw_file: Path, raw_sha: str,
                upstream_repo_path: str, format_name: str, output: Path,
                acquisition_receipt: Path, acquisition_sha: str,
                part_bytes: int = 128 * 1024 * 1024) -> dict:
    """Process every record from one pinned upstream shard, without admission."""
    candidate, source = _source(candidate_file, candidate_sha, n0_file, n0_sha,
                                candidate_id)
    if (not upstream_repo_path or upstream_repo_path.startswith("/") or
            "\\" in upstream_repo_path or
            any(p in ("", ".", "..") for p in upstream_repo_path.split("/"))):
        raise ValueError("upstream_repo_path must be an exact relative dataset file")
    _verified_file(raw_file, raw_sha)
    acquired = _bound_json(acquisition_receipt, acquisition_sha)
    if (acquired.get("schema") != "mfm-native-hf-exact-file-acquisition-v1" or
            acquired.get("status") != "bytes_verified_no_training_admission" or
            acquired.get("training_admitted") is not False or
            acquired.get("raw_origin_repo_metadata_verified") is not True or
            acquired.get("candidate_inventory_sha256") != candidate_sha or
            acquired.get("n0_manifest_sha256") != n0_sha or
            acquired.get("candidate_id") != candidate_id or
            acquired.get("dataset_repo_id") != source["repo_id"] or
            acquired.get("revision") != source["revision"] or
            acquired.get("upstream_repo_path") != upstream_repo_path or
            acquired.get("downloaded_sha256") != raw_sha or
            acquired.get("downloaded_bytes") != raw_file.stat().st_size or
            raw_file.resolve() != _safe_relative(acquisition_receipt.parent.resolve(),
                                                 upstream_repo_path)):
        raise ValueError("upstream bytes have no matching exact-revision acquisition receipt")
    output = output.resolve()
    if output.exists():
        receipt_file = output / "shard_receipt.json"
        if not receipt_file.is_file():
            raise ValueError("existing output is incomplete; choose a new output directory")
        receipt = json.loads(receipt_file.read_text("utf-8"))
        if (receipt.get("candidate_inventory_sha256") != candidate_sha or
                receipt.get("n0_manifest_sha256") != n0_sha or
                receipt.get("candidate_id") != candidate_id or
                receipt.get("upstream_sha256") != raw_sha or
                receipt.get("upstream_repo_path") != upstream_repo_path or
                receipt.get("acquisition_receipt_sha256") != acquisition_sha or
                receipt.get("format") != format_name or
                receipt.get("part_bytes") != part_bytes):
            raise ValueError("existing receipt differs from requested pinned shard")
        verify_shard(receipt_file, raw_file=raw_file)
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mfm-shard-", dir=output.parent) as temp:
        temp_root = Path(temp)
        writer = _PartWriter(temp_root, part_bytes)
        allowed = set(source["allowed_license_values"])
        counts = {"seen": 0, "candidates": 0, "held_license": 0,
                  "held_provenance": 0, "held_source_url": 0,
                  "held_invalid_url": 0,
                  "held_record_id": 0, "held_empty": 0}
        with (temp_root / "held_rows.jsonl").open("wb") as held:
            for ordinal, row in enumerate(_rows(raw_file, format_name)):
                counts["seen"] += 1
                license_value = _nested(row, source["license_field"])
                provenance = _text(_nested(row, source["provenance_field"]))
                raw_url = _nested(row, source["url_field"])
                source_url = _source_url(raw_url)
                record_id = _text(_nested(row, source["id_field"]))
                raw_text = _nested(row, source["text_field"])
                content = _normalize(raw_text) if isinstance(raw_text, str) else ""
                reason = ("held_license" if not isinstance(license_value, str) or
                          license_value not in allowed else
                          "held_provenance" if not provenance else
                          "held_source_url" if not _nonempty(raw_url) else
                          "held_invalid_url" if not source_url else
                          "held_record_id" if not record_id else
                          "held_empty" if not content else None)
                if reason:
                    counts[reason] += 1
                    held.write(_json_bytes({"row_ordinal": ordinal, "reason": reason,
                                            "upstream_row_sha256": _row_sha256(row)}))
                    continue
                content_sha = sha256(content.encode("utf-8")).hexdigest()
                identity = f"{source['repo_id']}@{source['revision']}:{upstream_repo_path}:{ordinal}:{record_id}"
                row_id = sha256(identity.encode("utf-8")).hexdigest()
                writer.write({"schema": STAGE_SCHEMA, "item_id": row_id,
                              "candidate_id": candidate_id, "category": candidate["category"],
                              "source_id": "public-url:" + sha256(source_url.encode("utf-8")).hexdigest(),
                              "modality": "text", "upstream_repo": source["repo_id"],
                              "upstream_revision": source["revision"],
                              "upstream_repo_path": upstream_repo_path,
                              "upstream_file_sha256": raw_sha, "row_ordinal": ordinal,
                              "upstream_item_ref": record_id, "source_url": source_url,
                              "source_provenance": provenance, "license_expression": license_value,
                              "content_sha256": content_sha, "text": content,
                              "source_lineage_key": sha256(source_url.encode("utf-8")).hexdigest(),
                              "status": "candidate_unreviewed_no_training_admission",
                              "rights_review": "pending", "privacy_review": "pending",
                              "contamination_review": "pending"})
                counts["candidates"] += 1
            held.flush()
            os.fsync(held.fileno())
        writer.close()
        hold_path = temp_root / "held_rows.jsonl"
        receipt = {"schema": STAGE_SCHEMA, "status": "candidate_unreviewed_no_training_admission",
                   "candidate_inventory_sha256": candidate_sha,
                   "n0_manifest_sha256": n0_sha, "candidate_id": candidate_id,
                   "upstream_repo": source["repo_id"],
                   "upstream_revision": source["revision"],
                   "upstream_repo_path": upstream_repo_path,
                   "upstream_sha256": raw_sha, "upstream_bytes": raw_file.stat().st_size,
                   "acquisition_receipt_sha256": acquisition_sha,
                   "format": format_name, "part_bytes": part_bytes,
                   "counts": counts, "parts": writer.parts,
                   "holds": {"path": "held_rows.jsonl", "sha256": _digest(hold_path),
                             "bytes": hold_path.stat().st_size,
                             "rows": counts["seen"] - counts["candidates"]},
                   "raw_origin_repo_metadata_verified": True,
                   "human_rights_review_completed": False,
                   "human_privacy_review_completed": False,
                   "semantic_contamination_cleared": False,
                   "training_admitted": False}
        (temp_root / "shard_receipt.json").write_bytes(_json_bytes(receipt))
        os.rename(temp_root, output)
        return receipt


def verify_shard(receipt_file: Path, *, raw_file: Path | None = None) -> dict:
    """Check transfer bytes; origin and per-row rights remain separate audits."""
    receipt = json.loads(receipt_file.read_text("utf-8"))
    if (receipt.get("schema") != STAGE_SCHEMA or
            receipt.get("status") != "candidate_unreviewed_no_training_admission" or
            receipt.get("training_admitted") is not False):
        raise ValueError("unexpected or purportedly admitted shard receipt")
    if raw_file is not None:
        _verified_file(raw_file, receipt["upstream_sha256"])
        if raw_file.stat().st_size != receipt["upstream_bytes"]:
            raise ValueError("upstream byte count changed")
    root = receipt_file.parent.resolve()
    seen = 0
    for part in receipt["parts"]:
        file = _safe_relative(root, part["path"])
        _verified_file(file, part["sha256"])
        if file.stat().st_size != part["bytes"]:
            raise ValueError("candidate part byte count changed")
        with file.open("rb") as stream:
            rows = sum(1 for _ in stream)
        if rows != part["rows"]:
            raise ValueError("candidate part row count changed")
        seen += rows
    if seen != receipt["counts"]["candidates"]:
        raise ValueError("candidate count changed")
    holds = receipt["holds"]
    held_file = _safe_relative(root, holds["path"])
    _verified_file(held_file, holds["sha256"])
    with held_file.open("rb") as stream:
        held_count = sum(1 for _ in stream)
    if (held_file.stat().st_size != holds["bytes"] or held_count != holds["rows"] or
            held_count + seen != receipt["counts"]["seen"]):
        raise ValueError("held row ledger changed")
    return receipt


def _find(db: sqlite3.Connection, key: str) -> str:
    visited = []
    while True:
        row = db.execute("SELECT parent FROM nodes WHERE key=?", (key,)).fetchone()
        if row is None:
            db.execute("INSERT INTO nodes VALUES (?,?)", (key, key))
            root = key
            break
        parent = row[0]
        if parent == key:
            root = key
            break
        visited.append(key)
        key = parent
    for child in visited:
        db.execute("UPDATE nodes SET parent=? WHERE key=?", (root, child))
    return root


def _union(db: sqlite3.Connection, keys: tuple[str, ...]) -> None:
    roots = [_find(db, key) for key in keys]
    smallest = min(roots)
    for root in roots:
        if root != smallest:
            db.execute("UPDATE nodes SET parent=? WHERE key=?", (smallest, root))


def reconcile(*, receipt_list: Path, receipt_list_sha: str, exclusions: Path,
              exclusions_sha: str, output: Path) -> dict:
    """Verify all parts and reconcile exact duplicates/source ancestry globally."""
    bundle = _bound_json(receipt_list, receipt_list_sha)
    blocked = _bound_json(exclusions, exclusions_sha)
    if (bundle.get("schema") != RECEIPT_LIST_SCHEMA or
            blocked.get("schema") != EXCLUSION_SCHEMA or
            not RESERVED.issubset(set(blocked.get("reserved_evaluations", []))) or
            not isinstance(bundle.get("receipts"), list) or not bundle["receipts"]):
        raise ValueError("frozen receipt list and reserved evaluation exclusions required")
    for name in ("candidate_ids", "content_sha256s", "source_urls", "upstream_item_refs"):
        if not isinstance(blocked.get(name), list):
            raise ValueError(f"exclusions.{name} must be a list")
    for value in blocked["content_sha256s"]:
        _checked_digest(value)
    excluded = {name: set(blocked[name]) for name in
                ("candidate_ids", "content_sha256s", "source_urls", "upstream_item_refs")}
    root = receipt_list.parent.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("global output already exists; frozen reconciliation is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mfm-global-", dir=output.parent) as temporary:
        receipt = _reconcile_into(bundle=bundle, excluded=excluded,
                                  root=root, output=Path(temporary),
                                  receipt_list_sha=receipt_list_sha,
                                  exclusions_sha=exclusions_sha)
        os.rename(temporary, output)
    return receipt


def _reconcile_into(*, bundle: dict, excluded: dict, root: Path, output: Path,
                    receipt_list_sha: str, exclusions_sha: str) -> dict:
    db = sqlite3.connect(output / "reconciliation.sqlite3")
    db.execute("CREATE TABLE nodes (key TEXT PRIMARY KEY, parent TEXT NOT NULL)")
    db.execute("CREATE TABLE candidates (item_id TEXT PRIMARY KEY, content_sha TEXT NOT NULL, "
               "lineage_key TEXT NOT NULL, source_url TEXT NOT NULL, candidate_id TEXT NOT NULL, "
               "upstream_ref TEXT NOT NULL, receipt_path TEXT NOT NULL, part_path TEXT NOT NULL, "
               "byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL)")
    db.execute("CREATE INDEX content_idx ON candidates(content_sha,item_id)")
    db.execute("CREATE TABLE blocked_keys (key TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE blocked_roots (root TEXT PRIMARY KEY)")
    source_sha = None
    seen_receipts: set[str] = set()
    count = 0
    with db:
        for link in bundle["receipts"]:
            receipt_name = link["path"]
            if receipt_name in seen_receipts:
                raise ValueError("same shard receipt registered twice")
            seen_receipts.add(receipt_name)
            receipt_file = _safe_relative(root, receipt_name)
            _verified_file(receipt_file, link["sha256"])
            receipt = verify_shard(receipt_file)
            if source_sha is None:
                source_sha = receipt["candidate_inventory_sha256"]
            if source_sha != receipt["candidate_inventory_sha256"]:
                raise ValueError("shards use different candidate inventories")
            for part in receipt["parts"]:
                path = _safe_relative(receipt_file.parent.resolve(), part["path"])
                with path.open("rb") as stream:
                    while True:
                        offset = stream.tell()
                        raw = stream.readline()
                        if not raw:
                            break
                        row = json.loads(raw)
                        if (row.get("schema") != STAGE_SCHEMA or
                                row.get("status") != "candidate_unreviewed_no_training_admission" or
                                row.get("candidate_id") != receipt["candidate_id"] or
                                row.get("upstream_repo") != receipt["upstream_repo"] or
                                row.get("upstream_file_sha256") != receipt["upstream_sha256"] or
                                row.get("upstream_revision") != receipt["upstream_revision"] or
                                row.get("upstream_repo_path") != receipt["upstream_repo_path"] or
                                row.get("source_id") != "public-url:" + row["source_lineage_key"] or
                                row.get("item_id") != sha256((
                                    f"{row['upstream_repo']}@{row['upstream_revision']}:"
                                    f"{row['upstream_repo_path']}:{row['row_ordinal']}:"
                                    f"{row['upstream_item_ref']}"
                                ).encode("utf-8")).hexdigest() or
                                sha256(row["text"].encode("utf-8")).hexdigest() != row["content_sha256"] or
                                sha256(row["source_url"].encode("utf-8")).hexdigest() != row["source_lineage_key"]):
                            raise ValueError("candidate row does not bind stage receipt")
                        lineage = "url:" + row["source_lineage_key"]
                        exact = "exact:" + row["content_sha256"]
                        upstream = (f"record:{row['upstream_repo']}@{row['upstream_revision']}:"
                                    f"{row['upstream_item_ref']}")
                        _union(db, (lineage, exact, upstream))
                        db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)",
                                   (row["item_id"], row["content_sha256"], lineage,
                                    row["source_url"], row["candidate_id"],
                                    row["upstream_item_ref"], receipt_name,
                                    part["path"], offset, len(raw)))
                        if (row["candidate_id"] in excluded["candidate_ids"] or
                                row["content_sha256"] in excluded["content_sha256s"] or
                                row["source_url"] in excluded["source_urls"] or
                                row["upstream_item_ref"] in excluded["upstream_item_refs"]):
                            db.execute("INSERT OR IGNORE INTO blocked_keys VALUES (?)", (lineage,))
                        count += 1
    for (key,) in db.execute("SELECT key FROM blocked_keys"):
        db.execute("INSERT OR IGNORE INTO blocked_roots VALUES (?)", (_find(db, key),))
    db.commit()
    totals = {"rows": count, "candidate_train": 0, "candidate_development": 0,
              "excluded_hold": 0, "exact_duplicate_hold": 0}
    # The first item for each exact payload is the sole candidate representative.
    # This does not purport to detect paraphrases or semantic near duplicates.
    selected_content = None
    with (output / "candidate_index.jsonl").open("wb") as stream:
        for row in db.execute("SELECT item_id,content_sha,lineage_key,source_url,"
                              "candidate_id,upstream_ref,receipt_path,part_path,"
                              "byte_offset,byte_length FROM candidates ORDER BY content_sha,item_id"):
            (item_id, content_sha, lineage, url, candidate_id, upstream_ref,
             receipt_path, part_path, offset, length) = row
            root_key = _find(db, lineage)
            bucket = int(sha256(root_key.encode("utf-8")).hexdigest()[:8], 16) % 10000
            split = "development" if bucket < 100 else "train"
            duplicate = content_sha == selected_content
            selected_content = content_sha
            is_blocked = db.execute("SELECT 1 FROM blocked_roots WHERE root=?",
                                    (root_key,)).fetchone() is not None
            status = ("excluded_hold" if is_blocked else
                      "exact_duplicate_hold" if duplicate else f"candidate_{split}")
            totals[status] += 1
            stream.write(_json_bytes({"schema": GLOBAL_SCHEMA, "item_id": item_id,
                                      "candidate_id": candidate_id,
                                      "source_id": "public-url:" + sha256(url.encode("utf-8")).hexdigest(),
                                      "source_url": url, "upstream_item_ref": upstream_ref,
                                      "content_sha256": content_sha,
                                      "lineage_component": root_key,
                                      "split_candidate": split,
                                      "status": status, "shard_receipt": receipt_path,
                                      "part_path": part_path,
                                      "byte_offset": offset, "byte_length": length,
                                      "rights_review": "pending", "privacy_review": "pending",
                                      "contamination_review": "pending",
                                      "training_admitted": False}))
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": GLOBAL_SCHEMA, "status": "candidate_unreviewed_no_training_admission",
               "receipt_list_sha256": receipt_list_sha,
               "exclusions_sha256": exclusions_sha,
               "candidate_inventory_sha256": source_sha,
               "candidate_index_sha256": _digest(output / "candidate_index.jsonl"),
               "totals": totals, "split_assignment": "source_lineage_and_exact_duplicate_component",
               "semantic_near_duplicates_cleared": False,
               "raw_origin_authenticated": False, "human_rights_review_completed": False,
               "human_privacy_review_completed": False,
               "independent_final_created": False, "training_admitted": False}
    (output / "global_receipt.json").write_bytes(_json_bytes(receipt))
    db.close()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="stream one exact upstream shard")
    stage.add_argument("--candidate-inventory", type=Path, required=True)
    stage.add_argument("--candidate-sha256", required=True)
    stage.add_argument("--n0-manifest", type=Path, required=True)
    stage.add_argument("--n0-sha256", required=True)
    stage.add_argument("--candidate-id", required=True)
    stage.add_argument("--raw-file", type=Path, required=True)
    stage.add_argument("--raw-sha256", required=True)
    stage.add_argument("--acquisition-receipt", type=Path, required=True)
    stage.add_argument("--acquisition-sha256", required=True)
    stage.add_argument("--upstream-repo-path", required=True)
    stage.add_argument("--format", choices=("jsonl", "jsonl.gz", "parquet"), required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--part-bytes", type=int, default=128 * 1024 * 1024)
    verify = commands.add_parser("verify", help="verify a transferred shard")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--raw-file", type=Path)
    global_cmd = commands.add_parser("reconcile", help="disk-backed cross-shard reconciliation")
    global_cmd.add_argument("--receipt-list", type=Path, required=True)
    global_cmd.add_argument("--receipt-list-sha256", required=True)
    global_cmd.add_argument("--exclusions", type=Path, required=True)
    global_cmd.add_argument("--exclusions-sha256", required=True)
    global_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage_shard(candidate_file=args.candidate_inventory,
                             candidate_sha=args.candidate_sha256, n0_file=args.n0_manifest,
                             n0_sha=args.n0_sha256, candidate_id=args.candidate_id,
                             raw_file=args.raw_file, raw_sha=args.raw_sha256,
                             upstream_repo_path=args.upstream_repo_path,
                             format_name=args.format, output=args.output,
                             acquisition_receipt=args.acquisition_receipt,
                             acquisition_sha=args.acquisition_sha256,
                             part_bytes=args.part_bytes)
    elif args.command == "verify":
        result = verify_shard(args.receipt, raw_file=args.raw_file)
    else:
        result = reconcile(receipt_list=args.receipt_list,
                           receipt_list_sha=args.receipt_list_sha256,
                           exclusions=args.exclusions, exclusions_sha=args.exclusions_sha256,
                           output=args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
