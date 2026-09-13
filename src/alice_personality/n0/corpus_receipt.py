from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus_receipt(
    corpus_dir: str | Path,
    source_config_path: str | Path,
) -> dict[str, Any]:
    corpus_root = Path(corpus_dir).resolve()
    source_config_path = Path(source_config_path)
    receipt_path = corpus_root / "corpus_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing corpus receipt: {receipt_path}")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("private_identity_data") is not False:
        raise ValueError("N0 corpus receipt must explicitly declare private_identity_data=false")

    expected_config_hash = sha256_file(source_config_path)
    if receipt.get("source_config_sha256") != expected_config_hash:
        raise ValueError("corpus receipt source_config_sha256 does not match active source config")

    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    expected_source_ids = [str(source["source_id"]) for source in source_config["sources"]]
    observed_sources = receipt.get("sources")
    if not isinstance(observed_sources, list) or not observed_sources:
        raise ValueError("corpus receipt contains no sources")
    observed_source_ids = [str(source.get("source_id", "")) for source in observed_sources]
    if observed_source_ids != expected_source_ids:
        raise ValueError(
            "corpus receipt source order/identity does not match active source config: "
            f"expected={expected_source_ids} observed={observed_source_ids}"
        )

    verified_shards = 0
    verified_bytes = 0
    accepted_rows = 0
    accepted_chars = 0
    shard_paths: list[str] = []

    for source in observed_sources:
        if not source.get("resolved_revision"):
            raise ValueError(f"source {source.get('source_id')} has no resolved revision")
        counters = source.get("counters") or {}
        accepted = int(counters.get("accepted", 0))
        chars = int(counters.get("accepted_chars", 0))
        if accepted < 1 or chars < 1:
            raise ValueError(f"source {source.get('source_id')} accepted no usable text")
        accepted_rows += accepted
        accepted_chars += chars

        shards = source.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"source {source.get('source_id')} has no manifested shards")
        for shard in shards:
            relative = Path(str(shard["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe shard path in corpus receipt: {relative}")
            path = (corpus_root / relative).resolve()
            if corpus_root not in path.parents:
                raise ValueError(f"shard path escapes corpus root: {relative}")
            if not path.is_file():
                raise ValueError(f"manifested corpus shard is missing: {relative}")
            expected_bytes = int(shard["bytes"])
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"corpus shard size mismatch for {relative}: "
                    f"expected={expected_bytes} actual={actual_bytes}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != str(shard["sha256"]):
                raise ValueError(f"corpus shard SHA256 mismatch for {relative}")
            verified_shards += 1
            verified_bytes += actual_bytes
            shard_paths.append(str(path))

    return {
        "status": "PASS",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "source_config_sha256": expected_config_hash,
        "source_count": len(observed_sources),
        "accepted_rows": accepted_rows,
        "accepted_chars": accepted_chars,
        "verified_shards": verified_shards,
        "verified_bytes": verified_bytes,
        "shard_paths": shard_paths,
        "private_identity_data": False,
    }
