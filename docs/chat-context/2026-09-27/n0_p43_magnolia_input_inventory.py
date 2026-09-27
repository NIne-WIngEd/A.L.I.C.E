#!/usr/bin/env python3
"""Read-only P43 portability inventory. Does not serialize corpus/teacher row content."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "a19f8e8893422702c138182f239064385addf91c"
MIXTURE_SHA = "791f342287d124770a193c175413f3df5d7043a963d1d3d655127baf5c38c8be"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def measured(path: Path, *, hash_file: bool = True) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {"path": str(path), "bytes": path.stat().st_size}
    if hash_file:
        result["sha256"] = digest(path)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--mixture", type=Path, required=True)
    a = p.parse_args()
    repo, work, mixture = a.repo.resolve(), a.work.resolve(), a.mixture.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != REVISION or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise SystemExit("STOP: N0 source must be clean at pinned a19f8e88 before inventory")
    teacher = work / "teacher-bank-v0.5"
    registry = teacher / "n0_v02_teacher_bank_v0.5.runtime.json"
    audit = teacher / "teacher-bank-v0.5-audit.json"
    manifest = mixture / "full_public_mixture_manifest.json"
    mix_audit = mixture / "full_public_mixture_audit.json"
    if digest(manifest) != MIXTURE_SHA:
        raise SystemExit("STOP: P39PN manifest hash mismatch")
    mix = json.loads(manifest.read_text(encoding="utf-8"))
    if mix.get("source_revision") != REVISION:
        raise SystemExit("STOP: P39PN source revision mismatch")
    bound = mix["training_lanes"]["governed_judgment_replay"]
    if bound["teacher_registry_sha256"] != digest(registry) or bound["teacher_audit_sha256"] != digest(audit):
        raise SystemExit("STOP: P39PN teacher binding mismatch")
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    if audit_value.get("private_identity_data") is not False:
        raise SystemExit("STOP: teacher bank contains private identity data")
    registry_rows = json.loads(registry.read_text(encoding="utf-8"))["shards"]
    teacher_rows = []
    for row in registry_rows:
        pair = {}
        for key in ("curriculum", "manifest"):
            raw = Path(row[key])
            path = raw if raw.is_absolute() else repo / raw
            pair[key] = {"path_kind": "absolute" if raw.is_absolute() else "repo_relative",
                         **measured(path.resolve())}
        teacher_rows.append(pair)
    corpus = work / "tokenizer-corpus-v0.2.1-offline"
    receipt = corpus / "corpus_receipt.json"
    if mix["training_lanes"]["broad_semantic_replay"]["corpus_receipt_sha256"] != digest(receipt):
        raise SystemExit("STOP: P39PN corpus binding mismatch")
    source_rows = json.loads(receipt.read_text(encoding="utf-8"))["sources"]
    shards = []
    source_inventory = []
    for source in source_rows:
        source_bytes = 0
        for shard in source["shards"]:
            rel = Path(shard["path"])
            if rel.is_absolute() or ".." in rel.parts:
                raise SystemExit("STOP: unsafe corpus shard path")
            path = (corpus / rel).resolve()
            if corpus not in path.parents or not path.is_file():
                raise SystemExit("STOP: missing/escaping corpus shard")
            if path.stat().st_size != shard["bytes"]:
                raise SystemExit("STOP: corpus shard size mismatch")
            source_bytes += path.stat().st_size
            shards.append(path)
        source_inventory.append({"source_id": source["source_id"],
                                 "shards": len(source["shards"]), "bytes": source_bytes})
    tokenizer = work / "tokenizer-v0.2.1"
    checkpoint = work / "targeted-repair-v0.1" / "checkpoints" / "step-00000080" / "alice_n0_v02.safetensors"
    lanes = {name: measured(mixture / name) for name in (
        "semantic/rows.jsonl", "semantic-long/rows.jsonl", "behavioral/rows.jsonl",
        "runtime-view/rows.jsonl", "long-context/rows.jsonl",
        "fewrel/train_dev_rows.jsonl", "fewrel/train_dev_bank.json")}
    bindings = {"semantic/rows.jsonl": "semantic_operator_intervention",
                "semantic-long/rows.jsonl": "semantic_operator_long_context",
                "behavioral/rows.jsonl": "full_envelope_behavioral",
                "runtime-view/rows.jsonl": "runtime_view_supplement",
                "long-context/rows.jsonl": "long_context_supplement",
                "fewrel/train_dev_rows.jsonl": "natural_relation"}
    for path, name in bindings.items():
        if lanes[path]["sha256"] != mix["training_lanes"][name]["rows_sha256"]:
            raise SystemExit(f"STOP: P39PN row binding mismatch: {path}")
    if lanes["fewrel/train_dev_bank.json"]["sha256"] != mix["training_lanes"]["natural_relation"]["bank_sha256"]:
        raise SystemExit("STOP: P39PN FewRel bank binding mismatch")
    result = {"schema": "alice.n0.p43.kaggle-input-inventory.v1", "source_revision": head,
              "private_identity_data": False, "training_authorized": False,
              "mixture_manifest": measured(manifest), "mixture_audit": measured(mix_audit),
              "teacher_registry": measured(registry), "teacher_audit": measured(audit),
              "teacher_shards": teacher_rows,
              "corpus_receipt": measured(receipt), "corpus_sources": len(source_rows),
              "corpus_sources_inventory": source_inventory, "corpus_shard_count": len(shards),
              "corpus_shard_total_bytes": sum(path.stat().st_size for path in shards),
              "tokenizer_json": measured(tokenizer / "tokenizer.json"),
              "checkpoint": measured(checkpoint), "optimizer_facing_lanes": lanes,
              "final_files_staged": False}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
