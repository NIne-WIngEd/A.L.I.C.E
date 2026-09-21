#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROW_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-row.v1"
BANK_SCHEMA = "alice.eipm.n0.fewrel-runtime-relation-bank.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-manifest.v1"
AUDIT_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-audit.v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_signature(row: dict[str, Any]) -> str:
    payload = {
        "sentence": row["sentence"],
        "head": row["head"],
        "tail": row["tail"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows_path = Path(args.rows)
    bank_path = Path(args.bank)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit("refusing to overwrite natural relation audit")

    data = rows(rows_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if bank.get("schema") != BANK_SCHEMA:
        errors.append("relation bank schema drift")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("materialization manifest schema drift")
    if manifest.get("rows_sha256") != sha256(rows_path):
        errors.append("row SHA-256 drift")
    if manifest.get("bank_sha256") != sha256(bank_path):
        errors.append("bank SHA-256 drift")
    relations = dict(bank.get("relations") or {})
    if not relations:
        errors.append("empty semantic relation bank")

    ids: set[str] = set()
    train_relations: set[str] = set()
    dev_relations: set[str] = set()
    train_signatures: set[str] = set()
    dev_signatures: set[str] = set()
    train_counts: set[int] = set()
    dev_counts: set[int] = set()

    for row in data:
        rid = str(row.get("id"))
        if row.get("schema") != ROW_SCHEMA:
            errors.append(f"{rid}: row schema drift")
            continue
        if rid in ids:
            errors.append(f"{rid}: duplicate row id")
        ids.add(rid)
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data")
        if row.get("natural_source_text") is not True:
            errors.append(f"{rid}: natural source flag lost")
        if row.get("generated_relation_label") is not False:
            errors.append(f"{rid}: relation label is not source-native")
        if row.get("relation_keys_are_metadata_only") is not True:
            errors.append(f"{rid}: relation keys not metadata-only")

        split = str(row.get("split"))
        if split not in {"train", "dev"}:
            errors.append(f"{rid}: invalid split")
            continue
        if split == "dev" and row.get("training_authorized") is not False:
            errors.append(f"{rid}: DEV row authorized for training")
        if split == "train" and row.get("training_authorized") is not True:
            errors.append(f"{rid}: TRAIN row unexpectedly blocked")

        candidates = list(row.get("candidate_relation_keys") or [])
        if not candidates:
            errors.append(f"{rid}: empty candidate set")
            continue
        if len(candidates) != len(set(candidates)):
            errors.append(f"{rid}: duplicate candidate relation")
        target = str(row.get("target_relation_key"))
        if target not in candidates:
            errors.append(f"{rid}: target relation absent from candidates")
            continue
        target_index = int(row.get("target_candidate_index", -1))
        if target_index < 0 or target_index >= len(candidates):
            errors.append(f"{rid}: target candidate index out of range")
        elif candidates[target_index] != target:
            errors.append(f"{rid}: target candidate index/key mismatch")
        if int(row.get("runtime_relation_count", -1)) != len(candidates):
            errors.append(f"{rid}: runtime relation count drift")
        for key in candidates:
            if key not in relations:
                errors.append(f"{rid}: candidate missing semantic description {key}")

        signature = source_signature(row)
        if split == "train":
            train_relations.add(target)
            train_signatures.add(signature)
            train_counts.add(len(candidates))
        else:
            dev_relations.add(target)
            dev_signatures.add(signature)
            dev_counts.add(len(candidates))

    relation_overlap = sorted(train_relations & dev_relations)
    source_overlap = sorted(train_signatures & dev_signatures)
    if relation_overlap:
        errors.append(f"train/dev relation-family leakage: {relation_overlap}")
    if source_overlap:
        errors.append(
            f"train/dev natural source leakage: {len(source_overlap)} exact rows"
        )
    if len(train_relations) != 64:
        errors.append(
            f"expected 64 natural TRAIN relations, observed {len(train_relations)}"
        )
    if len(dev_relations) != 16:
        errors.append(
            f"expected 16 heldout DEV relations, observed {len(dev_relations)}"
        )
    if not (dev_counts - train_counts):
        errors.append(
            "DEV lacks unseen candidate-cardinality operating points"
        )

    for key, item in relations.items():
        semantic_text = str(item.get("semantic_text", ""))
        if not semantic_text:
            errors.append(f"relation {key}: empty semantic text")
        if key.lower() in semantic_text.lower():
            errors.append(
                f"relation {key}: opaque relation key leaked into semantic text"
            )

    result = {
        "schema": AUDIT_SCHEMA,
        "status": (
            "PASS_FEWREL_NATURAL_RELATION_AUDIT"
            if not errors
            else "FAIL_FEWREL_NATURAL_RELATION_AUDIT"
        ),
        "errors": errors,
        "rows_sha256": sha256(rows_path),
        "bank_sha256": sha256(bank_path),
        "rows": len(data),
        "train_relations": len(train_relations),
        "dev_relations": len(dev_relations),
        "train_dev_relation_overlap": relation_overlap,
        "train_dev_source_overlap_count": len(source_overlap),
        "train_candidate_count_points": sorted(train_counts),
        "dev_candidate_count_points": sorted(dev_counts),
        "dev_unseen_candidate_count_points": sorted(dev_counts - train_counts),
        "natural_source_text": True,
        "human_relation_labels": True,
        "training_authorized_by_audit": False,
        "dev_training_authorized": False,
        "private_identity_data": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
