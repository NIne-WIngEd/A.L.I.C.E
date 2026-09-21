#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROW_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-row.v1"
BANK_SCHEMA = "alice.eipm.n0.fewrel-runtime-relation-bank.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-manifest.v2"
AUDIT_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-audit.v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def relation_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["target_relation_key"]) for row in rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dev-rows", required=True)
    p.add_argument("--train-dev-bank", required=True)
    p.add_argument("--final-rows", required=True)
    p.add_argument("--final-bank", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    train_dev_rows_path = Path(args.train_dev_rows)
    train_dev_bank_path = Path(args.train_dev_bank)
    final_rows_path = Path(args.final_rows)
    final_bank_path = Path(args.final_bank)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit("refusing to overwrite natural relation audit")

    train_dev_rows = read_jsonl(train_dev_rows_path)
    final_rows = read_jsonl(final_rows_path)
    train_dev_bank = json.loads(
        train_dev_bank_path.read_text(encoding="utf-8")
    )
    final_bank = json.loads(final_bank_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("materialization manifest schema drift")
    for bank_name, bank in (
        ("train_dev", train_dev_bank),
        ("final", final_bank),
    ):
        if bank.get("schema") != BANK_SCHEMA:
            errors.append(f"{bank_name} relation bank schema drift")

    expected_hashes = {
        "train_dev_rows_sha256": train_dev_rows_path,
        "train_dev_bank_sha256": train_dev_bank_path,
        "final_rows_sha256": final_rows_path,
        "final_bank_sha256": final_bank_path,
    }
    for key, path in expected_hashes.items():
        if manifest.get(key) != sha256(path):
            errors.append(f"{key} drift")

    train_rows = [row for row in train_dev_rows if row.get("split") == "train"]
    dev_rows = [row for row in train_dev_rows if row.get("split") == "dev"]
    malformed_train_dev = [
        row.get("id")
        for row in train_dev_rows
        if row.get("split") not in {"train", "dev"}
    ]
    if malformed_train_dev:
        errors.append("train/dev artifact contains non-train/dev split")
    if any(row.get("split") != "final" for row in final_rows):
        errors.append("sealed final artifact contains non-final row")

    train_relations = relation_set(train_rows)
    dev_relations = relation_set(dev_rows)
    final_relations = relation_set(final_rows)
    pairwise_overlap = {
        "train_dev": sorted(train_relations & dev_relations),
        "train_final": sorted(train_relations & final_relations),
        "dev_final": sorted(dev_relations & final_relations),
    }
    for name, overlap in pairwise_overlap.items():
        if overlap:
            errors.append(f"{name} relation-family leakage: {overlap}")

    if len(train_relations) != 64:
        errors.append(f"TRAIN relation count {len(train_relations)} != 64")
    if len(dev_relations) != 8:
        errors.append(f"DEV heldout relation count {len(dev_relations)} != 8")
    if len(final_relations) != 8:
        errors.append(
            f"FINAL heldout relation count {len(final_relations)} != 8"
        )

    train_dev_bank_relations = set(
        str(x) for x in train_dev_bank.get("relations", {})
    )
    final_bank_relations = set(str(x) for x in final_bank.get("relations", {}))
    if final_relations & train_dev_bank_relations:
        errors.append(
            "sealed FINAL relation descriptions leaked into TRAIN/DEV bank"
        )
    expected_train_dev = train_relations | dev_relations
    if train_dev_bank_relations != expected_train_dev:
        errors.append("TRAIN/DEV relation bank does not match active families")
    expected_final_bank = (
        train_relations | dev_relations | final_relations
    )
    if final_bank_relations != expected_final_bank:
        errors.append("FINAL relation bank does not contain exact full families")

    ids: set[str] = set()
    train_signatures: set[str] = set()
    dev_signatures: set[str] = set()
    final_signatures: set[str] = set()
    train_counts: set[int] = set()
    dev_counts: set[int] = set()
    final_counts: set[int] = set()

    def audit_row(
        row: dict[str, Any],
        *,
        allowed_candidates: set[str],
        expected_split: str,
    ) -> None:
        rid = str(row.get("id"))
        if row.get("schema") != ROW_SCHEMA:
            errors.append(f"{rid}: row schema drift")
            return
        if rid in ids:
            errors.append(f"{rid}: duplicate row id")
        ids.add(rid)
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data")
        if row.get("natural_source_text") is not True:
            errors.append(f"{rid}: natural-source flag lost")
        if row.get("generated_relation_label") is not False:
            errors.append(f"{rid}: relation label is not source-native")
        if row.get("relation_keys_are_metadata_only") is not True:
            errors.append(f"{rid}: relation keys not metadata-only")
        if row.get("split") != expected_split:
            errors.append(f"{rid}: unexpected split")

        if expected_split == "train":
            if row.get("training_authorized") is not True:
                errors.append(f"{rid}: TRAIN row blocked")
            if row.get("model_selection_authorized") is not False:
                errors.append(f"{rid}: TRAIN marked model-selection authority")
            if row.get("final_validation_only") is not False:
                errors.append(f"{rid}: TRAIN marked final-only")
        elif expected_split == "dev":
            if row.get("training_authorized") is not False:
                errors.append(f"{rid}: DEV row authorized for training")
            if row.get("model_selection_authorized") is not True:
                errors.append(f"{rid}: DEV row not marked model-selection")
            if row.get("final_validation_only") is not False:
                errors.append(f"{rid}: DEV marked final-only")
        else:
            if row.get("training_authorized") is not False:
                errors.append(f"{rid}: FINAL row authorized for training")
            if row.get("model_selection_authorized") is not False:
                errors.append(f"{rid}: FINAL row authorized for model selection")
            if row.get("final_validation_only") is not True:
                errors.append(f"{rid}: FINAL row missing final-only marker")

        candidates = list(row.get("candidate_relation_keys") or [])
        if not candidates:
            errors.append(f"{rid}: empty candidate set")
            return
        if len(candidates) != len(set(candidates)):
            errors.append(f"{rid}: duplicate candidate relation")
        if not set(candidates) <= allowed_candidates:
            errors.append(f"{rid}: candidate family outside allowed split bank")
        target = str(row.get("target_relation_key"))
        if target not in candidates:
            errors.append(f"{rid}: target absent from candidate set")
            return
        target_index = int(row.get("target_candidate_index", -1))
        if target_index < 0 or target_index >= len(candidates):
            errors.append(f"{rid}: target candidate index out of range")
        elif candidates[target_index] != target:
            errors.append(f"{rid}: target index/key mismatch")
        if int(row.get("runtime_relation_count", -1)) != len(candidates):
            errors.append(f"{rid}: runtime relation count drift")

    for row in train_rows:
        audit_row(
            row,
            allowed_candidates=train_relations,
            expected_split="train",
        )
        train_signatures.add(source_signature(row))
        train_counts.add(int(row["runtime_relation_count"]))
    for row in dev_rows:
        audit_row(
            row,
            allowed_candidates=train_relations | dev_relations,
            expected_split="dev",
        )
        dev_signatures.add(source_signature(row))
        dev_counts.add(int(row["runtime_relation_count"]))
    for row in final_rows:
        audit_row(
            row,
            allowed_candidates=(
                train_relations | dev_relations | final_relations
            ),
            expected_split="final",
        )
        final_signatures.add(source_signature(row))
        final_counts.add(int(row["runtime_relation_count"]))

    source_overlap = {
        "train_dev": len(train_signatures & dev_signatures),
        "train_final": len(train_signatures & final_signatures),
        "dev_final": len(dev_signatures & final_signatures),
    }
    for name, count in source_overlap.items():
        if count:
            errors.append(f"{name} exact natural-source leakage: {count}")

    if not (dev_counts - train_counts):
        errors.append("DEV lacks unseen candidate-cardinality points")
    if not (final_counts - (train_counts | dev_counts)):
        errors.append("FINAL lacks candidate-cardinality points unseen in TRAIN/DEV")

    for bank_name, relation_payload in (
        ("train_dev", train_dev_bank.get("relations", {})),
        ("final", final_bank.get("relations", {})),
    ):
        for key, item in relation_payload.items():
            semantic_text = str(item.get("semantic_text", ""))
            if not semantic_text:
                errors.append(f"{bank_name}/{key}: empty semantic text")
            if str(key).lower() in semantic_text.lower():
                errors.append(
                    f"{bank_name}/{key}: opaque key leaked into semantic text"
                )

    result = {
        "schema": AUDIT_SCHEMA,
        "status": (
            "PASS_FEWREL_NATURAL_RELATION_AUDIT_V2"
            if not errors
            else "FAIL_FEWREL_NATURAL_RELATION_AUDIT_V2"
        ),
        "errors": errors,
        "train_dev_rows_sha256": sha256(train_dev_rows_path),
        "train_dev_bank_sha256": sha256(train_dev_bank_path),
        "final_rows_sha256": sha256(final_rows_path),
        "final_bank_sha256": sha256(final_bank_path),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "final_rows": len(final_rows),
        "train_relation_count": len(train_relations),
        "dev_relation_count": len(dev_relations),
        "final_relation_count": len(final_relations),
        "relation_overlap": pairwise_overlap,
        "source_overlap": source_overlap,
        "train_candidate_count_points": sorted(train_counts),
        "dev_candidate_count_points": sorted(dev_counts),
        "final_candidate_count_points": sorted(final_counts),
        "dev_unseen_candidate_count_points": sorted(dev_counts - train_counts),
        "final_unseen_candidate_count_points": sorted(
            final_counts - (train_counts | dev_counts)
        ),
        "final_relation_descriptions_absent_from_train_dev_bank": not bool(
            final_relations & train_dev_bank_relations
        ),
        "natural_source_text": True,
        "human_relation_labels": True,
        "training_authorized_by_audit": False,
        "final_training_authorized": False,
        "final_model_selection_authorized": False,
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
