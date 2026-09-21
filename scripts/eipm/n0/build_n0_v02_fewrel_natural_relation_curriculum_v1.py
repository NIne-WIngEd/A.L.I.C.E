#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_REVISION = "278a2315d2138810a379cd8d5718914dc56e2582"
EXPECTED_BLOBS = {
    "LICENSE": "6fec9b6238e7b4f2037aefcf4110821a2fd16c70",
    "data/pid2name.json": "00b632be58e00b5337c054ac8a46bda1f4c27b0e",
    "data/train_wiki.json": "4fee283724b455bab1fcbf76bfdef02283bb44f5",
    "data/val_wiki.json": "fe103ba5f52de1f0c070d66c7a8ba9de20c9f96c",
}
ROW_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-row.v1"
BANK_SCHEMA = "alice.eipm.n0.fewrel-runtime-relation-bank.v1"
MANIFEST_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-manifest.v1"
PROPERTY_ID = re.compile(r"\bP\d+\b")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("utf-8")
    return hashlib.sha1(header + payload).hexdigest()


def require_source(root: Path) -> None:
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        raise SystemExit(
            "FewRel source must be an exact git checkout; unable to read HEAD"
        ) from exc
    if revision != EXPECTED_REVISION:
        raise SystemExit(
            f"FewRel revision drift: expected {EXPECTED_REVISION}, got {revision}"
        )

    for relative, expected in EXPECTED_BLOBS.items():
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"FewRel source file missing: {relative}")
        actual = git_blob_sha1(path)
        if actual != expected:
            raise SystemExit(
                f"FewRel git-blob drift for {relative}: expected {expected}, got {actual}"
            )

    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2018 THUNLP" not in license_text:
        raise SystemExit("FewRel MIT license receipt drift")


def semantic_relation_text(name: str, description: str) -> str:
    name = PROPERTY_ID.sub("referenced property", str(name)).strip()
    description = PROPERTY_ID.sub("referenced property", str(description)).strip()
    if not description or description.lower() == "no description defined":
        return f"Relation meaning: {name}."
    return f"Relation name: {name}. Relation meaning: {description}"


def relation_bank(pid2name: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, pair in pid2name.items():
        if not isinstance(pair, list) or len(pair) < 2:
            raise SystemExit(f"FewRel pid2name malformed for {key}")
        text = semantic_relation_text(str(pair[0]), str(pair[1]))
        if key.lower() in text.lower():
            raise SystemExit(f"opaque relation key leaked into semantic text: {key}")
        out[str(key)] = {
            "name": str(pair[0]),
            "description": str(pair[1]),
            "semantic_text": text,
        }
    return out


def flatten_positions(value: Any) -> list[int]:
    out: list[int] = []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_positions(item))
        return sorted(set(out))
    raise SystemExit("FewRel entity position structure is malformed")


def candidate_keys(
    *,
    target: str,
    pool: list[str],
    count: int,
    seed: int,
) -> list[str]:
    if target not in pool:
        raise SystemExit("target relation absent from candidate pool")
    if count <= 0:
        raise SystemExit("candidate count must be positive")
    count = min(int(count), len(pool))
    rng = random.Random(seed)
    distractors = [x for x in pool if x != target]
    rng.shuffle(distractors)
    chosen = [target] + distractors[: max(0, count - 1)]
    rng.shuffle(chosen)
    return chosen


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def compile_split(
    *,
    split: str,
    data: dict[str, list[dict[str, Any]]],
    candidate_pool: list[str],
    candidate_points: list[int],
    max_per_relation: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidate_points:
        raise SystemExit("candidate operating points must be non-empty")

    for relation_key in sorted(data):
        examples = data[relation_key]
        if max_per_relation > 0:
            examples = examples[:max_per_relation]
        for index, item in enumerate(examples):
            tokens = [str(x) for x in item["tokens"]]
            head = item["h"]
            tail = item["t"]
            if not isinstance(head, list) or len(head) < 3:
                raise SystemExit("FewRel head structure drift")
            if not isinstance(tail, list) or len(tail) < 3:
                raise SystemExit("FewRel tail structure drift")

            count = candidate_points[index % len(candidate_points)]
            candidates = candidate_keys(
                target=relation_key,
                pool=candidate_pool,
                count=count,
                seed=stable_seed(split, relation_key, str(index)),
            )
            target_index = candidates.index(relation_key)
            sentence = " ".join(tokens)
            source_key = hashlib.sha256(
                json.dumps(
                    {
                        "relation": relation_key,
                        "tokens": tokens,
                        "head": head,
                        "tail": tail,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            row = {
                "schema": ROW_SCHEMA,
                "id": f"fewrel:{split}:{source_key[:20]}",
                "split": split,
                "source_id": "thunlp_fewrel_1_0",
                "source_revision": EXPECTED_REVISION,
                "natural_source_text": True,
                "generated_instruction_only": True,
                "generated_relation_label": False,
                "sentence": sentence,
                "tokens": tokens,
                "head": {
                    "text": str(head[0]),
                    "type": str(head[1]),
                    "token_indices": flatten_positions(head[2]),
                },
                "tail": {
                    "text": str(tail[0]),
                    "type": str(tail[1]),
                    "token_indices": flatten_positions(tail[2]),
                },
                "instruction": (
                    "Given the natural sentence and the marked head/tail entities, "
                    "select the supplied relation description that best expresses "
                    "how the head entity relates to the tail entity."
                ),
                "candidate_relation_keys": candidates,
                "target_relation_key": relation_key,
                "target_candidate_index": target_index,
                "runtime_relation_count": len(candidates),
                "training_authorized": split == "train",
                "relation_keys_are_metadata_only": True,
                "private_identity_data": False,
            }
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fewrel-root", required=True)
    p.add_argument("--rows-output", required=True)
    p.add_argument("--bank-output", required=True)
    p.add_argument("--manifest-output", required=True)
    p.add_argument("--train-candidate-counts", default="4,8,16,32,64")
    p.add_argument("--dev-candidate-counts", default="3,7,12,24,48,80")
    p.add_argument(
        "--max-per-relation",
        type=int,
        default=0,
        help="0 means all examples; positive values are debug/operating caps only",
    )
    args = p.parse_args()

    root = Path(args.fewrel_root)
    require_source(root)

    pid2name = json.loads(
        (root / "data/pid2name.json").read_text(encoding="utf-8")
    )
    train_data = json.loads(
        (root / "data/train_wiki.json").read_text(encoding="utf-8")
    )
    dev_data = json.loads(
        (root / "data/val_wiki.json").read_text(encoding="utf-8")
    )
    bank = relation_bank(pid2name)

    train_relations = sorted(str(x) for x in train_data)
    dev_relations = sorted(str(x) for x in dev_data)
    overlap = sorted(set(train_relations) & set(dev_relations))
    if overlap:
        raise SystemExit(f"FewRel train/dev relation leakage: {overlap}")
    if len(train_relations) != 64:
        raise SystemExit(
            f"FewRel training relation count drift: {len(train_relations)}"
        )
    if len(dev_relations) != 16:
        raise SystemExit(
            f"FewRel heldout relation count drift: {len(dev_relations)}"
        )

    missing = [
        key
        for key in train_relations + dev_relations
        if key not in bank
    ]
    if missing:
        raise SystemExit(f"FewRel relation descriptions missing: {missing}")

    train_points = sorted(
        {int(x) for x in args.train_candidate_counts.split(",") if x.strip()}
    )
    dev_points = sorted(
        {int(x) for x in args.dev_candidate_counts.split(",") if x.strip()}
    )
    if min(train_points + dev_points) <= 0:
        raise SystemExit("candidate counts must be positive")
    if max(train_points) > len(train_relations):
        raise SystemExit("train candidate operating point exceeds training relation bank")
    all_relations = train_relations + dev_relations
    if max(dev_points) > len(all_relations):
        raise SystemExit("DEV candidate operating point exceeds full relation bank")

    train_rows = compile_split(
        split="train",
        data=train_data,
        candidate_pool=train_relations,
        candidate_points=train_points,
        max_per_relation=args.max_per_relation,
    )
    dev_rows = compile_split(
        split="dev",
        data=dev_data,
        candidate_pool=all_relations,
        candidate_points=dev_points,
        max_per_relation=args.max_per_relation,
    )
    rows = train_rows + dev_rows

    row_ids = [x["id"] for x in rows]
    if len(row_ids) != len(set(row_ids)):
        raise SystemExit("FewRel compiled row IDs are not unique")

    rows_path = Path(args.rows_output)
    bank_path = Path(args.bank_output)
    manifest_path = Path(args.manifest_output)
    for path in (rows_path, bank_path, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    rows_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    active_bank_keys = sorted(set(train_relations + dev_relations))
    bank_payload = {
        "schema": BANK_SCHEMA,
        "source_id": "thunlp_fewrel_1_0",
        "source_revision": EXPECTED_REVISION,
        "relations": {
            key: bank[key]
            for key in active_bank_keys
        },
        "relation_keys_are_metadata_only": True,
        "semantic_text_field": "semantic_text",
        "private_identity_data": False,
    }
    bank_path.write_text(
        json.dumps(bank_payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "MATERIALIZED_NATURAL_RELATION_CURRICULUM_NOT_TRAINING_AUTHORITY_BY_ITSELF",
        "source_id": "thunlp_fewrel_1_0",
        "source_revision": EXPECTED_REVISION,
        "source_git_blob_sha1": EXPECTED_BLOBS,
        "license": "MIT",
        "rows_sha256": sha256(rows_path),
        "bank_sha256": sha256(bank_path),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_relation_count": len(train_relations),
        "dev_relation_count": len(dev_relations),
        "train_dev_relation_overlap": overlap,
        "train_candidate_count_points": sorted(
            {x["runtime_relation_count"] for x in train_rows}
        ),
        "dev_candidate_count_points": sorted(
            {x["runtime_relation_count"] for x in dev_rows}
        ),
        "max_per_relation_operating_cap": args.max_per_relation,
        "operating_cap_is_capability_ceiling": False,
        "natural_source_text": True,
        "human_relation_labels": True,
        "synthetic_replacement": False,
        "dev_training_authorized": False,
        "private_identity_data": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
