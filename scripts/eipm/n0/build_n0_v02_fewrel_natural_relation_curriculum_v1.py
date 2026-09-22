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
MANIFEST_SCHEMA = "alice.eipm.n0.fewrel-natural-relation-manifest.v3"
PROPERTY_ID = re.compile(r"\bP\d+\b")
DEV_SPLIT_SALT = "alice-n0-fewrel-train-dev-family-split-v1"


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
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"FewRel source file missing: {relative}")
        actual = git_blob_sha1(source)
        if actual != expected:
            raise SystemExit(
                f"FewRel git-blob drift for {relative}: expected {expected}, got {actual}"
            )
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    if (
        "MIT License" not in license_text
        or "Copyright (c) 2018 THUNLP" not in license_text
    ):
        raise SystemExit("FewRel MIT license receipt drift")


def semantic_relation_text(name: str, description: str) -> str:
    name = PROPERTY_ID.sub("referenced property", str(name)).strip()
    description = PROPERTY_ID.sub(
        "referenced property", str(description)
    ).strip()
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
            raise SystemExit(
                f"opaque relation key leaked into semantic text: {key}"
            )
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


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


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


def training_family_partition(
    training_relations: list[str],
) -> tuple[list[str], list[str]]:
    """Use official FewRel validation families only for final evaluation.

    DEV is carved deterministically from the official training relation set.
    This avoids selecting or tuning a subset of the official validation
    relation families after seeing them.
    """
    if len(training_relations) != 64:
        raise SystemExit(
            "FewRel training relation count must be 64 before DEV partition"
        )
    ordered = sorted(
        training_relations,
        key=lambda key: hashlib.sha256(
            f"{DEV_SPLIT_SALT}:{key}".encode("utf-8")
        ).hexdigest(),
    )
    dev = sorted(ordered[:8])
    train = sorted(ordered[8:])
    return train, dev


def compile_split(
    *,
    split: str,
    data: dict[str, list[dict[str, Any]]],
    active_relations: list[str],
    candidate_pool: list[str],
    candidate_points: list[int],
    max_per_relation: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidate_points:
        raise SystemExit("candidate operating points must be non-empty")

    for relation_key in active_relations:
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
                "sentence": " ".join(tokens),
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
                    "Given the natural sentence and marked head/tail entities, "
                    "select the supplied relation description that best expresses "
                    "how the head entity relates to the tail entity."
                ),
                "candidate_relation_keys": candidates,
                "target_relation_key": relation_key,
                "target_candidate_index": target_index,
                "runtime_relation_count": len(candidates),
                "training_authorized": split == "train",
                "model_selection_authorized": split == "dev",
                "final_validation_only": split == "final",
                "relation_keys_are_metadata_only": True,
                "private_identity_data": False,
            }
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_bank(
    path: Path,
    *,
    bank: dict[str, dict[str, str]],
    keys: list[str],
    role: str,
) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    payload = {
        "schema": BANK_SCHEMA,
        "source_id": "thunlp_fewrel_1_0",
        "source_revision": EXPECTED_REVISION,
        "role": role,
        "relations": {key: bank[key] for key in keys},
        "relation_keys_are_metadata_only": True,
        "semantic_text_field": "semantic_text",
        "private_identity_data": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fewrel-root", required=True)
    p.add_argument("--train-dev-rows-output", required=True)
    p.add_argument("--train-dev-bank-output", required=True)
    p.add_argument("--final-rows-output", required=True)
    p.add_argument("--final-bank-output", required=True)
    p.add_argument("--manifest-output", required=True)
    p.add_argument("--train-candidate-counts", default="4,8,16,32,56")
    p.add_argument("--dev-candidate-counts", default="3,7,12,24,48,64")
    p.add_argument("--final-candidate-counts", default="5,9,20,40,73,80")
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
    validation_data = json.loads(
        (root / "data/val_wiki.json").read_text(encoding="utf-8")
    )
    bank = relation_bank(pid2name)

    official_train_relations = sorted(str(x) for x in train_data)
    validation_relations = sorted(str(x) for x in validation_data)
    if set(official_train_relations) & set(validation_relations):
        raise SystemExit("FewRel official train/validation relation leakage")
    if len(official_train_relations) != 64:
        raise SystemExit(
            f"FewRel training relation count drift: {len(official_train_relations)}"
        )
    if len(validation_relations) != 16:
        raise SystemExit(
            f"FewRel heldout relation count drift: {len(validation_relations)}"
        )
    train_relations, dev_relations = training_family_partition(
        official_train_relations
    )
    final_relations = validation_relations
    if set(train_relations) & set(dev_relations):
        raise SystemExit("TRAIN/DEV family split overlap")
    if (set(train_relations) | set(dev_relations)) != set(official_train_relations):
        raise SystemExit("TRAIN/DEV split does not exhaust official training families")

    active = train_relations + dev_relations + final_relations
    missing = [key for key in active if key not in bank]
    if missing:
        raise SystemExit(f"FewRel relation descriptions missing: {missing}")

    train_points = sorted(
        {int(x) for x in args.train_candidate_counts.split(",") if x.strip()}
    )
    dev_points = sorted(
        {int(x) for x in args.dev_candidate_counts.split(",") if x.strip()}
    )
    final_points = sorted(
        {int(x) for x in args.final_candidate_counts.split(",") if x.strip()}
    )
    if min(train_points + dev_points + final_points) <= 0:
        raise SystemExit("candidate counts must be positive")
    if max(train_points) > len(train_relations):
        raise SystemExit("TRAIN candidate count exceeds TRAIN relation bank")
    train_dev_relations = train_relations + dev_relations
    if max(dev_points) > len(train_dev_relations):
        raise SystemExit("DEV candidate count exceeds TRAIN+DEV relation bank")
    if max(final_points) > len(active):
        raise SystemExit("FINAL candidate count exceeds full relation bank")

    train_rows = compile_split(
        split="train",
        data=train_data,
        active_relations=train_relations,
        candidate_pool=train_relations,
        candidate_points=train_points,
        max_per_relation=args.max_per_relation,
    )
    dev_rows = compile_split(
        split="dev",
        data=train_data,
        active_relations=dev_relations,
        candidate_pool=train_dev_relations,
        candidate_points=dev_points,
        max_per_relation=args.max_per_relation,
    )
    final_rows = compile_split(
        split="final",
        data=validation_data,
        active_relations=final_relations,
        candidate_pool=active,
        candidate_points=final_points,
        max_per_relation=args.max_per_relation,
    )
    train_dev_rows = train_rows + dev_rows

    all_ids = [x["id"] for x in train_dev_rows + final_rows]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("FewRel compiled row IDs are not unique")

    train_dev_rows_path = Path(args.train_dev_rows_output)
    train_dev_bank_path = Path(args.train_dev_bank_output)
    final_rows_path = Path(args.final_rows_output)
    final_bank_path = Path(args.final_bank_output)
    manifest_path = Path(args.manifest_output)

    write_jsonl(train_dev_rows_path, train_dev_rows)
    write_jsonl(final_rows_path, final_rows)
    write_bank(
        train_dev_bank_path,
        bank=bank,
        keys=train_dev_relations,
        role="TRAIN_AND_MODEL_SELECTION_ONLY_NO_FINAL_RELATIONS",
    )
    write_bank(
        final_bank_path,
        bank=bank,
        keys=active,
        role="SEALED_FINAL_VALIDATION_RELATION_BANK",
    )
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "MATERIALIZED_NATURAL_RELATION_CURRICULUM_WITH_OFFICIAL_VALIDATION_FINAL_FAMILIES",
        "source_id": "thunlp_fewrel_1_0",
        "source_revision": EXPECTED_REVISION,
        "source_git_blob_sha1": EXPECTED_BLOBS,
        "license": "MIT",
        "dev_family_split_rule": DEV_SPLIT_SALT,
        "final_family_rule": "ALL_OFFICIAL_FEWREL_VAL_WIKI_RELATION_FAMILIES",
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
        "official_training_relation_count": len(official_train_relations),
        "official_validation_relation_count": len(validation_relations),
        "relation_ids_omitted_from_manifest": True,
        "train_candidate_count_points": sorted(
            {x["runtime_relation_count"] for x in train_rows}
        ),
        "dev_candidate_count_points": sorted(
            {x["runtime_relation_count"] for x in dev_rows}
        ),
        "final_candidate_count_points": sorted(
            {x["runtime_relation_count"] for x in final_rows}
        ),
        "max_per_relation_operating_cap": args.max_per_relation,
        "operating_cap_is_capability_ceiling": False,
        "natural_source_text": True,
        "human_relation_labels": True,
        "final_relation_descriptions_absent_from_train_dev_bank": True,
        "final_rows_separate_artifact": True,
        "all_official_validation_relation_families_are_final": True,
        "final_family_subset_selected_after_observation": False,
        "final_training_authorized": False,
        "final_model_selection_authorized": False,
        "private_identity_data": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
