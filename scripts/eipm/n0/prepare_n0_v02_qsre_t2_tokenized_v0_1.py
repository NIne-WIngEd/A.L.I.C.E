from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from alice_personality.n0.curriculum_data import load_tokenizer


EXPECTED_T1_PREPARED_SHA = (
    "03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564"
)
EXPECTED_T1_CHECKPOINT_SHA = (
    "483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expand_split(
    rows: list[dict],
    source: dict,
    tokenizer,
    *,
    max_length: int,
) -> dict:
    source_lookup = {
        row_id: index
        for index, row_id in enumerate(source["ids"])
    }
    indices = []
    for row in rows:
        source_id = row["source_t1_row_id"]
        if source_id not in source_lookup:
            raise SystemExit(f"T1 prepared row missing: {source_id}")
        indices.append(source_lookup[source_id])
    index = torch.tensor(indices, dtype=torch.long)

    text = [str(row["query_text"]) for row in rows]
    tokenized = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    tensor_keys = (
        "field_state",
        "field_metadata",
        "field_valid_mask",
        "edge_index",
        "edge_relation_id",
        "edge_metadata",
        "edge_valid_mask",
        "field_support_weight",
        "edge_support_weight",
        "focus_field_weight",
        "target_distribution",
        "expected_control",
    )
    out = {
        key: source[key][index].clone()
        for key in tensor_keys
    }
    out["input_ids"] = tokenized["input_ids"].long()
    out["attention_mask"] = tokenized["attention_mask"].bool()

    steps = 2
    none_relation = 6
    relation_target = torch.full(
        (len(rows), steps),
        none_relation,
        dtype=torch.long,
    )
    role_target = torch.zeros(len(rows), dtype=torch.long)
    operation_target = torch.zeros(len(rows), dtype=torch.long)
    control_target = torch.zeros(len(rows), dtype=torch.long)

    for i, row in enumerate(rows):
        target = row["operator_target"]
        seq = [int(x) for x in target["relation_sequence_id"]]
        relation_target[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        role_target[i] = int(target["role_id"])
        operation_target[i] = int(target["operation_id"])
        control_target[i] = int(target["control_id"])

    out["relation_target"] = relation_target
    out["role_target"] = role_target
    out["operation_target"] = operation_target
    out["control_target"] = control_target
    out["ids"] = [row["id"] for row in rows]
    out["source_t1_row_ids"] = [row["source_t1_row_id"] for row in rows]
    out["families"] = [row["family"] for row in rows]
    out["causal_groups"] = [
        f"{row['source_t1_causal_group']}:q{row['query_view']}"
        for row in rows
    ]
    out["query_views"] = [int(row["query_view"]) for row in rows]
    out["query_text"] = text
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--t1-prepared-cache", required=True)
    parser.add_argument("--t1-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    curriculum = Path(args.curriculum).resolve()
    t1_cache = Path(args.t1_prepared_cache).resolve()
    t1_checkpoint = Path(args.t1_checkpoint).resolve()
    output = Path(args.output).resolve()
    receipt_path = Path(args.receipt).resolve()

    if output.exists() or receipt_path.exists():
        raise SystemExit("refusing to overwrite T2 preparation evidence")
    if sha256(t1_cache) != EXPECTED_T1_PREPARED_SHA:
        raise SystemExit("T1 prepared cache hash drift")
    if sha256(t1_checkpoint) != EXPECTED_T1_CHECKPOINT_SHA:
        raise SystemExit("selected T1 checkpoint hash drift")

    rows = read_jsonl(curriculum)
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    if len(train_rows) != 720 or len(dev_rows) != 288:
        raise SystemExit("T2 curriculum split count drift")
    if any(row.get("private_identity_data") is not False for row in rows):
        raise SystemExit("private identity data present")

    prepared_t1 = torch.load(t1_cache, map_location="cpu")
    if prepared_t1.get("schema") != "alice.eipm.n0.qsre-t1-real-cache.v0.1":
        raise SystemExit("T1 prepared cache schema drift")
    if prepared_t1.get("test_present") is not False:
        raise SystemExit("T1 prepared cache unexpectedly contains TEST")

    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    payload = {
        "schema": "alice.eipm.n0.qsre-t2-tokenized-preparation.v0.1",
        "curriculum_sha256": sha256(curriculum),
        "t1_prepared_cache_sha256": sha256(t1_cache),
        "selected_t1_checkpoint_sha256": sha256(t1_checkpoint),
        "max_length": int(args.max_length),
        "none_relation_id": 6,
        "num_hidden_states_expected": 17,
        "semantic_dim_expected": 640,
        "train": expand_split(
            train_rows,
            prepared_t1["train"],
            tokenizer,
            max_length=int(args.max_length),
        ),
        "dev": expand_split(
            dev_rows,
            prepared_t1["dev"],
            tokenizer,
            max_length=int(args.max_length),
        ),
        "test_present": False,
        "private_identity_data": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    receipt = {
        "schema": "alice.eipm.n0.qsre-t2-tokenized-preparation-result.v0.1",
        "status": "PASS_QSRE_T2_TOKENIZED_PREPARATION",
        "curriculum_sha256": payload["curriculum_sha256"],
        "prepared_cache_sha256": sha256(output),
        "t1_prepared_cache_sha256": payload["t1_prepared_cache_sha256"],
        "selected_t1_checkpoint_sha256": payload[
            "selected_t1_checkpoint_sha256"
        ],
        "rows": {"train": len(train_rows), "dev": len(dev_rows)},
        "query_width": {
            "train": int(payload["train"]["input_ids"].size(1)),
            "dev": int(payload["dev"]["input_ids"].size(1)),
        },
        "semantic_hidden_states_materialized": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "test_present": False,
        "private_identity_data": False,
        "t2_training_authorized": False,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
