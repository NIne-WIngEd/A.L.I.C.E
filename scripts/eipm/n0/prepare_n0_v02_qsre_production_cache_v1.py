from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qsre_production_runtime import (
    _special_token_mask,
    load_frozen_semantic_model,
    load_tokenizer,
    sha256,
)


CACHE_SCHEMA = "alice.eipm.n0.qsre-production-public-cache.v1"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def encode_texts(
    *,
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    all_hidden_states: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    special_ids = set(
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", []) or [])
    )
    content_mask = _special_token_mask(
        tokens["input_ids"],
        tokens["attention_mask"],
        special_ids,
    )
    chunks = []
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        ids = tokens["input_ids"][start:stop].to(device)
        attention = tokens["attention_mask"][start:stop].to(device)
        outputs = model.backbone(
            input_ids=ids,
            attention_mask=attention,
            output_hidden_states=all_hidden_states,
            return_dict=True,
        )
        if all_hidden_states:
            states = outputs.hidden_states
            if states is None or len(states) != 17:
                raise RuntimeError("query hidden-state depth drift")
            value = torch.stack(states, dim=1)
        else:
            value = outputs.last_hidden_state
        chunks.append(value.detach().to(dtype=torch.float16).cpu())
    return torch.cat(chunks, dim=0), content_mask


def build_split(
    rows: list[dict],
    *,
    query_hidden: torch.Tensor,
    query_mask: torch.Tensor,
    field_token: torch.Tensor,
    field_token_mask_flat: torch.Tensor,
    relation_key_to_index: dict[str, int],
    type_to_index: dict[str, int],
) -> dict:
    count = len(rows)
    max_fields = max(len(row["fields"]) for row in rows)
    max_edges = max(len(row["edges"]) for row in rows)
    max_steps = max(len(row["operator_target"]["relation_sequence"]) for row in rows)
    field_tokens = field_token.size(1)

    field_state = torch.zeros(count, max_fields, 640)
    field_token_states = torch.zeros(count, max_fields, field_tokens, 640, dtype=torch.float16)
    field_token_mask = torch.zeros(count, max_fields, field_tokens, dtype=torch.bool)
    field_metadata = torch.zeros(count, max_fields, 3)
    field_type_id = torch.zeros(count, max_fields, dtype=torch.long)
    field_valid_mask = torch.zeros(count, max_fields, dtype=torch.bool)

    edge_index = torch.zeros(count, max_edges, 2, dtype=torch.long)
    edge_relation_index = torch.zeros(count, max_edges, dtype=torch.long)
    edge_valid_mask = torch.zeros(count, max_edges, dtype=torch.bool)
    edge_reliability = torch.zeros(count, max_edges)
    edge_recency = torch.zeros(count, max_edges)
    edge_temporal_match = torch.ones(count, max_edges)
    edge_provenance_match = torch.ones(count, max_edges)
    oracle_edge_support = torch.zeros(count, max_edges)

    relation_target = torch.zeros(count, max_steps, dtype=torch.long)
    relation_target_mask = torch.zeros(count, max_steps, dtype=torch.bool)
    role_target = torch.zeros(count, dtype=torch.long)
    traversal_target = torch.zeros(count, dtype=torch.long)
    modifier_target = torch.zeros(count, 4)
    applicability_target = torch.zeros(count)
    control_target = torch.zeros(count, dtype=torch.long)
    focus_field_weight = torch.zeros(count, max_fields)
    target_distribution = torch.zeros(count, max_fields)

    ids = []
    families = []
    causal_groups = []
    variants = []
    field_ids = []
    open_schema = torch.zeros(count, dtype=torch.bool)

    field_cursor = 0
    for row_i, row in enumerate(rows):
        ids.append(row["id"])
        families.append(row["family"])
        causal_groups.append(row["causal_group"])
        variants.append(row["variant"])
        open_schema[row_i] = bool(row["open_schema_relation"])

        positions = {}
        row_field_ids = []
        for field_i, field in enumerate(row["fields"]):
            fid = field["id"]
            positions[fid] = field_i
            row_field_ids.append(fid)
            token_state = field_token[field_cursor]
            token_mask = field_token_mask_flat[field_cursor]
            field_cursor += 1

            field_token_states[row_i, field_i, : token_state.size(0)] = token_state
            field_token_mask[row_i, field_i, : token_mask.size(0)] = token_mask
            maskf = token_mask.to(token_state.dtype).unsqueeze(-1)
            pooled = (token_state * maskf).sum(dim=0) / maskf.sum(dim=0).clamp_min(1.0)
            field_state[row_i, field_i] = pooled.float()
            field_metadata[row_i, field_i] = torch.tensor(
                [
                    float(field["metadata"]["reliability"]),
                    float(field["metadata"]["normalized_time"]),
                    0.0,
                ]
            )
            field_type_id[row_i, field_i] = int(type_to_index[field["type"]])
            field_valid_mask[row_i, field_i] = True

        field_ids.append(row_field_ids)
        support_ids = set(row["support_edge_ids"])
        for edge_i, edge in enumerate(row["edges"]):
            edge_index[row_i, edge_i, 0] = positions[edge["source"]]
            edge_index[row_i, edge_i, 1] = positions[edge["target"]]
            edge_relation_index[row_i, edge_i] = int(
                relation_key_to_index[edge["relation"]]
            )
            edge_valid_mask[row_i, edge_i] = True
            edge_reliability[row_i, edge_i] = float(edge["metadata"]["reliability"])
            edge_recency[row_i, edge_i] = float(edge["metadata"]["normalized_time"])
            edge_temporal_match[row_i, edge_i] = float(edge["metadata"]["temporal_match"])
            edge_provenance_match[row_i, edge_i] = float(edge["metadata"]["provenance_match"])
            if edge["id"] in support_ids:
                oracle_edge_support[row_i, edge_i] = 1.0

        target = row["operator_target"]
        sequence = target["relation_sequence"]
        for step, key in enumerate(sequence):
            relation_target[row_i, step] = int(relation_key_to_index[key])
            relation_target_mask[row_i, step] = True
        role_target[row_i] = int(target["role_id"])
        traversal_target[row_i] = int(target["traversal_id"])
        modifier_target[row_i] = torch.tensor(
            [
                float(target["modifier_target"]["RELIABILITY"]),
                float(target["modifier_target"]["RECENCY"]),
                float(target["modifier_target"]["TEMPORAL_CONSTRAINT"]),
                float(target["modifier_target"]["PROVENANCE_CONSTRAINT"]),
            ]
        )
        applicability_target[row_i] = float(target["applicability"])
        control_target[row_i] = int(target["control_id"])

        focus = target["focus_field_id"]
        if focus is not None:
            focus_field_weight[row_i, positions[focus]] = 1.0
        for fid, weight in row["target_distribution"].items():
            target_distribution[row_i, positions[fid]] = float(weight)

    if field_cursor != field_token.size(0):
        raise RuntimeError("field semantic cursor drift")

    return {
        "query_hidden_states": query_hidden,
        "query_token_mask": query_mask,
        "field_state": field_state,
        "field_token_states": field_token_states,
        "field_token_mask": field_token_mask,
        "field_metadata": field_metadata,
        "field_type_id": field_type_id,
        "field_valid_mask": field_valid_mask,
        "edge_index": edge_index,
        "edge_relation_index": edge_relation_index,
        "edge_valid_mask": edge_valid_mask,
        "edge_reliability": edge_reliability,
        "edge_recency": edge_recency,
        "edge_temporal_match": edge_temporal_match,
        "edge_provenance_match": edge_provenance_match,
        "oracle_edge_support": oracle_edge_support,
        "relation_target": relation_target,
        "relation_target_mask": relation_target_mask,
        "role_target": role_target,
        "traversal_target": traversal_target,
        "modifier_target": modifier_target,
        "applicability_target": applicability_target,
        "control_target": control_target,
        "focus_field_weight": focus_field_weight,
        "target_distribution": target_distribution,
        "open_schema": open_schema,
        "ids": ids,
        "families": families,
        "causal_groups": causal_groups,
        "variants": variants,
        "field_ids": field_ids,
    }


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curriculum", required=True)
    p.add_argument("--schema-cache", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--receipt", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    curriculum_path = Path(args.curriculum)
    schema_cache_path = Path(args.schema_cache)
    semantic_config_path = Path(args.semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_path = Path(args.output)
    receipt_path = Path(args.receipt)
    device = torch.device(args.device)

    rows = read_jsonl(curriculum_path)
    if not rows:
        raise SystemExit("empty production curriculum")
    if any(row.get("private_identity_data") is not False for row in rows):
        raise SystemExit("private identity data entered production cache")
    if any(row["split"] not in {"train", "dev"} for row in rows):
        raise SystemExit("production curriculum split drift")

    schema_cache = torch.load(schema_cache_path, map_location="cpu")
    if schema_cache.get("schema") != "alice.eipm.n0.qsre-production-schema-cache.v1":
        raise SystemExit("production schema-cache drift")
    relation_key_to_index = {
        str(k): int(v)
        for k, v in schema_cache["relation_key_to_index"].items()
    }
    type_to_index = {
        str(k): int(v)
        for k, v in schema_cache["type_to_index"].items()
    }

    model = load_frozen_semantic_model(
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer = load_tokenizer(tokenizer_dir)

    payload = {
        "schema": CACHE_SCHEMA,
        "curriculum_sha256": sha256(curriculum_path),
        "schema_cache_sha256": sha256(schema_cache_path),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "private_identity_data": False,
        "test_present": False,
    }

    for split in ("train", "dev"):
        split_rows = [row for row in rows if row["split"] == split]
        query_texts = [row["query"] for row in split_rows]
        flat_field_texts = [
            field["text"]
            for row in split_rows
            for field in row["fields"]
        ]

        query_hidden, query_mask = encode_texts(
            texts=query_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=128,
            batch_size=args.batch_size,
            all_hidden_states=True,
        )
        field_hidden, field_mask = encode_texts(
            texts=flat_field_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=96,
            batch_size=args.batch_size,
            all_hidden_states=False,
        )

        payload[split] = build_split(
            split_rows,
            query_hidden=query_hidden,
            query_mask=query_mask,
            field_token=field_hidden,
            field_token_mask_flat=field_mask,
            relation_key_to_index=relation_key_to_index,
            type_to_index=type_to_index,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    train = payload["train"]
    dev = payload["dev"]
    receipt = {
        "schema": "alice.eipm.n0.qsre-production-public-cache-result.v1",
        "status": "PASS_QSRE_PRODUCTION_PUBLIC_CACHE_PREPARATION",
        "curriculum_sha256": payload["curriculum_sha256"],
        "schema_cache_sha256": payload["schema_cache_sha256"],
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "prepared_cache_sha256": sha256(output_path),
        "rows": {
            "train": len(train["ids"]),
            "dev": len(dev["ids"]),
        },
        "query_width": {
            "train": int(train["query_hidden_states"].size(2)),
            "dev": int(dev["query_hidden_states"].size(2)),
        },
        "field_token_width": {
            "train": int(train["field_token_states"].size(2)),
            "dev": int(dev["field_token_states"].size(2)),
        },
        "open_schema_dev_rows": int(dev["open_schema"].sum().item()),
        "max_relation_steps": max(
            int(train["relation_target"].size(1)),
            int(dev["relation_target"].size(1)),
        ),
        "test_present": False,
        "private_identity_data": False,
        "optimizer": False,
        "gradient": False,
        "gpu": device.type == "cuda",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
