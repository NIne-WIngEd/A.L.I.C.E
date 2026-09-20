from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from alice_personality.n0.qsre_production_core import QSREDynamicRelationSchema


SCHEMA_CACHE_VERSION = "alice.eipm.n0.qsre-production-schema-cache.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _special_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in special_ids:
        mask &= input_ids.ne(int(token_id))
    empty = mask.sum(dim=-1).eq(0)
    if bool(empty.any()):
        mask[empty] = attention_mask[empty].bool()
    return mask


def load_frozen_semantic_model(
    *,
    semantic_config_path: Path,
    semantic_checkpoint: Path,
    device: torch.device,
):
    from safetensors.torch import load_file

    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.v02_model import AliceN0V02Model

    cfg = load_n0_config(semantic_config_path)
    model = AliceN0V02Model(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(str(semantic_checkpoint), device="cpu"),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            "semantic checkpoint mismatch "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def load_tokenizer(tokenizer_dir: Path):
    from alice_personality.n0.curriculum_data import load_tokenizer as _load

    return _load(tokenizer_dir)


def relation_schema_text(row: dict[str, Any]) -> str:
    descriptions = " ".join(str(value) for value in row["descriptions"])
    source = str(row["source_argument"])
    target = str(row["target_argument"])
    symmetry = (
        "The relation is symmetric."
        if bool(row.get("symmetric", False))
        else "The relation is directional."
    )
    return (
        f"Relation meaning: {descriptions} "
        f"Source argument: {source}. "
        f"Target argument: {target}. "
        f"{symmetry}"
    )


@torch.inference_mode()
def materialize_dynamic_relation_schema(
    *,
    schema_path: Path,
    semantic_config_path: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    output_path: Path,
    receipt_path: Path,
    device: torch.device,
    max_length: int = 128,
) -> dict[str, Any]:
    source = json.loads(schema_path.read_text(encoding="utf-8"))
    if source.get("schema") != "alice.eipm.n0.qsre-production-relation-schema.v1":
        raise RuntimeError("production relation schema version drift")
    if source.get("governance", {}).get("private_identity_data") is not False:
        raise RuntimeError("private identity data entered production relation schema")

    relations = list(source["relations"])
    relation_keys = [str(row["key"]) for row in relations]
    if len(relation_keys) != len(set(relation_keys)):
        raise RuntimeError("duplicate relation key")
    if not relations:
        raise RuntimeError("empty production relation schema")

    type_vocabulary = [str(value) for value in source["type_vocabulary"]]
    type_index = {value: index for index, value in enumerate(type_vocabulary)}
    if len(type_index) != len(type_vocabulary):
        raise RuntimeError("duplicate type vocabulary entry")

    model = load_frozen_semantic_model(
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer = load_tokenizer(tokenizer_dir)
    texts = [relation_schema_text(row) for row in relations]
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids = tokens["input_ids"].to(device)
    attention = tokens["attention_mask"].to(device)
    outputs = model.backbone(
        input_ids=ids,
        attention_mask=attention,
        output_hidden_states=True,
        return_dict=True,
    )
    states = outputs.hidden_states
    if states is None or len(states) != 17:
        raise RuntimeError(
            "semantic hidden-state depth drift while encoding schema: "
            f"{0 if states is None else len(states)}"
        )
    stacked = torch.stack(states, dim=1)
    if stacked.size(-1) != 640:
        raise RuntimeError("semantic hidden-state width drift while encoding schema")

    special_ids = set(
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", []) or [])
    )
    content_mask = _special_token_mask(
        tokens["input_ids"],
        tokens["attention_mask"],
        special_ids,
    )

    domain = torch.zeros(
        len(relations),
        len(type_vocabulary),
        dtype=torch.bool,
    )
    range_mask = torch.zeros_like(domain)
    symmetric = torch.zeros(len(relations), dtype=torch.bool)

    for relation_id, row in enumerate(relations):
        for type_name in row["domain_types"]:
            if type_name not in type_index:
                raise RuntimeError(
                    f"{row['key']}: unknown domain type {type_name}"
                )
            domain[relation_id, type_index[type_name]] = True
        for type_name in row["range_types"]:
            if type_name not in type_index:
                raise RuntimeError(
                    f"{row['key']}: unknown range type {type_name}"
                )
            range_mask[relation_id, type_index[type_name]] = True
        if not bool(domain[relation_id].any()):
            raise RuntimeError(f"{row['key']}: empty domain type set")
        if not bool(range_mask[relation_id].any()):
            raise RuntimeError(f"{row['key']}: empty range type set")
        symmetric[relation_id] = bool(row.get("symmetric", False))

    payload = {
        "schema": SCHEMA_CACHE_VERSION,
        "source_schema_sha256": sha256(schema_path),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "relation_keys": relation_keys,
        "relation_key_to_index": {
            key: index for index, key in enumerate(relation_keys)
        },
        "core_train_relation_keys": list(source["core_train_relation_keys"]),
        "open_schema_dev_relation_keys": list(source["open_schema_dev_relation_keys"]),
        "type_vocabulary": type_vocabulary,
        "type_to_index": type_index,
        "token_states": stacked.detach().to(dtype=torch.float16).cpu(),
        "token_mask": content_mask.cpu(),
        "domain_type_mask": domain,
        "range_type_mask": range_mask,
        "symmetric": symmetric,
        "private_identity_data": False,
    }
    runtime = QSREDynamicRelationSchema(
        token_states=payload["token_states"].float(),
        token_mask=payload["token_mask"],
        domain_type_mask=payload["domain_type_mask"],
        range_type_mask=payload["range_type_mask"],
        symmetric=payload["symmetric"],
    )
    runtime.validate(num_hidden_states=17, semantic_dim=640)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    receipt = {
        "schema": "alice.eipm.n0.qsre-production-schema-cache-result.v1",
        "status": "PASS_QSRE_PRODUCTION_DYNAMIC_SCHEMA_MATERIALIZATION",
        "source_schema_sha256": payload["source_schema_sha256"],
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "schema_cache_sha256": sha256(output_path),
        "relations": len(relations),
        "core_train_relations": len(payload["core_train_relation_keys"]),
        "open_schema_dev_relations": len(payload["open_schema_dev_relation_keys"]),
        "type_vocabulary_size": len(type_vocabulary),
        "hidden_layers": int(stacked.size(1)),
        "semantic_dim": int(stacked.size(-1)),
        "schema_token_width": int(stacked.size(2)),
        "optimizer": False,
        "gradient": False,
        "private_identity_data": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def load_dynamic_schema_cache(
    path: Path,
    *,
    device: torch.device,
) -> tuple[QSREDynamicRelationSchema, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != SCHEMA_CACHE_VERSION:
        raise RuntimeError("production schema-cache version drift")
    if payload.get("private_identity_data") is not False:
        raise RuntimeError("private identity data entered schema cache")
    schema = QSREDynamicRelationSchema(
        token_states=payload["token_states"].to(device=device, dtype=torch.float32),
        token_mask=payload["token_mask"].to(device),
        domain_type_mask=payload["domain_type_mask"].to(device),
        range_type_mask=payload["range_type_mask"].to(device),
        symmetric=payload["symmetric"].to(device),
    )
    schema.validate(num_hidden_states=17, semantic_dim=640)
    return schema, payload
