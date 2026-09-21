from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prepare_n0_v02_qsre_production_cache_v1 import encode_texts
from qsre_frozen_semantic_authority_runtime import (
    joint_preference_scores,
    principle_alignment_scores,
    semantic_projection_scores,
)
from qsre_production_runtime import (
    load_frozen_semantic_model,
    load_tokenizer,
    relation_schema_text,
    sha256,
)
from train_n0_v02_qsre_closure_matcher_v2 import factor_schema_text


PRODUCTION_SCHEMA = "alice.eipm.n0.qsre-frozen-semantic-authority-production-cache.v3"
FINAL_SCHEMA = "alice.eipm.n0.qsre-frozen-semantic-authority-final-cache.v3"
FACTOR_SCHEMA = "alice.eipm.n0.qsre-frozen-factor-schema-cache.v3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def query_views(rows: list[dict]) -> tuple[list[str], int]:
    values = [list(row["query_views"]) for row in rows]
    if not values or any(len(row) != 2 for row in values):
        raise RuntimeError("authority cache requires exactly two query views per row")
    return [str(text) for row in values for text in row], 2


@torch.inference_mode()
def score_components(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    joint = joint_preference_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    semantic = semantic_projection_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    principle = principle_alignment_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    return {
        "joint_preference": joint.to(dtype=torch.float32),
        "semantic_projection": semantic.to(dtype=torch.float32),
        "principle_alignment": principle.to(dtype=torch.float32),
    }


def reshape_views(
    row: dict[str, torch.Tensor],
    *,
    count: int,
    views: int,
) -> dict[str, torch.Tensor]:
    return {
        key: value.reshape(count, views, value.size(-1)).contiguous()
        for key, value in row.items()
    }


def factor_candidate_text(meta: dict) -> dict[str, list[str]]:
    result = {}
    for category in ("role", "traversal", "direction", "control"):
        result[category] = [
            factor_schema_text(category, str(row["description"]))
            for row in meta["factors"][category]
        ]
    modifiers = []
    for row in meta["factors"]["modifiers"]:
        modifiers.extend(
            [
                factor_schema_text("modifier state", str(row["off_description"])),
                factor_schema_text("modifier state", str(row["on_description"])),
            ]
        )
    result["modifiers"] = modifiers
    return result


@torch.inference_mode()
def build_factor_cache(
    *,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> dict:
    cache: dict[str, object] = {
        "schema": FACTOR_SCHEMA,
        "private_identity_data": False,
        "gradient": False,
        "optimizer": False,
    }
    candidates = factor_candidate_text(meta)
    for category in ("role", "traversal", "direction", "control"):
        states, mask = encode_texts(
            texts=candidates[category],
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=96,
            batch_size=batch_size,
            all_hidden_states=True,
        )
        cache[category] = {
            "keys": [str(row["key"]) for row in meta["factors"][category]],
            "token_states": states,
            "token_mask": mask,
        }
    modifier_states, modifier_mask = encode_texts(
        texts=candidates["modifiers"],
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=batch_size,
        all_hidden_states=True,
    )
    modifier_count = len(meta["factors"]["modifiers"])
    cache["modifiers"] = {
        "keys": [str(row["key"]) for row in meta["factors"]["modifiers"]],
        "token_states": modifier_states.reshape(
            modifier_count,
            2,
            *modifier_states.shape[1:],
        ),
        "token_mask": modifier_mask.reshape(
            modifier_count,
            2,
            modifier_mask.size(-1),
        ),
    }
    return cache


@torch.inference_mode()
def score_factors(
    *,
    queries: list[str],
    row_count: int,
    views: int,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> dict:
    candidates = factor_candidate_text(meta)
    result = {}
    for category in ("role", "traversal", "direction", "control", "modifiers"):
        raw = score_components(
            queries=queries,
            candidates=candidates[category],
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        shaped = reshape_views(raw, count=row_count, views=views)
        if category == "modifiers":
            modifier_count = len(meta["factors"]["modifiers"])
            shaped = {
                key: value.reshape(
                    row_count,
                    views,
                    modifier_count,
                    2,
                ).contiguous()
                for key, value in shaped.items()
            }
        result[category] = shaped
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("production", "final"), required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--relation-schema", required=True)
    p.add_argument("--schema-cache", required=True)
    p.add_argument("--meta-config", required=True)
    p.add_argument("--qualification-result", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--factor-schema-output")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("frozen semantic authority cache materialization requires CUDA")
    device = torch.device("cuda")

    rows_path = Path(args.rows)
    relation_schema_path = Path(args.relation_schema)
    schema_cache_path = Path(args.schema_cache)
    meta_path = Path(args.meta_config)
    qualification_path = Path(args.qualification_result)
    semantic_config_path = Path(args.semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit("refusing to overwrite frozen semantic authority cache")

    rows = read_jsonl(rows_path)
    if not rows:
        raise SystemExit("empty authority-cache row source")
    if any(row.get("private_identity_data") is not False for row in rows):
        raise SystemExit("private identity data entered authority-cache rows")

    meta = read_json(meta_path)
    if meta.get("schema") != "alice.eipm.n0.qsre-closure-schema-meta.v2":
        raise SystemExit("authority-cache meta schema drift")
    qualification = read_json(qualification_path)
    if qualification.get("schema") != "alice.eipm.n0.qsre-frozen-semantic-authority-result.v3":
        raise SystemExit("authority qualification result schema drift")
    if qualification.get("status") != "PASS_QSRE_FROZEN_SEMANTIC_AUTHORITY":
        raise SystemExit("authority cache requires passing frozen semantic qualification")
    if qualification.get("p2_authorized") is not True:
        raise SystemExit("authority qualification did not authorize Production P2")
    if qualification.get("gradient_performed") is not False or qualification.get("optimizer") is not False:
        raise SystemExit("authority qualification was not zero-gradient")
    if qualification.get("semantic_checkpoint_sha256") != sha256(semantic_checkpoint):
        raise SystemExit("authority qualification semantic checkpoint drift")
    relation_schema = read_json(relation_schema_path)
    schema_cache = torch.load(schema_cache_path, map_location="cpu")
    if schema_cache.get("schema") != "alice.eipm.n0.qsre-production-schema-cache.v1":
        raise SystemExit("authority-cache dynamic schema cache version drift")
    if schema_cache.get("source_schema_sha256") != sha256(relation_schema_path):
        raise SystemExit("authority-cache relation-schema source hash drift")
    if schema_cache.get("semantic_checkpoint_sha256") != sha256(semantic_checkpoint):
        raise SystemExit("authority-cache semantic checkpoint/schema-cache drift")
    if schema_cache.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered authority-cache dynamic schema")
    relation_rows = list(relation_schema["relations"])
    relation_by_key = {str(row["key"]): row for row in relation_rows}
    all_keys = [str(row["key"]) for row in relation_rows]
    if len(all_keys) != len(set(all_keys)):
        raise SystemExit("duplicate runtime relation key")
    if list(schema_cache["relation_keys"]) != all_keys:
        raise SystemExit("authority-cache runtime relation key order drift")
    relation_text = {
        key: relation_schema_text(relation_by_key[key])
        for key in all_keys
    }

    model = load_frozen_semantic_model(
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer = load_tokenizer(tokenizer_dir)

    payload: dict[str, object] = {
        "schema": PRODUCTION_SCHEMA if args.mode == "production" else FINAL_SCHEMA,
        "mode": args.mode,
        "rows_sha256": sha256(rows_path),
        "relation_schema_sha256": sha256(relation_schema_path),
        "schema_cache_sha256": sha256(schema_cache_path),
        "meta_config_sha256": sha256(meta_path),
        "qualification_result_sha256": sha256(qualification_path),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "authority_components": [
            "joint_preference",
            "semantic_projection",
            "principle_alignment",
            "runtime_parameter_free_token_evidence",
        ],
        "component_normalization": "candidate_axis_zscore_after_runtime_candidate_subset",
        "component_fusion": "equal_weight_mean",
        "relation_keys_used_as_semantic_tokens": False,
        "gradient": False,
        "optimizer": False,
        "private_identity_data": False,
    }

    if args.mode == "production":
        split_payload = {}
        for split in ("train", "dev"):
            split_rows = [row for row in rows if str(row["split"]) == split]
            if not split_rows:
                raise SystemExit(f"empty Production authority split: {split}")
            queries, views = query_views(split_rows)
            if split == "train":
                relation_keys = [
                    str(x) for x in relation_schema["core_train_relation_keys"]
                ]
            else:
                relation_keys = all_keys
            raw = score_components(
                queries=queries,
                candidates=[relation_text[key] for key in relation_keys],
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=int(args.batch_size),
            )
            split_payload[split] = {
                "ids": [str(row["id"]) for row in split_rows],
                "query_views": views,
                "relation_keys": relation_keys,
                "relation": reshape_views(
                    raw,
                    count=len(split_rows),
                    views=views,
                ),
                "factors": score_factors(
                    queries=queries,
                    row_count=len(split_rows),
                    views=views,
                    meta=meta,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=int(args.batch_size),
                ),
            }
        payload["train"] = split_payload["train"]
        payload["dev"] = split_payload["dev"]

        if not args.factor_schema_output:
            raise SystemExit("production authority cache requires --factor-schema-output")
        factor_path = Path(args.factor_schema_output)
        if factor_path.exists():
            raise SystemExit("refusing to overwrite frozen factor schema cache")
        factor_cache = build_factor_cache(
            meta=meta,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=int(args.batch_size),
        )
        factor_cache["semantic_checkpoint_sha256"] = sha256(semantic_checkpoint)
        factor_cache["meta_config_sha256"] = sha256(meta_path)
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(factor_cache, factor_path)
        payload["factor_schema_cache_sha256"] = sha256(factor_path)
    else:
        queries, views = query_views(rows)
        raw = score_components(
            queries=queries,
            candidates=[relation_text[key] for key in all_keys],
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=int(args.batch_size),
        )
        payload["rows"] = {
            "ids": [str(row["id"]) for row in rows],
            "query_views": views,
            "relation_keys": all_keys,
            "relation": reshape_views(
                raw,
                count=len(rows),
                views=views,
            ),
            "factors": score_factors(
                queries=queries,
                row_count=len(rows),
                views=views,
                meta=meta,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=int(args.batch_size),
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    receipt = {
        "schema": payload["schema"] + ".result",
        "status": "PASS_QSRE_FROZEN_SEMANTIC_AUTHORITY_CACHE",
        "mode": args.mode,
        "cache_sha256": sha256(output_path),
        "rows": len(rows),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "schema_cache_sha256": sha256(schema_cache_path),
        "qualification_result_sha256": sha256(qualification_path),
        "relation_keys_used_as_semantic_tokens": False,
        "gradient": False,
        "optimizer": False,
        "private_identity_data": False,
    }
    print("FROZEN_AUTHORITY_CACHE=" + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
