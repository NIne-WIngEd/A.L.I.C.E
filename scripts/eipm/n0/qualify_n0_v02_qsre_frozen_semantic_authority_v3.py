from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_semantic_authority import (
    combine_authority_components,
)
from qsre_frozen_semantic_authority_runtime import frozen_authority_scores
from qsre_production_runtime import (
    load_frozen_semantic_model,
    load_tokenizer,
    relation_schema_text,
    sha256,
)
from train_n0_v02_qsre_closure_matcher_v2 import (
    auxiliary_relation_schema_text,
    factor_holdout_examples,
    factor_schema_text,
    relation_examples,
)


RESULT_SCHEMA = "alice.eipm.n0.qsre-frozen-semantic-authority-result.v3"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def accuracy(scores: torch.Tensor, target: torch.Tensor) -> float:
    if scores.ndim != 2:
        raise ValueError("authority accuracy requires [N,C] scores")
    if target.numel() != scores.size(0):
        raise ValueError("authority accuracy target count drift")
    return float(scores.argmax(dim=-1).cpu().eq(target.long()).float().mean().item())


def component_accuracy(bundle: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, float]:
    result = {
        "joint_preference": accuracy(bundle["joint_preference"], target),
        "semantic_projection": accuracy(bundle["semantic_projection"], target),
        "token_evidence": accuracy(bundle["token_evidence"], target),
        "combined": accuracy(bundle["combined"], target),
    }
    return result


def score(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    num_hidden_states: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    return frozen_authority_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_dim=semantic_dim,
        num_hidden_states=num_hidden_states,
        batch_size=batch_size,
        include_token_evidence=True,
    )


def min_combined(view_scores: list[dict[str, float]]) -> float:
    return min(float(row["combined"]) for row in view_scores)


def relation_eval(
    *,
    view0: list[str],
    view1: list[str],
    targets: torch.Tensor,
    candidates: list[str],
    target_offset: int,
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    num_hidden_states: int,
    batch_size: int,
) -> tuple[float, dict]:
    target = targets.long() + int(target_offset)
    views = []
    for query in (view0, view1):
        bundle = score(
            queries=query,
            candidates=candidates,
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_dim=semantic_dim,
            num_hidden_states=num_hidden_states,
            batch_size=batch_size,
        )
        views.append(component_accuracy(bundle, target))
    return min_combined(views), {"view0": views[0], "view1": views[1]}


def factor_eval(
    *,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    num_hidden_states: int,
    batch_size: int,
) -> tuple[float, dict]:
    source = factor_holdout_examples(meta)
    scores: dict[str, float] = {}
    diagnostics: dict[str, object] = {}

    for category in ("role", "traversal", "direction", "control"):
        rows = list(meta["factors"][category])
        candidates = [
            factor_schema_text(category, str(row["description"]))
            for row in rows
        ]
        texts, target = source[category]
        bundle = score(
            queries=texts,
            candidates=candidates,
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_dim=semantic_dim,
            num_hidden_states=num_hidden_states,
            batch_size=batch_size,
        )
        diag = component_accuracy(bundle, target)
        diagnostics[category] = diag
        scores[category] = float(diag["combined"])

    modifier_texts, modifier_target = source["modifiers"]
    modifier_rows = list(meta["factors"]["modifiers"])
    correct = {key: 0 for key in ("joint_preference", "semantic_projection", "token_evidence", "combined")}
    count = int(modifier_target.size(0))
    per_row = []
    for row_index in range(count):
        modifier_index = int(modifier_target[row_index, 0].item())
        binary_target = modifier_target[row_index, 1:2].long()
        row = modifier_rows[modifier_index]
        candidates = [
            factor_schema_text("modifier state", str(row["off_description"])),
            factor_schema_text("modifier state", str(row["on_description"])),
        ]
        bundle = score(
            queries=[modifier_texts[row_index]],
            candidates=candidates,
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_dim=semantic_dim,
            num_hidden_states=num_hidden_states,
            batch_size=1,
        )
        diag = component_accuracy(bundle, binary_target)
        per_row.append(diag)
        for key, value in diag.items():
            correct[key] += int(value == 1.0)
    modifier_diag = {key: value / max(count, 1) for key, value in correct.items()}
    modifier_diag["rows"] = count
    diagnostics["modifiers"] = modifier_diag
    scores["modifiers"] = float(modifier_diag["combined"])

    macro = sum(scores.values()) / max(len(scores), 1)
    diagnostics["combined_factor_accuracy"] = scores
    diagnostics["combined_macro"] = macro
    return macro, diagnostics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--meta-config", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--production-cache", required=True)
    p.add_argument("--production-curriculum", required=True)
    p.add_argument("--production-relation-schema", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("frozen semantic authority qualification requires CUDA")
    device = torch.device("cuda")

    plan_path = Path(args.plan)
    meta_path = Path(args.meta_config)
    semantic_config_path = Path(args.semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    production_cache_path = Path(args.production_cache)
    production_curriculum_path = Path(args.production_curriculum)
    production_relation_schema_path = Path(args.production_relation_schema)
    output_path = Path(args.output)

    if output_path.exists():
        raise SystemExit("refusing to overwrite frozen semantic authority qualification")

    plan = read_json(plan_path)
    if plan.get("schema") != "alice.eipm.n0.qsre-frozen-semantic-authority-plan.v3":
        raise SystemExit("frozen semantic authority plan schema drift")
    if plan["semantic_authority"]["gradient"] is not False:
        raise SystemExit("semantic authority qualification unexpectedly authorizes gradient")
    if plan["semantic_authority"]["optimizer"] is not False:
        raise SystemExit("semantic authority qualification unexpectedly authorizes optimizer")

    meta = read_json(meta_path)
    if meta.get("schema") != "alice.eipm.n0.qsre-closure-schema-meta.v2":
        raise SystemExit("semantic authority meta schema drift")
    if meta["schema_text_contract"]["relation_key_in_semantic_text"] is not False:
        raise SystemExit("relation key entered semantic authority text")

    production = torch.load(production_cache_path, map_location="cpu")
    if production.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered Production cache")
    if production.get("test_present") is not False:
        raise SystemExit("TEST entered Production cache")
    raw_rows = read_jsonl(production_curriculum_path)
    raw_train = [row for row in raw_rows if row["split"] == "train"]
    raw_by_id = {str(row["id"]): row for row in raw_train}
    if len(raw_by_id) != len(raw_train):
        raise SystemExit("duplicate Production TRAIN row id")

    relation_schema = read_json(production_relation_schema_path)
    relation_by_key = {str(row["key"]): row for row in relation_schema["relations"]}
    core_keys = [str(x) for x in relation_schema["core_train_relation_keys"]]
    open_keys = [str(x) for x in relation_schema["open_schema_dev_relation_keys"]]
    forbidden = set(str(x) for x in meta["forbidden_semantics"])
    if not set(open_keys).issubset(forbidden):
        raise SystemExit("Production open relation escaped forbidden semantic set")
    core_candidates = [relation_schema_text(relation_by_key[key]) for key in core_keys]

    model = load_frozen_semantic_model(
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer = load_tokenizer(tokenizer_dir)
    semantic_dim = int(production["train"]["query_hidden_states"].size(-1))
    num_hidden_states = int(production["train"]["query_hidden_states"].size(2))

    generation = meta["generation"]
    relation_train = list(meta["relation_train"])
    relation_holdout = list(meta["relation_holdout"])

    seen_v0, seen_v1, seen_target = relation_examples(
        relation_train,
        phrase_field="heldout_phrases",
        seed=int(generation["seed"]) + 1,
        examples_per_relation=max(
            4,
            int(generation["holdout_examples_per_relation"]) // 2,
        ),
    )
    hold_v0, hold_v1, hold_target = relation_examples(
        relation_holdout,
        phrase_field="phrases",
        seed=int(generation["seed"]) + 2,
        examples_per_relation=int(generation["holdout_examples_per_relation"]),
    )
    train_candidates = [
        auxiliary_relation_schema_text(row)
        for row in relation_train
    ]
    full_candidates = train_candidates + [
        auxiliary_relation_schema_text(row)
        for row in relation_holdout
    ]

    seen_score, seen_diag = relation_eval(
        view0=seen_v0,
        view1=seen_v1,
        targets=seen_target,
        candidates=train_candidates,
        target_offset=0,
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_dim=semantic_dim,
        num_hidden_states=num_hidden_states,
        batch_size=int(args.batch_size),
    )
    holdout_score, holdout_diag = relation_eval(
        view0=hold_v0,
        view1=hold_v1,
        targets=hold_target,
        candidates=full_candidates,
        target_offset=len(relation_train),
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_dim=semantic_dim,
        num_hidden_states=num_hidden_states,
        batch_size=int(args.batch_size),
    )

    train_split = production["train"]
    relation_mask = train_split["relation_target_mask"].bool()
    relational = train_split["control_target"].long().eq(1)
    core_single = relation_mask.sum(dim=-1).eq(1) & relational
    indices = core_single.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        raise SystemExit("no Production core single-relation rows")
    core_target = train_split["relation_target"][indices, 0].long()
    production_views: list[dict[str, float]] = []
    for view in (0, 1):
        queries = [
            str(raw_by_id[str(train_split["ids"][int(i)])]["query_views"][view])
            for i in indices.tolist()
        ]
        bundle = score(
            queries=queries,
            candidates=core_candidates,
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_dim=semantic_dim,
            num_hidden_states=num_hidden_states,
            batch_size=int(args.batch_size),
        )
        production_views.append(component_accuracy(bundle, core_target))
    production_score = min_combined(production_views)

    factor_macro, factor_diag = factor_eval(
        meta=meta,
        model=model,
        tokenizer=tokenizer,
        device=device,
        semantic_dim=semantic_dim,
        num_hidden_states=num_hidden_states,
        batch_size=int(args.batch_size),
    )

    metrics = {
        "auxiliary_seen_relation_top1": seen_score,
        "auxiliary_holdout_relation_top1": holdout_score,
        "production_core_single_relation_top1": production_score,
        "heldout_factor_macro_accuracy": factor_macro,
    }
    threshold = plan["eligibility"]
    checks = {
        key: float(metrics[key]) >= float(threshold[key])
        for key in metrics
    }
    passed = all(checks.values())

    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_QSRE_FROZEN_SEMANTIC_AUTHORITY"
            if passed
            else "FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY"
        ),
        "p2_authorized": passed,
        "metrics": metrics,
        "eligibility": threshold,
        "gate_checks": checks,
        "diagnostics": {
            "auxiliary_seen": seen_diag,
            "auxiliary_holdout": holdout_diag,
            "production_core": {
                "view0": production_views[0],
                "view1": production_views[1],
            },
            "factors": factor_diag,
        },
        "authority_components": list(plan["semantic_authority"]["components"]),
        "component_normalization": plan["semantic_authority"]["component_normalization"],
        "component_fusion": plan["semantic_authority"]["component_fusion"],
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "plan_sha256": sha256(plan_path),
        "meta_config_sha256": sha256(meta_path),
        "production_cache_sha256": sha256(production_cache_path),
        "production_curriculum_sha256": sha256(production_curriculum_path),
        "production_relation_schema_sha256": sha256(production_relation_schema_path),
        "gradient_performed": False,
        "optimizer": False,
        "matcher_training": False,
        "trainable_authority_parameters": 0,
        "relation_keys_used_as_semantic_tokens": False,
        "production_open_schema_descriptions_used_in_gradient": False,
        "final_only_relation_descriptions_used_in_gradient": False,
        "auxiliary_holdout_relation_descriptions_used_in_gradient": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "automatic_rerun": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FROZEN_SEMANTIC_AUTHORITY_RESULT=" + json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
