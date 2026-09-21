#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_semantic_authority import candidate_zscore
from prepare_n0_v02_qsre_production_cache_v1 import encode_texts
from qsre_frozen_semantic_authority_runtime import (
    JOINT_MATCH_PROMPT,
    frozen_authority_scores,
)
from qsre_production_runtime import (
    load_frozen_semantic_model,
    load_tokenizer,
    relation_schema_text,
)
from train_n0_v02_qsre_closure_matcher_v2 import (
    auxiliary_relation_schema_text,
    factor_holdout_examples,
    factor_schema_text,
    relation_examples,
)

SCHEMA = "alice.eipm.n0.qsre-p2a-semantic-localization.v1"
SEMANTIC_SHA = "6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"
SOURCE_METRICS = {
    "production_core_single_relation_top1": 0.4068181812763214,
    "auxiliary_seen_relation_top1": 0.2708333432674408,
    "auxiliary_holdout_relation_top1": 0.2604166567325592,
    "heldout_factor_macro_accuracy": 0.36666667064030967,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def acc(scores: torch.Tensor, target: torch.Tensor) -> float:
    return float(scores.argmax(-1).eq(target.long()).float().mean().item())


def confusion(scores: torch.Tensor, target: torch.Tensor, labels: list[str]) -> dict[str, dict[str, int]]:
    out = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(target.long().tolist(), scores.argmax(-1).tolist()):
        out[labels[g]][labels[p]] += 1
    return out


def zmean(*values: torch.Tensor) -> torch.Tensor:
    return sum(candidate_zscore(x.float()) for x in values) / float(len(values))


@torch.inference_mode()
def pair_scores(
    queries: list[str],
    candidates: list[str],
    *,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    meta_prompt: bool,
    principle: bool,
) -> torch.Tensor:
    rationale = None
    if principle:
        chunks = []
        for start in range(0, len(candidates), batch_size):
            stop = min(start + batch_size, len(candidates))
            t = tokenizer(
                candidates[start:stop],
                padding=True,
                truncation=True,
                max_length=192,
                return_tensors="pt",
            )
            pooled = model.encode(t["input_ids"].to(device), t["attention_mask"].to(device))
            chunks.append(model.project_rationale_pooled(pooled).detach().float().cpu())
        rationale = torch.cat(chunks, 0)

    prompts: list[str] = []
    texts: list[str] = []
    indices: list[int] = []
    for q in queries:
        p = (JOINT_MATCH_PROMPT + q) if meta_prompt else q
        for i, c in enumerate(candidates):
            prompts.append(p)
            texts.append(c)
            indices.append(i)

    values = []
    for start in range(0, len(prompts), batch_size):
        stop = min(start + batch_size, len(prompts))
        t = tokenizer(
            prompts[start:stop],
            texts[start:stop],
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )
        ids = t["input_ids"].to(device)
        mask = t["attention_mask"].to(device)
        if principle:
            semantic = model.project_semantic_pooled(model.encode(ids, mask))
            rr = rationale[torch.tensor(indices[start:stop], dtype=torch.long)].to(
                device=device, dtype=semantic.dtype
            )
            value = model.principle_alignment_from_projected(semantic, rr)
        else:
            value = model.score_candidates(ids, mask)
        values.append(value.detach().float().cpu())
    return torch.cat(values, 0).reshape(len(queries), len(candidates))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.softmax(logits.masked_fill(~mask, -1.0e4), dim=dim) * mask.to(logits.dtype)
    return x / x.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)


def token_layer_batch(
    q: torch.Tensor,
    qm: torch.Tensor,
    s: torch.Tensor,
    sm: torch.Tensor,
) -> torch.Tensor:
    q = F.normalize(q.float(), dim=-1)
    s = F.normalize(s.float(), dim=-1)
    sim = torch.einsum("btd,csd->bcts", q, s)
    b, c, tq, ts = sim.shape
    valid = qm[:, None, :, None].expand(b, c, tq, ts) & sm[None, :, None, :].expand(b, c, tq, ts)
    sim = sim.masked_fill(~valid, -1.0e4)
    qc = sim.max(-1).values
    sc = sim.max(-2).values
    qw = masked_softmax(4.0 * qc, qm[:, None, :].expand(b, c, tq), -1)
    sw = masked_softmax(4.0 * sc, sm[None, :, :].expand(b, c, ts), -1)
    return 0.5 * (qw * qc).sum(-1) + 0.5 * (sw * sc).sum(-1)


@torch.inference_mode()
def layer_scores(
    queries: list[str],
    candidates: list[str],
    *,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> list[torch.Tensor]:
    qs, qm = encode_texts(
        texts=queries, model=model, tokenizer=tokenizer, device=device,
        max_length=128, batch_size=batch_size, all_hidden_states=True
    )
    ss, sm = encode_texts(
        texts=candidates, model=model, tokenizer=tokenizer, device=device,
        max_length=128, batch_size=batch_size, all_hidden_states=True
    )
    out = []
    for layer in range(qs.size(1)):
        chunks = []
        for start in range(0, qs.size(0), batch_size):
            stop = min(start + batch_size, qs.size(0))
            chunks.append(token_layer_batch(
                qs[start:stop, layer], qm[start:stop].bool(),
                ss[:, layer], sm.bool()
            ).cpu())
        out.append(torch.cat(chunks, 0))
    return out


def evaluate_task(
    name: str,
    queries: list[str],
    candidates: list[str],
    target: torch.Tensor,
    labels: list[str],
    *,
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    hidden_layers: int,
    batch_size: int,
) -> dict[str, Any]:
    current = frozen_authority_scores(
        queries=queries, candidates=candidates, model=model, tokenizer=tokenizer,
        device=device, semantic_dim=semantic_dim, num_hidden_states=hidden_layers,
        batch_size=batch_size, include_token_evidence=True
    )
    native_joint = pair_scores(
        queries, candidates, model=model, tokenizer=tokenizer, device=device,
        batch_size=batch_size, meta_prompt=False, principle=False
    )
    native_principle = pair_scores(
        queries, candidates, model=model, tokenizer=tokenizer, device=device,
        batch_size=batch_size, meta_prompt=False, principle=True
    )
    layers = layer_scores(
        queries, candidates, model=model, tokenizer=tokenizer,
        device=device, batch_size=batch_size
    )
    layer_acc = [acc(x, target) for x in layers]
    best_layer = max(range(len(layer_acc)), key=lambda i: (layer_acc[i], -i))
    best_token = layers[best_layer]
    native_fusion = zmean(
        native_joint,
        current["semantic_projection"],
        native_principle,
        current["token_evidence"],
    )
    native_best_layer = zmean(
        native_joint,
        current["semantic_projection"],
        native_principle,
        best_token,
    )

    components = {
        "meta_joint_preference": current["joint_preference"],
        "teacher_native_joint_preference": native_joint,
        "semantic_projection": current["semantic_projection"],
        "meta_principle_alignment": current["principle_alignment"],
        "teacher_native_principle_alignment": native_principle,
        "mixed_token_evidence": current["token_evidence"],
        "best_layer_token_oracle": best_token,
    }
    component_metrics = {
        key: {
            "accuracy": acc(value, target),
            "confusion": confusion(value, target, labels),
        }
        for key, value in components.items()
    }

    any_component = torch.stack(
        [value.argmax(-1).eq(target.long()) for value in components.values()], 0
    ).any(0).float().mean().item()

    return {
        "task": name,
        "rows": len(queries),
        "candidate_labels": labels,
        "component_metrics": component_metrics,
        "layer_token_accuracy": {
            str(i): value for i, value in enumerate(layer_acc)
        },
        "best_layer_token": {
            "layer": best_layer,
            "accuracy": layer_acc[best_layer],
        },
        "current_four_surface_accuracy": acc(current["combined"], target),
        "teacher_native_four_surface_accuracy": acc(native_fusion, target),
        "teacher_native_plus_best_layer_oracle_accuracy": acc(native_best_layer, target),
        "per_row_any_component_oracle_accuracy": float(any_component),
        "oracle_warning": "Label-aware upper bound only; not an authorized production selector.",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-failure-result", required=True)
    p.add_argument("--meta-config", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--production-cache", required=True)
    p.add_argument("--production-curriculum", required=True)
    p.add_argument("--production-relation-schema", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    if torch.cuda.is_available():
        raise SystemExit("This diagnostic is deliberately CPU-only.")

    source_path = Path(args.source_failure_result)
    meta_path = Path(args.meta_config)
    semantic_path = Path(args.semantic_checkpoint)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Refusing to overwrite semantic-localization result.")

    source = read_json(source_path)
    if source.get("status") != "FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY":
        raise SystemExit("Source is not the preserved P2A failure.")
    if source.get("p2_authorized") is not False:
        raise SystemExit("Source P2A unexpectedly authorized P2.")
    for key, expected in SOURCE_METRICS.items():
        actual = float(source["metrics"][key])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise SystemExit(f"Job575986 metric drift: {key}={actual}")

    if sha256(semantic_path) != SEMANTIC_SHA:
        raise SystemExit("Semantic checkpoint SHA drift.")

    meta = read_json(meta_path)
    production = torch.load(Path(args.production_cache), map_location="cpu")
    rows = read_jsonl(Path(args.production_curriculum))
    train_rows = [row for row in rows if row["split"] == "train"]
    row_by_id = {str(row["id"]): row for row in train_rows}
    schema = read_json(Path(args.production_relation_schema))

    if production.get("private_identity_data") is not False:
        raise SystemExit("Private identity data entered Production cache.")
    if production.get("test_present") is not False:
        raise SystemExit("TEST entered Production cache.")

    device = torch.device("cpu")
    model = load_frozen_semantic_model(
        semantic_config_path=Path(args.semantic_config),
        semantic_checkpoint=semantic_path,
        device=device,
    )
    tokenizer = load_tokenizer(Path(args.tokenizer_dir))
    split = production["train"]
    semantic_dim = int(split["query_hidden_states"].size(-1))
    hidden_layers = int(split["query_hidden_states"].size(2))
    if hidden_layers != 17:
        raise SystemExit("Semantic hidden-state depth drift.")

    generation = meta["generation"]
    relation_train = list(meta["relation_train"])
    relation_holdout = list(meta["relation_holdout"])

    seen0, seen1, seen_t = relation_examples(
        relation_train, phrase_field="heldout_phrases",
        seed=int(generation["seed"]) + 1,
        examples_per_relation=max(4, int(generation["holdout_examples_per_relation"]) // 2),
    )
    hold0, hold1, hold_local = relation_examples(
        relation_holdout, phrase_field="phrases",
        seed=int(generation["seed"]) + 2,
        examples_per_relation=int(generation["holdout_examples_per_relation"]),
    )
    train_candidates = [auxiliary_relation_schema_text(x) for x in relation_train]
    train_labels = [str(x["key"]) for x in relation_train]
    full_candidates = train_candidates + [auxiliary_relation_schema_text(x) for x in relation_holdout]
    full_labels = train_labels + [str(x["key"]) for x in relation_holdout]
    hold_t = hold_local.long() + len(relation_train)

    relation_by_key = {str(x["key"]): x for x in schema["relations"]}
    core_keys = [str(x) for x in schema["core_train_relation_keys"]]
    core_candidates = [relation_schema_text(relation_by_key[x]) for x in core_keys]
    relation_mask = split["relation_target_mask"].bool()
    relational = split["control_target"].long().eq(1)
    idx = (relation_mask.sum(-1).eq(1) & relational).nonzero(as_tuple=False).flatten()
    core_t = split["relation_target"][idx, 0].long()
    prod0, prod1 = [], []
    for i in idx.tolist():
        row = row_by_id[str(split["ids"][int(i)])]
        prod0.append(str(row["query_views"][0]))
        prod1.append(str(row["query_views"][1]))

    tasks: dict[str, dict[str, Any]] = {}
    for name, q, c, t, labels in (
        ("auxiliary_seen_view0", seen0, train_candidates, seen_t, train_labels),
        ("auxiliary_seen_view1", seen1, train_candidates, seen_t, train_labels),
        ("auxiliary_holdout_view0", hold0, full_candidates, hold_t, full_labels),
        ("auxiliary_holdout_view1", hold1, full_candidates, hold_t, full_labels),
        ("production_core_view0", prod0, core_candidates, core_t, core_keys),
        ("production_core_view1", prod1, core_candidates, core_t, core_keys),
    ):
        tasks[name] = evaluate_task(
            name, q, c, t, labels, model=model, tokenizer=tokenizer, device=device,
            semantic_dim=semantic_dim, hidden_layers=hidden_layers,
            batch_size=int(args.batch_size)
        )

    factor_source = factor_holdout_examples(meta)
    factor_tasks: dict[str, dict[str, Any]] = {}
    for category in ("role", "traversal", "direction", "control"):
        entries = list(meta["factors"][category])
        q, t = factor_source[category]
        c = [factor_schema_text(category, str(x["description"])) for x in entries]
        labels = [str(x["key"]) for x in entries]
        factor_tasks[category] = evaluate_task(
            category, q, c, t, labels, model=model, tokenizer=tokenizer, device=device,
            semantic_dim=semantic_dim, hidden_layers=hidden_layers,
            batch_size=int(args.batch_size)
        )

    modifier_q, modifier_t = factor_source["modifiers"]
    modifier_names = []
    for mi, entry in enumerate(meta["factors"]["modifiers"]):
        selected = (modifier_t[:, 0] == mi).nonzero(as_tuple=False).flatten()
        q = [modifier_q[int(i)] for i in selected.tolist()]
        t = modifier_t.index_select(0, selected)[:, 1].long()
        c = [
            factor_schema_text("modifier state", str(entry["off_description"])),
            factor_schema_text("modifier state", str(entry["on_description"])),
        ]
        key = "modifier:" + str(entry["key"])
        modifier_names.append(key)
        factor_tasks[key] = evaluate_task(
            key, q, c, t, ["OFF", "ON"], model=model, tokenizer=tokenizer, device=device,
            semantic_dim=semantic_dim, hidden_layers=hidden_layers,
            batch_size=int(args.batch_size)
        )

    def pair_min(prefix: str, field: str) -> float:
        return min(float(tasks[prefix + "_view0"][field]), float(tasks[prefix + "_view1"][field]))

    def factor_macro(field: str) -> float:
        categorical = [float(factor_tasks[x][field]) for x in ("role", "traversal", "direction", "control")]
        modifier = sum(float(factor_tasks[x][field]) for x in modifier_names) / len(modifier_names)
        return sum(categorical + [modifier]) / 5.0

    current = {
        "auxiliary_seen_relation_top1": pair_min("auxiliary_seen", "current_four_surface_accuracy"),
        "auxiliary_holdout_relation_top1": pair_min("auxiliary_holdout", "current_four_surface_accuracy"),
        "production_core_single_relation_top1": pair_min("production_core", "current_four_surface_accuracy"),
        "heldout_factor_macro_accuracy": factor_macro("current_four_surface_accuracy"),
    }
    native = {
        "auxiliary_seen_relation_top1": pair_min("auxiliary_seen", "teacher_native_four_surface_accuracy"),
        "auxiliary_holdout_relation_top1": pair_min("auxiliary_holdout", "teacher_native_four_surface_accuracy"),
        "production_core_single_relation_top1": pair_min("production_core", "teacher_native_four_surface_accuracy"),
        "heldout_factor_macro_accuracy": factor_macro("teacher_native_four_surface_accuracy"),
    }
    native_best_layer = {
        "auxiliary_seen_relation_top1": pair_min("auxiliary_seen", "teacher_native_plus_best_layer_oracle_accuracy"),
        "auxiliary_holdout_relation_top1": pair_min("auxiliary_holdout", "teacher_native_plus_best_layer_oracle_accuracy"),
        "production_core_single_relation_top1": pair_min("production_core", "teacher_native_plus_best_layer_oracle_accuracy"),
        "heldout_factor_macro_accuracy": factor_macro("teacher_native_plus_best_layer_oracle_accuracy"),
    }
    reconstructed = all(
        math.isclose(float(current[k]), float(source["metrics"][k]), rel_tol=0.0, abs_tol=1e-6)
        for k in current
    )

    payload = {
        "schema": SCHEMA,
        "status": "COMPLETE_ZERO_GRADIENT_SEMANTIC_LOCALIZATION",
        "gradient": False,
        "optimizer": False,
        "model_training": False,
        "gpu": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "test_opened": False,
        "threshold_changed": False,
        "automatic_rerun": False,
        "production_p2_authorized": False,
        "semantic_backbone_retraining_authorized": False,
        "diagnostic_only": True,
        "source_failure_sha256": sha256(source_path),
        "semantic_checkpoint_sha256": sha256(semantic_path),
        "reconstruction": {
            "metrics": current,
            "matches_job575986_within_1e-6": reconstructed,
        },
        "teacher_native_prompt_geometry": native,
        "teacher_native_plus_best_layer_oracle": {
            "metrics": native_best_layer,
            "warning": "Best layer is selected posthoc per task; diagnostic upper bound only.",
        },
        "relation_tasks": tasks,
        "factor_tasks": factor_tasks,
        "decision_contract": {
            "representation_recoverable": "Intermediate/token views approach the existing gates while pooled/trained heads do not.",
            "readout_or_prompt_mismatch": "Teacher-native prompt/candidate geometry materially improves a frozen trained head.",
            "fusion_or_calibration": "One frozen component is strong while equal-weight z-score fusion degrades it.",
            "semantic_objective_reopen": "No frozen view exposes broad held-out schema semantics near the existing gates while prior learned adaptation remains substantially stronger.",
            "no_automatic_architecture_change": True,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("===== N0 P2A SEMANTIC LOCALIZATION V1 =====")
    print("source_reconstruction_matches=" + str(reconstructed).lower())
    print("current=" + json.dumps(current, sort_keys=True))
    print("teacher_native=" + json.dumps(native, sort_keys=True))
    print("native_plus_best_layer_oracle=" + json.dumps(native_best_layer, sort_keys=True))
    print("result=" + str(output))


if __name__ == "__main__":
    main()
