#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.v02_model import AliceN0V02Model

import audit_n0_v02_query_semantic_representations_v0_1 as base

SCHEMA = "alice.eipm.n0.query-semantics-layerwise-audit.v0.1"


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def masked_layer_tokens(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for i in range(hidden.size(0)):
        mask = attention_mask[i].bool().clone()
        for special in special_ids:
            mask &= input_ids[i].ne(int(special))
        values = hidden[i][mask]
        if values.numel() == 0:
            values = hidden[i][attention_mask[i].bool()]
        rows.append(F.normalize(values.float(), dim=-1))
    return rows


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def late(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = a @ b.transpose(0, 1)
    return float(0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0))[0])


def build_texts() -> tuple[list[dict[str, Any]], list[str], list[tuple[int, str]], list[str], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    for relation, spec in base.RELATIONS.items():
        for index, (source_template, target_template) in enumerate(spec["templates"]):
            subject = base.SUBJECTS[(index + len(rows)) % len(base.SUBJECTS)]
            attribute = base.ATTRIBUTES[(index + 2 * len(rows)) % len(base.ATTRIBUTES)]
            rows.append({
                "relation": relation,
                "pair_index": index,
                "source_query": source_template.format(subject=subject, attribute=attribute),
                "target_query": target_template.format(subject=subject, attribute=attribute),
            })

    query_texts: list[str] = []
    query_keys: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        for role in ("source", "target"):
            query_texts.append(str(row[f"{role}_query"]))
            query_keys.append((i, role))

    gloss_texts: list[str] = []
    gloss_keys: list[tuple[str, str]] = []
    for relation, spec in base.RELATIONS.items():
        for role in ("source", "target"):
            gloss_texts.append(str(spec[f"{role}_gloss"]))
            gloss_keys.append((relation, role))
    return rows, query_texts, query_keys, gloss_texts, gloss_keys


@torch.inference_mode()
def hidden_views(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, Any]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    outputs = model.backbone(
        input_ids=ids,
        attention_mask=mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("semantic backbone did not expose hidden states")
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "hidden_states": hidden_states,
        "special_ids": specials,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--previous-audit", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=96)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise SystemExit("layerwise query-semantics audit requires CUDA")
    device = torch.device("cuda")

    root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    previous = Path(args.previous_audit).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
    if previous_payload.get("schema") != "alice.eipm.n0.query-semantics-architecture-audit.v0.1":
        raise SystemExit("previous query semantics audit schema drift")
    if previous_payload.get("gradient_performed") is not False:
        raise SystemExit("previous audit gradient contract drift")

    cfg = load_n0_config(config_path)
    model = AliceN0V02Model(cfg)
    missing, unexpected = model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    rows, query_texts, query_keys, gloss_texts, gloss_keys = build_texts()
    q = hidden_views(model, tokenizer, query_texts, device=device, max_length=args.max_length)
    g = hidden_views(model, tokenizer, gloss_texts, device=device, max_length=args.max_length)

    if len(q["hidden_states"]) != len(g["hidden_states"]):
        raise RuntimeError("query/gloss hidden-state depth mismatch")

    query_lookup = {key: i for i, key in enumerate(query_keys)}
    gloss_lookup = {key: i for i, key in enumerate(gloss_keys)}

    layer_results: dict[str, Any] = {}
    best_token = (-1.0, None)
    best_pool = (-1.0, None)

    for layer_index, (qh, gh) in enumerate(zip(q["hidden_states"], g["hidden_states"])):
        q_tokens = masked_layer_tokens(qh, q["input_ids"], q["attention_mask"], q["special_ids"])
        g_tokens = masked_layer_tokens(gh, g["input_ids"], g["attention_mask"], g["special_ids"])
        q_pool = mean_pool(qh, q["attention_mask"]).detach().float().cpu()
        g_pool = mean_pool(gh, g["attention_mask"]).detach().float().cpu()

        total = 0
        token_correct = 0
        pool_correct = 0
        per_relation = {
            relation: {"token_correct": 0, "pool_correct": 0, "total": 0}
            for relation in base.RELATIONS
        }
        pair_token_dist: list[float] = []
        pair_pool_dist: list[float] = []

        for row_index, row in enumerate(rows):
            relation = str(row["relation"])
            source_idx = query_lookup[(row_index, "source")]
            target_idx = query_lookup[(row_index, "target")]
            sg = gloss_lookup[(relation, "source")]
            tg = gloss_lookup[(relation, "target")]

            pair_token_dist.append(
                1.0 - late(q_tokens[source_idx], q_tokens[target_idx])
            )
            pair_pool_dist.append(
                1.0 - cosine(q_pool[source_idx], q_pool[target_idx])
            )

            for role, qi in (("source", source_idx), ("target", target_idx)):
                opposite = "target" if role == "source" else "source"
                token_scores = {
                    "source": late(q_tokens[qi], g_tokens[sg]),
                    "target": late(q_tokens[qi], g_tokens[tg]),
                }
                pool_scores = {
                    "source": cosine(q_pool[qi], g_pool[sg]),
                    "target": cosine(q_pool[qi], g_pool[tg]),
                }
                token_hit = max(token_scores, key=token_scores.get) == role
                pool_hit = max(pool_scores, key=pool_scores.get) == role
                token_correct += int(token_hit)
                pool_correct += int(pool_hit)
                total += 1
                per_relation[relation]["token_correct"] += int(token_hit)
                per_relation[relation]["pool_correct"] += int(pool_hit)
                per_relation[relation]["total"] += 1

        token_accuracy = token_correct / max(total, 1)
        pool_accuracy = pool_correct / max(total, 1)
        if token_accuracy > best_token[0]:
            best_token = (token_accuracy, layer_index)
        if pool_accuracy > best_pool[0]:
            best_pool = (pool_accuracy, layer_index)

        layer_results[str(layer_index)] = {
            "layer_index": layer_index,
            "is_embedding_output": layer_index == 0,
            "is_final_layer": layer_index == len(q["hidden_states"]) - 1,
            "token_late_interaction_role_accuracy": token_accuracy,
            "mean_pool_role_accuracy": pool_accuracy,
            "token_mean_opposite_role_pair_distance": sum(pair_token_dist) / len(pair_token_dist),
            "pool_mean_opposite_role_pair_distance": sum(pair_pool_dist) / len(pair_pool_dist),
            "per_relation": {
                relation: {
                    "token_late_interaction_role_accuracy":
                        values["token_correct"] / max(values["total"], 1),
                    "mean_pool_role_accuracy":
                        values["pool_correct"] / max(values["total"], 1),
                }
                for relation, values in per_relation.items()
            },
        }

    final_index = len(q["hidden_states"]) - 1
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_NO_GRADIENT_LAYERWISE_RELATION_SEMANTICS_AUDIT",
        "git_revision": git_revision(root),
        "num_backbone_hidden_states": len(q["hidden_states"]),
        "transformer_layer_count": len(q["hidden_states"]) - 1,
        "final_layer_index": final_index,
        "gradient_performed": False,
        "optimizer_created": False,
        "model_parameters_mutated": False,
        "heldout_rows_used": False,
        "prior_repair_rows_used": False,
        "qrr_rows_used": False,
        "frozen_challenge_rows_used": False,
        "best_token_late_interaction": {
            "accuracy": best_token[0],
            "layer_index": best_token[1],
        },
        "best_mean_pool": {
            "accuracy": best_pool[0],
            "layer_index": best_pool[1],
        },
        "final_layer_token_late_interaction_accuracy":
            layer_results[str(final_index)]["token_late_interaction_role_accuracy"],
        "final_layer_mean_pool_accuracy":
            layer_results[str(final_index)]["mean_pool_role_accuracy"],
        "layer_results": layer_results,
        "decision_contract": {
            "no_automatic_model_change": True,
            "if_mid_layer_restores_causes_supports": (
                "directional relation semantics exist upstream but are lost or transformed by "
                "later semantic layers; next architecture should expose selected token layers "
                "to a relation-aware language-graph interface rather than retraining graph selection"
            ),
            "if_no_layer_restores_causes_supports": (
                "the current semantic training objective/data does not reliably encode those "
                "directional relation families; reopen semantic curriculum/objectives before graph work"
            ),
        },
        "scale_authorized": False,
        "n0_complete": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "best_token_late_interaction": result["best_token_late_interaction"],
        "best_mean_pool": result["best_mean_pool"],
        "final_layer_token_late_interaction_accuracy": result["final_layer_token_late_interaction_accuracy"],
        "final_layer_mean_pool_accuracy": result["final_layer_mean_pool_accuracy"],
        "layer_results": {
            key: {
                "token_accuracy": value["token_late_interaction_role_accuracy"],
                "pool_accuracy": value["mean_pool_role_accuracy"],
                "token_pair_distance": value["token_mean_opposite_role_pair_distance"],
                "causes_token": value["per_relation"]["causes"]["token_late_interaction_role_accuracy"],
                "supports_token": value["per_relation"]["supports"]["token_late_interaction_role_accuracy"],
            }
            for key, value in layer_results.items()
        },
    }
    print("===== LAYERWISE QUERY SEMANTICS AUDIT =====")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"result={output}")
    print("gradient_performed=false")
    print("model_parameters_mutated=false")
    print("scale_authorized=false")
    print("n0_complete=false")


if __name__ == "__main__":
    main()
