#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.v02_model import AliceN0V02Model

SCHEMA = "alice.eipm.n0.query-semantics-architecture-audit.v0.1"

RELATIONS: dict[str, dict[str, Any]] = {
    "corrects": {
        "source_gloss": "the record that provides the corrected current value after an error is fixed",
        "target_gloss": "the earlier record whose incorrect value is being corrected and replaced",
        "templates": [
            ("What {attribute} should be used after the correction for {subject}?",
             "What {attribute} was replaced by the correction for {subject}?"),
            ("Which {attribute} remains valid once the mistake is fixed for {subject}?",
             "Which {attribute} belonged to the mistaken record for {subject}?"),
            ("What is the corrected current {attribute} for {subject}?",
             "What was the pre-correction {attribute} for {subject}?"),
            ("Which {attribute} survives the corrective update for {subject}?",
             "Which {attribute} is displaced by the corrective update for {subject}?"),
        ],
    },
    "supersedes": {
        "source_gloss": "the newer replacement record that takes precedence after supersession",
        "target_gloss": "the older record that loses precedence and becomes superseded",
        "templates": [
            ("Which {attribute} takes precedence after supersession for {subject}?",
             "Which {attribute} belonged to the superseded state for {subject}?"),
            ("What {attribute} is current after the replacement state takes over for {subject}?",
             "What {attribute} was replaced by the newer state for {subject}?"),
            ("Which {attribute} belongs to the state that prevails for {subject}?",
             "Which {attribute} belongs to the state that no longer prevails for {subject}?"),
            ("What {attribute} should be retained after supersession for {subject}?",
             "What {attribute} should be treated as historical after supersession for {subject}?"),
        ],
    },
    "temporal_successor": {
        "source_gloss": "the later successor state that comes next in the temporal sequence",
        "target_gloss": "the earlier predecessor state that comes before the successor",
        "templates": [
            ("What {attribute} belongs to the later state for {subject}?",
             "What {attribute} belongs to the earlier state for {subject}?"),
            ("Which {attribute} comes next in time for {subject}?",
             "Which {attribute} comes before the next state for {subject}?"),
            ("After the temporal transition, what {attribute} applies to {subject}?",
             "Before the temporal transition, what {attribute} applied to {subject}?"),
            ("Which {attribute} is associated with the successor for {subject}?",
             "Which {attribute} is associated with the predecessor for {subject}?"),
        ],
    },
    "causes": {
        "source_gloss": "the causal source condition that produces or initiates the effect",
        "target_gloss": "the downstream effect or consequence produced by the cause",
        "templates": [
            ("Which {attribute} caused the linked outcome for {subject}?",
             "Which {attribute} resulted from the linked cause for {subject}?"),
            ("What {attribute} initiated the effect for {subject}?",
             "What {attribute} is the effect for {subject}?"),
            ("Which {attribute} is upstream in the causal relation for {subject}?",
             "Which {attribute} is downstream in the causal relation for {subject}?"),
            ("What {attribute} produced the other linked state for {subject}?",
             "What {attribute} was produced by the other linked state for {subject}?"),
        ],
    },
    "supports": {
        "source_gloss": "the supporting evidence record that provides corroboration for another claim",
        "target_gloss": "the claim or value that receives support from the evidence record",
        "templates": [
            ("Which {attribute} comes from the record providing support for {subject}?",
             "Which {attribute} is supported by the linked evidence for {subject}?"),
            ("What {attribute} is on the evidence-provider side for {subject}?",
             "What {attribute} is the claim receiving support for {subject}?"),
            ("Which {attribute} belongs to the corroborating record for {subject}?",
             "Which {attribute} is corroborated by the other record for {subject}?"),
            ("What {attribute} comes from the supporting record for {subject}?",
             "What {attribute} is the supported conclusion for {subject}?"),
        ],
    },
    "derived_from": {
        "source_gloss": "the derived result or dependent record produced from an underlying basis",
        "target_gloss": "the underlying basis or source material from which the result is derived",
        "templates": [
            ("Which {attribute} belongs to the derived record for {subject}?",
             "Which {attribute} belongs to the basis record for {subject}?"),
            ("What {attribute} was produced from the linked basis for {subject}?",
             "What {attribute} underlies the derived result for {subject}?"),
            ("Which {attribute} is the derivation result for {subject}?",
             "Which {attribute} is the derivation basis for {subject}?"),
            ("What {attribute} belongs to the dependent derived item for {subject}?",
             "What {attribute} belongs to the source material for {subject}?"),
        ],
    },
}

SUBJECTS = [
    "Audit Alder", "Audit Bramble", "Audit Cirrus", "Audit Drift",
    "Audit Estuary", "Audit Fjord", "Audit Grove", "Audit Harbor",
]
ATTRIBUTES = [
    "handoff mode", "control epoch", "service posture", "refresh window",
    "routing lane", "snapshot period", "sampling regime", "archive tier",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def masked_tokens(
    token_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for i in range(token_states.size(0)):
        mask = attention_mask[i].bool().clone()
        for special_id in special_ids:
            mask &= input_ids[i].ne(int(special_id))
        values = token_states[i][mask]
        if values.numel() == 0:
            values = token_states[i][attention_mask[i].bool()]
        output.append(F.normalize(values.float(), dim=-1))
    return output


def symmetric_late_interaction(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = a @ b.transpose(0, 1)
    forward = sim.max(dim=1).values.mean()
    backward = sim.max(dim=0).values.mean()
    return float(0.5 * (forward + backward))


def mean_pairwise_cosine(vectors: list[torch.Tensor]) -> float | None:
    if len(vectors) < 2:
        return None
    sims: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(float(cosine(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))[0]))
    return sum(sims) / len(sims)


def encode(
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
    with torch.inference_mode():
        views = model.encode_views(ids, mask)
        pooled = views["pooled_state"]
        projected = model.project_semantic_pooled(pooled)
    specials = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) or [])
    token_lists = masked_tokens(views["token_states"], ids, mask, specials)
    return {
        "input_ids": ids.detach().cpu(),
        "attention_mask": mask.detach().cpu(),
        "pooled": pooled.detach().float().cpu(),
        "projected": projected.detach().float().cpu(),
        "tokens": [x.detach().float().cpu() for x in token_lists],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=96)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise SystemExit("query semantics audit requires one CUDA device")
    device = torch.device("cuda")

    root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite audit result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_n0_config(config_path)
    model = AliceN0V02Model(cfg)
    state = load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    rows: list[dict[str, Any]] = []
    gloss_texts: list[str] = []
    gloss_keys: list[tuple[str, str]] = []
    for relation, spec in RELATIONS.items():
        for role in ("source", "target"):
            gloss_texts.append(str(spec[f"{role}_gloss"]))
            gloss_keys.append((relation, role))
        for index, (source_template, target_template) in enumerate(spec["templates"]):
            subject = SUBJECTS[(index + len(rows)) % len(SUBJECTS)]
            attribute = ATTRIBUTES[(index + 2 * len(rows)) % len(ATTRIBUTES)]
            rows.append({
                "relation": relation,
                "pair_index": index,
                "source_query": source_template.format(subject=subject, attribute=attribute),
                "target_query": target_template.format(subject=subject, attribute=attribute),
            })

    query_texts: list[str] = []
    query_keys: list[tuple[int, str]] = []
    for row_index, row in enumerate(rows):
        for role in ("source", "target"):
            query_texts.append(str(row[f"{role}_query"]))
            query_keys.append((row_index, role))

    query_rep = encode(model, tokenizer, query_texts, device=device, max_length=args.max_length)
    gloss_rep = encode(model, tokenizer, gloss_texts, device=device, max_length=args.max_length)
    gloss_index = {key: i for i, key in enumerate(gloss_keys)}

    representation_names = ("raw_mean_pool", "trained_semantic_projection", "token_late_interaction")
    correct = {name: 0 for name in representation_names}
    total = 0
    family_correct: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in representation_names})
    family_total: dict[str, int] = defaultdict(int)
    row_results: list[dict[str, Any]] = []
    pair_metrics: list[dict[str, Any]] = []
    raw_directions: dict[str, list[torch.Tensor]] = defaultdict(list)
    projected_directions: dict[str, list[torch.Tensor]] = defaultdict(list)

    query_lookup = {key: i for i, key in enumerate(query_keys)}
    for row_index, row in enumerate(rows):
        relation = str(row["relation"])
        source_idx = query_lookup[(row_index, "source")]
        target_idx = query_lookup[(row_index, "target")]
        sg = gloss_index[(relation, "source")]
        tg = gloss_index[(relation, "target")]

        raw_source = query_rep["pooled"][source_idx]
        raw_target = query_rep["pooled"][target_idx]
        proj_source = query_rep["projected"][source_idx]
        proj_target = query_rep["projected"][target_idx]
        raw_delta = raw_source - raw_target
        proj_delta = proj_source - proj_target
        raw_directions[relation].append(F.normalize(raw_delta.float(), dim=0))
        projected_directions[relation].append(F.normalize(proj_delta.float(), dim=0))

        pair_metrics.append({
            "relation": relation,
            "pair_index": int(row["pair_index"]),
            "raw_mean_pool_cosine_distance": float(1.0 - cosine(raw_source.unsqueeze(0), raw_target.unsqueeze(0))[0]),
            "trained_semantic_projection_cosine_distance": float(1.0 - cosine(proj_source.unsqueeze(0), proj_target.unsqueeze(0))[0]),
            "token_set_cosine_distance": float(
                1.0 - symmetric_late_interaction(
                    query_rep["tokens"][source_idx],
                    query_rep["tokens"][target_idx],
                )
            ),
        })

        for role, q_idx in (("source", source_idx), ("target", target_idx)):
            raw_q = query_rep["pooled"][q_idx]
            proj_q = query_rep["projected"][q_idx]
            tok_q = query_rep["tokens"][q_idx]

            raw_scores = {
                "source": float(cosine(raw_q.unsqueeze(0), gloss_rep["pooled"][sg].unsqueeze(0))[0]),
                "target": float(cosine(raw_q.unsqueeze(0), gloss_rep["pooled"][tg].unsqueeze(0))[0]),
            }
            proj_scores = {
                "source": float(cosine(proj_q.unsqueeze(0), gloss_rep["projected"][sg].unsqueeze(0))[0]),
                "target": float(cosine(proj_q.unsqueeze(0), gloss_rep["projected"][tg].unsqueeze(0))[0]),
            }
            token_scores = {
                "source": symmetric_late_interaction(tok_q, gloss_rep["tokens"][sg]),
                "target": symmetric_late_interaction(tok_q, gloss_rep["tokens"][tg]),
            }

            predictions = {
                "raw_mean_pool": max(raw_scores, key=raw_scores.get),
                "trained_semantic_projection": max(proj_scores, key=proj_scores.get),
                "token_late_interaction": max(token_scores, key=token_scores.get),
            }
            margins = {
                "raw_mean_pool": raw_scores[role] - raw_scores["target" if role == "source" else "source"],
                "trained_semantic_projection": proj_scores[role] - proj_scores["target" if role == "source" else "source"],
                "token_late_interaction": token_scores[role] - token_scores["target" if role == "source" else "source"],
            }

            for name in representation_names:
                hit = predictions[name] == role
                correct[name] += int(hit)
                family_correct[relation][name] += int(hit)
            family_total[relation] += 1
            total += 1

            row_results.append({
                "relation": relation,
                "pair_index": int(row["pair_index"]),
                "role": role,
                "query": row[f"{role}_query"],
                "predictions": predictions,
                "correct_role_margins": margins,
                "raw_scores": raw_scores,
                "projected_scores": proj_scores,
                "token_late_interaction_scores": token_scores,
            })

    accuracies = {name: correct[name] / max(total, 1) for name in representation_names}
    per_relation = {
        relation: {
            name: family_correct[relation][name] / max(family_total[relation], 1)
            for name in representation_names
        }
        for relation in sorted(family_total)
    }
    mean_pair_distance = {
        "raw_mean_pool": sum(x["raw_mean_pool_cosine_distance"] for x in pair_metrics) / len(pair_metrics),
        "trained_semantic_projection": sum(x["trained_semantic_projection_cosine_distance"] for x in pair_metrics) / len(pair_metrics),
        "token_late_interaction": sum(x["token_set_cosine_distance"] for x in pair_metrics) / len(pair_metrics),
    }
    direction_consistency = {
        relation: {
            "raw_mean_pool_source_minus_target_mean_pairwise_cosine": mean_pairwise_cosine(raw_directions[relation]),
            "trained_semantic_projection_source_minus_target_mean_pairwise_cosine": mean_pairwise_cosine(projected_directions[relation]),
        }
        for relation in sorted(raw_directions)
    }

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_NO_GRADIENT_ARCHITECTURE_AUDIT",
        "git_revision": git_revision(root),
        "semantic_checkpoint_sha256": sha(checkpoint_path),
        "semantic_config_sha256": sha(config_path),
        "tokenizer_sha256": sha(tokenizer_dir / "tokenizer.json"),
        "rows": len(row_results),
        "query_pairs": len(pair_metrics),
        "relations": sorted(RELATIONS),
        "gradient_performed": False,
        "optimizer_created": False,
        "model_parameters_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "frozen_challenge_rows_used": False,
        "prior_semantic_repair_rows_used": False,
        "qrr_v0_3_rows_used": False,
        "heldout_rows_used": False,
        "representation_role_accuracy": accuracies,
        "per_relation_role_accuracy": per_relation,
        "mean_opposite_role_pair_distance": mean_pair_distance,
        "role_direction_consistency": direction_consistency,
        "row_results": row_results,
        "pair_metrics": pair_metrics,
        "interpretation_contract": {
            "no_automatic_model_change": True,
            "no_threshold_gate": True,
            "purpose": (
                "determine whether directional relation semantics are present in raw mean pooled "
                "state, the trained semantic projection, or token-level contextual states before "
                "choosing any new graph/language architecture"
            ),
            "if_token_or_projection_substantially_outperforms_raw_pool": (
                "the graph has been consuming an information-poor compatibility readout; "
                "redesign the query-to-graph interface rather than training another router"
            ),
            "if_all_views_fail_to_separate_roles": (
                "the semantic backbone/objective lacks the needed relation-role distinction; "
                "audit semantic representation learning before any graph repair"
            ),
        },
        "n0_complete": False,
        "scale_authorized": False,
        "production_promotion_authorized": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("===== QUERY SEMANTICS ARCHITECTURE AUDIT =====")
    print(json.dumps({
        "representation_role_accuracy": accuracies,
        "mean_opposite_role_pair_distance": mean_pair_distance,
        "per_relation_role_accuracy": per_relation,
        "role_direction_consistency": direction_consistency,
    }, indent=2, sort_keys=True))
    print(f"result={output}")
    print("gradient_performed=false")
    print("model_parameters_mutated=false")
    print("scale_authorized=false")
    print("n0_complete=false")


if __name__ == "__main__":
    main()
