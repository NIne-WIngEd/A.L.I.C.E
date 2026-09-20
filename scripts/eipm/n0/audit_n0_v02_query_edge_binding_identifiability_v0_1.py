#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml


SCHEMA = "alice.eipm.n0.query-edge-binding-identifiability-audit.v0.1"
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_HIDDEN_STATES = 17
SPECIAL_IDS = {0, 1, 2, 3, 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=1e-8)


def edge_scores_from_field_scores(
    field_scores: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    source = edge_index[:, 0].long()
    target = edge_index[:, 1].long()
    return 0.5 * (field_scores[source] + field_scores[target])


def score_result(
    *,
    row: dict[str, Any],
    edge_scores: torch.Tensor,
) -> dict[str, Any]:
    relevant = int(row["relevant_edge_index"])
    all_scores = edge_scores[:4]
    same_scores = edge_scores[:3]
    all_pred = int(all_scores.argmax().item())
    same_pred = int(same_scores.argmax().item())

    all_other = torch.cat(
        [all_scores[:relevant], all_scores[relevant + 1 :]]
    )
    same_other = torch.cat(
        [same_scores[:relevant], same_scores[relevant + 1 :]]
    )
    return {
        "all_correct": int(all_pred == relevant),
        "same_correct": int(same_pred == relevant),
        "all_margin": float(
            (all_scores[relevant] - all_other.max()).item()
        ),
        "same_margin": float(
            (same_scores[relevant] - same_other.max()).item()
        ),
        "target_score": float(all_scores[relevant].item()),
        "different_relation_score": float(all_scores[3].item()),
    }


def aggregate(
    values: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not values:
        raise ValueError("empty identifiability metric")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation, metric in values:
        families[relation].append(metric)

    def mean(key: str, items: list[dict[str, Any]]) -> float:
        return sum(float(item[key]) for item in items) / len(items)

    flat = [metric for _, metric in values]
    return {
        "rows": len(values),
        "all_four_edge_top1_accuracy": mean("all_correct", flat),
        "same_relation_three_edge_top1_accuracy": mean("same_correct", flat),
        "mean_all_edge_margin": mean("all_margin", flat),
        "mean_same_relation_margin": mean("same_margin", flat),
        "mean_target_score": mean("target_score", flat),
        "mean_different_relation_score": mean("different_relation_score", flat),
        "chance_reference_all_four": 0.25,
        "chance_reference_same_relation_three": 1.0 / 3.0,
        "family": {
            relation: {
                "rows": len(items),
                "all_four_edge_top1_accuracy": mean("all_correct", items),
                "same_relation_three_edge_top1_accuracy": mean("same_correct", items),
                "mean_all_edge_margin": mean("all_margin", items),
                "mean_same_relation_margin": mean("same_margin", items),
            }
            for relation, items in sorted(families.items())
        },
    }


def lexical_field_scores(
    *,
    tokenizer: Any,
    query_text: str,
    field_texts: list[str],
) -> torch.Tensor:
    query_ids = [
        int(x)
        for x in tokenizer.encode(
            query_text,
            add_special_tokens=False,
        )
        if int(x) not in SPECIAL_IDS
    ]
    field_sets = [
        {
            int(x)
            for x in tokenizer.encode(
                text,
                add_special_tokens=False,
            )
            if int(x) not in SPECIAL_IDS
        }
        for text in field_texts
    ]
    df: Counter[int] = Counter()
    for tokens in field_sets:
        df.update(tokens)
    query_unique = set(query_ids)
    scores: list[float] = []
    n = len(field_sets)
    for tokens in field_sets:
        score = 0.0
        for token in query_unique & tokens:
            score += math.log((n + 1.0) / (float(df[token]) + 1.0))
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)


def cosine_field_scores(
    query: torch.Tensor,
    fields: torch.Tensor,
) -> torch.Tensor:
    q = normalize(query).unsqueeze(0)
    f = normalize(fields)
    return (f * q).sum(dim=-1)


def evaluate_static_stage(
    *,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    indices: list[int],
    stage: str,
    tokenizer: Any | None = None,
    adapter: EvidenceViewAdapter | None = None,
    graph: DualEndpointEvidenceGraphEncoder | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    metrics: list[tuple[str, dict[str, Any]]] = []

    adapter_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    graph_cache: dict[int, torch.Tensor] = {}

    if stage in {"adapter_field_weight", "parent_graph_field_weight"}:
        if adapter is None:
            raise ValueError("adapter required")
        adapter.eval()
        if graph is not None:
            graph.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                idx = indices[start : start + batch_size]
                batch = {
                    key: payload[key][idx]
                    for key in (
                        "parent_field_states",
                        "parent_field_weights",
                        "valid_mask",
                        "query_semantic",
                        "edge_index",
                        "edge_type_ids",
                        "edge_confidence",
                        "edge_valid_mask",
                    )
                }
                adapted = ml.adapter_forward(adapter, batch)
                for pos, global_index in enumerate(idx):
                    adapter_cache[int(global_index)] = (
                        adapted["field_weights"][pos].detach().cpu(),
                        adapted["field_states"][pos].detach().cpu(),
                    )
                if stage == "parent_graph_field_weight":
                    if graph is None:
                        raise ValueError("graph required")
                    out = graph(
                        field_states=adapted["field_states"],
                        valid_mask=batch["valid_mask"],
                        edge_index=batch["edge_index"],
                        edge_type_ids=batch["edge_type_ids"],
                        edge_confidence=batch["edge_confidence"],
                        edge_valid_mask=batch["edge_valid_mask"],
                        query_semantic=batch["query_semantic"],
                        base_field_weights=adapted["field_weights"],
                    )
                    for pos, global_index in enumerate(idx):
                        graph_cache[int(global_index)] = (
                            out["field_weights"][pos].detach().cpu()
                        )

    for index in indices:
        row = rows[index]
        edge_index = payload["edge_index"][index].cpu()
        if stage == "lexical_token_id_overlap":
            if tokenizer is None:
                raise ValueError("tokenizer required")
            field_scores = lexical_field_scores(
                tokenizer=tokenizer,
                query_text=str(row["query_text"]),
                field_texts=[str(f["text"]) for f in row["fields"]],
            )
        elif stage == "final_mean_pooled_field_semantic":
            field_scores = cosine_field_scores(
                payload["query_semantic"][index],
                payload["field_semantic"][index],
            )
        elif stage == "structured_parent_field_state":
            field_scores = cosine_field_scores(
                payload["query_semantic"][index],
                payload["parent_field_states"][index],
            )
        elif stage == "adapter_field_weight":
            field_scores = adapter_cache[index][0]
        elif stage == "parent_graph_field_weight":
            field_scores = graph_cache[index]
        else:
            raise ValueError(f"unsupported stage: {stage}")

        metrics.append(
            (
                str(row["relation"]),
                score_result(
                    row=row,
                    edge_scores=edge_scores_from_field_scores(
                        field_scores,
                        edge_index,
                    ),
                ),
            )
        )
    return aggregate(metrics)


@torch.inference_mode()
def encode_dev_field_token_states(
    *,
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    unique = list(dict.fromkeys(texts))
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    model.eval().to(device)

    for start in range(0, len(unique), batch_size):
        chunk = unique[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].bool().to(device)
        result = model.backbone(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = result.hidden_states
        if states is None or len(states) != EXPECTED_HIDDEN_STATES:
            raise SystemExit("field hidden-state depth drift")
        stack = torch.stack(states, dim=1).detach().to(torch.float16).cpu()
        ids_cpu = ids.cpu()
        mask_cpu = mask.cpu()
        for pos, text in enumerate(chunk):
            semantic_mask = mask_cpu[pos].clone()
            for token_id in SPECIAL_IDS:
                semantic_mask &= ids_cpu[pos].ne(token_id)
            output[text] = (
                stack[pos],
                semantic_mask,
            )
    return output


def reencode_query_masks(
    *,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    indices: list[int],
    payload: dict[str, Any],
    max_length: int,
) -> dict[int, torch.Tensor]:
    output: dict[int, torch.Tensor] = {}
    for index in indices:
        encoded = tokenizer(
            str(rows[index]["query_text"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention = encoded["attention_mask"][0].bool()
        cached = payload["query_attention_mask"][index].bool()
        if not torch.equal(attention, cached):
            raise SystemExit(f"query attention-mask/cache drift: {rows[index]['id']}")
        ids = encoded["input_ids"][0]
        semantic = attention.clone()
        for token_id in SPECIAL_IDS:
            semantic &= ids.ne(token_id)
        output[index] = semantic
    return output


def maxsim(
    query_tokens: torch.Tensor,
    query_mask: torch.Tensor,
    field_tokens: torch.Tensor,
    field_mask: torch.Tensor,
) -> float:
    q = normalize(query_tokens[query_mask])
    f = normalize(field_tokens[field_mask])
    if q.numel() == 0 or f.numel() == 0:
        return -1.0
    similarity = q @ f.transpose(0, 1)
    return float(similarity.max(dim=1).values.mean().item())


def evaluate_token_late_interaction(
    *,
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    indices: list[int],
    tokenizer: Any,
    field_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    max_length: int,
) -> dict[str, Any]:
    query_masks = reencode_query_masks(
        tokenizer=tokenizer,
        rows=rows,
        indices=indices,
        payload=payload,
        max_length=max_length,
    )

    layers: dict[str, Any] = {}
    for layer in range(EXPECTED_HIDDEN_STATES):
        metrics: list[tuple[str, dict[str, Any]]] = []
        for index in indices:
            row = rows[index]
            q = payload["query_hidden_states"][index, layer].float()
            qmask = query_masks[index]
            scores: list[float] = []
            for field in row["fields"]:
                hidden, fmask = field_states[str(field["text"])]
                scores.append(
                    maxsim(
                        q,
                        qmask,
                        hidden[layer].float(),
                        fmask,
                    )
                )
            field_scores = torch.tensor(scores, dtype=torch.float32)
            metrics.append(
                (
                    str(row["relation"]),
                    score_result(
                        row=row,
                        edge_scores=edge_scores_from_field_scores(
                            field_scores,
                            payload["edge_index"][index].cpu(),
                        ),
                    ),
                )
            )
        layers[str(layer)] = aggregate(metrics)

    best_all = max(
        layers,
        key=lambda key: (
            layers[key]["all_four_edge_top1_accuracy"],
            layers[key]["mean_all_edge_margin"],
        ),
    )
    best_same = max(
        layers,
        key=lambda key: (
            layers[key]["same_relation_three_edge_top1_accuracy"],
            layers[key]["mean_same_relation_margin"],
        ),
    )
    return {
        "layers": layers,
        "best_all_edge_layer": int(best_all),
        "best_all_edge_metrics": layers[best_all],
        "best_same_relation_layer": int(best_same),
        "best_same_relation_metrics": layers[best_same],
        "selection_note": (
            "layer scan is diagnostic on DEV architecture-development data only; "
            "it does not authorize hard-coding or training"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=32)
    args = p.parse_args()

    cache_path = Path(args.cache).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    semantic_config_path = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    output = Path(args.output).resolve()

    for path in (
        cache_path,
        curriculum_path,
        semantic_config_path,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json",
        adapter_path,
        graph_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing identifiability-audit input: {path}")
    if output.exists():
        raise SystemExit("refusing to overwrite identifiability audit")
    output.parent.mkdir(parents=True, exist_ok=True)

    if sha256(curriculum_path) != EXPECTED_CURRICULUM_SHA256:
        raise SystemExit("curriculum lineage drift")
    if sha256(adapter_path) != EXPECTED_PARENT_ADAPTER_SHA256:
        raise SystemExit("parent adapter lineage drift")
    if sha256(graph_path) != EXPECTED_PARENT_GRAPH_SHA256:
        raise SystemExit("parent graph lineage drift")

    rows_all = read_jsonl(curriculum_path)
    selected_rows = [
        row for row in rows_all
        if str(row["split"]) in {"train", "dev"}
    ]
    payload = torch.load(cache_path, map_location="cpu")

    required = (
        "ids",
        "splits",
        "relations",
        "field_semantic",
        "query_semantic",
        "query_hidden_states",
        "query_attention_mask",
        "parent_field_states",
        "parent_field_weights",
        "edge_index",
        "edge_type_ids",
        "edge_confidence",
        "edge_valid_mask",
        "valid_mask",
        "relevant_edge_index",
        "same_relation_edge_mask",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"cache missing identifiability keys: {missing}")
    if len(selected_rows) != len(payload["ids"]):
        raise SystemExit("cache/curriculum row-count drift")
    for index, row in enumerate(selected_rows):
        if str(row["id"]) != str(payload["ids"][index]):
            raise SystemExit(f"cache/curriculum row order drift at {index}")
        if int(row["relevant_edge_index"]) != int(
            payload["relevant_edge_index"][index]
        ):
            raise SystemExit(f"relevant-edge cache drift: {row['id']}")

    train_indices = [
        i for i, split in enumerate(payload["splits"])
        if str(split) == "train"
    ]
    dev_indices = [
        i for i, split in enumerate(payload["splits"])
        if str(split) == "dev"
    ]
    if len(train_indices) != 432 or len(dev_indices) != 144:
        raise SystemExit(
            f"train/dev row-count drift: {len(train_indices)}/{len(dev_indices)}"
        )

    tokenizer = load_tokenizer(tokenizer_dir)

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config())
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False

    graph = DualEndpointEvidenceGraphEncoder(rr.expanded_graph_config())
    graph.load_state_dict(load_file(str(graph_path), device="cpu"), strict=True)
    for parameter in graph.parameters():
        parameter.requires_grad = False

    stages: dict[str, Any] = {}
    for split, indices in (("train", train_indices), ("dev", dev_indices)):
        stages[split] = {}
        for stage in (
            "lexical_token_id_overlap",
            "final_mean_pooled_field_semantic",
            "structured_parent_field_state",
            "adapter_field_weight",
            "parent_graph_field_weight",
        ):
            stages[split][stage] = evaluate_static_stage(
                rows=selected_rows,
                payload=payload,
                indices=indices,
                stage=stage,
                tokenizer=tokenizer,
                adapter=adapter,
                graph=graph,
            )

    config = load_n0_config(semantic_config_path)
    semantic = AliceN0V02Model(config)
    state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing_state, unexpected_state = semantic.load_state_dict(state, strict=False)
    if missing_state or unexpected_state:
        raise SystemExit(
            "semantic checkpoint mismatch "
            f"missing={list(missing_state)} unexpected={list(unexpected_state)}"
        )
    for parameter in semantic.parameters():
        parameter.requires_grad = False

    dev_field_texts = [
        str(field["text"])
        for index in dev_indices
        for field in selected_rows[index]["fields"]
    ]
    field_token_states = encode_dev_field_token_states(
        model=semantic,
        tokenizer=tokenizer,
        texts=dev_field_texts,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        device=torch.device("cpu"),
    )
    token_late = evaluate_token_late_interaction(
        rows=selected_rows,
        payload=payload,
        indices=dev_indices,
        tokenizer=tokenizer,
        field_states=field_token_states,
        max_length=args.max_length,
    )

    lexical_dev = stages["dev"]["lexical_token_id_overlap"]
    pooled_dev = stages["dev"]["final_mean_pooled_field_semantic"]
    token_best = token_late["best_same_relation_metrics"]

    if lexical_dev["same_relation_three_edge_top1_accuracy"] < 0.90:
        localization = (
            "RAW_TEXT_BINDING_NOT_CLEANLY_IDENTIFIABLE_REOPEN_CURRICULUM_SEMANTICS"
        )
    elif (
        token_best["same_relation_three_edge_top1_accuracy"]
        >= pooled_dev["same_relation_three_edge_top1_accuracy"] + 0.10
        and token_best["same_relation_three_edge_top1_accuracy"] > 0.50
    ):
        localization = (
            "FIELD_POOLING_BOTTLENECK_SUPPORTED_TOKEN_LATE_INTERACTION_RECOVERS_BINDING"
        )
    elif token_best["same_relation_three_edge_top1_accuracy"] <= 0.40:
        localization = (
            "RAW_TEXT_IDENTIFIABLE_BUT_FROZEN_SEMANTIC_TOKEN_BINDING_WEAK"
        )
    elif pooled_dev["same_relation_three_edge_top1_accuracy"] > 0.50:
        localization = (
            "POOLED_FIELD_SIGNAL_ALREADY_IDENTIFIABLE_REOPEN_ROUTER_OPTIMIZATION"
        )
    else:
        localization = (
            "MIXED_REPRESENTATION_SIGNAL_REQUIRES_REVIEW_NO_AUTOMATIC_ARCHITECTURE_CHANGE"
        )

    result = {
        "schema": SCHEMA,
        "status": "PASS_BINDING_IDENTIFIABILITY_AUDIT_NO_GRADIENT",
        "cache_sha256": sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "curriculum_sha256": sha256(curriculum_path),
        "parent_adapter_sha256": sha256(adapter_path),
        "parent_graph_sha256": sha256(graph_path),
        "train_rows": len(train_indices),
        "dev_rows": len(dev_indices),
        "causal_test_evaluated": False,
        "frozen_challenge_evaluated": False,
        "optimizer_created": False,
        "gradient_performed": False,
        "model_parameter_update_performed": False,
        "gpu_required": False,
        "stages": stages,
        "dev_token_level_late_interaction": token_late,
        "localization": localization,
        "diagnostic_comparisons": {
            "dev_lexical_same_relation_accuracy": lexical_dev[
                "same_relation_three_edge_top1_accuracy"
            ],
            "dev_pooled_same_relation_accuracy": pooled_dev[
                "same_relation_three_edge_top1_accuracy"
            ],
            "dev_best_token_same_relation_accuracy": token_best[
                "same_relation_three_edge_top1_accuracy"
            ],
            "dev_pooled_to_best_token_accuracy_delta": (
                token_best["same_relation_three_edge_top1_accuracy"]
                - pooled_dev["same_relation_three_edge_top1_accuracy"]
            ),
            "dev_lexical_all_edge_accuracy": lexical_dev[
                "all_four_edge_top1_accuracy"
            ],
            "dev_pooled_all_edge_accuracy": pooled_dev[
                "all_four_edge_top1_accuracy"
            ],
            "dev_best_token_all_edge_accuracy": token_late[
                "best_all_edge_metrics"
            ]["all_four_edge_top1_accuracy"],
        },
        "setwise_training_authorized": False,
        "replacement_gpu_run_authorized": False,
        "heldout_opening_authorized": False,
        "scale_authorized": False,
        "semantic_retraining_authorized": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "review information-localization result before choosing one model/data change"
        ),
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
