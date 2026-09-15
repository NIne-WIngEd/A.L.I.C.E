#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import EvidenceGraphConfig, EvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter, EvidenceViewAdapterConfig
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model


EXPECTED_SEMANTIC_PARAMETERS = 136_594_435


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_structured_config(path: Path) -> StructuredStateConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arch = raw["architecture"]
    return StructuredStateConfig(
        semantic_size=int(arch["semantic_size"]),
        state_size=int(arch["state_size"]),
        num_layers=int(arch["num_layers"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        max_fields=int(arch["max_fields"]),
    )


def encode_texts(
    model: AliceN0V02Model,
    tokenizer: Any,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            pooled = model.encode(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
        chunks.append(pooled.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def build_cache(
    rows: list[dict[str, Any]],
    *,
    semantic_model: AliceN0V02Model,
    tokenizer: Any,
    structured: StructuredStateEncoder,
    device: torch.device,
) -> dict[str, Any]:
    max_fields = max(len(row["fields"]) for row in rows)
    max_edges = max(len(row["relations"]) for row in rows)
    field_texts = [str(field["text"]) for row in rows for field in row["fields"]]
    flat_field = encode_texts(semantic_model, tokenizer, field_texts, device=device)
    query = encode_texts(
        semantic_model, tokenizer, [str(row["query_text"]) for row in rows], device=device
    )
    summary = encode_texts(
        semantic_model,
        tokenizer,
        [str(row["target_summary_text"]) for row in rows],
        device=device,
    )

    n = len(rows)
    semantic_size = int(query.shape[-1])
    field_semantic = torch.zeros(n, max_fields, semantic_size)
    field_type_ids = torch.zeros(n, max_fields, dtype=torch.long)
    provenance_ids = torch.zeros(n, max_fields, dtype=torch.long)
    relation_role_ids = torch.zeros(n, max_fields, dtype=torch.long)
    temporal_scope_ids = torch.zeros(n, max_fields, dtype=torch.long)
    confidence = torch.zeros(n, max_fields, 1)
    missing_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    valid_mask = torch.zeros(n, max_fields, dtype=torch.bool)
    target_distribution = torch.zeros(n, max_fields)
    edge_index = torch.zeros(n, max_edges, 2, dtype=torch.long)
    edge_type_ids = torch.zeros(n, max_edges, dtype=torch.long)
    edge_confidence = torch.zeros(n, max_edges, 1)
    edge_valid_mask = torch.zeros(n, max_edges, dtype=torch.bool)

    cursor = 0
    for i, row in enumerate(rows):
        fields = row["fields"]
        f = len(fields)
        field_semantic[i, :f] = flat_field[cursor : cursor + f]
        cursor += f
        valid_mask[i, :f] = True
        target_distribution[i, :f] = torch.tensor(row["target_evidence_distribution"])
        for j, item in enumerate(fields):
            field_type_ids[i, j] = int(item["field_type_id"])
            provenance_ids[i, j] = int(item["provenance_id"])
            relation_role_ids[i, j] = int(item["relation_role_id"])
            temporal_scope_ids[i, j] = int(item["temporal_scope_id"])
            confidence[i, j, 0] = float(item["confidence"])
            missing_mask[i, j] = bool(item["missing"])
        for e, rel in enumerate(row["relations"]):
            edge_index[i, e, 0] = int(rel["source"])
            edge_index[i, e, 1] = int(rel["target"])
            edge_type_ids[i, e] = int(rel["relation_type_id"])
            edge_confidence[i, e, 0] = float(rel["confidence"])
            edge_valid_mask[i, e] = True

    structured = structured.to(device).eval()
    with torch.inference_mode():
        parent = structured(
            semantic_values=field_semantic.to(device),
            field_type_ids=field_type_ids.to(device),
            provenance_ids=provenance_ids.to(device),
            relation_role_ids=relation_role_ids.to(device),
            temporal_scope_ids=temporal_scope_ids.to(device),
            confidence=confidence.to(device),
            missing_mask=missing_mask.to(device),
            valid_mask=valid_mask.to(device),
        )
    return {
        "field_semantic": field_semantic,
        "valid_mask": valid_mask,
        "target_distribution": target_distribution,
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_confidence": edge_confidence,
        "edge_valid_mask": edge_valid_mask,
        "query_semantic": query,
        "summary_semantic": summary,
        "parent_field_states": parent["field_states"].detach().float().cpu(),
        "parent_field_weights": parent["field_weights"].detach().float().cpu(),
        "ids": [str(row["id"]) for row in rows],
        "pair_ids": [str(row["pair_id"]) for row in rows],
        "families": [str(row["family"]) for row in rows],
        "variants": [str(row["variant"]) for row in rows],
    }


def candidate_config(variant: str) -> tuple[EvidenceViewAdapterConfig, EvidenceGraphConfig]:
    if variant == "specialized_compact_384x2_graph256x1":
        return (
            EvidenceViewAdapterConfig(
                semantic_size=640, adapter_size=384, num_layers=2, num_heads=6,
                feedforward_size=1536, dropout=0.0, max_fields=64, base_prior_scale=0.5,
            ),
            EvidenceGraphConfig(
                semantic_size=640, graph_size=256, graph_layers=1, max_fields=64, max_edges=256,
            ),
        )
    if variant == "specialized_expanded_640x3_graph512x2":
        return (
            EvidenceViewAdapterConfig(
                semantic_size=640, adapter_size=640, num_layers=3, num_heads=10,
                feedforward_size=2560, dropout=0.0, max_fields=64, base_prior_scale=0.5,
            ),
            EvidenceGraphConfig(
                semantic_size=640, graph_size=512, graph_layers=2, max_fields=64, max_edges=256,
            ),
        )
    raise ValueError(f"unknown specialist variant: {variant}")


def permute_cache(cache: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    fields = int(cache["valid_mask"].shape[1])
    order = torch.arange(fields - 1, -1, -1)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(fields)
    out: dict[str, torch.Tensor] = {}
    for key in ("field_semantic", "valid_mask", "target_distribution", "parent_field_states", "parent_field_weights"):
        out[key] = cache[key].index_select(1, order).to(device)
    out["edge_index"] = inverse[cache["edge_index"]].to(device)
    for key in ("edge_type_ids", "edge_confidence", "edge_valid_mask", "query_semantic", "summary_semantic"):
        out[key] = cache[key].to(device)
    return out


def forward_candidate(
    adapter: EvidenceViewAdapter,
    graph: EvidenceGraphEncoder,
    batch: dict[str, torch.Tensor],
    *,
    ablate_edges: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    adapted = adapter(
        parent_field_states=batch["parent_field_states"],
        valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"],
        parent_field_weights=batch["parent_field_weights"],
    )
    edge_valid = torch.zeros_like(batch["edge_valid_mask"]) if ablate_edges else batch["edge_valid_mask"]
    output = graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=edge_valid,
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
    )
    return adapted, output


def evaluate_candidate(
    *,
    variant: str,
    checkpoint: Path,
    cache: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    adapter_config, graph_config = candidate_config(variant)
    adapter = EvidenceViewAdapter(adapter_config).to(device)
    graph = EvidenceGraphEncoder(graph_config).to(device)
    adapter.load_state_dict(load_file(str(checkpoint / "evidence_view_adapter.safetensors"), device="cpu"), strict=True)
    graph.load_state_dict(load_file(str(checkpoint / "evidence_graph.safetensors"), device="cpu"), strict=True)
    adapter.eval()
    graph.eval()

    batch = {key: value.to(device) for key, value in cache.items() if torch.is_tensor(value)}
    with torch.inference_mode():
        _adapted, output = forward_candidate(adapter, graph, batch)
        _ab_adapted, ablated = forward_candidate(adapter, graph, batch, ablate_edges=True)
        pbatch = permute_cache(cache, device)
        _padapted, permuted = forward_candidate(adapter, graph, pbatch)

    weights = output["field_weights"].detach().cpu()
    ablated_weights = ablated["field_weights"].detach().cpu()
    target = cache["target_distribution"]
    predictions = weights.argmax(dim=-1)
    target_index = target.argmax(dim=-1)
    correct = predictions.eq(target_index)
    target_mass = weights.gather(1, target_index.unsqueeze(1)).squeeze(1)
    summary_cos = F.cosine_similarity(
        output["pooled_state"], batch["summary_semantic"], dim=-1
    ).detach().cpu()
    permutation_cos = F.cosine_similarity(
        output["pooled_state"], permuted["pooled_state"], dim=-1
    ).detach().cpu()

    by_pair: dict[str, list[int]] = defaultdict(list)
    by_family_pairs: dict[str, list[str]] = defaultdict(list)
    for i, pair_id in enumerate(cache["pair_ids"]):
        by_pair[pair_id].append(i)
        family = cache["families"][i]
        if pair_id not in by_family_pairs[family]:
            by_family_pairs[family].append(pair_id)

    pair_pass: dict[str, bool] = {}
    pair_margin: dict[str, float] = {}
    ablated_pair_l1: list[float] = []
    for pair_id, indices in by_pair.items():
        if len(indices) != 2:
            raise RuntimeError(f"pair {pair_id} is incomplete")
        a, b = sorted(indices, key=lambda i: cache["variants"][i])
        pair_pass[pair_id] = bool(correct[a] and correct[b] and predictions[a] != predictions[b])
        ta = int(target_index[a])
        tb = int(target_index[b])
        margin_a = float(weights[a, ta] - weights[a, tb])
        margin_b = float(weights[b, tb] - weights[b, ta])
        pair_margin[pair_id] = 0.5 * (margin_a + margin_b)
        ablated_pair_l1.append(float((ablated_weights[a] - ablated_weights[b]).abs().sum()))

    family_pair_accuracy = {
        family: sum(pair_pass[pair_id] for pair_id in pairs) / len(pairs)
        for family, pairs in sorted(by_family_pairs.items())
    }
    family_margin = {
        family: sum(pair_margin[pair_id] for pair_id in pairs) / len(pairs)
        for family, pairs in sorted(by_family_pairs.items())
    }
    return {
        "variant": variant,
        "checkpoint": checkpoint.name,
        "adapter_parameter_count": adapter.parameter_report()["total_parameters"],
        "graph_parameter_count": graph.parameter_report()["total_parameters"],
        "specialist_parameter_count": adapter.parameter_report()["total_parameters"] + graph.parameter_report()["total_parameters"],
        "example_top1_accuracy": float(correct.float().mean()),
        "pair_flip_accuracy": sum(pair_pass.values()) / len(pair_pass),
        "family_pair_flip_accuracy": family_pair_accuracy,
        "family_min_pair_flip_accuracy": min(family_pair_accuracy.values()),
        "mean_pair_relation_flip_margin": sum(pair_margin.values()) / len(pair_margin),
        "family_pair_relation_flip_margin": family_margin,
        "mean_target_mass": float(target_mass.mean()),
        "min_target_mass": float(target_mass.min()),
        "mean_summary_cosine": float(summary_cos.mean()),
        "mean_permutation_cosine": float(permutation_cos.mean()),
        "mean_ablated_pair_distribution_l1": sum(ablated_pair_l1) / len(ablated_pair_l1),
        "shared_structured_parent_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }


def score(result: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(result["family_min_pair_flip_accuracy"]),
        float(result["pair_flip_accuracy"]),
        float(result["mean_pair_relation_flip_margin"]),
        float(result["mean_target_mass"]),
        float(result["mean_summary_cosine"]),
        float(result["mean_permutation_cosine"]),
        -float(result["specialist_parameter_count"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--specialist-root", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--challenge-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("relation-essential evaluation requires one CUDA device")
    device = torch.device("cuda")

    challenge_path = Path(args.challenge).resolve()
    manifest = json.loads(Path(args.challenge_manifest).read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-relation-essential-challenge.v0.1":
        raise SystemExit("relation-essential challenge schema mismatch")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("relation-essential challenge must remain eval-only")
    rows = read_jsonl(challenge_path)
    if len(rows) != 80:
        raise SystemExit("relation-essential challenge row count drift")

    from safetensors.torch import load_file

    n0_config = load_n0_config(Path(args.config))
    semantic_model = AliceN0V02Model(n0_config)
    semantic_state = load_file(str(Path(args.semantic_checkpoint) / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir))

    structured = StructuredStateEncoder(load_structured_config(Path(args.structured_config)))
    structured.load_state_dict(
        load_file(str(Path(args.structured_checkpoint) / "structured_state.safetensors"), device="cpu"),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    cache = build_cache(
        rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        device=device,
    )
    del semantic_model, semantic_state
    torch.cuda.empty_cache()

    root = Path(args.specialist_root).resolve()
    candidates: list[tuple[str, Path]] = []
    for variant in ("specialized_compact_384x2_graph256x1", "specialized_expanded_640x3_graph512x2"):
        for step in (80, 160, 240):
            checkpoint = root / variant / f"step-{step:08d}"
            if not (checkpoint / "evidence_view_adapter.safetensors").is_file():
                raise SystemExit(f"missing specialist checkpoint: {checkpoint}")
            candidates.append((variant, checkpoint))

    results = [
        evaluate_candidate(variant=variant, checkpoint=checkpoint, cache=cache, device=device)
        for variant, checkpoint in candidates
    ]
    winner = max(results, key=score)
    gate_pass = (
        float(winner["pair_flip_accuracy"]) >= 0.90
        and float(winner["family_min_pair_flip_accuracy"]) >= 0.75
        and float(winner["mean_pair_relation_flip_margin"]) > 0.20
        and float(winner["mean_permutation_cosine"]) >= 0.999
        and float(winner["mean_ablated_pair_distribution_l1"]) <= 1e-5
    )
    compact_best = max(
        (result for result in results if result["variant"].startswith("specialized_compact")),
        key=score,
    )
    expanded_best = max(
        (result for result in results if result["variant"].startswith("specialized_expanded")),
        key=score,
    )
    comparison = {
        "schema": "alice.eipm.n0.v02-relation-essential-comparison.v0.1",
        "status": "PASS" if gate_pass else "NEEDS_RELATION_REPAIR",
        "challenge_schema": manifest["schema"],
        "examples": 80,
        "pairs": 40,
        "families": 10,
        "selection_policy": "worst-family-pair-flip_then_pair-flip_then_relation-flip-margin_then_target-mass_then-summary-fidelity_then_efficiency",
        "results": results,
        "winner": winner,
        "gate_pass": gate_pass,
        "diagnostics": {
            "expanded_minus_compact_pair_flip": float(expanded_best["pair_flip_accuracy"]) - float(compact_best["pair_flip_accuracy"]),
            "expanded_minus_compact_min_family_pair_flip": float(expanded_best["family_min_pair_flip_accuracy"]) - float(compact_best["family_min_pair_flip_accuracy"]),
            "expanded_minus_compact_relation_flip_margin": float(expanded_best["mean_pair_relation_flip_margin"]) - float(compact_best["mean_pair_relation_flip_margin"]),
            "capacity_material_on_relation_essential": (
                float(expanded_best["pair_flip_accuracy"]) - float(compact_best["pair_flip_accuracy"]) >= 0.05
                or float(expanded_best["family_min_pair_flip_accuracy"]) - float(compact_best["family_min_pair_flip_accuracy"]) >= 0.25
            ),
        },
        "training_authorized_on_challenge": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": (
            "ratify_relation_specialist_checkpoint_and_move_to_cross_context_fusion"
            if gate_pass
            else "build_failure_driven_relation_repair_from_independent_analogues_then_retest_frozen_pairs"
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
