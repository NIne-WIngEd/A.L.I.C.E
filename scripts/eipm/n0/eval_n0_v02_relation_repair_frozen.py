#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from safetensors.torch import load_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph import EvidenceGraphConfig
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter, EvidenceViewAdapterConfig
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model
from eval_n0_v02_relation_essential_challenge import build_cache, read_jsonl

EXPECTED_SEMANTIC_PARAMETERS = 136_594_435
STEPS = (40, 80, 120, 160)


def structured_config(path: Path) -> StructuredStateConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))["architecture"]
    return StructuredStateConfig(
        semantic_size=int(raw["semantic_size"]), state_size=int(raw["state_size"]),
        num_layers=int(raw["num_layers"]), num_heads=int(raw["num_heads"]),
        feedforward_size=int(raw["feedforward_size"]), dropout=float(raw["dropout"]),
        max_fields=int(raw["max_fields"]),
    )


def adapter_config() -> EvidenceViewAdapterConfig:
    return EvidenceViewAdapterConfig(
        semantic_size=640, adapter_size=640, num_layers=3, num_heads=10,
        feedforward_size=2560, dropout=0.0, max_fields=64, base_prior_scale=0.5,
    )


def graph_config() -> EvidenceGraphConfig:
    return EvidenceGraphConfig(
        semantic_size=640, graph_size=512, graph_layers=2,
        max_fields=64, max_edges=256,
    )


def permuted(cache: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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


def forward(adapter: EvidenceViewAdapter, graph: DualEndpointEvidenceGraphEncoder,
            batch: dict[str, torch.Tensor], *, ablate: bool = False) -> dict[str, torch.Tensor]:
    adapted = adapter(
        parent_field_states=batch["parent_field_states"], valid_mask=batch["valid_mask"],
        query_semantic=batch["query_semantic"], parent_field_weights=batch["parent_field_weights"],
    )
    edge_valid = torch.zeros_like(batch["edge_valid_mask"]) if ablate else batch["edge_valid_mask"]
    return graph(
        field_states=adapted["field_states"], valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"], edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"], edge_valid_mask=edge_valid,
        query_semantic=batch["query_semantic"], base_field_weights=adapted["field_weights"],
    )


def evaluate(*, adapter: EvidenceViewAdapter, graph_path: Path, cache: dict[str, Any],
             device: torch.device, receipt: dict[str, Any]) -> dict[str, Any]:
    graph = DualEndpointEvidenceGraphEncoder(graph_config()).to(device)
    graph.load_state_dict(load_file(str(graph_path), device="cpu"), strict=True)
    graph.eval()
    batch = {key: value.to(device) for key, value in cache.items() if torch.is_tensor(value)}
    pbatch = permuted(cache, device)
    with torch.inference_mode():
        output = forward(adapter, graph, batch)
        ablated = forward(adapter, graph, batch, ablate=True)
        poutput = forward(adapter, graph, pbatch)

    weights = output["field_weights"].cpu()
    ablated_weights = ablated["field_weights"].cpu()
    target = cache["target_distribution"]
    target_idx = target.argmax(dim=-1)
    pred = weights.argmax(dim=-1)
    correct = pred.eq(target_idx)
    target_mass = weights.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    summary_cos = F.cosine_similarity(output["pooled_state"], batch["summary_semantic"], dim=-1).cpu()
    perm_cos = F.cosine_similarity(output["pooled_state"], poutput["pooled_state"], dim=-1).cpu()

    by_pair: dict[str, list[int]] = defaultdict(list)
    by_family: dict[str, list[str]] = defaultdict(list)
    for i, pair_id in enumerate(cache["pair_ids"]):
        by_pair[str(pair_id)].append(i)
        fam = str(cache["families"][i])
        if pair_id not in by_family[fam]:
            by_family[fam].append(str(pair_id))

    pair_ok: dict[str, bool] = {}
    pair_margin: dict[str, float] = {}
    ablated_l1: list[float] = []
    for pair_id, idxs in by_pair.items():
        if len(idxs) != 2:
            raise SystemExit(f"incomplete frozen pair: {pair_id}")
        a, b = sorted(idxs, key=lambda i: cache["variants"][i])
        ta, tb = int(target_idx[a]), int(target_idx[b])
        pair_ok[pair_id] = bool(correct[a] and correct[b] and pred[a] != pred[b])
        ma = float(weights[a, ta] - weights[a, tb])
        mb = float(weights[b, tb] - weights[b, ta])
        pair_margin[pair_id] = 0.5 * (ma + mb)
        ablated_l1.append(float((ablated_weights[a] - ablated_weights[b]).abs().sum()))

    fam_acc = {fam: sum(pair_ok[p] for p in pairs) / len(pairs) for fam, pairs in sorted(by_family.items())}
    fam_margin = {fam: sum(pair_margin[p] for p in pairs) / len(pairs) for fam, pairs in sorted(by_family.items())}
    replay = receipt["ordinary_replay_dev_metrics"]
    result = {
        "checkpoint": graph_path.parent.name,
        "step": int(receipt["step"]),
        "graph_sha256": receipt["graph_sha256"],
        "graph_parameter_count": int(receipt["graph_parameter_count"]),
        "example_top1_accuracy": float(correct.float().mean()),
        "pair_flip_accuracy": sum(pair_ok.values()) / len(pair_ok),
        "family_pair_flip_accuracy": fam_acc,
        "family_min_pair_flip_accuracy": min(fam_acc.values()),
        "family_pair_relation_flip_margin": fam_margin,
        "mean_pair_relation_flip_margin": sum(pair_margin.values()) / len(pair_margin),
        "mean_target_mass": float(target_mass.mean()),
        "min_target_mass": float(target_mass.min()),
        "mean_summary_cosine": float(summary_cos.mean()),
        "mean_permutation_cosine": float(perm_cos.mean()),
        "mean_ablated_pair_distribution_l1": sum(ablated_l1) / len(ablated_l1),
        "ordinary_replay_macro_target_support_mass": float(replay["family_macro_target_support_mass"]),
        "ordinary_replay_min_target_support_mass": float(replay["family_min_target_support_mass"]),
        "ordinary_replay_summary_cosine": float(replay["mean_summary_cosine"]),
        "frozen_relation_essential_challenge_used_for_training": False,
        "shared_structured_parent_mutated": False,
        "evidence_view_adapter_mutated": False,
        "semantic_core_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    result["gate_pass"] = (
        result["pair_flip_accuracy"] >= 0.90
        and result["family_min_pair_flip_accuracy"] >= 0.75
        and result["mean_pair_relation_flip_margin"] > 0.20
        and result["mean_permutation_cosine"] >= 0.999
        and result["mean_ablated_pair_distribution_l1"] <= 1e-5
        and result["ordinary_replay_macro_target_support_mass"] >= 0.970
        and result["ordinary_replay_min_target_support_mass"] >= 0.90
        and result["ordinary_replay_summary_cosine"] >= 0.95
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--repair-root", required=True)
    p.add_argument("--challenge", required=True)
    p.add_argument("--challenge-manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("frozen repair ratification eval requires CUDA")
    device = torch.device("cuda")
    manifest = json.loads(Path(args.challenge_manifest).read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-relation-essential-challenge.v0.1":
        raise SystemExit("frozen challenge schema mismatch")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("frozen challenge must remain eval-only")
    rows = read_jsonl(Path(args.challenge))
    if len(rows) != 80:
        raise SystemExit("frozen challenge row-count drift")

    semantic = AliceN0V02Model(load_n0_config(Path(args.config)))
    state = load_file(str(Path(args.semantic_checkpoint) / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for param in semantic.parameters(): param.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir))

    structured = StructuredStateEncoder(structured_config(Path(args.structured_config)))
    structured.load_state_dict(load_file(str(Path(args.structured_checkpoint) / "structured_state.safetensors"), device="cpu"), strict=True)
    for param in structured.parameters(): param.requires_grad = False
    cache = build_cache(rows, semantic_model=semantic, tokenizer=tokenizer, structured=structured, device=device)
    del semantic, state
    torch.cuda.empty_cache()

    adapter = EvidenceViewAdapter(adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(Path(args.parent_adapter)), device="cpu"), strict=True)
    for param in adapter.parameters(): param.requires_grad = False
    adapter.eval()

    repair_root = Path(args.repair_root)
    results: list[dict[str, Any]] = []
    for step in STEPS:
        cp = repair_root / f"step-{step:08d}"
        receipt = json.loads((cp / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("frozen_relation_essential_challenge_used_for_training") is not False:
            raise SystemExit(f"frozen challenge contamination at step {step}")
        if receipt.get("evidence_view_adapter_mutated") is not False:
            raise SystemExit(f"adapter mutation at step {step}")
        graph_path = cp / "evidence_graph_dual_endpoint.safetensors"
        if not graph_path.is_file():
            raise SystemExit(f"missing repaired graph checkpoint: {graph_path}")
        results.append(evaluate(adapter=adapter, graph_path=graph_path, cache=cache, device=device, receipt=receipt))

    passing = [r for r in results if r["gate_pass"]]
    ratified = min(passing, key=lambda r: r["step"]) if passing else None
    comparison = {
        "schema": "alice.eipm.n0.v02-relation-repair-frozen-ratification.v0.1",
        "status": "PASS" if ratified else "REPAIR_NOT_RATIFIED",
        "challenge_schema": manifest["schema"],
        "examples": 80,
        "pairs": 40,
        "families": 10,
        "selection_policy": "earliest_checkpoint_clearing_all_frozen_relation_and_ordinary_replay_gates",
        "results": results,
        "ratified_candidate": ratified,
        "training_authorized_on_challenge": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": "ratify_evidence_specialist_and_begin_cross_context_fusion" if ratified else "diagnose_remaining_frozen_relation_failures_without_training_on_challenge",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
