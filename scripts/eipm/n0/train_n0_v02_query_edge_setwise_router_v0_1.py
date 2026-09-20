#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_graph_query_edge_setwise_router import (
    SetwiseQueryEdgeEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.query_edge_setwise_router_bridge import (
    SetwiseQueryEdgeBridge,
)
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml
import train_n0_v02_query_edge_cross_attention_bridge_v0_1 as old


QUAL_SCHEMA = "alice.eipm.n0.query-edge-setwise-router-qualification.v0.1"
CURRICULUM_SCHEMA = "alice.eipm.n0.v02-query-edge-binding-curriculum.v0.1"
ANCHOR_SCHEMA = "alice.eipm.n0.query-edge-preservation-train-anchors.v0.1"
CHECKPOINT_SCHEMA = "alice.eipm.n0.setwise-edge-router-training-checkpoint.v0.1"
RESULT_SCHEMA = "alice.eipm.n0.setwise-edge-router-training-result.v0.1"

EXPECTED_QUAL_SHA256 = (
    "4b1591e1eec5c450a872264a1bdf1130def5a29e443c8c35e974ae05f2861666"
)
EXPECTED_FAILED_RESULT_SHA256 = (
    "08bef659779c2e6fcfb0d60ef209d9f6c7f03b0198bcf370e097e278f5436328"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_MANIFEST_SHA256 = (
    "0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
)
EXPECTED_ANCHORS_SHA256 = (
    "f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_CALIBRATION_SHA256 = (
    "d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b"
)
EXPECTED_HIDDEN_STATES = 17


def candidate_forward(
    graph: SetwiseQueryEdgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = ml.adapter_forward(adapter, batch)
    hidden = batch["query_hidden_states"].float()
    hidden_tuple = tuple(hidden[:, layer] for layer in range(hidden.size(1)))
    return graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=batch["edge_valid_mask"],
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
        query_hidden_states=hidden_tuple,
        query_attention_mask=batch["query_attention_mask"],
    )


def route_targets(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # Route class 0 is parent/no-op. Edge index e maps to route class e+1.
    return batch["relevant_edge_index"].long() + 1


def causal_route_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return F.cross_entropy(out["route_logits_with_noop"], route_targets(batch))


def noop_route_loss(out: dict[str, torch.Tensor]) -> torch.Tensor:
    targets = torch.zeros(
        out["route_logits_with_noop"].size(0),
        dtype=torch.long,
        device=out["route_logits_with_noop"].device,
    )
    return F.cross_entropy(out["route_logits_with_noop"], targets)


def evaluate_causal(
    graph: SetwiseQueryEdgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: old.QueryEdgeQuadDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()

    rows = row_ok = quads = quad_ok = 0
    route_total = route_ok = 0
    margins: list[float] = []
    target_route_probabilities: list[float] = []
    noop_probabilities: list[float] = []
    relation_rows: dict[str, list[int]] = defaultdict(list)
    relation_quads: dict[str, list[int]] = defaultdict(list)
    relation_routes: dict[str, list[int]] = defaultdict(list)
    relation_route_prob: dict[str, list[float]] = defaultdict(list)
    relation_margins: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            global_indices = batch["global_indices"]
            flat = old.to_device(old.flatten_quad_batch(batch), device)
            out = candidate_forward(graph, adapter, flat)

            target = flat["target_distribution"].argmax(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            ok = pred.eq(target)
            margin = ml.target_margin(
                out["field_weights"],
                flat["target_distribution"],
                flat["valid_mask"],
            )

            route_target = route_targets(flat)
            route_prob = out["route_probabilities_with_noop"]
            route_pred = route_prob.argmax(dim=-1)
            route_ok_row = route_pred.eq(route_target)
            row_index = torch.arange(route_target.size(0), device=device)
            target_prob = route_prob[row_index, route_target]
            noop_prob = route_prob[:, 0]

            batch_quads = global_indices.size(0)
            ok4 = ok.reshape(batch_quads, 4)
            margin4 = margin.reshape(batch_quads, 4)
            route4 = route_ok_row.reshape(batch_quads, 4)
            route_prob4 = target_prob.reshape(batch_quads, 4)
            qok = ok4.all(dim=1)

            rows += int(ok.numel())
            row_ok += int(ok.sum().item())
            quads += int(qok.numel())
            quad_ok += int(qok.sum().item())
            route_total += int(route_ok_row.numel())
            route_ok += int(route_ok_row.sum().item())
            margins.extend(float(x) for x in margin.cpu().tolist())
            target_route_probabilities.extend(
                float(x) for x in target_prob.cpu().tolist()
            )
            noop_probabilities.extend(float(x) for x in noop_prob.cpu().tolist())

            for local in range(batch_quads):
                first = int(global_indices[local, 0].item())
                relation = str(payload["relations"][first])
                relation_quads[relation].append(int(qok[local].item()))
                relation_rows[relation].extend(
                    int(x) for x in ok4[local].cpu().tolist()
                )
                relation_routes[relation].extend(
                    int(x) for x in route4[local].cpu().tolist()
                )
                relation_route_prob[relation].extend(
                    float(x) for x in route_prob4[local].cpu().tolist()
                )
                relation_margins[relation].extend(
                    float(x) for x in margin4[local].cpu().tolist()
                )

    family_row = {
        key: sum(v) / len(v) for key, v in sorted(relation_rows.items())
    }
    family_quad = {
        key: sum(v) / len(v) for key, v in sorted(relation_quads.items())
    }
    family_route = {
        key: sum(v) / len(v) for key, v in sorted(relation_routes.items())
    }
    family_route_prob = {
        key: sum(v) / len(v) for key, v in sorted(relation_route_prob.items())
    }
    family_margin = {
        key: sum(v) / len(v) for key, v in sorted(relation_margins.items())
    }

    return {
        "rows": rows,
        "row_accuracy": row_ok / max(rows, 1),
        "quads": quads,
        "quad_accuracy": quad_ok / max(quads, 1),
        "routing_accuracy": route_ok / max(route_total, 1),
        "family_row_accuracy": family_row,
        "family_min_row_accuracy": min(family_row.values()),
        "family_quad_accuracy": family_quad,
        "family_min_quad_accuracy": min(family_quad.values()),
        "family_routing_accuracy": family_route,
        "family_min_routing_accuracy": min(family_route.values()),
        "mean_target_route_probability": (
            sum(target_route_probabilities) / max(len(target_route_probabilities), 1)
        ),
        "family_mean_target_route_probability": family_route_prob,
        "mean_noop_probability_on_causal_rows": (
            sum(noop_probabilities) / max(len(noop_probabilities), 1)
        ),
        "mean_target_margin": sum(margins) / max(len(margins), 1),
        "family_mean_target_margin": family_margin,
    }


def evaluate_noop_rows(
    graph: SetwiseQueryEdgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    probabilities: list[float] = []
    graph.eval()
    adapter.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = old.to_device(batch, device)
            out = candidate_forward(graph, adapter, batch)
            probs = out["route_probabilities_with_noop"]
            correct += int(probs.argmax(dim=-1).eq(0).sum().item())
            total += int(probs.size(0))
            probabilities.extend(float(x) for x in probs[:, 0].cpu().tolist())
    return {
        "noop_top1_accuracy": correct / max(total, 1),
        "mean_noop_probability": sum(probabilities) / max(len(probabilities), 1),
    }


def evaluate_noop_pairs(
    graph: SetwiseQueryEdgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = total = 0
    probabilities: list[float] = []
    graph.eval()
    adapter.eval()
    with torch.inference_mode():
        for batch in loader:
            for prefix in ("a", "b"):
                row = {
                    key[2:]: value
                    for key, value in batch.items()
                    if key.startswith(prefix + "_")
                    and key != prefix + "_global_index"
                }
                row = old.to_device(row, device)
                out = candidate_forward(graph, adapter, row)
                probs = out["route_probabilities_with_noop"]
                correct += int(probs.argmax(dim=-1).eq(0).sum().item())
                total += int(probs.size(0))
                probabilities.extend(
                    float(x) for x in probs[:, 0].cpu().tolist()
                )
    return {
        "noop_top1_accuracy": correct / max(total, 1),
        "mean_noop_probability": sum(probabilities) / max(len(probabilities), 1),
    }


def causal_ready(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    no_family_regression = all(
        float(current["family_row_accuracy"][relation]) + 1e-12
        >= float(baseline["family_row_accuracy"][relation])
        for relation in baseline["family_row_accuracy"]
    )
    detail = {
        "no_relation_family_row_accuracy_regression": no_family_regression,
        "overall_row_accuracy_strictly_improves": (
            float(current["row_accuracy"])
            > float(baseline["row_accuracy"]) + 1e-12
        ),
        "overall_quad_accuracy_strictly_improves": (
            float(current["quad_accuracy"])
            > float(baseline["quad_accuracy"]) + 1e-12
        ),
        "worst_family_quad_accuracy_not_worse": (
            float(current["family_min_quad_accuracy"]) + 1e-12
            >= float(baseline["family_min_quad_accuracy"])
        ),
        "mean_target_margin_strictly_improves": (
            float(current["mean_target_margin"])
            > float(baseline["mean_target_margin"]) + 1e-12
        ),
        "routing_accuracy_above_half": (
            float(current["routing_accuracy"]) > 0.5
        ),
        "worst_family_routing_strictly_above_five_route_uniform_chance": (
            float(current["family_min_routing_accuracy"]) > 0.2 + 1e-12
        ),
        "routing_accuracy_strictly_improves_vs_initial": (
            float(current["routing_accuracy"])
            > float(baseline["routing_accuracy"]) + 1e-12
        ),
        "target_route_probability_strictly_improves_vs_initial": (
            float(current["mean_target_route_probability"])
            > float(baseline["mean_target_route_probability"]) + 1e-12
        ),
    }
    return all(detail.values()), detail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--qualification-receipt", required=True)
    p.add_argument("--failed-result", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--curriculum-manifest", required=True)
    p.add_argument("--preservation-anchors", required=True)
    p.add_argument("--calibration-receipt", required=True)
    p.add_argument("--ordinary-source-curriculum", required=True)
    p.add_argument("--endpoint-source-curriculum", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=24)
    p.add_argument("--quad-batch-size", type=int, default=4)
    p.add_argument("--preservation-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=12)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--margin-weight", type=float, default=0.25)
    p.add_argument("--causal-route-weight", type=float, default=1.0)
    p.add_argument("--preservation-weight", type=float, default=1.0)
    p.add_argument("--noop-route-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("setwise edge-router training requires one CUDA device")
    device = torch.device("cuda")
    old.seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    qual_path = Path(args.qualification_receipt).resolve()
    failed_path = Path(args.failed_result).resolve()
    map_path = Path(args.layer_map).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    anchors_path = Path(args.preservation_anchors).resolve()
    cal_path = Path(args.calibration_receipt).resolve()
    ordinary_source = Path(args.ordinary_source_curriculum).resolve()
    endpoint_source = Path(args.endpoint_source_curriculum).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    structured_config = Path(args.structured_config).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_replay_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_replay_path = Path(args.endpoint_replay_cache).resolve()
    output = Path(args.output_dir).resolve()

    for path in (
        qual_path, failed_path, map_path, curriculum_path, manifest_path,
        anchors_path, cal_path, ordinary_source, endpoint_source, semantic_config,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json", structured_config,
        structured_checkpoint / "structured_state.safetensors",
        adapter_path, graph_path, ordinary_replay_path, endpoint_replay_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing setwise training input: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite setwise training output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    exact_hashes = {
        qual_path: EXPECTED_QUAL_SHA256,
        failed_path: EXPECTED_FAILED_RESULT_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        manifest_path: EXPECTED_MANIFEST_SHA256,
        anchors_path: EXPECTED_ANCHORS_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
        cal_path: EXPECTED_CALIBRATION_SHA256,
    }
    for path, expected in exact_hashes.items():
        if old.sha(path) != expected:
            raise SystemExit(f"exact competitive training lineage drift: {path.name}")

    qual = json.loads(qual_path.read_text(encoding="utf-8"))
    if qual.get("schema") != QUAL_SCHEMA:
        raise SystemExit("setwise qualification schema drift")
    if qual.get("status") != "PASS_COMPETITIVE_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT":
        raise SystemExit("competitive runtime qualification did not pass")
    for key in (
        "parent_parameters_exactly_unchanged",
        "exact_parent_field_weights",
        "exact_parent_pooled_state",
        "zero_initialized_source_residual_readout",
        "zero_initialized_target_residual_readout",
        "route_probabilities_sum_to_one",
        "explicit_parent_noop_route",
        "all_active_directed_edges_jointly_contextualized",
        "edge_set_permutation_equivariance",
        "nonlocal_competitor_context",
        "same_relation_edges_produce_distinct_edge_queries",
        "edge_content_conditions_query_token_attention",
        "route_probability_is_actual_source_contribution_control",
        "route_probability_is_actual_target_contribution_control",
    ):
        if qual.get(key) is not True:
            raise SystemExit(f"setwise qualification invariant failed: {key}")
    if qual.get("independent_per_edge_route_scoring") is not False:
        raise SystemExit("independent edge route scoring returned")
    if qual.get("proxy_dot_product_router") is not False:
        raise SystemExit("proxy router returned")
    if qual.get("training_authorized") is not False:
        raise SystemExit("qualification receipt must not self-authorize training")

    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    if failed.get("status") != (
        "FAIL_COMPETITIVE_EDGE_ROUTER_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("source failed experiment status drift")
    if failed.get("selected_checkpoint") is not None:
        raise SystemExit("source failed experiment unexpectedly selected a checkpoint")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CURRICULUM_SCHEMA:
        raise SystemExit("query-edge curriculum schema drift")
    if manifest.get("compiled_sha256") != old.sha(curriculum_path):
        raise SystemExit("query-edge curriculum hash drift")
    if manifest.get("test_split_opening_authorized") is not False:
        raise SystemExit("test split unexpectedly opened")

    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if anchors.get("schema") != ANCHOR_SCHEMA:
        raise SystemExit("preservation anchor schema drift")
    if anchors["gradient_eligibility"]["dev_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation DEV leakage")
    if anchors["gradient_eligibility"]["heldout_or_test_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation heldout leakage")

    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal.get("status") != "PASS_PARENT_ONLY_PRESERVATION_CALIBRATION":
        raise SystemExit("frozen parent preservation calibration drift")
    policy = cal["policy"]

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    structured_cfg = rr.load_structured_config(structured_config)
    structured = StructuredStateEncoder(structured_cfg)
    structured.load_state_dict(
        load_file(
            str(structured_checkpoint / "structured_state.safetensors"),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    rows = old.read_jsonl(curriculum_path)
    causal = old.compile_query_edge_payload(
        rows,
        semantic=semantic,
        tokenizer=tokenizer,
        structured=structured,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    cache_path = output / "setwise_query_edge_train_dev_hidden_cache.pt"
    torch.save(causal, cache_path)

    ordinary_payload = torch.load(ordinary_replay_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_replay_path, map_location="cpu")

    ordinary_train_indices = old.indices_from_ids(
        ordinary_payload,
        [str(x) for x in anchors["ordinary"]["train_anchor_ids"]],
        "ordinary",
    )
    endpoint_train_indices = old.indices_from_ids(
        endpoint_payload,
        [str(x) for x in anchors["endpoint"]["train_anchor_ids"]],
        "endpoint",
    )
    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev_pairs = rr.pair_indices(endpoint_payload, "dev")
    endpoint_dev_indices = sorted({i for pair in endpoint_dev_pairs for i in pair})

    ordinary_needed = sorted(set(ordinary_train_indices + ordinary_dev_indices))
    endpoint_needed = sorted(set(endpoint_train_indices + endpoint_dev_indices))
    ordinary_hidden = ml.attach_query_hidden_states(
        ordinary_payload,
        old.read_jsonl(ordinary_source),
        indices=ordinary_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_hidden = ml.attach_query_hidden_states(
        endpoint_payload,
        old.read_jsonl(endpoint_source),
        indices=endpoint_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )

    del semantic, state, structured
    torch.cuda.empty_cache()

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_state = load_file(str(graph_path), device="cpu")
    baseline_graph = DualEndpointEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    baseline_graph.load_state_dict(parent_state, strict=True)
    for parameter in baseline_graph.parameters():
        parameter.requires_grad = False
    baseline_graph.eval()

    bridge = SetwiseQueryEdgeBridge.from_layer_map_file(
        layer_map_path=map_path,
        semantic_size=rr.expanded_graph_config().semantic_size,
        graph_size=rr.expanded_graph_config().graph_size,
        num_relation_types=rr.expanded_graph_config().num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=rr.expanded_graph_config().graph_size,
        dropout=0.0,
        route_context_layers=2,
        route_context_heads=8,
        route_context_ffn_multiplier=4,
    )
    graph = SetwiseQueryEdgeEvidenceGraphEncoder(
        config=rr.expanded_graph_config(),
        bridge=bridge,
    ).to(device)
    missing, unexpected = graph.load_state_dict(parent_state, strict=False)
    if any(
        not name.startswith("setwise_query_edge_bridge.")
        for name in missing
    ) or unexpected:
        raise SystemExit(
            f"setwise parent load drift missing={list(missing)} unexpected={list(unexpected)}"
        )

    trainable = graph.freeze_parent_for_bridge_training()
    parent_hash_before = old.tensor_state_sha256(
        graph, exclude_prefixes=("setwise_query_edge_bridge.",)
    )

    train_quads = old.QueryEdgeQuadDataset(causal, "train")
    dev_quads = old.QueryEdgeQuadDataset(causal, "dev")
    if len(train_quads) != 108 or len(dev_quads) != 36:
        raise SystemExit(
            f"query-edge quad split drift train={len(train_quads)} dev={len(dev_quads)}"
        )

    ordinary_train = ml.RowHiddenDataset(
        ordinary_payload, ordinary_train_indices, ordinary_hidden
    )
    endpoint_train = ml.RowHiddenDataset(
        endpoint_payload, endpoint_train_indices, endpoint_hidden
    )
    ordinary_dev = ml.RowHiddenDataset(
        ordinary_payload, ordinary_dev_indices, ordinary_hidden
    )
    endpoint_dev = ml.PairHiddenDataset(
        endpoint_payload, endpoint_dev_pairs, endpoint_hidden
    )

    baseline_dev = evaluate_causal(
        graph, adapter, dev_quads, causal, device, args.eval_batch_size
    )
    initial_ordinary = old.evaluate_ordinary(
        graph, adapter, ordinary_dev, ordinary_payload, device, args.eval_batch_size
    )
    initial_endpoint = old.evaluate_endpoint(
        graph, adapter, endpoint_dev, endpoint_payload, device, args.eval_batch_size
    )
    initial_preserved, initial_checks = old.preservation_pass(
        initial_ordinary, initial_endpoint, policy
    )
    if not initial_preserved:
        raise SystemExit("zero-init setwise bridge failed preservation policy")

    initial_ordinary_noop = evaluate_noop_rows(
        graph, adapter, ordinary_dev, device, args.eval_batch_size
    )
    initial_endpoint_noop = evaluate_noop_pairs(
        graph, adapter, endpoint_dev, device, args.eval_batch_size
    )

    first_dev = next(iter(DataLoader(dev_quads, batch_size=1, shuffle=False)))
    flat_first = old.to_device(old.flatten_quad_batch(first_dev), device)
    with torch.inference_mode():
        parent_out = old.parent_forward(baseline_graph, adapter, flat_first)
        candidate_out = candidate_forward(graph, adapter, flat_first)
    if not torch.equal(parent_out["field_weights"], candidate_out["field_weights"]):
        raise SystemExit("setwise zero-init field weights lost exact parent")
    if not torch.equal(parent_out["pooled_state"], candidate_out["pooled_state"]):
        raise SystemExit("setwise zero-init pooled state lost exact parent")

    print(
        "baseline_setwise_dev=" + json.dumps(baseline_dev, sort_keys=True),
        flush=True,
    )
    print(
        "initial_preservation=" + json.dumps(
            {
                "ordinary": initial_ordinary,
                "endpoint": initial_endpoint,
                "ordinary_noop": initial_ordinary_noop,
                "endpoint_noop": initial_endpoint_noop,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(
            args.max_steps - args.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    causal_loader = DataLoader(
        train_quads,
        batch_size=args.quad_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    ordinary_loader = DataLoader(
        ordinary_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    endpoint_loader = DataLoader(
        endpoint_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
        num_workers=0,
    )

    causal_it = iter(causal_loader)
    ordinary_it = iter(ordinary_loader)
    endpoint_it = iter(endpoint_loader)
    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        causal_batch, causal_it = old.next_batch(causal_it, causal_loader)
        ordinary_batch, ordinary_it = old.next_batch(ordinary_it, ordinary_loader)
        endpoint_batch, endpoint_it = old.next_batch(endpoint_it, endpoint_loader)

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)

        causal_flat = old.to_device(old.flatten_quad_batch(causal_batch), device)
        out = candidate_forward(graph, adapter, causal_flat)

        selection_ce = ml.dist_ce(
            out["field_weights"], causal_flat["target_distribution"]
        )
        margins = ml.target_margin(
            out["field_weights"],
            causal_flat["target_distribution"],
            causal_flat["valid_mask"],
        )
        margin_loss = F.relu(args.margin - margins).mean()
        route_ce = causal_route_loss(out, causal_flat)

        causal_total = (
            selection_ce
            + args.margin_weight * margin_loss
            + args.causal_route_weight * route_ce
        )
        causal_total.backward()

        preservation_log: list[dict[str, float]] = []
        for label, preservation_batch in (
            ("ordinary", ordinary_batch),
            ("endpoint", endpoint_batch),
        ):
            preserve = old.to_device(preservation_batch, device)
            with torch.inference_mode():
                teacher = old.parent_forward(baseline_graph, adapter, preserve)
            student = candidate_forward(graph, adapter, preserve)
            kl = old.distill_kl(
                student["field_weights"],
                teacher["field_weights"],
            )
            noop_ce = noop_route_loss(student)
            weighted = (
                0.5 * args.preservation_weight * kl
                + 0.5 * args.noop_route_weight * noop_ce
            )
            weighted.backward()
            preservation_log.append(
                {
                    "parent_kl": float(kl.detach().cpu()),
                    "noop_route_ce": float(noop_ce.detach().cpu()),
                }
            )

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "selection_ce": float(selection_ce.detach().cpu()),
                        "margin_loss": float(margin_loss.detach().cpu()),
                        "causal_route_ce": float(route_ce.detach().cpu()),
                        "ordinary_parent_kl": preservation_log[0]["parent_kl"],
                        "ordinary_noop_route_ce": preservation_log[0]["noop_route_ce"],
                        "endpoint_parent_kl": preservation_log[1]["parent_kl"],
                        "endpoint_noop_route_ce": preservation_log[1]["noop_route_ce"],
                        "lr": scheduler.get_last_lr()[0],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if step % args.save_every == 0 or step == args.max_steps:
            dev = evaluate_causal(
                graph, adapter, dev_quads, causal, device, args.eval_batch_size
            )
            ordinary = old.evaluate_ordinary(
                graph, adapter, ordinary_dev, ordinary_payload,
                device, args.eval_batch_size
            )
            endpoint = old.evaluate_endpoint(
                graph, adapter, endpoint_dev, endpoint_payload,
                device, args.eval_batch_size
            )
            ordinary_noop = evaluate_noop_rows(
                graph, adapter, ordinary_dev, device, args.eval_batch_size
            )
            endpoint_noop = evaluate_noop_pairs(
                graph, adapter, endpoint_dev, device, args.eval_batch_size
            )

            preserved, preservation_checks = old.preservation_pass(
                ordinary, endpoint, policy
            )
            causal_ok, causal_checks = causal_ready(dev, baseline_dev)

            cp = output / f"step-{step:08d}"
            cp.mkdir(parents=True, exist_ok=True)
            candidate_path = cp / "setwise_query_edge_bridge.safetensors"
            bridge_state = {
                name: value.detach().cpu().contiguous()
                for name, value in graph.state_dict().items()
                if name.startswith("setwise_query_edge_bridge.")
            }
            save_file(bridge_state, str(candidate_path))

            receipt = {
                "schema": CHECKPOINT_SCHEMA,
                "status": "TRAINED_NOT_RATIFIED",
                "step": step,
                "git_revision": old.git_revision(root),
                "candidate_sha256": old.sha(candidate_path),
                "qualification_receipt_sha256": old.sha(qual_path),
                "source_failed_result_sha256": old.sha(failed_path),
                "curriculum_sha256": old.sha(curriculum_path),
                "curriculum_manifest_sha256": old.sha(manifest_path),
                "preservation_anchors_sha256": old.sha(anchors_path),
                "calibration_receipt_sha256": old.sha(cal_path),
                "parent_graph_sha256": old.sha(graph_path),
                "parent_adapter_sha256": old.sha(adapter_path),
                "trainable_scope": "setwise_query_edge_bridge_only",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "training_objective": {
                    "causal_field_selection_ce": True,
                    "causal_margin_weight": args.margin_weight,
                    "causal_actual_route_ce_weight": args.causal_route_weight,
                    "parent_distillation_weight": args.preservation_weight,
                    "parent_noop_route_ce_weight": args.noop_route_weight,
                    "ordinary_and_endpoint_equal_share": True,
                    "proxy_routing_loss": False,
                    "irrelevant_edge_gate_penalty": False,
                    "load_balancing_loss": False,
                    "gradient_surgery": False,
                    "route_probability_detachment": False,
                },
                "parent_parameters_exactly_unchanged": (
                    old.tensor_state_sha256(
                        graph,
                        exclude_prefixes=("setwise_query_edge_bridge.",),
                    )
                    == parent_hash_before
                ),
                "causal_dev_metrics": dev,
                "causal_dev_checks": causal_checks,
                "causal_dev_ready": causal_ok,
                "ordinary_preservation_metrics": ordinary,
                "endpoint_preservation_metrics": endpoint,
                "ordinary_noop_route_metrics": ordinary_noop,
                "endpoint_noop_route_metrics": endpoint_noop,
                "preservation_checks": preservation_checks,
                "preservation_pass": preserved,
                "eligible_for_heldout_decision": preserved and causal_ok,
                "preservation_train_anchors_used_for_gradient": True,
                "preservation_dev_rows_used_for_gradient": False,
                "causal_test_split_evaluated": False,
                "frozen_challenge_evaluated": False,
                "semantic_parent_mutated": False,
                "structured_parent_mutated": False,
                "adapter_mutated": False,
                "parent_graph_mutated": False,
                "scale_authorized": False,
                "private_identity_gradient": False,
                "production_promotion_authorized": False,
                "n0_complete": False,
            }
            if receipt["parent_parameters_exactly_unchanged"] is not True:
                raise SystemExit("frozen parent changed during competitive training")

            (cp / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipts[cp.name] = receipt
            print(
                "setwise_router_checkpoint="
                + json.dumps(receipt, sort_keys=True),
                flush=True,
            )

    eligible = {
        key: value for key, value in receipts.items()
        if value["eligible_for_heldout_decision"]
    }

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["causal_dev_metrics"]
        return (
            float(dev["family_min_routing_accuracy"]),
            float(dev["family_min_quad_accuracy"]),
            float(dev["routing_accuracy"]),
            float(dev["quad_accuracy"]),
            float(dev["row_accuracy"]),
            float(dev["mean_target_margin"]),
            -int(receipt["step"]),
        )

    selected = max(
        eligible,
        key=lambda key: score(eligible[key]),
    ) if eligible else None

    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_SETWISE_EDGE_ROUTER_CAUSAL_DEV_AND_PRESERVATION_READY_FOR_SEPARATE_HELDOUT_DECISION"
            if selected is not None
            else
            "FAIL_SETWISE_EDGE_ROUTER_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": old.git_revision(root),
        "qualification_receipt_sha256": old.sha(qual_path),
        "source_failed_result_sha256": old.sha(failed_path),
        "curriculum_sha256": old.sha(curriculum_path),
        "curriculum_manifest_sha256": old.sha(manifest_path),
        "preservation_anchors_sha256": old.sha(anchors_path),
        "calibration_receipt_sha256": old.sha(cal_path),
        "parent_graph_sha256": old.sha(graph_path),
        "parent_adapter_sha256": old.sha(adapter_path),
        "causal_train_dev_hidden_cache_sha256": old.sha(cache_path),
        "baseline_setwise_dev": baseline_dev,
        "initial_parent_preservation": {
            "ordinary": initial_ordinary,
            "endpoint": initial_endpoint,
            "ordinary_noop": initial_ordinary_noop,
            "endpoint_noop": initial_endpoint_noop,
            "checks": initial_checks,
        },
        "checkpoints": receipts,
        "eligible_checkpoint_keys": sorted(eligible),
        "selected_checkpoint": selected,
        "selected_candidate_sha256": (
            eligible[selected]["candidate_sha256"] if selected else None
        ),
        "training_objective": {
            "causal_field_selection_ce": True,
            "causal_margin_weight": args.margin_weight,
            "causal_actual_route_ce_weight": args.causal_route_weight,
            "parent_distillation_weight": args.preservation_weight,
            "parent_noop_route_ce_weight": args.noop_route_weight,
            "ordinary_and_endpoint_equal_share": True,
            "proxy_routing_loss": False,
            "irrelevant_edge_gate_penalty": False,
            "load_balancing_loss": False,
            "gradient_surgery": False,
            "route_probability_detachment": False,
            "rationale": (
                "train the exact runtime route directly from causal edge labels "
                "and teach parent/no-op on old-task TRAIN anchors"
            ),
        },
        "selection_policy": (
            "preservation_first_then_worst_family_actual_route_then_worst_family_quad_"
            "then_overall_actual_route_then_quad_then_row_then_margin_then_earlier"
        ),
        "causal_readiness_policy": {
            "no_relation_family_row_accuracy_regression_vs_initial_parent": True,
            "overall_row_accuracy_strictly_improves": True,
            "overall_quad_accuracy_strictly_improves": True,
            "worst_family_quad_accuracy_may_not_regress": True,
            "mean_target_margin_strictly_improves": True,
            "actual_route_accuracy_must_exceed_half": True,
            "worst_family_actual_route_accuracy_strictly_above_uniform_five_route_chance": True,
            "actual_route_accuracy_must_strictly_improve_vs_initial": True,
            "target_route_probability_must_strictly_improve_vs_initial": True,
        },
        "preservation_train_anchors_used_for_gradient": True,
        "preservation_dev_rows_used_for_gradient": False,
        "causal_test_split_evaluated": False,
        "causal_test_split_opened_after_training": False,
        "frozen_challenge_evaluated": False,
        "parent_graph_parameters_exactly_unchanged": (
            old.tensor_state_sha256(
                graph,
                exclude_prefixes=("setwise_query_edge_bridge.",),
            )
            == parent_hash_before
        ),
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "adapter_mutated": False,
        "scale_authorized": False,
        "automatic_rerun_or_hotfix_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "make_one_explicit_heldout_opening_decision_without_changing_candidate"
            if selected is not None
            else
            "stop_and_reassess_setwise_route_or_training_boundary_from_first_principles"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
