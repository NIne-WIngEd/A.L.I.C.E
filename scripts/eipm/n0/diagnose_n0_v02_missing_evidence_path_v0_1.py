#!/usr/bin/env python3
"""Trace a fresh evidence-only relation signal through the full public N0 stack.

The diagnostic uses only fresh relation-flip rows materialized by the paired
builder. It performs no gradient and has no ratification effect. Within each
pair, text/query/fields are identical and only the direction of a `corrects`
edge flips, so pair-flip accuracy is a direct test of whether relation-derived
value identity survives graph -> fusion -> latent processing.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 as prior_eval
import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2 as v02_eval
from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
)
from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
    LatentPoolDataset,
    latent_forward,
    load_ratified_fusion,
    precompute_fusion_cache,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import (
    load_config as load_latent_config,
)
from train_n0_v02_cross_context_fusion_full_scale import (
    build_parent_cache,
    encode_pooled,
    read_jsonl,
    sha256_file,
    to_device,
)

EXPECTED_DATASET_SCHEMA = (
    "alice.eipm.n0.missing-evidence-fresh-relation-flip-diagnostic-manifest.v0.1"
)
EXPECTED_DATASET_STATUS = "FRESH_DIAGNOSTIC_ROWS_MATERIALIZED_NOT_EVALUATED"
EXPECTED_REPAIRED_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_LATENT_SHA256 = (
    "503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4"
)


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{label} must be an object")
    return value


def capture_hook(store: list[dict[str, torch.Tensor]]):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, dict):
            raise RuntimeError("evidence graph output is not a dict")
        captured: dict[str, torch.Tensor] = {}
        for key, value in output.items():
            if torch.is_tensor(value):
                captured[key] = value.detach().float().cpu()
        store.append(captured)

    return hook


def concat_capture(
    store: list[dict[str, torch.Tensor]], key: str
) -> torch.Tensor:
    chunks = [item[key] for item in store if key in item]
    if not chunks:
        raise RuntimeError(f"missing graph capture: {key}")
    return torch.cat(chunks, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)
    ).sum(dim=-1)


def vector_margin(
    vector: torch.Tensor, target: torch.Tensor, foil: torch.Tensor
) -> torch.Tensor:
    return cosine(vector, target) - cosine(vector, foil)


def slot_margin(
    slots: torch.Tensor, target: torch.Tensor, foil: torch.Tensor
) -> torch.Tensor:
    target_cos = (
        F.normalize(slots.float(), dim=-1)
        * F.normalize(target.float(), dim=-1).unsqueeze(1)
    ).sum(dim=-1)
    foil_cos = (
        F.normalize(slots.float(), dim=-1)
        * F.normalize(foil.float(), dim=-1).unsqueeze(1)
    ).sum(dim=-1)
    return (target_cos - foil_cos).max(dim=-1).values


def best_target_cosine(slots: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        F.normalize(slots.float(), dim=-1)
        * F.normalize(target.float(), dim=-1).unsqueeze(1)
    ).sum(dim=-1).max(dim=-1).values


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def row_mean_view_attention(value: torch.Tensor, view: int) -> torch.Tensor:
    selected = value[..., view].float()
    while selected.ndim > 1:
        selected = selected.mean(dim=1)
    return selected


def run_latent(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    parent: dict[str, Any],
    fusion: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    dataset = LatentPoolDataset(parent, fusion, list(range(len(parent["ids"]))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                result = latent_forward(model, batch)
            for key in (
                "latent_slots",
                "pooled_state",
                "view_attention_mass",
                "channel_attention_mass",
            ):
                outputs[key].append(result[key].detach().float().cpu())

    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


def stage_summary(
    margins: torch.Tensor,
    pair_ids: Sequence[str],
) -> dict[str, Any]:
    values = [float(value) for value in margins.detach().float().cpu().tolist()]
    by_pair: dict[str, list[float]] = defaultdict(list)
    for pair_id, value in zip(pair_ids, values):
        by_pair[str(pair_id)].append(value)
    pair_correct = [
        len(pair_values) == 2 and all(value > 0.0 for value in pair_values)
        for pair_values in by_pair.values()
    ]
    return {
        "rows": len(values),
        "mean_margin": sum(values) / max(len(values), 1),
        "min_margin": min(values) if values else None,
        "max_margin": max(values) if values else None,
        "target_preference_rate": (
            sum(value > 0.0 for value in values) / max(len(values), 1)
        ),
        "relation_flip_pair_accuracy": (
            sum(pair_correct) / max(len(pair_correct), 1)
        ),
        "pairs": len(pair_correct),
    }


def load_and_validate_dataset(
    dataset_path: Path, manifest_path: Path
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    rows = read_jsonl(dataset_path)
    manifest = require_mapping(load_json(manifest_path), "diagnostic manifest")
    if manifest.get("schema") != EXPECTED_DATASET_SCHEMA:
        raise SystemExit("fresh diagnostic manifest schema drift")
    if manifest.get("status") != EXPECTED_DATASET_STATUS:
        raise SystemExit("fresh diagnostic manifest status drift")
    if manifest.get("dataset_sha256") != sha256_file(dataset_path):
        raise SystemExit("fresh diagnostic dataset hash drift")
    if manifest.get("frozen_challenge_rows_reused") is not False:
        raise SystemExit("frozen challenge rows leaked into fresh diagnostic")
    if manifest.get("prior_diagnostic_rows_reused") is not False:
        raise SystemExit("prior diagnostic rows leaked into fresh diagnostic")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("fresh diagnostic may not authorize training")
    if len(rows) != 32:
        raise SystemExit(f"expected 32 fresh diagnostic rows, got {len(rows)}")
    return rows, manifest


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_result_trigger(path: Path) -> Mapping[str, Any]:
    receipt = require_mapping(load_json(path), "final challenge execution receipt")
    if receipt.get("status") != "COMPLETE":
        raise SystemExit("final challenge execution receipt is not COMPLETE")
    if receipt.get("model_conclusion") != (
        "FAIL_STOP_AND_LOCALIZE_WITHOUT_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("fresh localization is not bound to the expected failed challenge")
    if receipt.get("gate_pass") is not False:
        raise SystemExit("fresh localization requires gate_pass=false")
    if receipt.get("training_authorized") is not False:
        raise SystemExit("failed challenge unexpectedly authorized training")
    if receipt.get("scale_authorized") is not False:
        raise SystemExit("failed challenge unexpectedly authorized scaling")
    if receipt.get("n0_complete") is not False:
        raise SystemExit("failed challenge unexpectedly marked N0 complete")
    graph = require_mapping(
        receipt.get("selected_repaired_graph"), "selected_repaired_graph"
    )
    if graph.get("sha256") != EXPECTED_REPAIRED_GRAPH_SHA256:
        raise SystemExit("failed challenge did not use the selected repaired graph")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--trigger-receipt", required=True, type=Path)
    parser.add_argument("--latent-checkpoint", required=True, type=Path)
    parser.add_argument("--latent-config", required=True, type=Path)
    parser.add_argument("--semantic-config", required=True, type=Path)
    parser.add_argument("--semantic-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--structured-config", required=True, type=Path)
    parser.add_argument("--structured-checkpoint", required=True, type=Path)
    parser.add_argument("--evidence-adapter", required=True, type=Path)
    parser.add_argument("--repaired-graph", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-config", required=True, type=Path)
    parser.add_argument("--fusion-ratification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("fresh missing-evidence path localization requires CUDA")
    device = torch.device("cuda")

    dataset_path = args.dataset.resolve()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    rows, manifest = load_and_validate_dataset(dataset_path, manifest_path)
    trigger = verify_result_trigger(args.trigger_receipt.resolve())

    repaired_graph_path = args.repaired_graph.resolve()
    if sha256_file(repaired_graph_path) != EXPECTED_REPAIRED_GRAPH_SHA256:
        raise SystemExit("selected repaired graph hash drift")
    latent_path = args.latent_checkpoint.resolve()
    if sha256_file(latent_path) != EXPECTED_LATENT_SHA256:
        raise SystemExit("latent checkpoint hash drift")

    semantic_model, tokenizer, structured, adapter, graph = prior_eval.load_public_parents(
        semantic_config_path=args.semantic_config.resolve(),
        semantic_checkpoint=args.semantic_checkpoint.resolve(),
        tokenizer_dir=args.tokenizer_dir.resolve(),
        structured_config_path=args.structured_config.resolve(),
        structured_checkpoint=args.structured_checkpoint.resolve(),
        evidence_adapter_path=args.evidence_adapter.resolve(),
        evidence_graph_path=repaired_graph_path,
        device=device,
    )

    target_probe = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["diagnostic_target_probe_text"]) for row in rows],
        device=device,
        batch_size=32,
        max_length=64,
    ).float().cpu()
    foil_probe = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["diagnostic_foil_probe_text"]) for row in rows],
        device=device,
        batch_size=32,
        max_length=64,
    ).float().cpu()
    foil_summary = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["diagnostic_foil_summary_text"]) for row in rows],
        device=device,
        batch_size=32,
        max_length=96,
    ).float().cpu()

    graph_captures: list[dict[str, torch.Tensor]] = []
    handle = graph.register_forward_hook(capture_hook(graph_captures))
    try:
        normal_parent = build_parent_cache(
            rows,
            semantic_model=semantic_model,
            tokenizer=tokenizer,
            structured=structured,
            evidence_adapter=adapter,
            evidence_graph=graph,
            device=device,
            encode_batch_size=32,
            raw_max_length=96,
            field_max_length=96,
        )
    finally:
        handle.remove()

    counterfactual_rows = v02_eval.build_prefusion_counterfactual_rows(rows)
    counterfactual_parent = build_parent_cache(
        counterfactual_rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=adapter,
        evidence_graph=graph,
        device=device,
        encode_batch_size=32,
        raw_max_length=96,
        field_max_length=96,
    )

    graph_pooled = concat_capture(graph_captures, "pooled_state")
    graph_field_weights = concat_capture(graph_captures, "field_weights")
    relation_bias = concat_capture(graph_captures, "relation_status_bias")

    graph_parity = float(
        (
            graph_pooled
            - normal_parent["source_view_summaries"][:, 2].float()
        )
        .abs()
        .max()
    )
    if graph_parity > 1e-6:
        raise SystemExit(f"graph capture parity failure: {graph_parity}")

    pair_ids = [str(row["pair_id"]) for row in rows]
    target_indices = torch.tensor(
        [int(row["diagnostic_target_field_index"]) for row in rows],
        dtype=torch.long,
    )
    foil_indices = torch.tensor(
        [int(row["diagnostic_foil_field_index"]) for row in rows],
        dtype=torch.long,
    )
    row_index = torch.arange(len(rows), dtype=torch.long)

    target_field_weight = graph_field_weights[row_index, target_indices]
    foil_field_weight = graph_field_weights[row_index, foil_indices]
    graph_field_margin = target_field_weight - foil_field_weight
    graph_argmax = graph_field_weights.argmax(dim=-1)
    graph_argmax_target = graph_argmax.eq(target_indices)

    relation_target = relation_bias[row_index, target_indices]
    relation_foil = relation_bias[row_index, foil_indices]
    relation_bias_margin = relation_target - relation_foil

    raw_semantic_margin = vector_margin(
        normal_parent["source_view_summaries"][:, 0], target_probe, foil_probe
    )
    graph_pooled_margin = vector_margin(graph_pooled, target_probe, foil_probe)
    graph_pooled_summary_margin = vector_margin(
        graph_pooled, normal_parent["semantic_target"], foil_summary
    )

    fusion = load_ratified_fusion(
        checkpoint_dir=args.fusion_checkpoint.resolve(),
        fusion_config_path=args.fusion_config.resolve(),
        ratification_path=args.fusion_ratification.resolve(),
        device=device,
    )
    normal_fusion = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=normal_parent,
        device=device,
        batch_size=16,
    )
    counterfactual_fusion = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=counterfactual_parent,
        device=device,
        batch_size=16,
    )

    normal_semantic_context = masked_mean(
        normal_fusion["contextualized_view_tokens"][0],
        normal_parent["semantic_valid_mask"],
    )
    normal_evidence_context = masked_mean(
        normal_fusion["contextualized_view_tokens"][2],
        normal_parent["evidence_valid_mask"],
    )
    counterfactual_semantic_context = masked_mean(
        counterfactual_fusion["contextualized_view_tokens"][0],
        counterfactual_parent["semantic_valid_mask"],
    )

    fusion_semantic_margin = vector_margin(
        normal_semantic_context, target_probe, foil_probe
    )
    fusion_evidence_margin = vector_margin(
        normal_evidence_context, target_probe, foil_probe
    )
    fusion_semantic_counterfactual_margin = vector_margin(
        counterfactual_semantic_context, target_probe, foil_probe
    )
    fusion_semantic_transfer_delta = (
        fusion_semantic_margin - fusion_semantic_counterfactual_margin
    )
    normal_fusion_evidence_weight = normal_fusion["fusion_view_weights"][:, 2]
    counterfactual_fusion_evidence_weight = (
        counterfactual_fusion["fusion_view_weights"][:, 2]
    )

    latent_model = AdaptiveMultiViewLatentPoolV02(
        load_latent_config(args.latent_config.resolve())
    )
    latent_model.load_state_dict(
        load_file(str(latent_path), device="cpu"), strict=True
    )
    for parameter in latent_model.parameters():
        parameter.requires_grad = False
    latent_model = latent_model.to(device).eval()

    normal_latent = run_latent(
        model=latent_model,
        parent=normal_parent,
        fusion=normal_fusion,
        device=device,
        batch_size=16,
    )
    counterfactual_latent = run_latent(
        model=latent_model,
        parent=counterfactual_parent,
        fusion=counterfactual_fusion,
        device=device,
        batch_size=16,
    )

    normal_latent_probe_margin = slot_margin(
        normal_latent["latent_slots"], target_probe, foil_probe
    )
    counterfactual_latent_probe_margin = slot_margin(
        counterfactual_latent["latent_slots"], target_probe, foil_probe
    )
    latent_probe_margin_drop = (
        normal_latent_probe_margin - counterfactual_latent_probe_margin
    )

    normal_latent_summary_margin = slot_margin(
        normal_latent["latent_slots"],
        normal_parent["semantic_target"],
        foil_summary,
    )
    counterfactual_latent_summary_margin = slot_margin(
        counterfactual_latent["latent_slots"],
        normal_parent["semantic_target"],
        foil_summary,
    )
    latent_summary_margin_drop = (
        normal_latent_summary_margin - counterfactual_latent_summary_margin
    )

    normal_target_cosine = best_target_cosine(
        normal_latent["latent_slots"], normal_parent["semantic_target"]
    )
    counterfactual_target_cosine = best_target_cosine(
        counterfactual_latent["latent_slots"], normal_parent["semantic_target"]
    )
    absolute_target_cosine_drop = (
        normal_target_cosine - counterfactual_target_cosine
    )

    normal_pooled_probe_margin = vector_margin(
        normal_latent["pooled_state"], target_probe, foil_probe
    )
    counterfactual_pooled_probe_margin = vector_margin(
        counterfactual_latent["pooled_state"], target_probe, foil_probe
    )

    normal_evidence_attention = row_mean_view_attention(
        normal_latent["view_attention_mass"], 2
    )
    counterfactual_evidence_attention = row_mean_view_attention(
        counterfactual_latent["view_attention_mass"], 2
    )

    per_row: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        per_row.append(
            {
                "id": str(row["id"]),
                "pair_id": str(row["pair_id"]),
                "orientation": int(row["orientation"]),
                "target_value": str(row["diagnostic_target_value"]),
                "foil_value": str(row["diagnostic_foil_value"]),
                "target_field_index": int(target_indices[index]),
                "foil_field_index": int(foil_indices[index]),
                "graph_target_field_weight": float(target_field_weight[index]),
                "graph_foil_field_weight": float(foil_field_weight[index]),
                "graph_target_minus_foil_weight": float(graph_field_margin[index]),
                "graph_argmax_is_target_field": bool(graph_argmax_target[index]),
                "relation_bias_target_minus_foil": float(relation_bias_margin[index]),
                "raw_semantic_probe_margin": float(raw_semantic_margin[index]),
                "graph_pooled_probe_margin": float(graph_pooled_margin[index]),
                "graph_pooled_summary_margin": float(
                    graph_pooled_summary_margin[index]
                ),
                "fusion_semantic_probe_margin": float(fusion_semantic_margin[index]),
                "fusion_evidence_probe_margin": float(fusion_evidence_margin[index]),
                "fusion_semantic_counterfactual_probe_margin": float(
                    fusion_semantic_counterfactual_margin[index]
                ),
                "fusion_semantic_transfer_delta": float(
                    fusion_semantic_transfer_delta[index]
                ),
                "fusion_evidence_view_weight": float(
                    normal_fusion_evidence_weight[index]
                ),
                "counterfactual_fusion_evidence_view_weight": float(
                    counterfactual_fusion_evidence_weight[index]
                ),
                "latent_best_slot_probe_margin": float(
                    normal_latent_probe_margin[index]
                ),
                "counterfactual_latent_best_slot_probe_margin": float(
                    counterfactual_latent_probe_margin[index]
                ),
                "latent_probe_margin_drop": float(
                    latent_probe_margin_drop[index]
                ),
                "latent_best_slot_summary_margin": float(
                    normal_latent_summary_margin[index]
                ),
                "counterfactual_latent_best_slot_summary_margin": float(
                    counterfactual_latent_summary_margin[index]
                ),
                "latent_summary_margin_drop": float(
                    latent_summary_margin_drop[index]
                ),
                "absolute_target_cosine_drop": float(
                    absolute_target_cosine_drop[index]
                ),
                "latent_pooled_probe_margin": float(
                    normal_pooled_probe_margin[index]
                ),
                "counterfactual_latent_pooled_probe_margin": float(
                    counterfactual_pooled_probe_margin[index]
                ),
                "latent_evidence_view_attention": float(
                    normal_evidence_attention[index]
                ),
                "counterfactual_latent_evidence_view_attention": float(
                    counterfactual_evidence_attention[index]
                ),
            }
        )

    graph_field_stage = stage_summary(graph_field_margin, pair_ids)
    raw_stage = stage_summary(raw_semantic_margin, pair_ids)
    graph_pooled_stage = stage_summary(graph_pooled_margin, pair_ids)
    fusion_evidence_stage = stage_summary(fusion_evidence_margin, pair_ids)
    fusion_semantic_stage = stage_summary(fusion_semantic_margin, pair_ids)
    latent_stage = stage_summary(normal_latent_probe_margin, pair_ids)
    latent_cf_stage = stage_summary(counterfactual_latent_probe_margin, pair_ids)
    latent_pooled_stage = stage_summary(normal_pooled_probe_margin, pair_ids)

    graph_exact = (
        graph_field_stage["relation_flip_pair_accuracy"] == 1.0
        and float(graph_argmax_target.float().mean()) == 1.0
    )
    fusion_preserves = (
        fusion_evidence_stage["relation_flip_pair_accuracy"] == 1.0
    )
    latent_preserves = latent_stage["relation_flip_pair_accuracy"] == 1.0
    ablation_breaks_relation_signal = (
        latent_cf_stage["relation_flip_pair_accuracy"]
        < latent_stage["relation_flip_pair_accuracy"]
    )

    if not graph_exact:
        localization = "EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE"
    elif not fusion_preserves:
        localization = "RELATION_SIGNAL_PRESENT_AT_GRAPH_SELECTION_BUT_NOT_PRESERVED_BY_FUSION_VALUE_READOUT"
    elif not latent_preserves:
        localization = "RELATION_SIGNAL_PRESERVED_THROUGH_FUSION_BUT_NOT_BY_LATENT_VALUE_READOUT"
    elif ablation_breaks_relation_signal:
        localization = "FRESH_RELATION_SIGNAL_SURVIVES_GRAPH_FUSION_LATENT_AND_IS_CAUSALLY_USED;_FROZEN_ABSOLUTE_COSINE_GATE_REQUIRES_METRIC_INTERPRETATION"
    else:
        localization = "RELATION_SIGNAL_SURVIVES_BUT_PRE_FUSION_ABLATION_DID_NOT_REMOVE_PAIR_DISCRIMINATION;CHECK_LEAKAGE_OR_ALTERNATE_PATH"

    result = {
        "schema": "alice.eipm.n0.missing-evidence-path-localization-result.v0.1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": len(rows),
            "pairs": int(manifest["pairs"]),
            "frozen_challenge_rows_reused": False,
            "prior_diagnostic_rows_reused": False,
        },
        "trigger": {
            "execution_receipt_path": str(args.trigger_receipt.resolve()),
            "execution_receipt_sha256": sha256_file(args.trigger_receipt.resolve()),
            "model_conclusion": trigger.get("model_conclusion"),
        },
        "artifacts": {
            "repaired_graph_sha256": sha256_file(repaired_graph_path),
            "latent_checkpoint_sha256": sha256_file(latent_path),
        },
        "parity": {
            "graph_capture_to_parent_summary_max_abs_error": graph_parity,
            "normal_fusion_source_anchor_summary_max_abs_error": normal_fusion[
                "source_anchor_summary_max_abs_error"
            ],
            "normal_fusion_source_anchor_token_max_abs_error": normal_fusion[
                "source_anchor_token_max_abs_error"
            ],
            "counterfactual_fusion_source_anchor_summary_max_abs_error": (
                counterfactual_fusion["source_anchor_summary_max_abs_error"]
            ),
            "counterfactual_fusion_source_anchor_token_max_abs_error": (
                counterfactual_fusion["source_anchor_token_max_abs_error"]
            ),
        },
        "stage_summaries": {
            "raw_semantic_negative_control": raw_stage,
            "graph_field_selection": graph_field_stage,
            "graph_pooled_value_probe": graph_pooled_stage,
            "fusion_evidence_context_value_probe": fusion_evidence_stage,
            "fusion_semantic_context_value_probe": fusion_semantic_stage,
            "latent_best_slot_value_probe": latent_stage,
            "latent_best_slot_value_probe_without_evidence": latent_cf_stage,
            "latent_pooled_value_probe": latent_pooled_stage,
        },
        "aggregate_causal_metrics": {
            "graph_argmax_target_rate": float(graph_argmax_target.float().mean()),
            "mean_relation_bias_target_minus_foil": float(
                relation_bias_margin.mean()
            ),
            "mean_fusion_evidence_view_weight": float(
                normal_fusion_evidence_weight.mean()
            ),
            "mean_counterfactual_fusion_evidence_view_weight": float(
                counterfactual_fusion_evidence_weight.mean()
            ),
            "mean_fusion_semantic_transfer_delta": float(
                fusion_semantic_transfer_delta.mean()
            ),
            "mean_latent_probe_margin_drop": float(
                latent_probe_margin_drop.mean()
            ),
            "mean_latent_summary_margin_drop": float(
                latent_summary_margin_drop.mean()
            ),
            "mean_absolute_target_cosine_drop": float(
                absolute_target_cosine_drop.mean()
            ),
            "mean_latent_evidence_view_attention": float(
                normal_evidence_attention.mean()
            ),
            "mean_counterfactual_latent_evidence_view_attention": float(
                counterfactual_evidence_attention.mean()
            ),
        },
        "localization": localization,
        "interpretation_contract": {
            "graph_field_pair_flip_is_relation_direction_ground_truth": True,
            "full_sentence_cosine_is_not_used_as_the_only_localization_signal": True,
            "pair_text_query_fields_identical_except_relation_direction": True,
            "frozen_challenge_reclassified": False,
            "training_authorized": False,
            "gradient_performed": False,
            "ratification_effect": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "private_identity_data": False,
            "private_identity_gradient": False,
            "n0_complete": False,
        },
        "per_row": per_row,
    }

    if output_path.exists():
        raise SystemExit(f"refusing to overwrite diagnostic result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
