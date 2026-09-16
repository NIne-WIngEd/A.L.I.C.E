#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 as prior_eval
from train_n0_v02_cross_context_fusion_full_scale import (
    build_parent_cache,
    encode_pooled,
    read_jsonl,
    sha256_file,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=-1).cpu())


def margin(vector: torch.Tensor, target: torch.Tensor, foil: torch.Tensor) -> float:
    return cosine(vector, target) - cosine(vector, foil)


def contains_value(text: str, value: str) -> bool:
    # Values such as `4` must not match `34`; multi-word values still work.
    pattern = rf"(?<!\w){re.escape(value.strip())}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def field_indices(fields: list[dict[str, Any]], value: str) -> list[int]:
    return [i for i, item in enumerate(fields) if contains_value(str(item["text"]), value)]


def capture_hook(store: dict[str, list[dict[str, torch.Tensor]]], name: str):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, dict):
            raise RuntimeError(f"{name} output is not a dict")
        captured: dict[str, torch.Tensor] = {}
        for key, value in output.items():
            if torch.is_tensor(value):
                captured[key] = value.detach().float().cpu()
        store[name].append(captured)
    return hook


def concat_capture(store: dict[str, list[dict[str, torch.Tensor]]], name: str, key: str) -> torch.Tensor:
    chunks = [item[key] for item in store[name] if key in item]
    if not chunks:
        raise RuntimeError(f"missing capture {name}.{key}")
    if chunks[0].ndim == 0:
        return chunks[-1]
    return torch.cat(chunks, dim=0)


def max_field_margin(states: torch.Tensor, indices: list[int], target: torch.Tensor, foil: torch.Tensor) -> float | None:
    if not indices:
        return None
    return max(margin(states[index], target, foil) for index in indices)


def max_foil_margin(states: torch.Tensor, indices: list[int], target: torch.Tensor, foil: torch.Tensor) -> float | None:
    if not indices:
        return None
    return max(margin(states[index], foil, target) for index in indices)


def weight_mass(weights: torch.Tensor, indices: list[int]) -> float | None:
    if not indices:
        return None
    return float(weights[indices].sum().cpu())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                numeric[key].append(float(value))
            elif isinstance(value, (int, float)) and key not in {"required_view"}:
                numeric[key].append(float(value))
    out = {key: sum(values) / len(values) for key, values in sorted(numeric.items()) if values}
    out["rows"] = len(rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--contrast-contract", required=True)
    parser.add_argument("--contrast-manifest", required=True)
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--evidence-adapter", required=True)
    parser.add_argument("--evidence-graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("parent value-path diagnostic requires CUDA")
    device = torch.device("cuda")

    challenge_path = Path(args.challenge).resolve()
    contract_path = Path(args.contrast_contract).resolve()
    manifest_path = Path(args.contrast_manifest).resolve()
    output_path = Path(args.output).resolve()

    rows_all = read_jsonl(challenge_path)
    cases = read_jsonl(contract_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPILED_DIAGNOSTIC_CONTRACT_NOT_EVALUATED":
        raise SystemExit("value-contrast manifest is not the frozen compiled contract")
    if manifest.get("challenge_sha256") != sha256_file(challenge_path):
        raise SystemExit("value-contrast contract is not bound to this challenge")
    if manifest.get("contrast_contract_sha256") != sha256_file(contract_path):
        raise SystemExit("value-contrast contract hash drift")
    if len(cases) != 32:
        raise SystemExit(f"expected 32 contrast cases, observed {len(cases)}")

    rows_by_id = {str(row["id"]): row for row in rows_all}
    selected_rows: list[dict[str, Any]] = []
    for case in cases:
        row_id = str(case["row_id"])
        row = rows_by_id.get(row_id)
        if row is None:
            raise SystemExit(f"contrast row missing from challenge: {row_id}")
        if not bool(row.get("counterfactual_required")):
            raise SystemExit(f"non-counterfactual row leaked into parent diagnostic: {row_id}")
        if str(case["target_text"]) != str(row["target_summary_text"]):
            raise SystemExit(f"target text drift for {row_id}")
        selected_rows.append(row)

    semantic_model, tokenizer, structured, adapter, graph = prior_eval.load_public_parents(
        semantic_config_path=Path(args.semantic_config).resolve(),
        semantic_checkpoint=Path(args.semantic_checkpoint).resolve(),
        tokenizer_dir=Path(args.tokenizer_dir).resolve(),
        structured_config_path=Path(args.structured_config).resolve(),
        structured_checkpoint=Path(args.structured_checkpoint).resolve(),
        evidence_adapter_path=Path(args.evidence_adapter).resolve(),
        evidence_graph_path=Path(args.evidence_graph).resolve(),
        device=device,
    )

    captures: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    handles = [
        structured.register_forward_hook(capture_hook(captures, "structured")),
        adapter.register_forward_hook(capture_hook(captures, "adapter")),
        graph.register_forward_hook(capture_hook(captures, "graph")),
    ]
    try:
        parent = build_parent_cache(
            selected_rows,
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
        for handle in handles:
            handle.remove()

    structured_fields = concat_capture(captures, "structured", "field_states")
    structured_pooled = concat_capture(captures, "structured", "pooled_state")
    structured_weights = concat_capture(captures, "structured", "field_weights")
    adapter_fields = concat_capture(captures, "adapter", "field_states")
    adapter_pooled = concat_capture(captures, "adapter", "pooled_state")
    adapter_weights = concat_capture(captures, "adapter", "field_weights")
    graph_fields = concat_capture(captures, "graph", "field_states")
    graph_pooled = concat_capture(captures, "graph", "pooled_state")
    graph_weights = concat_capture(captures, "graph", "field_weights")
    relation_bias = concat_capture(captures, "graph", "relation_status_bias")

    # The traced path must be byte-for-byte equivalent in meaning to the canonical
    # parent-cache path before any diagnostic interpretation is permitted.
    parity = {
        "structured_summary_max_abs_error": float((structured_pooled - parent["source_view_summaries"][:, 1]).abs().max()),
        "evidence_summary_max_abs_error": float((graph_pooled - parent["source_view_summaries"][:, 2]).abs().max()),
    }
    if parity["structured_summary_max_abs_error"] > 1e-6 or parity["evidence_summary_max_abs_error"] > 1e-6:
        raise SystemExit(f"parent trace parity failure: {parity}")

    raw_pooled = encode_pooled(
        semantic_model,
        tokenizer,
        [str(row["raw_text"]) for row in selected_rows],
        device=device,
        batch_size=32,
        max_length=96,
    ).float().cpu()
    foil_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        [str(case["foil_text"]) for case in cases],
        device=device,
        batch_size=32,
        max_length=96,
    ).float().cpu()
    flat_field_texts = [str(field["text"]) for row in selected_rows for field in row["fields"]]
    flat_field_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        flat_field_texts,
        device=device,
        batch_size=32,
        max_length=96,
    ).float().cpu()

    field_offsets: list[tuple[int, int]] = []
    cursor = 0
    for row in selected_rows:
        start = cursor
        cursor += len(row["fields"])
        field_offsets.append((start, cursor))

    per_row: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, (row, case) in enumerate(zip(selected_rows, cases)):
        target = parent["semantic_target"][i].float()
        foil = foil_semantic[i].float()
        required_view = int(case["counterfactual_view"])
        target_value = str(case["target_value"])
        foil_value = str(case["foil_value"])
        fields = list(row["fields"])
        target_indices = field_indices(fields, target_value)
        foil_indices = field_indices(fields, foil_value)
        start, end = field_offsets[i]
        raw_fields = flat_field_semantic[start:end]
        valid_count = len(fields)

        sw = structured_weights[i, :valid_count]
        aw = adapter_weights[i, :valid_count]
        gw = graph_weights[i, :valid_count]
        rb = relation_bias[i, :valid_count]

        source_margin = margin(parent["source_view_summaries"][i, required_view], target, foil)
        target_graph_mass = weight_mass(gw, target_indices)
        foil_graph_mass = weight_mass(gw, foil_indices)
        target_bias_mass = weight_mass(rb, target_indices)
        foil_bias_mass = weight_mass(rb, foil_indices)

        record: dict[str, Any] = {
            "row_id": str(case["row_id"]),
            "family": str(case["family"]),
            "required_view": required_view,
            "target_value": target_value,
            "foil_value": foil_value,
            "target_value_present_in_raw_text": contains_value(str(row["raw_text"]), target_value),
            "foil_value_present_in_raw_text": contains_value(str(row["raw_text"]), foil_value),
            "target_field_indices": target_indices,
            "foil_field_indices": foil_indices,
            "target_foil_semantic_separation": 1.0 - cosine(target, foil),
            "canonical_required_source_margin": source_margin,
            "canonical_required_source_prefers_target": source_margin > 0.0,
            "raw_text_masked_mean_margin": margin(parent["source_view_summaries"][i, 0], target, foil),
            "raw_text_pooled_encoder_margin": margin(raw_pooled[i], target, foil),
            "structured_pooled_margin": margin(structured_pooled[i], target, foil),
            "adapter_pooled_margin": margin(adapter_pooled[i], target, foil),
            "graph_pooled_margin": margin(graph_pooled[i], target, foil),
            "raw_target_field_semantic_margin": max_field_margin(raw_fields, target_indices, target, foil),
            "raw_foil_field_inverse_margin": max_foil_margin(raw_fields, foil_indices, target, foil),
            "structured_target_field_margin": max_field_margin(structured_fields[i, :valid_count], target_indices, target, foil),
            "adapter_target_field_margin": max_field_margin(adapter_fields[i, :valid_count], target_indices, target, foil),
            "graph_target_field_margin": max_field_margin(graph_fields[i, :valid_count], target_indices, target, foil),
            "structured_target_field_weight": weight_mass(sw, target_indices),
            "structured_foil_field_weight": weight_mass(sw, foil_indices),
            "adapter_target_field_weight": weight_mass(aw, target_indices),
            "adapter_foil_field_weight": weight_mass(aw, foil_indices),
            "graph_target_field_weight": target_graph_mass,
            "graph_foil_field_weight": foil_graph_mass,
            "relation_bias_target_mass": target_bias_mass,
            "relation_bias_foil_mass": foil_bias_mass,
        }
        if target_graph_mass is not None and foil_graph_mass is not None:
            record["graph_target_minus_foil_weight"] = target_graph_mass - foil_graph_mass
            record["graph_prefers_target_field"] = target_graph_mass > foil_graph_mass
        if target_bias_mass is not None and foil_bias_mass is not None:
            record["relation_bias_target_minus_foil"] = target_bias_mass - foil_bias_mass
            record["relation_bias_prefers_target_field"] = target_bias_mass > foil_bias_mass
        if target_indices:
            record["graph_argmax_is_target_field"] = int(torch.argmax(gw).item()) in set(target_indices)
            record["adapter_argmax_is_target_field"] = int(torch.argmax(aw).item()) in set(target_indices)
            record["structured_argmax_is_target_field"] = int(torch.argmax(sw).item()) in set(target_indices)

        per_row.append(record)
        by_family[record["family"]].append(record)

    family_summary = {family: summarize(items) for family, items in sorted(by_family.items())}
    overall = summarize(per_row)
    missing = family_summary.get("missing_structured_evidence", {})

    output = {
        "schema": "alice.eipm.n0.v02-parent-value-path-diagnostic.v0.1",
        "status": "DIAGNOSTIC_ONLY_NO_RATIFICATION_EFFECT",
        "challenge_sha256": sha256_file(challenge_path),
        "contrast_contract_sha256": sha256_file(contract_path),
        "contrast_manifest_sha256": sha256_file(manifest_path),
        "rows": len(per_row),
        "parent_trace_parity": parity,
        "families": family_summary,
        "overall": overall,
        "localization_signals": {
            "missing_structured_graph_target_field_preference_rate": missing.get("graph_prefers_target_field"),
            "missing_structured_graph_argmax_target_rate": missing.get("graph_argmax_is_target_field"),
            "missing_structured_relation_bias_target_preference_rate": missing.get("relation_bias_prefers_target_field"),
            "missing_structured_graph_target_minus_foil_weight": missing.get("graph_target_minus_foil_weight"),
            "missing_structured_relation_bias_target_minus_foil": missing.get("relation_bias_target_minus_foil"),
            "missing_structured_graph_pooled_margin": missing.get("graph_pooled_margin"),
            "semantic_required_families_raw_masked_mean_margin": {
                family: summary.get("raw_text_masked_mean_margin")
                for family, summary in family_summary.items()
                if family in {"novel_semantic_authority", "semantic_reliability_reversal"}
            },
            "semantic_required_families_raw_pooled_encoder_margin": {
                family: summary.get("raw_text_pooled_encoder_margin")
                for family, summary in family_summary.items()
                if family in {"novel_semantic_authority", "semantic_reliability_reversal"}
            },
        },
        "interpretation_contract": {
            "historical_latent_challenge_reclassified": False,
            "latent_weights_changed": False,
            "challenge_rows_used_for_training": False,
            "gradient_performed": False,
            "ratification_effect": False,
            "purpose": "localize_value_discrimination_or_query_conditioned_selection_failure_before_latent_repair_or_scale",
            "graph_selection_defect_signal": "target_and_foil_fields_exist_but_graph_field_weight_or_relation_status_bias_does_not_reliably_prefer_target_for_the_query",
            "representation_metric_mismatch_signal": "graph_selects_target_field_reliably_but_graph_pooled_sentence_cosine_margin_is_weak_or_inverted",
            "semantic_pooling_defect_signal": "pooled_semantic_encoder_prefers_target_but_masked_mean_semantic_source_summary_does_not",
            "semantic_value_geometry_defect_signal": "direct_pooled_semantic_raw_or_target_field_representation_does_not_reliably_distinguish_target_from_foil",
            "scale_decision_from_this_diagnostic_alone": False,
        },
        "private_identity_data": False,
        "private_identity_gradient": False,
    }

    if output_path.exists():
        raise SystemExit(f"refusing to overwrite parent value-path diagnostic: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
