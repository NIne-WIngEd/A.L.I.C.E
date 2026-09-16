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
from torch.utils.data import DataLoader

import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 as prior_eval
import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2 as v02_eval
from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import AdaptiveMultiViewLatentPoolV02
from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
    LatentPoolDataset,
    latent_forward,
    load_ratified_fusion,
    precompute_fusion_cache,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import load_config as load_latent_config
from train_n0_v02_cross_context_fusion_full_scale import (
    build_parent_cache,
    encode_pooled,
    read_jsonl,
    sha256_file,
    to_device,
)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(dim=-1)


def summarize(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: mean(item) for key, item in sorted(values.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--contrast-contract", required=True)
    parser.add_argument("--contrast-manifest", required=True)
    parser.add_argument("--v02-result", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--latent-config", required=True)
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--evidence-adapter", required=True)
    parser.add_argument("--evidence-graph", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--fusion-ratification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("value-contrast diagnostic v0.2 requires CUDA")
    device = torch.device("cuda")

    challenge_path = Path(args.challenge).resolve()
    contract_path = Path(args.contrast_contract).resolve()
    contract_manifest_path = Path(args.contrast_manifest).resolve()
    result_path = Path(args.v02_result).resolve()
    candidate_path = Path(args.candidate).resolve()
    output_path = Path(args.output).resolve()

    rows = read_jsonl(challenge_path)
    cases = read_jsonl(contract_path)
    contract_manifest = json.loads(contract_manifest_path.read_text(encoding="utf-8"))
    historical = json.loads(result_path.read_text(encoding="utf-8"))

    if historical.get("status") != "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION":
        raise SystemExit("expected immutable failed v0.2 result")
    if historical.get("gradient_performed") is not False:
        raise SystemExit("unexpected gradient in historical v0.2 challenge")
    if historical.get("candidate_latent_pool_sha256") != sha256_file(candidate_path):
        raise SystemExit("candidate hash differs from v0.2 result")
    if contract_manifest.get("status") != "COMPILED_DIAGNOSTIC_CONTRACT_NOT_EVALUATED":
        raise SystemExit("contrast contract manifest status mismatch")
    if contract_manifest.get("challenge_sha256") != sha256_file(challenge_path):
        raise SystemExit("contrast contract is not bound to this frozen challenge")
    if contract_manifest.get("contrast_contract_sha256") != sha256_file(contract_path):
        raise SystemExit("contrast contract hash mismatch")
    if contract_manifest.get("global_current_value_assumption") is not False:
        raise SystemExit("contrast contract incorrectly permits a global current-value assumption")
    if contract_manifest.get("ratification_effect") is not False:
        raise SystemExit("diagnostic contract cannot affect ratification")

    rows_by_id = {str(row["id"]): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise SystemExit("challenge row ids are not unique")
    case_ids = [str(case["row_id"]) for case in cases]
    if len(cases) != 32 or len(set(case_ids)) != 32:
        raise SystemExit(f"expected 32 unique diagnostic cases, got {len(cases)}")
    selected_rows: list[dict[str, Any]] = []
    for case in cases:
        row_id = str(case["row_id"])
        if row_id not in rows_by_id:
            raise SystemExit(f"contrast row missing from challenge: {row_id}")
        row = rows_by_id[row_id]
        if not bool(row.get("counterfactual_required")):
            raise SystemExit(f"contrast contract includes non-counterfactual row: {row_id}")
        if str(case["target_text"]) != str(row["target_summary_text"]):
            raise SystemExit(f"contrast target text drift for {row_id}")
        if int(case["counterfactual_view"]) != int(row["counterfactual_view"]):
            raise SystemExit(f"contrast counterfactual view drift for {row_id}")
        selected_rows.append(row)

    counterfactual_rows = v02_eval.build_prefusion_counterfactual_rows(selected_rows)
    foil_texts = [str(case["foil_text"]) for case in cases]

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
    foil_semantic = encode_pooled(
        semantic_model,
        tokenizer,
        foil_texts,
        device=device,
        batch_size=32,
        max_length=96,
    ).float().cpu()
    normal_parent = build_parent_cache(
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
    cf_parent = build_parent_cache(
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
    del semantic_model, structured, adapter, graph
    torch.cuda.empty_cache()

    fusion = load_ratified_fusion(
        checkpoint_dir=Path(args.fusion_checkpoint).resolve(),
        fusion_config_path=Path(args.fusion_config).resolve(),
        ratification_path=Path(args.fusion_ratification).resolve(),
        device=device,
    )
    normal_fusion = precompute_fusion_cache(
        fusion=fusion, parent_cache=normal_parent, device=device, batch_size=16
    )
    cf_fusion = precompute_fusion_cache(
        fusion=fusion, parent_cache=cf_parent, device=device, batch_size=16
    )
    del fusion
    torch.cuda.empty_cache()

    model = AdaptiveMultiViewLatentPoolV02(load_latent_config(Path(args.latent_config).resolve()))
    model.load_state_dict(load_file(str(candidate_path), device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device).eval()

    indices = list(range(len(selected_rows)))
    normal_ds = LatentPoolDataset(normal_parent, normal_fusion, indices)
    cf_ds = LatentPoolDataset(cf_parent, cf_fusion, indices)
    normal_loader = DataLoader(normal_ds, batch_size=16, shuffle=False, num_workers=0)
    cf_loader = DataLoader(cf_ds, batch_size=16, shuffle=False, num_workers=0)

    per_family: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_row: list[dict[str, Any]] = []

    with torch.inference_mode():
        for normal_batch, cf_batch in zip(normal_loader, cf_loader):
            normal_batch = to_device(normal_batch, device)
            cf_batch = to_device(cf_batch, device)
            batch_indices = [int(value) for value in normal_batch["global_index"].cpu().tolist()]
            cf_indices = [int(value) for value in cf_batch["global_index"].cpu().tolist()]
            if batch_indices != cf_indices:
                raise RuntimeError("normal/counterfactual diagnostic ordering drift")

            with torch.amp.autocast("cuda", dtype=torch.float16):
                normal_out = latent_forward(model, normal_batch)
                cf_out = latent_forward(model, cf_batch)

            target = normal_batch["semantic_target"].float()
            foil = foil_semantic[batch_indices].to(device)
            target_foil_cos = cosine(target, foil)

            normal_slots = F.normalize(normal_out["latent_slots"].float(), dim=-1)
            cf_slots = F.normalize(cf_out["latent_slots"].float(), dim=-1)
            target_n = F.normalize(target, dim=-1).unsqueeze(1)
            foil_n = F.normalize(foil, dim=-1).unsqueeze(1)

            normal_slot_target = (normal_slots * target_n).sum(dim=-1)
            normal_slot_foil = (normal_slots * foil_n).sum(dim=-1)
            cf_slot_target = (cf_slots * target_n).sum(dim=-1)
            cf_slot_foil = (cf_slots * foil_n).sum(dim=-1)

            normal_best_target = normal_slot_target.max(dim=-1).values
            normal_best_foil = normal_slot_foil.max(dim=-1).values
            cf_best_target = cf_slot_target.max(dim=-1).values
            cf_best_foil = cf_slot_foil.max(dim=-1).values
            normal_separate_margin = normal_best_target - normal_best_foil
            cf_separate_margin = cf_best_target - cf_best_foil

            # Permutation-free set contrast: any latent slot may carry the
            # target-vs-foil distinction. This avoids assigning a fixed meaning
            # to a slot while still asking whether the set preserves the value.
            normal_set_margin = (normal_slot_target - normal_slot_foil).max(dim=-1).values
            cf_set_margin = (cf_slot_target - cf_slot_foil).max(dim=-1).values

            normal_pooled = normal_out["pooled_state"].float()
            cf_pooled = cf_out["pooled_state"].float()
            normal_pool_margin = cosine(normal_pooled, target) - cosine(normal_pooled, foil)
            cf_pool_margin = cosine(cf_pooled, target) - cosine(cf_pooled, foil)

            for local, index in enumerate(batch_indices):
                case = cases[index]
                family = str(case["family"])
                view = int(case["counterfactual_view"])
                source = normal_parent["source_view_summaries"][index, view].to(device)
                source_margin = float(
                    (
                        cosine(source.unsqueeze(0), target[local].unsqueeze(0))
                        - cosine(source.unsqueeze(0), foil[local].unsqueeze(0))
                    ).cpu()
                )
                values = {
                    "target_foil_semantic_cosine": float(target_foil_cos[local].cpu()),
                    "target_foil_semantic_separation": float((1.0 - target_foil_cos[local]).cpu()),
                    "required_source_target_vs_foil_margin": source_margin,
                    "normal_best_target_cosine": float(normal_best_target[local].cpu()),
                    "counterfactual_best_target_cosine": float(cf_best_target[local].cpu()),
                    "absolute_target_cosine_drop": float((normal_best_target[local] - cf_best_target[local]).cpu()),
                    "normal_best_separate_target_vs_foil_margin": float(normal_separate_margin[local].cpu()),
                    "counterfactual_best_separate_target_vs_foil_margin": float(cf_separate_margin[local].cpu()),
                    "best_separate_target_vs_foil_margin_drop": float((normal_separate_margin[local] - cf_separate_margin[local]).cpu()),
                    "normal_set_target_vs_foil_margin": float(normal_set_margin[local].cpu()),
                    "counterfactual_set_target_vs_foil_margin": float(cf_set_margin[local].cpu()),
                    "set_target_vs_foil_margin_drop": float((normal_set_margin[local] - cf_set_margin[local]).cpu()),
                    "normal_pooled_target_vs_foil_margin": float(normal_pool_margin[local].cpu()),
                    "counterfactual_pooled_target_vs_foil_margin": float(cf_pool_margin[local].cpu()),
                    "pooled_target_vs_foil_margin_drop": float((normal_pool_margin[local] - cf_pool_margin[local]).cpu()),
                    "required_source_prefers_target": float(source_margin > 0.0),
                    "normal_set_prefers_target": float(normal_set_margin[local].cpu() > 0.0),
                    "counterfactual_set_prefers_target": float(cf_set_margin[local].cpu() > 0.0),
                    "normal_pooled_prefers_target": float(normal_pool_margin[local].cpu() > 0.0),
                    "counterfactual_pooled_prefers_target": float(cf_pool_margin[local].cpu() > 0.0),
                }
                for key, value in values.items():
                    per_family[family][key].append(value)
                per_row.append({**case, **values})

    family_summary: dict[str, Any] = {}
    for family, metrics in sorted(per_family.items()):
        family_summary[family] = summarize(metrics)
        family_summary[family]["rows"] = len(next(iter(metrics.values())))

    overall_values: dict[str, list[float]] = defaultdict(list)
    for row in per_row:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in {"counterfactual_view"}:
                overall_values[key].append(float(value))

    output = {
        "schema": "alice.eipm.n0.v02-latent-pool-counterfactual-value-contrast-diagnostic.v0.2",
        "status": "DIAGNOSTIC_ONLY_NO_RATIFICATION_EFFECT",
        "historical_v0_2_result_sha256": sha256_file(result_path),
        "candidate_latent_pool_sha256": sha256_file(candidate_path),
        "challenge_sha256": sha256_file(challenge_path),
        "contrast_contract_sha256": sha256_file(contract_path),
        "contrast_manifest_sha256": sha256_file(contract_manifest_path),
        "rows": len(per_row),
        "families": family_summary,
        "overall": summarize(overall_values),
        "per_row": per_row,
        "interpretation_contract": {
            "historical_v0_2_result_reclassified": False,
            "challenge_rows_used_for_training": False,
            "gradient_performed": False,
            "global_current_value_assumption": False,
            "query_conditioned_contrast_contract": True,
            "purpose": "trace value discrimination from semantic target and required source through normal latent state and pre_fusion_ablation",
            "metric_insensitivity_evidence": "target_and_foil_are_semantically_separable_required_source_prefers_target_normal_latent_prefers_target_and_ablation_reduces_target_vs_foil_margin_even_if_absolute_target_cosine_drop_is_small",
            "real_causal_failure_evidence": "target_and_foil_are_semantically_separable_required_source_prefers_target_but_normal_latent_does_not_preserve_the_margin_or_prefusion_ablation_does_not_reduce_it",
            "scale_decision_from_this_diagnostic_alone": False,
        },
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite diagnostic v0.2 output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
