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

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2 as builder
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


def build_foil_texts(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, str]]]:
    builder.prior.base.SUBJECTS = builder.NEW_SUBJECTS
    builder.prior.base.ATTRIBUTES = builder.NEW_ATTRIBUTES
    texts: list[str] = []
    metadata: list[dict[str, str]] = []
    for item in rows:
        family = str(item["family"])
        suffix = int(str(item["id"]).rsplit("-", 1)[1]) - 1
        family_index = builder.prior.base.FAMILIES.index(family)
        subject, attribute, previous, current = builder.prior.base.parts(suffix, family_index)
        target = str(item["target_summary_text"])
        if current not in target:
            raise SystemExit(f"current value {current!r} not found in target for {item['id']}")
        foil = target.replace(current, previous, 1)
        if foil == target:
            raise SystemExit(f"foil construction failed for {item['id']}")
        texts.append(foil)
        metadata.append({
            "id": str(item["id"]),
            "family": family,
            "subject": subject,
            "attribute": attribute,
            "previous": previous,
            "current": current,
            "target_text": target,
            "foil_text": foil,
        })
    return texts, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
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
        raise SystemExit("value-contrastive diagnostic requires CUDA")
    device = torch.device("cuda")
    challenge_path = Path(args.challenge).resolve()
    result_path = Path(args.v02_result).resolve()
    candidate_path = Path(args.candidate).resolve()
    rows = read_jsonl(challenge_path)
    historical = json.loads(result_path.read_text(encoding="utf-8"))
    if historical.get("status") != "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION":
        raise SystemExit("expected immutable failed v0.2 result")
    if historical.get("gradient_performed") is not False:
        raise SystemExit("unexpected gradient in historical v0.2 challenge")
    if historical.get("candidate_latent_pool_sha256") != sha256_file(candidate_path):
        raise SystemExit("candidate hash differs from v0.2 result")

    counterfactual_rows = v02_eval.build_prefusion_counterfactual_rows(rows)
    foil_texts, foil_metadata = build_foil_texts(rows)

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
    normal_fusion = precompute_fusion_cache(fusion=fusion, parent_cache=normal_parent, device=device, batch_size=16)
    cf_fusion = precompute_fusion_cache(fusion=fusion, parent_cache=cf_parent, device=device, batch_size=16)
    del fusion
    torch.cuda.empty_cache()

    model = AdaptiveMultiViewLatentPoolV02(load_latent_config(Path(args.latent_config).resolve()))
    model.load_state_dict(load_file(str(candidate_path), device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device).eval()

    indices = list(range(len(rows)))
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
            ids = [int(v) for v in normal_batch["global_index"].cpu().tolist()]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                normal_out = latent_forward(model, normal_batch)
                cf_out = latent_forward(model, cf_batch)

            target = normal_batch["semantic_target"].float()
            foil = foil_semantic[ids].to(device)
            normal_slots = normal_out["latent_slots"].float()
            cf_slots = cf_out["latent_slots"].float()
            normal_pooled = normal_out["pooled_state"].float()
            cf_pooled = cf_out["pooled_state"].float()

            target_n = F.normalize(target, dim=-1).unsqueeze(1)
            foil_n = F.normalize(foil, dim=-1).unsqueeze(1)
            normal_target = (F.normalize(normal_slots, dim=-1) * target_n).sum(dim=-1).max(dim=-1).values
            normal_foil = (F.normalize(normal_slots, dim=-1) * foil_n).sum(dim=-1).max(dim=-1).values
            cf_target = (F.normalize(cf_slots, dim=-1) * target_n).sum(dim=-1).max(dim=-1).values
            cf_foil = (F.normalize(cf_slots, dim=-1) * foil_n).sum(dim=-1).max(dim=-1).values
            normal_best_margin = normal_target - normal_foil
            cf_best_margin = cf_target - cf_foil
            best_margin_drop = normal_best_margin - cf_best_margin

            normal_pool_margin = cosine(normal_pooled, target) - cosine(normal_pooled, foil)
            cf_pool_margin = cosine(cf_pooled, target) - cosine(cf_pooled, foil)
            pooled_margin_drop = normal_pool_margin - cf_pool_margin

            for local, global_index in enumerate(ids):
                row = rows[global_index]
                if not row.get("counterfactual_required"):
                    continue
                family = str(row["family"])
                view = int(row["counterfactual_view"])
                source = normal_parent["source_view_summaries"][global_index, view].to(device)
                source_margin = float((cosine(source.unsqueeze(0), target[local].unsqueeze(0)) - cosine(source.unsqueeze(0), foil[local].unsqueeze(0))).cpu())
                values = {
                    "normal_best_target_cosine": float(normal_target[local].cpu()),
                    "counterfactual_best_target_cosine": float(cf_target[local].cpu()),
                    "absolute_target_cosine_drop": float((normal_target[local] - cf_target[local]).cpu()),
                    "normal_best_target_vs_foil_margin": float(normal_best_margin[local].cpu()),
                    "counterfactual_best_target_vs_foil_margin": float(cf_best_margin[local].cpu()),
                    "best_target_vs_foil_margin_drop": float(best_margin_drop[local].cpu()),
                    "normal_pooled_target_vs_foil_margin": float(normal_pool_margin[local].cpu()),
                    "counterfactual_pooled_target_vs_foil_margin": float(cf_pool_margin[local].cpu()),
                    "pooled_target_vs_foil_margin_drop": float(pooled_margin_drop[local].cpu()),
                    "required_source_target_vs_foil_margin": source_margin,
                }
                for key, value in values.items():
                    per_family[family][key].append(value)
                per_row.append({**foil_metadata[global_index], "counterfactual_view": view, **values})

    family_summary: dict[str, Any] = {}
    for family, metrics in sorted(per_family.items()):
        family_summary[family] = {key: mean(values) for key, values in sorted(metrics.items())}
        family_summary[family]["rows"] = len(next(iter(metrics.values())))

    output = {
        "schema": "alice.eipm.n0.v02-latent-pool-counterfactual-value-contrast-diagnostic.v0.1",
        "status": "DIAGNOSTIC_ONLY_NO_RATIFICATION_EFFECT",
        "historical_v0_2_result_sha256": sha256_file(result_path),
        "candidate_latent_pool_sha256": sha256_file(candidate_path),
        "challenge_sha256": sha256_file(challenge_path),
        "rows": len(per_row),
        "families": family_summary,
        "per_row": per_row,
        "interpretation_contract": {
            "historical_v0_2_result_reclassified": False,
            "challenge_rows_used_for_training": False,
            "gradient_performed": False,
            "purpose": "distinguish_absolute_sentence_cosine_insensitivity_from_value_level_causal_failure",
            "evidence_for_metric_insensitivity": "required source has positive target-vs-foil margin and latent target-vs-foil margin drops after pre-fusion ablation even when absolute target cosine does not",
            "evidence_for_real_causal_failure": "required source is value-discriminative but latent target-vs-foil margin does not decrease after pre-fusion ablation",
        },
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    out_path = Path(args.output).resolve()
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite diagnostic output: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
