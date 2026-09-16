#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

import train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale as base
from alice_personality.n0.adaptive_multi_view_latent_pool_objectives_v0_2 import (
    LatentPoolV02ObjectiveWeights,
    adaptive_latent_pool_v0_2_objective,
    centered_slot_effective_rank,
    counterfactual_target_margin_loss,
    normalized_view_specialization,
    source_view_best_slot_cosine,
    source_view_disagreement_weight,
)
from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
)
from train_n0_v02_cross_context_fusion_full_scale import git_revision, sha256_file, to_device


def corrected_loss_bundle(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    batch: dict[str, torch.Tensor],
    weights: LatentPoolV02ObjectiveWeights,
) -> dict[str, torch.Tensor]:
    output = base.latent_forward(model, batch)
    losses = adaptive_latent_pool_v0_2_objective(
        latent_slots=output["latent_slots"],
        pooled_state=output["pooled_state"],
        target_semantic=batch["semantic_target"],
        source_view_summaries=batch["source_view_summaries"],
        view_attention_mass=output["view_attention_mass"],
        view_available=output["view_available"],
        channel_attention_mass=output["channel_attention_mass"],
        counterfactual_slots=None,
        weights=weights,
    )
    decisive, best_view = base.decisive_view_mask(batch)
    counterfactual_output = base.counterfactual_forward(model, batch, decisive, best_view)
    if counterfactual_output is None:
        counterfactual = output["latent_slots"].sum() * 0.0
    else:
        counterfactual = counterfactual_target_margin_loss(
            output["latent_slots"][decisive],
            counterfactual_output["latent_slots"][decisive],
            batch["semantic_target"][decisive],
        )
    total = losses["total"] + weights.counterfactual_margin * counterfactual
    return {
        **losses,
        "counterfactual_margin": counterfactual,
        "total": total,
    }


def _offdiag_values(slot_cosine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    slots = slot_cosine.shape[1]
    eye = torch.eye(slots, device=slot_cosine.device, dtype=torch.bool).unsqueeze(0)
    selected = slot_cosine.masked_select(~eye.expand_as(slot_cosine)).view(slot_cosine.shape[0], -1)
    return selected.mean(dim=-1), selected.max(dim=-1).values


def corrected_evaluate(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    dataset: Any,
    parent_cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    best_values: list[float] = []
    pooled_values: list[float] = []
    offdiag_mean_values: list[float] = []
    offdiag_max_values: list[float] = []
    effective_rank_values: list[float] = []
    raw_specialization_values: list[float] = []
    source_semantic_coverage_values: list[float] = []
    entropy_values: list[float] = []
    view_coverage_values: list[float] = []
    channel_coverage_values: list[float] = []
    family_best: dict[str, list[float]] = defaultdict(list)
    family_pooled: dict[str, list[float]] = defaultdict(list)
    missing_attention = 0.0
    specialization_weighted_sum = 0.0
    specialization_weight_sum = 0.0

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = base.latent_forward(model, batch)

            target = F.normalize(batch["semantic_target"], dim=-1).unsqueeze(1)
            slot_cos = (F.normalize(output["latent_slots"], dim=-1) * target).sum(dim=-1)
            best = slot_cos.max(dim=-1).values
            pooled = F.cosine_similarity(output["pooled_state"], batch["semantic_target"], dim=-1)
            offdiag_mean, offdiag_max = _offdiag_values(output["slot_cosine"])
            effective_rank = centered_slot_effective_rank(output["latent_slots"])
            specialization = normalized_view_specialization(
                output["view_attention_mass"], batch["view_available"]
            )
            disagreement = source_view_disagreement_weight(
                batch["source_view_summaries"], batch["view_available"]
            )
            source_best = source_view_best_slot_cosine(
                output["latent_slots"], batch["source_view_summaries"]
            )
            source_coverage_masked = source_best.masked_fill(~batch["view_available"], 2.0)
            source_semantic_coverage = source_coverage_masked.min(dim=-1).values

            entropy = -(
                output["slot_weights"] * output["slot_weights"].clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(output["slot_weights"].shape[-1])
            best_by_view = output["view_attention_mass"].max(dim=1).values
            coverage_masked = best_by_view.masked_fill(~batch["view_available"], 2.0)
            view_coverage = coverage_masked.min(dim=-1).values
            channel_coverage = output["channel_attention_mass"].max(dim=1).values.min(dim=-1).values

            unavailable = ~batch["view_available"]
            if unavailable.any():
                expanded = unavailable.unsqueeze(1).expand_as(output["view_attention_mass"])
                values = output["view_attention_mass"].masked_select(expanded)
                if values.numel():
                    missing_attention = max(missing_attention, float(values.max().float().cpu()))

            best_cpu = best.float().cpu().tolist()
            pooled_cpu = pooled.float().cpu().tolist()
            best_values.extend(float(x) for x in best_cpu)
            pooled_values.extend(float(x) for x in pooled_cpu)
            offdiag_mean_values.extend(float(x) for x in offdiag_mean.float().cpu().tolist())
            offdiag_max_values.extend(float(x) for x in offdiag_max.float().cpu().tolist())
            effective_rank_values.extend(float(x) for x in effective_rank.float().cpu().tolist())
            raw_specialization_values.extend(float(x) for x in specialization.float().cpu().tolist())
            source_semantic_coverage_values.extend(float(x) for x in source_semantic_coverage.float().cpu().tolist())
            entropy_values.extend(float(x) for x in entropy.float().cpu().tolist())
            view_coverage_values.extend(float(x) for x in view_coverage.float().cpu().tolist())
            channel_coverage_values.extend(float(x) for x in channel_coverage.float().cpu().tolist())
            specialization_weighted_sum += float((specialization * disagreement).sum().float().cpu())
            specialization_weight_sum += float(disagreement.sum().float().cpu())

            for local, global_index in enumerate(batch["global_index"].cpu().tolist()):
                family = str(parent_cache["families"][int(global_index)])
                family_best[family].append(float(best_cpu[local]))
                family_pooled[family].append(float(pooled_cpu[local]))

    family_best_mean = {k: sum(v) / len(v) for k, v in sorted(family_best.items())}
    family_pooled_mean = {k: sum(v) / len(v) for k, v in sorted(family_pooled.items())}
    weighted_specialization = (
        specialization_weighted_sum / specialization_weight_sum
        if specialization_weight_sum > 1e-8
        else 1.0
    )
    return {
        "rows": len(dataset),
        "best_slot_semantic_cosine": sum(best_values) / len(best_values),
        "family_best_slot_semantic_cosine": family_best_mean,
        "family_min_best_slot_semantic_cosine": min(family_best_mean.values()),
        "pooled_semantic_cosine": sum(pooled_values) / len(pooled_values),
        "family_pooled_semantic_cosine": family_pooled_mean,
        "family_min_pooled_semantic_cosine": min(family_pooled_mean.values()),
        "mean_pairwise_offdiag_slot_cosine": sum(offdiag_mean_values) / len(offdiag_mean_values),
        "mean_max_offdiag_slot_cosine": sum(offdiag_max_values) / len(offdiag_max_values),
        "mean_centered_slot_effective_rank": sum(effective_rank_values) / len(effective_rank_values),
        "mean_view_specialization": sum(raw_specialization_values) / len(raw_specialization_values),
        "mean_disagreement_weighted_view_specialization": weighted_specialization,
        "mean_source_view_disagreement_weight": specialization_weight_sum / max(len(dataset), 1),
        "mean_min_available_view_best_slot_semantic_cosine": (
            sum(source_semantic_coverage_values) / len(source_semantic_coverage_values)
        ),
        "mean_normalized_slot_weight_entropy": sum(entropy_values) / len(entropy_values),
        "mean_min_available_view_best_slot_attention": sum(view_coverage_values) / len(view_coverage_values),
        "mean_min_channel_best_slot_attention": sum(channel_coverage_values) / len(channel_coverage_values),
        "missing_view_attention_max": missing_attention,
    }


def corrected_training_gate(
    metrics: dict[str, Any],
    gate: dict[str, Any],
    random_baseline: dict[str, Any],
) -> bool:
    source_semantic = float(metrics["mean_min_available_view_best_slot_semantic_cosine"])
    baseline_source = float(random_baseline["mean_min_available_view_best_slot_semantic_cosine"])
    specialization = float(metrics["mean_disagreement_weighted_view_specialization"])
    baseline_specialization = float(random_baseline["mean_disagreement_weighted_view_specialization"])
    return (
        float(metrics["best_slot_semantic_cosine"]) >= float(gate["best_slot_semantic_cosine_min"])
        and float(metrics["pooled_semantic_cosine"]) >= float(gate["pooled_semantic_cosine_min"])
        and float(metrics["family_min_best_slot_semantic_cosine"]) >= float(gate["family_min_best_slot_semantic_cosine_min"])
        and float(metrics["family_min_pooled_semantic_cosine"]) >= float(gate["family_min_pooled_semantic_cosine_min"])
        and float(metrics["mean_pairwise_offdiag_slot_cosine"]) <= float(gate["mean_pairwise_offdiag_slot_cosine_max"])
        and float(metrics["mean_centered_slot_effective_rank"]) >= float(gate["mean_centered_slot_effective_rank_min"])
        and source_semantic >= float(gate["mean_min_available_view_best_slot_semantic_cosine_min"])
        and source_semantic >= baseline_source + float(gate["source_view_semantic_coverage_improvement_over_random_min"])
        and specialization >= float(gate["mean_disagreement_weighted_view_specialization_min"])
        and specialization >= baseline_specialization + float(gate["view_specialization_improvement_over_random_min"])
        and float(metrics["mean_min_available_view_best_slot_attention"]) >= float(gate["mean_min_available_view_best_slot_attention_min"])
        and float(metrics["mean_min_channel_best_slot_attention"]) >= float(gate["mean_min_channel_best_slot_attention_min"])
        and float(metrics["missing_view_attention_max"]) <= float(gate["missing_view_attention_max"])
    )


def corrected_capability_score(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    return (
        float(metrics["family_min_best_slot_semantic_cosine"]),
        float(metrics["family_min_pooled_semantic_cosine"]),
        float(metrics["mean_min_available_view_best_slot_semantic_cosine"]),
        float(metrics["best_slot_semantic_cosine"]),
        float(metrics["pooled_semantic_cosine"]),
        float(metrics["mean_centered_slot_effective_rank"]),
        float(metrics["mean_disagreement_weighted_view_specialization"]),
        float(metrics["mean_min_available_view_best_slot_attention"]),
        -float(metrics["mean_pairwise_offdiag_slot_cosine"]),
        -float(step),
    )


def corrected_save_checkpoint(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    root: Path,
    step: int,
    metrics: dict[str, Any],
    random_baseline: dict[str, Any],
    gate: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint / "adaptive_multi_view_latent_pool_v0_2.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(model_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-checkpoint.v0.2",
        "status": "TRAINED_PUBLIC_IDENTITY_NEUTRAL_COMPETITIVE_FULL_SCALE_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(),
        "latent_pool_sha256": sha256_file(model_path),
        "parameter_report": model.parameter_report(),
        "dev_metrics": metrics,
        "random_baseline": random_baseline,
        "training_gate": gate,
        "training_gate_pass": corrected_training_gate(metrics, gate, random_baseline),
        "lineage": lineage,
        "fresh_latent_initialization": True,
        "failed_v0_1_latent_weights_reused": False,
        "full_scale_model": True,
        "reduced_capability_pilot": False,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (checkpoint / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


base.loss_bundle = corrected_loss_bundle
base.evaluate = corrected_evaluate
base.capability_score = corrected_capability_score
base.save_checkpoint = corrected_save_checkpoint


if __name__ == "__main__":
    base.main()
