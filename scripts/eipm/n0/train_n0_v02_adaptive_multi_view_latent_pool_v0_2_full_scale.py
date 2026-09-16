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
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
    AdaptiveMultiViewLatentPoolV02Config,
)
from alice_personality.n0.adaptive_multi_view_latent_pool_objectives_v0_2 import (
    LatentPoolV02ObjectiveWeights,
    adaptive_latent_pool_v0_2_objective,
    centered_slot_effective_rank,
    normalized_view_specialization,
)

from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import LatentPoolDataset
from train_n0_v02_cross_context_fusion_full_scale import (
    git_revision,
    seed_everything,
    sha256_file,
    to_device,
)


def load_config(path: Path) -> AdaptiveMultiViewLatentPoolV02Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arch = raw["architecture"]
    return AdaptiveMultiViewLatentPoolV02Config(
        semantic_size=int(arch["semantic_size"]),
        latent_size=int(arch["latent_size"]),
        num_slots=int(arch["current_instantiated_slots"]),
        num_layers=int(arch["latent_layers"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        num_views=int(arch["current_instantiated_view_count"]),
        competition_temperature=float(arch["competition_temperature"]),
        initial_self_exchange_gate=float(arch["initial_self_exchange_gate_logit"]),
    )


def latent_forward(
    model: AdaptiveMultiViewLatentPoolV02,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return model(
        contextualized_view_tokens=[
            batch["context_semantic"],
            batch["context_structured"],
            batch["context_evidence"],
        ],
        source_view_tokens=[
            batch["source_semantic"],
            batch["source_structured"],
            batch["source_evidence"],
        ],
        view_valid_masks=[
            batch["semantic_mask"],
            batch["structured_mask"],
            batch["evidence_mask"],
        ],
        query_semantic=batch["query_semantic"],
        view_reliability=batch["view_reliability"],
        fusion_view_weights=batch["fusion_view_weights"],
    )


def decisive_view_mask(
    batch: dict[str, torch.Tensor],
    *,
    margin: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.normalize(batch["semantic_target"], dim=-1).unsqueeze(1)
    source = F.normalize(batch["source_view_summaries"], dim=-1)
    cosine = (target * source).sum(dim=-1)
    cosine = cosine.masked_fill(~batch["view_available"], -2.0)
    values, indices = cosine.topk(k=2, dim=-1)
    enough_views = batch["view_available"].sum(dim=-1) >= 2
    decisive = enough_views & ((values[:, 0] - values[:, 1]) >= margin)
    return decisive, indices[:, 0]


def counterfactual_forward(
    model: AdaptiveMultiViewLatentPoolV02,
    batch: dict[str, torch.Tensor],
    decisive: torch.Tensor,
    best_view: torch.Tensor,
) -> dict[str, Any] | None:
    if not decisive.any():
        return None
    masks = [
        batch["semantic_mask"].clone(),
        batch["structured_mask"].clone(),
        batch["evidence_mask"].clone(),
    ]
    reliability = batch["view_reliability"].clone()
    routing = batch["fusion_view_weights"].clone()
    for row in decisive.nonzero(as_tuple=False).squeeze(-1).tolist():
        view = int(best_view[row].item())
        masks[view][row].zero_()
        reliability[row, view] = 0.0
        routing[row, view] = 0.0
    routing = routing / routing.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return model(
        contextualized_view_tokens=[
            batch["context_semantic"],
            batch["context_structured"],
            batch["context_evidence"],
        ],
        source_view_tokens=[
            batch["source_semantic"],
            batch["source_structured"],
            batch["source_evidence"],
        ],
        view_valid_masks=masks,
        query_semantic=batch["query_semantic"],
        view_reliability=reliability,
        fusion_view_weights=routing,
    )


def loss_bundle(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    batch: dict[str, torch.Tensor],
    weights: LatentPoolV02ObjectiveWeights,
) -> dict[str, torch.Tensor]:
    output = latent_forward(model, batch)
    decisive, best_view = decisive_view_mask(batch)
    counterfactual = counterfactual_forward(model, batch, decisive, best_view)
    counterfactual_slots = None
    if counterfactual is not None:
        # The objective accepts one tensor for the whole batch. Rows without a
        # decisive source should exert no counterfactual pressure, so copy the
        # correct slots for those rows and replace only decisive rows.
        counterfactual_slots = output["latent_slots"].detach().clone()
        counterfactual_slots[decisive] = counterfactual["latent_slots"][decisive]
    return adaptive_latent_pool_v0_2_objective(
        latent_slots=output["latent_slots"],
        pooled_state=output["pooled_state"],
        target_semantic=batch["semantic_target"],
        view_attention_mass=output["view_attention_mass"],
        view_available=output["view_available"],
        channel_attention_mass=output["channel_attention_mass"],
        counterfactual_slots=counterfactual_slots,
        weights=weights,
    )


def _offdiag_values(slot_cosine: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    slots = slot_cosine.shape[1]
    eye = torch.eye(slots, device=slot_cosine.device, dtype=torch.bool).unsqueeze(0)
    selected = slot_cosine.masked_select(~eye.expand_as(slot_cosine)).view(slot_cosine.shape[0], -1)
    return selected.mean(dim=-1), selected.max(dim=-1).values


def evaluate(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    dataset: LatentPoolDataset,
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
    specialization_values: list[float] = []
    entropy_values: list[float] = []
    view_coverage_values: list[float] = []
    channel_coverage_values: list[float] = []
    family_best: dict[str, list[float]] = defaultdict(list)
    family_pooled: dict[str, list[float]] = defaultdict(list)
    missing_attention = 0.0

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = latent_forward(model, batch)
            target = F.normalize(batch["semantic_target"], dim=-1).unsqueeze(1)
            slot_cos = (F.normalize(output["latent_slots"], dim=-1) * target).sum(dim=-1)
            best = slot_cos.max(dim=-1).values
            pooled = F.cosine_similarity(output["pooled_state"], batch["semantic_target"], dim=-1)
            offdiag_mean, offdiag_max = _offdiag_values(output["slot_cosine"])
            effective_rank = centered_slot_effective_rank(output["latent_slots"])
            specialization = normalized_view_specialization(
                output["view_attention_mass"],
                batch["view_available"],
            )
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
            specialization_values.extend(float(x) for x in specialization.float().cpu().tolist())
            entropy_values.extend(float(x) for x in entropy.float().cpu().tolist())
            view_coverage_values.extend(float(x) for x in view_coverage.float().cpu().tolist())
            channel_coverage_values.extend(float(x) for x in channel_coverage.float().cpu().tolist())
            for local, global_index in enumerate(batch["global_index"].cpu().tolist()):
                family = str(parent_cache["families"][int(global_index)])
                family_best[family].append(float(best_cpu[local]))
                family_pooled[family].append(float(pooled_cpu[local]))

    family_best_mean = {k: sum(v) / len(v) for k, v in sorted(family_best.items())}
    family_pooled_mean = {k: sum(v) / len(v) for k, v in sorted(family_pooled.items())}
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
        "mean_view_specialization": sum(specialization_values) / len(specialization_values),
        "mean_normalized_slot_weight_entropy": sum(entropy_values) / len(entropy_values),
        "mean_min_available_view_best_slot_attention": sum(view_coverage_values) / len(view_coverage_values),
        "mean_min_channel_best_slot_attention": sum(channel_coverage_values) / len(channel_coverage_values),
        "missing_view_attention_max": missing_attention,
    }


def training_gate(
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> bool:
    return (
        float(metrics["best_slot_semantic_cosine"]) >= float(gate["best_slot_semantic_cosine_min"])
        and float(metrics["pooled_semantic_cosine"]) >= float(gate["pooled_semantic_cosine_min"])
        and float(metrics["family_min_best_slot_semantic_cosine"]) >= float(gate["family_min_best_slot_semantic_cosine_min"])
        and float(metrics["family_min_pooled_semantic_cosine"]) >= float(gate["family_min_pooled_semantic_cosine_min"])
        and float(metrics["mean_pairwise_offdiag_slot_cosine"]) <= float(gate["mean_pairwise_offdiag_slot_cosine_max"])
        and float(metrics["mean_centered_slot_effective_rank"]) >= float(gate["mean_centered_slot_effective_rank_min"])
        and float(metrics["mean_view_specialization"]) >= float(gate["mean_view_specialization_min"])
        and float(metrics["mean_min_available_view_best_slot_attention"]) >= float(gate["mean_min_available_view_best_slot_attention_min"])
        and float(metrics["mean_min_channel_best_slot_attention"]) >= float(gate["mean_min_channel_best_slot_attention_min"])
        and float(metrics["missing_view_attention_max"]) <= float(gate["missing_view_attention_max"])
    )


def capability_score(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    return (
        float(metrics["family_min_best_slot_semantic_cosine"]),
        float(metrics["family_min_pooled_semantic_cosine"]),
        float(metrics["best_slot_semantic_cosine"]),
        float(metrics["pooled_semantic_cosine"]),
        float(metrics["mean_centered_slot_effective_rank"]),
        float(metrics["mean_view_specialization"]),
        float(metrics["mean_min_available_view_best_slot_attention"]),
        -float(metrics["mean_pairwise_offdiag_slot_cosine"]),
        -float(step),
    )


def save_checkpoint(
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
        "training_gate_pass": training_gate(metrics, gate),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-cache", required=True)
    parser.add_argument("--fusion-cache", required=True)
    parser.add_argument("--latent-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--prep-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("competitive latent-pool v0.2 training requires CUDA")
    device = torch.device("cuda")
    training_path = Path(args.training_config).resolve()
    training_cfg = json.loads(training_path.read_text(encoding="utf-8"))
    if training_cfg.get("status") != "PUBLIC_IDENTITY_NEUTRAL_FULL_SCALE_TRAINING_GATE_READY_AFTER_CPU_PREP":
        raise SystemExit("latent-pool v0.2 training config status mismatch")
    if training_cfg.get("private_identity_gradient") is not False:
        raise SystemExit("latent-pool v0.2 crossed private gradient boundary")

    prep_path = Path(args.prep_receipt).resolve()
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("status") != "PASS" or prep.get("git_revision") != git_revision():
        raise SystemExit("latent-pool v0.2 prep receipt missing or stale")
    if prep.get("public_latent_pool_v0_2_gradient_authorized_after_prep") is not True:
        raise SystemExit("latent-pool v0.2 gradient authorization missing")

    parent_cache_path = Path(args.parent_cache).resolve()
    fusion_cache_path = Path(args.fusion_cache).resolve()
    if sha256_file(parent_cache_path) != prep.get("parent_cache_sha256"):
        raise SystemExit("parent cache hash mismatch")
    if sha256_file(fusion_cache_path) != prep.get("fusion_cache_sha256"):
        raise SystemExit("fusion cache hash mismatch")
    parent_cache = torch.load(parent_cache_path, map_location="cpu", weights_only=False)
    fusion_cache = torch.load(fusion_cache_path, map_location="cpu", weights_only=False)

    train_indices = [i for i, split in enumerate(parent_cache["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(parent_cache["splits"]) if split == "dev"]
    data_cfg = training_cfg["training_data"]
    if len(parent_cache["ids"]) != int(data_cfg["expected_rows"]):
        raise SystemExit("latent-pool v0.2 cache row count mismatch")
    if len(train_indices) != int(data_cfg["expected_train_rows"]) or len(dev_indices) != int(data_cfg["expected_dev_rows"]):
        raise SystemExit("latent-pool v0.2 split count mismatch")

    opt = training_cfg["optimization"]
    seed = int(opt["seed"])
    seed_everything(seed)
    batch_size = int(opt["batch_size"])
    max_steps = int(opt["steps"])
    save_steps = [int(value) for value in opt["save_steps"]]
    learning_rate = float(opt["learning_rate"])
    warmup_steps = int(opt["warmup_steps"])
    weight_decay = float(opt["weight_decay"])
    clip = float(opt["gradient_clip_norm"])

    train_dataset = LatentPoolDataset(parent_cache, fusion_cache, train_indices)
    dev_dataset = LatentPoolDataset(parent_cache, fusion_cache, dev_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    model = AdaptiveMultiViewLatentPoolV02(load_config(Path(args.latent_config).resolve())).to(device)
    report = model.parameter_report()
    print("latent_pool_v0_2_parameter_report=" + json.dumps(report, sort_keys=True), flush=True)
    if report.get("hard_parameter_ceiling") is not None or report.get("slot_count_ceiling") is not None:
        raise SystemExit("latent-pool v0.2 contains an unexpected capability ceiling")
    if report.get("competitive_cross_attention") is not True:
        raise SystemExit("latent-pool v0.2 competitive attention missing")

    weights = LatentPoolV02ObjectiveWeights(
        **{key: float(value) for key, value in training_cfg["objective_weights"].items()}
    )
    gate = training_cfg["training_dev_gate"]
    random_baseline = evaluate(
        model=model,
        dataset=dev_dataset,
        parent_cache=parent_cache,
        device=device,
        batch_size=batch_size,
    )
    print("latent_pool_v0_2_random_baseline=" + json.dumps(random_baseline, sort_keys=True), flush=True)

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty latent-pool v0.2 output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    lineage = {
        "git_revision": git_revision(),
        "latent_config_sha256": sha256_file(Path(args.latent_config).resolve()),
        "training_config_sha256": sha256_file(training_path),
        "prep_receipt_sha256": sha256_file(prep_path),
        "parent_cache_sha256": sha256_file(parent_cache_path),
        "fusion_cache_sha256": sha256_file(fusion_cache_path),
        "ratified_fusion_sha256": prep["ratified_fusion_sha256"],
        "failed_v0_1_job": 575670,
        "failed_v0_1_weights_reused": False,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    iterator = iter(train_loader)
    running: defaultdict[str, float] = defaultdict(float)
    saved: list[dict[str, Any]] = []

    model.train()
    for step in range(1, max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            losses = loss_bundle(model=model, batch=batch, weights=weights)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key, value in losses.items():
            running[key] += float(value.detach().float().cpu())
        if step % 20 == 0:
            payload = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
            payload.update({key: value / 20.0 for key, value in running.items()})
            print("latent_pool_v0_2_train_step=" + json.dumps(payload, sort_keys=True), flush=True)
            running.clear()
        if step in save_steps:
            metrics = evaluate(
                model=model,
                dataset=dev_dataset,
                parent_cache=parent_cache,
                device=device,
                batch_size=batch_size,
            )
            receipt = save_checkpoint(
                model=model,
                root=output_root,
                step=step,
                metrics=metrics,
                random_baseline=random_baseline,
                gate=gate,
                lineage=lineage,
            )
            saved.append(receipt)
            print("latent_pool_v0_2_checkpoint_eval=" + json.dumps(receipt, sort_keys=True), flush=True)
            model.train()

    eligible = [item for item in saved if item["training_gate_pass"]]
    comparison = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-comparison.v0.2",
        "status": "PASS_COMPETITIVE_LATENT_POOL_READY_FOR_UNTOUCHED_CHALLENGE" if eligible else "FAIL_COMPETITIVE_LATENT_POOL_NO_CHECKPOINT_CLEARED_TRAINING_DEV_GATE",
        "git_revision": git_revision(),
        "failed_v0_1_job": 575670,
        "failed_v0_1_weights_reused": False,
        "random_baseline": random_baseline,
        "checkpoints": saved,
        "winner": None,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "competitive_cross_attention": True,
        "frozen_challenge_rows_used_for_training": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    if eligible:
        winner = max(
            eligible,
            key=lambda item: capability_score(item["dev_metrics"], int(item["step"])),
        )
        comparison["winner"] = {
            "step": int(winner["step"]),
            "latent_pool_sha256": winner["latent_pool_sha256"],
            "dev_metrics": winner["dev_metrics"],
            "selection_policy": "capability_first_training_dev_only_with_earlier_step_tie_break",
        }
    comparison_path = output_root / "adaptive_multi_view_latent_pool_v0_2_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
