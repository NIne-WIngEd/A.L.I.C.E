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
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, save_file

from alice_personality.n0.adaptive_multi_view_latent_pool import (
    AdaptiveMultiViewLatentPool,
    AdaptiveMultiViewLatentPoolConfig,
)
from alice_personality.n0.adaptive_multi_view_latent_pool_objectives import (
    LatentPoolObjectiveWeights,
    adaptive_latent_pool_objective,
    counterfactual_target_margin_loss,
)
from alice_personality.n0.config import load_n0_config
from alice_personality.n0.cross_context_fusion_anchored import SourceAnchoredCrossContextFusion
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

from train_n0_v02_cross_context_fusion_full_scale import (
    EXPECTED_SEMANTIC_PARAMETERS,
    EXPECTED_STRUCTURED_PARENT_STEP,
    FusionDataset,
    build_parent_cache,
    expanded_adapter_config,
    expanded_graph_config,
    git_revision,
    load_fusion_config,
    load_structured_config,
    read_jsonl,
    seed_everything,
    sha256_file,
    to_device,
)
from train_n0_v02_cross_context_fusion_repair_full_scale import forward_fusion


def load_latent_config(path: Path) -> AdaptiveMultiViewLatentPoolConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arch = raw["architecture"]
    return AdaptiveMultiViewLatentPoolConfig(
        semantic_size=int(arch["semantic_size"]),
        latent_size=int(arch["latent_size"]),
        num_slots=int(arch["current_instantiated_slots"]),
        num_layers=int(arch["latent_layers"]),
        num_heads=int(arch["num_heads"]),
        feedforward_size=int(arch["feedforward_size"]),
        dropout=float(arch["dropout"]),
        num_views=int(arch["current_instantiated_view_count"]),
    )


def load_ratified_fusion(
    *,
    checkpoint_dir: Path,
    fusion_config_path: Path,
    ratification_path: Path,
    device: torch.device,
) -> SourceAnchoredCrossContextFusion:
    manifest = json.loads(ratification_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RATIFIED_PUBLIC_N0_FUSION_BASE_NOT_FULL_EIPM_PROMOTION":
        raise SystemExit("fusion ratification manifest status mismatch")
    selected = manifest["selected_fusion"]
    model_path = checkpoint_dir / "cross_context_fusion.safetensors"
    if sha256_file(model_path) != selected["sha256"]:
        raise SystemExit("ratified fusion checkpoint hash mismatch")
    model = SourceAnchoredCrossContextFusion(load_fusion_config(fusion_config_path))
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def precompute_fusion_cache(
    *,
    fusion: SourceAnchoredCrossContextFusion,
    parent_cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = FusionDataset(parent_cache, list(range(len(parent_cache["ids"]))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    contextualized: list[list[torch.Tensor]] = [[], [], []]
    weights: list[torch.Tensor] = []
    max_summary_error = 0.0
    max_token_error = 0.0

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = forward_fusion(fusion, batch)
            for view_id, value in enumerate(output["contextualized_view_tokens"]):
                contextualized[view_id].append(value.detach().float().cpu())
            weights.append(output["view_weights"].detach().float().cpu())
            active = batch["view_available"]
            if active.any():
                diff = (output["source_view_summaries"] - batch["source_view_summaries"]).abs()
                max_summary_error = max(
                    max_summary_error,
                    float(diff[active].max().detach().float().cpu()),
                )
            source_targets = [
                batch["semantic_tokens"],
                batch["structured_tokens"],
                batch["evidence_tokens"],
            ]
            source_masks = [
                batch["semantic_valid_mask"],
                batch["structured_valid_mask"],
                batch["evidence_valid_mask"],
            ]
            for actual, expected, mask in zip(output["source_view_tokens"], source_targets, source_masks):
                if mask.any():
                    max_token_error = max(
                        max_token_error,
                        float((actual - expected).abs()[mask].max().detach().float().cpu()),
                    )
    if max_summary_error != 0.0 or max_token_error != 0.0:
        raise SystemExit("ratified fusion source-anchor path changed during latent cache construction")
    return {
        "contextualized_view_tokens": [torch.cat(parts, dim=0) for parts in contextualized],
        "fusion_view_weights": torch.cat(weights, dim=0),
        "source_anchor_summary_max_abs_error": max_summary_error,
        "source_anchor_token_max_abs_error": max_token_error,
    }


class LatentPoolDataset(Dataset):
    def __init__(self, parent: dict[str, Any], fusion: dict[str, Any], indices: list[int]) -> None:
        self.parent = parent
        self.fusion = fusion
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        return {
            "context_semantic": self.fusion["contextualized_view_tokens"][0][index],
            "context_structured": self.fusion["contextualized_view_tokens"][1][index],
            "context_evidence": self.fusion["contextualized_view_tokens"][2][index],
            "source_semantic": self.parent["semantic_tokens"][index],
            "source_structured": self.parent["structured_tokens"][index],
            "source_evidence": self.parent["evidence_tokens"][index],
            "semantic_mask": self.parent["semantic_valid_mask"][index],
            "structured_mask": self.parent["structured_valid_mask"][index],
            "evidence_mask": self.parent["evidence_valid_mask"][index],
            "query_semantic": self.parent["query_semantic"][index],
            "semantic_target": self.parent["semantic_target"][index],
            "source_view_summaries": self.parent["source_view_summaries"][index],
            "view_reliability": self.parent["view_reliability"][index],
            "view_available": self.parent["view_available"][index],
            "fusion_view_weights": self.fusion["fusion_view_weights"][index],
            "global_index": index,
        }


def latent_forward(model: AdaptiveMultiViewLatentPool, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
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


def decisive_view_mask(batch: dict[str, torch.Tensor], *, margin: float = 0.08) -> tuple[torch.Tensor, torch.Tensor]:
    target = F.normalize(batch["semantic_target"], dim=-1).unsqueeze(1)
    source = F.normalize(batch["source_view_summaries"], dim=-1)
    cosine = (target * source).sum(dim=-1)
    cosine = cosine.masked_fill(~batch["view_available"], -2.0)
    values, indices = cosine.topk(k=2, dim=-1)
    enough_views = batch["view_available"].sum(dim=-1) >= 2
    decisive = enough_views & ((values[:, 0] - values[:, 1]) >= margin)
    return decisive, indices[:, 0]


def counterfactual_forward(
    model: AdaptiveMultiViewLatentPool,
    batch: dict[str, torch.Tensor],
    decisive: torch.Tensor,
    best_view: torch.Tensor,
) -> dict[str, Any] | None:
    if not decisive.any():
        return None
    masks = [batch["semantic_mask"].clone(), batch["structured_mask"].clone(), batch["evidence_mask"].clone()]
    reliability = batch["view_reliability"].clone()
    routing = batch["fusion_view_weights"].clone()
    rows = decisive.nonzero(as_tuple=False).squeeze(-1)
    for row in rows.tolist():
        view = int(best_view[row].item())
        masks[view][row].zero_()
        reliability[row, view] = 0.0
        routing[row, view] = 0.0
    denom = routing.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    routing = routing / denom
    return model(
        contextualized_view_tokens=[batch["context_semantic"], batch["context_structured"], batch["context_evidence"]],
        source_view_tokens=[batch["source_semantic"], batch["source_structured"], batch["source_evidence"]],
        view_valid_masks=masks,
        query_semantic=batch["query_semantic"],
        view_reliability=reliability,
        fusion_view_weights=routing,
    )


def loss_bundle(
    *,
    model: AdaptiveMultiViewLatentPool,
    batch: dict[str, torch.Tensor],
    weights: LatentPoolObjectiveWeights,
) -> dict[str, torch.Tensor]:
    output = latent_forward(model, batch)
    losses = adaptive_latent_pool_objective(
        latent_slots=output["latent_slots"],
        pooled_state=output["pooled_state"],
        target_semantic=batch["semantic_target"],
        view_attention_mass=output["view_attention_mass"],
        view_available=output["view_available"],
        channel_attention_mass=output["channel_attention_mass"],
        counterfactual_slots=None,
        weights=weights,
    )
    decisive, best_view = decisive_view_mask(batch)
    counterfactual_output = counterfactual_forward(model, batch, decisive, best_view)
    if counterfactual_output is None:
        counterfactual = output["latent_slots"].sum() * 0.0
    else:
        counterfactual = counterfactual_target_margin_loss(
            output["latent_slots"][decisive],
            counterfactual_output["latent_slots"][decisive],
            batch["semantic_target"][decisive],
        )
    total = losses["total"] + weights.counterfactual_margin * counterfactual
    return {**losses, "counterfactual_margin": counterfactual, "total": total}


def _offdiag_max(slot_cosine: torch.Tensor) -> torch.Tensor:
    slots = slot_cosine.shape[1]
    eye = torch.eye(slots, device=slot_cosine.device, dtype=torch.bool).unsqueeze(0)
    masked = slot_cosine.masked_fill(eye, -2.0)
    return masked.max(dim=-1).values.max(dim=-1).values


def evaluate(
    *,
    model: AdaptiveMultiViewLatentPool,
    dataset: LatentPoolDataset,
    parent_cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    best_slot_values: list[float] = []
    pooled_values: list[float] = []
    offdiag_values: list[float] = []
    entropy_values: list[float] = []
    view_coverage_values: list[float] = []
    channel_coverage_values: list[float] = []
    missing_attention = 0.0
    family_best: dict[str, list[float]] = defaultdict(list)
    family_pooled: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = latent_forward(model, batch)
            target = F.normalize(batch["semantic_target"], dim=-1).unsqueeze(1)
            slot_cos = (F.normalize(output["latent_slots"], dim=-1) * target).sum(dim=-1)
            best = slot_cos.max(dim=-1).values
            pooled = F.cosine_similarity(output["pooled_state"], batch["semantic_target"], dim=-1)
            offdiag = _offdiag_max(output["slot_cosine"])
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
            best_slot_values.extend(float(x) for x in best_cpu)
            pooled_values.extend(float(x) for x in pooled_cpu)
            offdiag_values.extend(float(x) for x in offdiag.float().cpu().tolist())
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
        "best_slot_semantic_cosine": sum(best_slot_values) / len(best_slot_values),
        "family_best_slot_semantic_cosine": family_best_mean,
        "family_min_best_slot_semantic_cosine": min(family_best_mean.values()),
        "pooled_semantic_cosine": sum(pooled_values) / len(pooled_values),
        "family_pooled_semantic_cosine": family_pooled_mean,
        "family_min_pooled_semantic_cosine": min(family_pooled_mean.values()),
        "mean_max_offdiag_slot_cosine": sum(offdiag_values) / len(offdiag_values),
        "mean_normalized_slot_weight_entropy": sum(entropy_values) / len(entropy_values),
        "mean_min_available_view_best_slot_attention": sum(view_coverage_values) / len(view_coverage_values),
        "mean_min_channel_best_slot_attention": sum(channel_coverage_values) / len(channel_coverage_values),
        "missing_view_attention_max": missing_attention,
    }


def capability_score(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    return (
        float(metrics["family_min_best_slot_semantic_cosine"]),
        float(metrics["best_slot_semantic_cosine"]),
        float(metrics["family_min_pooled_semantic_cosine"]),
        float(metrics["pooled_semantic_cosine"]),
        float(metrics["mean_min_available_view_best_slot_attention"]),
        float(metrics["mean_min_channel_best_slot_attention"]),
        -float(metrics["mean_max_offdiag_slot_cosine"]),
        float(metrics["mean_normalized_slot_weight_entropy"]),
        -float(step),
    )


def training_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        float(metrics["best_slot_semantic_cosine"]) >= float(baseline["best_slot_semantic_cosine"]) + 0.03
        and float(metrics["pooled_semantic_cosine"]) >= float(baseline["pooled_semantic_cosine"]) + 0.03
        and float(metrics["family_min_best_slot_semantic_cosine"])
        >= float(baseline["family_min_best_slot_semantic_cosine"]) + 0.02
        and float(metrics["mean_min_available_view_best_slot_attention"]) >= 0.12
        and float(metrics["mean_min_channel_best_slot_attention"]) >= 0.10
        and float(metrics["mean_max_offdiag_slot_cosine"]) <= 0.97
        and float(metrics["missing_view_attention_max"]) <= 1e-6
    )


def save_checkpoint(
    *,
    model: AdaptiveMultiViewLatentPool,
    root: Path,
    step: int,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint / "adaptive_multi_view_latent_pool.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(model_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-checkpoint.v0.1",
        "status": "TRAINED_PUBLIC_IDENTITY_NEUTRAL_FULL_SCALE_NOT_RATIFIED",
        "step": step,
        "git_revision": git_revision(),
        "latent_pool_sha256": sha256_file(model_path),
        "parameter_report": model.parameter_report(),
        "dev_metrics": metrics,
        "random_baseline": baseline,
        "training_gate_pass": training_gate(metrics, baseline),
        "lineage": lineage,
        "full_scale_model": True,
        "reduced_capability_pilot": False,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (checkpoint / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--structured-config", required=True)
    parser.add_argument("--structured-checkpoint", required=True)
    parser.add_argument("--evidence-adapter", required=True)
    parser.add_argument("--evidence-graph", required=True)
    parser.add_argument("--fusion-config", required=True)
    parser.add_argument("--fusion-ratification", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--latent-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--prep-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--fusion-cache-batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("full-scale adaptive latent-pool training requires CUDA")
    device = torch.device("cuda")
    training_cfg = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    if training_cfg.get("status") != "PUBLIC_IDENTITY_NEUTRAL_FULL_SCALE_TRAINING_GATE_READY_AFTER_CPU_PREP":
        raise SystemExit("latent-pool training config status mismatch")
    if training_cfg.get("private_identity_gradient") is not False:
        raise SystemExit("latent-pool training crossed private gradient boundary")

    prep = json.loads(Path(args.prep_receipt).read_text(encoding="utf-8"))
    if prep.get("status") != "PASS" or prep.get("git_revision") != git_revision():
        raise SystemExit("latent-pool prep receipt missing or stale")
    if prep.get("public_latent_pool_gradient_authorized_after_prep") is not True:
        raise SystemExit("public latent-pool gradient was not authorized by same-revision prep")

    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(curriculum_path)
    if manifest.get("compiled_sha256") != sha256_file(curriculum_path):
        raise SystemExit("latent-pool source curriculum hash mismatch")
    if len(rows) != 512 or int(manifest.get("rows", -1)) != 512:
        raise SystemExit("latent-pool training expects the frozen 512-row public training tranche")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("latent-pool source curriculum crossed private boundary")

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty latent-pool output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    opt = training_cfg["optimization"]
    seed = int(opt["seed"])
    seed_everything(seed)
    max_steps = int(opt["steps"])
    save_steps = [int(x) for x in opt["save_steps"]]
    batch_size = int(opt["batch_size"])
    lr = float(opt["learning_rate"])
    warmup_steps = int(opt["warmup_steps"])
    weight_decay = float(opt["weight_decay"])
    clip = float(opt["gradient_clip_norm"])

    semantic_config = load_n0_config(Path(args.semantic_config).resolve())
    semantic_model = AliceN0V02Model(semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    semantic_state = load_file(str(semantic_checkpoint / "alice_n0_v02.safetensors"), device="cpu")
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_receipt = json.loads((structured_checkpoint / "receipt.json").read_text(encoding="utf-8"))
    if int(structured_receipt.get("step", -1)) != EXPECTED_STRUCTURED_PARENT_STEP:
        raise SystemExit("latent pool requires ratified structured step80")
    structured = StructuredStateEncoder(load_structured_config(Path(args.structured_config).resolve()))
    structured.load_state_dict(load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu"), strict=True)
    for parameter in structured.parameters():
        parameter.requires_grad = False

    evidence_adapter_path = Path(args.evidence_adapter).resolve()
    evidence_adapter = EvidenceViewAdapter(expanded_adapter_config())
    evidence_adapter.load_state_dict(load_file(str(evidence_adapter_path), device="cpu"), strict=True)
    for parameter in evidence_adapter.parameters():
        parameter.requires_grad = False

    evidence_graph_path = Path(args.evidence_graph).resolve()
    evidence_graph = DualEndpointEvidenceGraphEncoder(expanded_graph_config())
    evidence_graph.load_state_dict(load_file(str(evidence_graph_path), device="cpu"), strict=True)
    for parameter in evidence_graph.parameters():
        parameter.requires_grad = False

    print("latent_pool_parent_cache_start=true", flush=True)
    parent_cache = build_parent_cache(
        rows,
        semantic_model=semantic_model,
        tokenizer=tokenizer,
        structured=structured,
        evidence_adapter=evidence_adapter,
        evidence_graph=evidence_graph,
        device=device,
        encode_batch_size=args.encode_batch_size,
        raw_max_length=args.raw_max_length,
        field_max_length=args.field_max_length,
    )
    parent_cache_path = output_root / "parent_cache.pt"
    torch.save(parent_cache, parent_cache_path)

    del semantic_model, semantic_state, structured, evidence_adapter, evidence_graph
    torch.cuda.empty_cache()

    fusion_checkpoint = Path(args.fusion_checkpoint).resolve()
    fusion = load_ratified_fusion(
        checkpoint_dir=fusion_checkpoint,
        fusion_config_path=Path(args.fusion_config).resolve(),
        ratification_path=Path(args.fusion_ratification).resolve(),
        device=device,
    )
    print("latent_pool_fusion_cache_start=true", flush=True)
    fusion_cache = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=parent_cache,
        device=device,
        batch_size=args.fusion_cache_batch_size,
    )
    fusion_cache_path = output_root / "ratified_fusion_cache.pt"
    torch.save(fusion_cache, fusion_cache_path)
    del fusion
    torch.cuda.empty_cache()

    train_indices = [i for i, split in enumerate(parent_cache["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(parent_cache["splits"]) if split == "dev"]
    if len(train_indices) != 384 or len(dev_indices) != 128:
        raise SystemExit(f"latent-pool split drift: train={len(train_indices)} dev={len(dev_indices)}")
    train_dataset = LatentPoolDataset(parent_cache, fusion_cache, train_indices)
    dev_dataset = LatentPoolDataset(parent_cache, fusion_cache, dev_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    model = AdaptiveMultiViewLatentPool(load_latent_config(Path(args.latent_config).resolve())).to(device)
    report = model.parameter_report()
    print("latent_pool_parameter_report=" + json.dumps(report, sort_keys=True), flush=True)
    if report.get("hard_parameter_ceiling") is not None or report.get("slot_count_ceiling") is not None:
        raise SystemExit("latent pool unexpectedly contains hard capacity ceiling")

    w = training_cfg["objective_weights"]
    weights = LatentPoolObjectiveWeights(**{key: float(value) for key, value in w.items()})
    baseline = evaluate(model=model, dataset=dev_dataset, parent_cache=parent_cache, device=device, batch_size=batch_size)
    print("latent_pool_random_baseline=" + json.dumps(baseline, sort_keys=True), flush=True)

    lineage = {
        "semantic_sha256": sha256_file(semantic_checkpoint / "alice_n0_v02.safetensors"),
        "structured_sha256": sha256_file(structured_checkpoint / "structured_state.safetensors"),
        "evidence_adapter_sha256": sha256_file(evidence_adapter_path),
        "evidence_graph_sha256": sha256_file(evidence_graph_path),
        "fusion_sha256": sha256_file(fusion_checkpoint / "cross_context_fusion.safetensors"),
        "fusion_ratification_sha256": sha256_file(Path(args.fusion_ratification).resolve()),
        "curriculum_sha256": sha256_file(curriculum_path),
        "curriculum_manifest_sha256": sha256_file(manifest_path),
        "prep_receipt_sha256": sha256_file(Path(args.prep_receipt).resolve()),
        "parent_cache_sha256": sha256_file(parent_cache_path),
        "fusion_cache_sha256": sha256_file(fusion_cache_path),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            print("latent_pool_train_step=" + json.dumps(payload, sort_keys=True), flush=True)
            running.clear()
        if step in save_steps:
            metrics = evaluate(model=model, dataset=dev_dataset, parent_cache=parent_cache, device=device, batch_size=batch_size)
            receipt = save_checkpoint(
                model=model,
                root=output_root,
                step=step,
                metrics=metrics,
                baseline=baseline,
                lineage=lineage,
            )
            saved.append(receipt)
            print("latent_pool_checkpoint_eval=" + json.dumps(receipt, sort_keys=True), flush=True)
            model.train()

    eligible = [receipt for receipt in saved if receipt["training_gate_pass"]]
    comparison = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-comparison.v0.1",
        "status": "PASS_FULL_SCALE_LATENT_POOL_READY_FOR_UNTOUCHED_CHALLENGE" if eligible else "FAIL_NO_LATENT_POOL_CHECKPOINT_CLEARED_TRAINING_DEV_GATE",
        "git_revision": git_revision(),
        "random_baseline": baseline,
        "checkpoints": saved,
        "winner": None,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
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
    comparison_path = output_root / "adaptive_multi_view_latent_pool_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
