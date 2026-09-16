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
from torch.utils.data import DataLoader

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.cross_context_fusion_anchored import (
    SourceAnchoredCrossContextFusion,
)
from alice_personality.n0.cross_context_fusion_repair_objectives import (
    FusionRepairObjectiveWeights,
    cross_context_fusion_repair_objective,
)
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


def forward_fusion(
    model: SourceAnchoredCrossContextFusion,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return model(
        semantic_tokens=batch["semantic_tokens"],
        semantic_valid_mask=batch["semantic_valid_mask"],
        structured_tokens=batch["structured_tokens"],
        structured_valid_mask=batch["structured_valid_mask"],
        evidence_tokens=batch["evidence_tokens"],
        evidence_valid_mask=batch["evidence_valid_mask"],
        query_semantic=batch["query_semantic"],
        view_reliability=batch["view_reliability"],
        source_view_summaries=batch["source_view_summaries"],
    )


def objective(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    weights: FusionRepairObjectiveWeights,
) -> dict[str, torch.Tensor]:
    return cross_context_fusion_repair_objective(
        fused_state=output["fused_state"],
        semantic_target=batch["semantic_target"],
        view_weights=output["view_weights"],
        target_view_distribution=batch["target_view_distribution"],
        contextualized_view_summaries=output["contextualized_view_summaries"],
        source_view_summaries=batch["source_view_summaries"],
        view_available=batch["view_available"],
        weights=weights,
    )


def _active_max_abs_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    active: torch.Tensor,
) -> float:
    if not active.any():
        return 0.0
    expanded = active
    while expanded.ndim < actual.ndim:
        expanded = expanded.unsqueeze(-1)
    values = (actual - expected).abs().masked_select(expanded.expand_as(actual))
    return float(values.max().detach().float().cpu()) if values.numel() else 0.0


def evaluate(
    *,
    model: SourceAnchoredCrossContextFusion,
    dataset: FusionDataset,
    cache: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    family_routing: dict[str, list[float]] = defaultdict(list)
    semantic_cosines: list[float] = []
    contextualized_source_cosines: list[float] = []
    disagreement_errors: list[float] = []
    decisive_correct = 0
    decisive_total = 0
    missing_leak = 0.0
    anchor_summary_max_error = 0.0
    anchor_token_max_error = 0.0
    gate_values: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = forward_fusion(model, batch)

            routing_similarity = 1.0 - 0.5 * (
                out["view_weights"] - batch["target_view_distribution"]
            ).abs().sum(dim=-1)
            semantic_cos = F.cosine_similarity(
                out["fused_state"], batch["semantic_target"], dim=-1
            )
            contextualized_cos = F.cosine_similarity(
                out["contextualized_view_summaries"],
                batch["source_view_summaries"],
                dim=-1,
            )
            active = batch["view_available"]
            contextualized_source_cosines.extend(
                float(x)
                for x in contextualized_cos[active].detach().float().cpu().tolist()
            )

            anchor_summary_max_error = max(
                anchor_summary_max_error,
                _active_max_abs_error(
                    out["source_view_summaries"],
                    batch["source_view_summaries"],
                    active,
                ),
            )
            source_token_targets = [
                batch["semantic_tokens"],
                batch["structured_tokens"],
                batch["evidence_tokens"],
            ]
            source_token_masks = [
                batch["semantic_valid_mask"],
                batch["structured_valid_mask"],
                batch["evidence_valid_mask"],
            ]
            for actual, expected, mask in zip(
                out["source_view_tokens"], source_token_targets, source_token_masks
            ):
                anchor_token_max_error = max(
                    anchor_token_max_error,
                    _active_max_abs_error(actual, expected, mask),
                )

            out_norm = F.normalize(out["contextualized_view_summaries"], dim=-1)
            src_norm = F.normalize(batch["source_view_summaries"], dim=-1)
            out_sim = torch.einsum("bvd,bwd->bvw", out_norm, out_norm)
            src_sim = torch.einsum("bvd,bwd->bvw", src_norm, src_norm)
            pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
            eye = torch.eye(active.size(1), device=device, dtype=torch.bool).unsqueeze(0)
            pair_mask = pair_mask & ~eye
            for row in range(active.size(0)):
                row_mask = pair_mask[row]
                if row_mask.any():
                    disagreement_errors.append(
                        float((out_sim[row] - src_sim[row]).abs()[row_mask].mean().float().cpu())
                    )

            unavailable = ~active
            if unavailable.any():
                missing_leak = max(
                    missing_leak,
                    float(out["view_weights"][unavailable].max().float().cpu()),
                )

            target_sorted, _ = batch["target_view_distribution"].sort(dim=-1, descending=True)
            decisive = (target_sorted[:, 0] - target_sorted[:, 1]) >= 0.25
            if decisive.any():
                pred = out["view_weights"].argmax(dim=-1)
                target = batch["target_view_distribution"].argmax(dim=-1)
                decisive_correct += int((pred[decisive] == target[decisive]).sum().item())
                decisive_total += int(decisive.sum().item())

            semantic_cosines.extend(float(x) for x in semantic_cos.float().cpu().tolist())
            gate_values.extend(
                float(x) for x in out["cross_gate_means"].float().cpu().reshape(-1).tolist()
            )
            global_indices = batch["global_index"].cpu().tolist()
            routing_cpu = routing_similarity.float().cpu().tolist()
            for local, global_index in enumerate(global_indices):
                family = str(cache["families"][int(global_index)])
                family_routing[family].append(float(routing_cpu[local]))

    family_means = {
        family: sum(values) / len(values)
        for family, values in sorted(family_routing.items())
    }
    return {
        "rows": len(dataset),
        "family_routing_similarity": family_means,
        "family_macro_routing_similarity": sum(family_means.values()) / len(family_means),
        "family_min_routing_similarity": min(family_means.values()),
        "decisive_view_accuracy": decisive_correct / max(decisive_total, 1),
        "decisive_rows": decisive_total,
        "fused_semantic_cosine": sum(semantic_cosines) / max(len(semantic_cosines), 1),
        "contextualized_source_cosine": (
            sum(contextualized_source_cosines) / max(len(contextualized_source_cosines), 1)
        ),
        "source_anchor_summary_max_abs_error": anchor_summary_max_error,
        "source_anchor_token_max_abs_error": anchor_token_max_error,
        "disagreement_geometry_mae": sum(disagreement_errors) / max(len(disagreement_errors), 1),
        "missing_view_max_weight": missing_leak,
        "mean_cross_gate_activation": sum(gate_values) / max(len(gate_values), 1),
    }


def capability_score(metrics: dict[str, Any], repair_step: int) -> tuple[float, ...]:
    return (
        float(metrics["family_min_routing_similarity"]),
        float(metrics["family_macro_routing_similarity"]),
        float(metrics["decisive_view_accuracy"]),
        float(metrics["fused_semantic_cosine"]),
        -float(metrics["source_anchor_summary_max_abs_error"]),
        -float(metrics["source_anchor_token_max_abs_error"]),
        float(metrics["contextualized_source_cosine"]),
        -float(metrics["disagreement_geometry_mae"]),
        -float(repair_step),
    )


def repair_gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return (
        float(metrics["source_anchor_summary_max_abs_error"]) == 0.0
        and float(metrics["source_anchor_token_max_abs_error"]) == 0.0
        and float(metrics["missing_view_max_weight"]) <= 1e-6
        and float(metrics["family_min_routing_similarity"])
        > float(baseline["family_min_routing_similarity"])
        and float(metrics["family_macro_routing_similarity"])
        >= float(baseline["family_macro_routing_similarity"]) - 1e-6
        and float(metrics["decisive_view_accuracy"])
        >= float(baseline["decisive_view_accuracy"]) - 1e-6
        and float(metrics["fused_semantic_cosine"])
        >= float(baseline["fused_semantic_cosine"]) - 0.01
    )


def save_checkpoint(
    *,
    model: SourceAnchoredCrossContextFusion,
    root: Path,
    repair_step: int,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    lineage: dict[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint = root / f"repair-step-{repair_step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint / "cross_context_fusion.safetensors"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()},
        str(model_path),
    )
    receipt = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-repair-checkpoint.v0.2",
        "status": "TRAINED_FULL_SCALE_REPAIR_NOT_RATIFIED",
        "repair_step": repair_step,
        "parent_fusion_step": 240,
        "total_fusion_update_count": 240 + repair_step,
        "git_revision": git_revision(),
        "fusion_sha256": sha256_file(model_path),
        "fusion_parameter_report": model.parameter_report(),
        "dev_metrics": metrics,
        "repair_gate_pass": repair_gate(metrics, baseline),
        "lineage": lineage,
        "full_scale_model": True,
        "reduced_capability_pilot": False,
        "hard_parameter_ceiling": None,
        "view_count_ceiling": None,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    (checkpoint / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    parser.add_argument("--fusion-parent-checkpoint", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--curriculum-manifest", required=True)
    parser.add_argument("--prep-receipt", required=True)
    parser.add_argument("--failed-challenge-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("full-scale source-anchored fusion repair requires CUDA")
    device = torch.device("cuda")

    training_cfg = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    if training_cfg.get("status") != "CONDITIONALLY_AUTHORIZED_AFTER_SAME_REVISION_REPAIR_PREP_PASS":
        raise SystemExit("fusion repair training authorization config status mismatch")
    auth = training_cfg["authorization"]
    if auth.get("public_fusion_gradient_authorized") is not True:
        raise SystemExit("public fusion repair gradient is not authorized")
    if auth.get("failed_frozen_challenge_rows_may_be_used_for_training") is not False:
        raise SystemExit("frozen challenge rows must remain excluded from repair training")
    if auth.get("private_identity_gradient") is not False:
        raise SystemExit("fusion repair crossed private gradient boundary")

    current_revision = git_revision()
    prep_receipt = json.loads(Path(args.prep_receipt).read_text(encoding="utf-8"))
    if prep_receipt.get("status") != "PASS" or prep_receipt.get("git_revision") != current_revision:
        raise SystemExit("repair preparation receipt is stale or did not pass")

    failed_challenge_path = Path(args.failed_challenge_result).resolve()
    failed_challenge = json.loads(failed_challenge_path.read_text(encoding="utf-8"))
    if failed_challenge.get("status") != "FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION":
        raise SystemExit("expected the frozen challenge failure receipt")
    if failed_challenge.get("challenge_rows_used_for_training") is not False:
        raise SystemExit("frozen challenge contamination detected")

    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-repair-curriculum.v0.3":
        raise SystemExit("fusion repair curriculum schema mismatch")
    if manifest.get("compiled_sha256") != sha256_file(curriculum_path):
        raise SystemExit("fusion repair curriculum hash mismatch")
    if manifest.get("frozen_challenge_rows_reused") is not False:
        raise SystemExit("fusion repair curriculum reused frozen challenge rows")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("fusion repair curriculum crossed private boundary")

    rows = read_jsonl(curriculum_path)
    if len(rows) != int(training_cfg["curriculum"]["rows"]):
        raise SystemExit("fusion repair curriculum row count mismatch")

    opt_cfg = training_cfg["optimization"]
    seed_everything(int(opt_cfg["seed"]))
    max_steps = int(opt_cfg["max_steps"])
    save_steps = [int(step) for step in opt_cfg["save_steps"]]
    train_batch_size = int(opt_cfg["train_batch_size"])
    eval_batch_size = int(opt_cfg["eval_batch_size"])
    learning_rate = float(opt_cfg["learning_rate"])
    warmup_steps = int(opt_cfg["warmup_steps"])
    weight_decay = float(opt_cfg["weight_decay"])

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import load_file

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

    structured = StructuredStateEncoder(load_structured_config(Path(args.structured_config).resolve()))
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_receipt = json.loads((structured_checkpoint / "receipt.json").read_text(encoding="utf-8"))
    if int(structured_receipt.get("step", -1)) != EXPECTED_STRUCTURED_PARENT_STEP:
        raise SystemExit("fusion repair requires ratified structured step80")
    structured.load_state_dict(
        load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu"),
        strict=True,
    )
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

    print("fusion_repair_parent_cache_start=true", flush=True)
    cache = build_parent_cache(
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
    cache_path = output_root / "fusion_repair_parent_cache.pt"
    torch.save(cache, cache_path)
    print(f"fusion_repair_parent_cache_sha256={sha256_file(cache_path)}", flush=True)

    lineage = {
        "semantic_sha256": sha256_file(semantic_checkpoint / "alice_n0_v02.safetensors"),
        "structured_sha256": sha256_file(structured_checkpoint / "structured_state.safetensors"),
        "evidence_adapter_sha256": sha256_file(evidence_adapter_path),
        "evidence_graph_sha256": sha256_file(evidence_graph_path),
        "fusion_parent_sha256": sha256_file(Path(args.fusion_parent_checkpoint).resolve() / "cross_context_fusion.safetensors"),
        "curriculum_sha256": sha256_file(curriculum_path),
        "curriculum_manifest_sha256": sha256_file(manifest_path),
        "prep_receipt_sha256": sha256_file(Path(args.prep_receipt).resolve()),
        "failed_challenge_result_sha256": sha256_file(failed_challenge_path),
        "parent_cache_sha256": sha256_file(cache_path),
    }
    if lineage["fusion_parent_sha256"] != training_cfg["fusion_parent"]["sha256"]:
        raise SystemExit("fusion repair parent hash mismatch")

    del semantic_model, semantic_state, structured, evidence_adapter, evidence_graph
    torch.cuda.empty_cache()

    train_indices = [i for i, split in enumerate(cache["splits"]) if split == "train"]
    dev_indices = [i for i, split in enumerate(cache["splits"]) if split == "dev"]
    if len(train_indices) != int(training_cfg["curriculum"]["train_rows"]):
        raise SystemExit("fusion repair train split count mismatch")
    if len(dev_indices) != int(training_cfg["curriculum"]["dev_rows"]):
        raise SystemExit("fusion repair dev split count mismatch")
    train_dataset = FusionDataset(cache, train_indices)
    dev_dataset = FusionDataset(cache, dev_indices)

    fusion_config = load_fusion_config(Path(args.fusion_config).resolve())
    model = SourceAnchoredCrossContextFusion(fusion_config).to(device)
    parent_state = load_file(
        str(Path(args.fusion_parent_checkpoint).resolve() / "cross_context_fusion.safetensors"),
        device="cpu",
    )
    model.load_state_dict(parent_state, strict=True)
    del parent_state
    report = model.parameter_report()
    print("fusion_repair_parameter_report=" + json.dumps(report, sort_keys=True), flush=True)
    if report.get("hard_parameter_ceiling") is not None or report.get("view_count_ceiling") is not None:
        raise SystemExit("source-anchored fusion unexpectedly contains a hard capability ceiling")
    if report.get("explicit_source_view_anchor_channel") is not True:
        raise SystemExit("source summary anchor channel missing")
    if report.get("explicit_source_token_anchor_channel") is not True:
        raise SystemExit("source token anchor channel missing")

    weight_cfg = training_cfg["objective"]
    loss_weights = FusionRepairObjectiveWeights(
        fused_semantic=float(weight_cfg["fused_semantic"]),
        view_routing=float(weight_cfg["view_routing"]),
        contextualized_source_alignment=float(weight_cfg["contextualized_source_alignment"]),
        disagreement_geometry=float(weight_cfg["disagreement_geometry"]),
    )

    baseline = evaluate(
        model=model,
        dataset=dev_dataset,
        cache=cache,
        device=device,
        batch_size=eval_batch_size,
    )
    print("fusion_repair_parent_baseline=" + json.dumps(baseline, sort_keys=True), flush=True)
    if baseline["source_anchor_summary_max_abs_error"] != 0.0:
        raise SystemExit("source summary anchor is not exact before repair gradient")
    if baseline["source_anchor_token_max_abs_error"] != 0.0:
        raise SystemExit("source token anchor is not exact before repair gradient")

    generator = torch.Generator().manual_seed(int(opt_cfg["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    loader_iter = iter(train_loader)
    saved: list[dict[str, Any]] = []
    running: defaultdict[str, float] = defaultdict(float)

    model.train()
    for repair_step in range(1, max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = forward_fusion(model, batch)
            losses = objective(output, batch, loss_weights)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for key, value in losses.items():
            running[key] += float(value.detach().float().cpu())
        if repair_step % 20 == 0:
            averaged = {key: value / 20.0 for key, value in running.items()}
            print(
                "fusion_repair_train_step="
                + json.dumps(
                    {"repair_step": repair_step, "lr": optimizer.param_groups[0]["lr"], **averaged},
                    sort_keys=True,
                ),
                flush=True,
            )
            running.clear()

        if repair_step in save_steps:
            metrics = evaluate(
                model=model,
                dataset=dev_dataset,
                cache=cache,
                device=device,
                batch_size=eval_batch_size,
            )
            receipt = save_checkpoint(
                model=model,
                root=output_root,
                repair_step=repair_step,
                metrics=metrics,
                baseline=baseline,
                lineage=lineage,
            )
            saved.append(receipt)
            print("fusion_repair_checkpoint_eval=" + json.dumps(receipt, sort_keys=True), flush=True)
            model.train()

    if not saved:
        raise SystemExit("fusion repair produced no checkpoints")
    winner = max(
        saved,
        key=lambda item: capability_score(item["dev_metrics"], int(item["repair_step"])),
    )
    comparison = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-repair-comparison.v0.2",
        "status": (
            "PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE"
            if winner["repair_gate_pass"]
            else "FULL_SCALE_REPAIR_NEEDS_MORE_FAILURE_DRIVEN_WORK"
        ),
        "architecture": (
            "full_scale_frontier_multi_stream_gated_bidirectional_cross_attention"
            "_with_explicit_source_anchors"
        ),
        "parent_baseline": baseline,
        "checkpoints": saved,
        "provisional_winner_repair_step": winner["repair_step"],
        "winner": winner,
        "repair_gate_pass": bool(winner["repair_gate_pass"]),
        "selection_policy": (
            "worst_family_routing_then_macro_routing_then_decisive_then_semantic_then_"
            "anchor_exactness_then_contextualized_alignment_then_geometry_then_earlier"
        ),
        "frozen_challenge_rows_used_for_training": False,
        "failed_challenge_is_diagnostic_only": True,
        "old_frozen_challenge_is_retired_for_ratification_after_informing_repair": True,
        "next_action": (
            "build_new_untouched_fusion_challenge_v0.2_before_ratification"
            if winner["repair_gate_pass"]
            else "continue_failure_driven_repair_without_using_frozen_challenge_rows"
        ),
        "hard_parameter_ceiling": None,
        "view_count_ceiling": None,
        "run_step_bound_is_compute_control_not_model_limit": True,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
    }
    comparison_path = output_root / "cross_context_fusion_repair_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
