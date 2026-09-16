#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import AdaptiveMultiViewLatentPoolV02

import eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 as prior_eval
from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
    LatentPoolDataset,
    latent_forward,
    load_ratified_fusion,
    precompute_fusion_cache,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import load_config as load_latent_config
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale import corrected_evaluate
from train_n0_v02_cross_context_fusion_full_scale import (
    build_parent_cache,
    git_revision,
    read_jsonl,
    sha256_file,
    to_device,
)


def _rebalance_distribution(available: list[bool]) -> list[float]:
    count = sum(1 for value in available if value)
    if count < 1:
        raise ValueError("counterfactual must leave at least one source view available")
    return [1.0 / count if value else 0.0 for value in available]


def build_prefusion_counterfactual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modified = copy.deepcopy(rows)
    for item in modified:
        if not item.get("counterfactual_required"):
            continue
        view = int(item["counterfactual_view"])
        available = [bool(value) for value in item["view_available"]]
        if not available[view]:
            raise ValueError(f"counterfactual view {view} is already unavailable for {item['id']}")
        available[view] = False
        reliability = [float(value) for value in item["view_reliability"]]
        reliability[view] = 0.0
        item["view_available"] = available
        item["view_reliability"] = reliability
        item["target_view_distribution"] = _rebalance_distribution(available)
        item["counterfactual_applied_before_parent_cache"] = True
        item["counterfactual_applied_before_fusion"] = True
    return modified


def prefusion_counterfactual_metrics(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    normal_dataset: LatentPoolDataset,
    counterfactual_dataset: LatentPoolDataset,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cf_loader = DataLoader(counterfactual_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    drops: list[float] = []
    family_drops: dict[str, list[float]] = defaultdict(list)
    model.eval()

    with torch.inference_mode():
        for normal_batch, cf_batch in zip(normal_loader, cf_loader):
            normal_batch = to_device(normal_batch, device)
            cf_batch = to_device(cf_batch, device)
            normal_indices = [int(value) for value in normal_batch["global_index"].cpu().tolist()]
            cf_indices = [int(value) for value in cf_batch["global_index"].cpu().tolist()]
            if normal_indices != cf_indices:
                raise RuntimeError("normal/counterfactual dataset ordering drift")

            required = torch.tensor(
                [bool(rows[index].get("counterfactual_required")) for index in normal_indices],
                device=device,
                dtype=torch.bool,
            )
            if not required.any():
                continue

            with torch.amp.autocast("cuda", dtype=torch.float16):
                normal_output = latent_forward(model, normal_batch)
                cf_output = latent_forward(model, cf_batch)

            target = F.normalize(normal_batch["semantic_target"].float(), dim=-1).unsqueeze(1)
            normal_best = (
                F.normalize(normal_output["latent_slots"].float(), dim=-1) * target
            ).sum(dim=-1).max(dim=-1).values
            cf_best = (
                F.normalize(cf_output["latent_slots"].float(), dim=-1) * target
            ).sum(dim=-1).max(dim=-1).values
            delta = normal_best - cf_best

            for local, global_index in enumerate(normal_indices):
                if not required[local]:
                    continue
                value = float(delta[local].detach().cpu())
                drops.append(value)
                family_drops[str(rows[global_index]["family"])].append(value)

    family_mean = {key: sum(values) / len(values) for key, values in sorted(family_drops.items())}
    return {
        "counterfactual_required_rows": len(drops),
        "counterfactual_intervention_point": "before_parent_cache_and_before_cross_context_fusion",
        "counterfactual_rebuilt_parent_representations": True,
        "counterfactual_rebuilt_fusion_representations": True,
        "counterfactual_mean_target_drop": sum(drops) / max(len(drops), 1),
        "counterfactual_family_mean_target_drop": family_mean,
        "counterfactual_family_min_mean_target_drop": min(family_mean.values()) if family_mean else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--freeze-receipt", required=True)
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
        raise SystemExit("latent frozen challenge v0.2 requires CUDA")
    device = torch.device("cuda")

    challenge_path = Path(args.challenge).resolve()
    manifest_path = Path(args.manifest).resolve()
    spec_path = Path(args.spec).resolve()
    freeze_path = Path(args.freeze_receipt).resolve()
    candidate_path = Path(args.candidate).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    if freeze.get("status") != "FROZEN_CONFIRMATORY_BEFORE_EVALUATION":
        raise SystemExit("latent v0.2 challenge freeze receipt missing")
    if freeze.get("git_revision") != git_revision():
        raise SystemExit("latent v0.2 challenge freeze receipt stale")
    for path, key in [
        (challenge_path, "challenge_sha256"),
        (manifest_path, "manifest_sha256"),
        (spec_path, "spec_sha256"),
        (candidate_path, "candidate_latent_pool_sha256"),
    ]:
        if freeze.get(key) != sha256_file(path):
            raise SystemExit(f"latent v0.2 frozen artifact drift: {key}")
    if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False:
        raise SystemExit("latent v0.2 challenge governance drift")
    if manifest.get("counterfactual_intervention_point") != "before_parent_cache_and_before_cross_context_fusion":
        raise SystemExit("latent v0.2 intervention-point drift")
    expected = str(spec["candidate_selection"]["latent_pool_sha256"])
    if sha256_file(candidate_path) != expected:
        raise SystemExit("latent v0.2 candidate hash drift")

    rows = read_jsonl(challenge_path)
    if len(rows) != int(spec["challenge"]["rows"]):
        raise SystemExit("latent v0.2 challenge row count drift")
    counterfactual_rows = build_prefusion_counterfactual_rows(rows)

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

    normal_parent_cache = build_parent_cache(
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
    cf_parent_cache = build_parent_cache(
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
    normal_fusion_cache = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=normal_parent_cache,
        device=device,
        batch_size=16,
    )
    cf_fusion_cache = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=cf_parent_cache,
        device=device,
        batch_size=16,
    )
    del fusion
    torch.cuda.empty_cache()

    model = AdaptiveMultiViewLatentPoolV02(load_latent_config(Path(args.latent_config).resolve()))
    model.load_state_dict(load_file(str(candidate_path), device="cpu"), strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device).eval()

    indices = list(range(len(rows)))
    normal_dataset = LatentPoolDataset(normal_parent_cache, normal_fusion_cache, indices)
    cf_dataset = LatentPoolDataset(cf_parent_cache, cf_fusion_cache, indices)
    metrics = corrected_evaluate(
        model=model,
        dataset=normal_dataset,
        parent_cache=normal_parent_cache,
        device=device,
        batch_size=16,
    )
    cf = prefusion_counterfactual_metrics(
        model=model,
        normal_dataset=normal_dataset,
        counterfactual_dataset=cf_dataset,
        rows=rows,
        device=device,
        batch_size=16,
    )
    passed, checks = prior_eval.gate(metrics, cf, spec["ratification_gate"])

    result = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-result.v0.2",
        "status": "PASS_PRESELECTED_CANDIDATE_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION" if passed else "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION",
        "git_revision": git_revision(),
        "candidate_step": 360,
        "candidate_latent_pool_sha256": sha256_file(candidate_path),
        "candidate_preselected_before_challenge": True,
        "candidate_weights_changed_after_v0_1": False,
        "checkpoint_selection_performed_on_challenge": False,
        "challenge_rows_used_for_training": False,
        "gradient_performed": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "metrics": metrics,
        "counterfactual_metrics": cf,
        "counterfactual_intervention_point": "before_parent_cache_and_before_cross_context_fusion",
        "historical_v0_1_reclassified_as_pass": False,
        "frozen_gate": spec["ratification_gate"],
        "gate_checks": checks,
        "gate_pass": passed,
        "mean_max_offdiag_slot_cosine_role": "diagnostic_only_not_a_gate",
        "next_action": (
            "write_latent_pool_ratification_manifest_and_advance_to_multi_head_identity_decision_packet_scaffolding"
            if passed
            else "diagnose_valid_pre_fusion_capability_failure_without_training_on_challenge_rows"
        ),
        "production_promotion_authorized": False,
        "n0_complete": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
