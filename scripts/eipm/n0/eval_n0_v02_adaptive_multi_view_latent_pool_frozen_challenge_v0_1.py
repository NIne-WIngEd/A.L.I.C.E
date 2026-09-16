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

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import AdaptiveMultiViewLatentPoolV02
from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

from train_n0_v02_adaptive_multi_view_latent_pool_full_scale import (
    LatentPoolDataset,
    latent_forward,
    load_ratified_fusion,
    precompute_fusion_cache,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import load_config as load_latent_config
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale import corrected_evaluate
from train_n0_v02_cross_context_fusion_full_scale import (
    EXPECTED_SEMANTIC_PARAMETERS,
    build_parent_cache,
    expanded_adapter_config,
    expanded_graph_config,
    git_revision,
    load_structured_config,
    read_jsonl,
    sha256_file,
    to_device,
)


def load_public_parents(
    *,
    semantic_config_path: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    structured_config_path: Path,
    structured_checkpoint: Path,
    evidence_adapter_path: Path,
    evidence_graph_path: Path,
    device: torch.device,
) -> tuple[AliceN0V02Model, Any, StructuredStateEncoder, EvidenceViewAdapter, DualEndpointEvidenceGraphEncoder]:
    semantic_model = AliceN0V02Model(load_n0_config(semantic_config_path))
    semantic_model.load_state_dict(
        load_file(str(semantic_checkpoint / "alice_n0_v02.safetensors"), device="cpu"),
        strict=True,
    )
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parent parameter drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model = semantic_model.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    structured = StructuredStateEncoder(load_structured_config(structured_config_path))
    structured.load_state_dict(
        load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu"),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    adapter = EvidenceViewAdapter(expanded_adapter_config())
    adapter.load_state_dict(load_file(str(evidence_adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False

    graph = DualEndpointEvidenceGraphEncoder(expanded_graph_config())
    graph.load_state_dict(load_file(str(evidence_graph_path), device="cpu"), strict=True)
    for parameter in graph.parameters():
        parameter.requires_grad = False

    return semantic_model, tokenizer, structured, adapter, graph


def counterfactual_metrics(
    *,
    model: AdaptiveMultiViewLatentPoolV02,
    dataset: LatentPoolDataset,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    drops: list[float] = []
    family_drops: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                normal = latent_forward(model, batch)
            indices = [int(value) for value in batch["global_index"].cpu().tolist()]
            required = torch.tensor(
                [bool(rows[index].get("counterfactual_required")) for index in indices],
                device=device,
                dtype=torch.bool,
            )
            if not required.any():
                continue

            masks = [batch["semantic_mask"].clone(), batch["structured_mask"].clone(), batch["evidence_mask"].clone()]
            reliability = batch["view_reliability"].clone()
            routing = batch["fusion_view_weights"].clone()
            for local, global_index in enumerate(indices):
                if not required[local]:
                    continue
                view = int(rows[global_index]["counterfactual_view"])
                masks[view][local].zero_()
                reliability[local, view] = 0.0
                routing[local, view] = 0.0
            routing = routing / routing.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                counterfactual = model(
                    contextualized_view_tokens=[batch["context_semantic"], batch["context_structured"], batch["context_evidence"]],
                    source_view_tokens=[batch["source_semantic"], batch["source_structured"], batch["source_evidence"]],
                    view_valid_masks=masks,
                    query_semantic=batch["query_semantic"],
                    view_reliability=reliability,
                    fusion_view_weights=routing,
                )

            target = F.normalize(batch["semantic_target"].float(), dim=-1).unsqueeze(1)
            normal_best = (F.normalize(normal["latent_slots"].float(), dim=-1) * target).sum(dim=-1).max(dim=-1).values
            cf_best = (F.normalize(counterfactual["latent_slots"].float(), dim=-1) * target).sum(dim=-1).max(dim=-1).values
            delta = normal_best - cf_best
            for local, global_index in enumerate(indices):
                if not required[local]:
                    continue
                value = float(delta[local].cpu())
                drops.append(value)
                family_drops[str(rows[global_index]["family"])].append(value)

    family_mean = {key: sum(values) / len(values) for key, values in sorted(family_drops.items())}
    return {
        "counterfactual_required_rows": len(drops),
        "counterfactual_mean_target_drop": sum(drops) / max(len(drops), 1),
        "counterfactual_family_mean_target_drop": family_mean,
        "counterfactual_family_min_mean_target_drop": min(family_mean.values()) if family_mean else 0.0,
    }


def gate(metrics: dict[str, Any], cf: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "best_slot_semantic": float(metrics["best_slot_semantic_cosine"]) >= float(thresholds["best_slot_semantic_cosine_min"]),
        "family_min_best_slot_semantic": float(metrics["family_min_best_slot_semantic_cosine"]) >= float(thresholds["family_min_best_slot_semantic_cosine_min"]),
        "pooled_semantic": float(metrics["pooled_semantic_cosine"]) >= float(thresholds["pooled_semantic_cosine_min"]),
        "family_min_pooled_semantic": float(metrics["family_min_pooled_semantic_cosine"]) >= float(thresholds["family_min_pooled_semantic_cosine_min"]),
        "pairwise_slot_diversity": float(metrics["mean_pairwise_offdiag_slot_cosine"]) <= float(thresholds["mean_pairwise_offdiag_slot_cosine_max"]),
        "effective_rank": float(metrics["mean_centered_slot_effective_rank"]) >= float(thresholds["mean_centered_slot_effective_rank_min"]),
        "source_view_semantic_recoverability": float(metrics["mean_min_available_view_best_slot_semantic_cosine"]) >= float(thresholds["mean_min_available_view_best_slot_semantic_cosine_min"]),
        "disagreement_weighted_specialization": float(metrics["mean_disagreement_weighted_view_specialization"]) >= float(thresholds["mean_disagreement_weighted_view_specialization_min"]),
        "view_attention_coverage": float(metrics["mean_min_available_view_best_slot_attention"]) >= float(thresholds["mean_min_available_view_best_slot_attention_min"]),
        "channel_attention_coverage": float(metrics["mean_min_channel_best_slot_attention"]) >= float(thresholds["mean_min_channel_best_slot_attention_min"]),
        "missing_view_safety": float(metrics["missing_view_attention_max"]) <= float(thresholds["missing_view_attention_max"]),
        "counterfactual_mean_drop": float(cf["counterfactual_mean_target_drop"]) >= float(thresholds["counterfactual_required_rows_mean_target_drop_min"]),
        "counterfactual_family_min_drop": float(cf["counterfactual_family_min_mean_target_drop"]) >= float(thresholds["counterfactual_family_min_mean_target_drop_min"]),
    }
    return all(checks.values()), checks


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
        raise SystemExit("latent frozen challenge requires CUDA")
    device = torch.device("cuda")
    challenge_path = Path(args.challenge).resolve()
    manifest_path = Path(args.manifest).resolve()
    spec_path = Path(args.spec).resolve()
    freeze_path = Path(args.freeze_receipt).resolve()
    candidate_path = Path(args.candidate).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    if freeze.get("status") != "FROZEN_CONFIRMATORY_BEFORE_EVALUATION" or freeze.get("git_revision") != git_revision():
        raise SystemExit("latent challenge freeze receipt missing or stale")
    for path, key in [
        (challenge_path, "challenge_sha256"),
        (manifest_path, "manifest_sha256"),
        (spec_path, "spec_sha256"),
        (candidate_path, "candidate_latent_pool_sha256"),
    ]:
        if freeze.get(key) != sha256_file(path):
            raise SystemExit(f"latent challenge artifact drift: {key}")
    if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False:
        raise SystemExit("latent challenge governance drift")
    expected = str(spec["candidate_selection"]["latent_pool_sha256"])
    if sha256_file(candidate_path) != expected:
        raise SystemExit("latent challenge candidate hash drift")

    rows = read_jsonl(challenge_path)
    if len(rows) != int(spec["challenge"]["rows"]):
        raise SystemExit("latent challenge row count drift")

    semantic_model, tokenizer, structured, adapter, graph = load_public_parents(
        semantic_config_path=Path(args.semantic_config).resolve(),
        semantic_checkpoint=Path(args.semantic_checkpoint).resolve(),
        tokenizer_dir=Path(args.tokenizer_dir).resolve(),
        structured_config_path=Path(args.structured_config).resolve(),
        structured_checkpoint=Path(args.structured_checkpoint).resolve(),
        evidence_adapter_path=Path(args.evidence_adapter).resolve(),
        evidence_graph_path=Path(args.evidence_graph).resolve(),
        device=device,
    )
    parent_cache = build_parent_cache(
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
    del semantic_model, structured, adapter, graph
    torch.cuda.empty_cache()

    fusion = load_ratified_fusion(
        checkpoint_dir=Path(args.fusion_checkpoint).resolve(),
        fusion_config_path=Path(args.fusion_config).resolve(),
        ratification_path=Path(args.fusion_ratification).resolve(),
        device=device,
    )
    fusion_cache = precompute_fusion_cache(
        fusion=fusion,
        parent_cache=parent_cache,
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

    dataset = LatentPoolDataset(parent_cache, fusion_cache, list(range(len(rows))))
    metrics = corrected_evaluate(
        model=model,
        dataset=dataset,
        parent_cache=parent_cache,
        device=device,
        batch_size=16,
    )
    cf = counterfactual_metrics(
        model=model,
        dataset=dataset,
        rows=rows,
        device=device,
        batch_size=16,
    )
    passed, checks = gate(metrics, cf, spec["ratification_gate"])
    result = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-result.v0.1",
        "status": "PASS_PRESELECTED_CANDIDATE_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION" if passed else "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION",
        "git_revision": git_revision(),
        "candidate_step": 360,
        "candidate_latent_pool_sha256": sha256_file(candidate_path),
        "candidate_preselected_before_challenge": True,
        "checkpoint_selection_performed_on_challenge": False,
        "challenge_rows_used_for_training": False,
        "gradient_performed": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "metrics": metrics,
        "counterfactual_metrics": cf,
        "frozen_gate": spec["ratification_gate"],
        "gate_checks": checks,
        "gate_pass": passed,
        "mean_max_offdiag_slot_cosine_role": "diagnostic_only_not_a_gate",
        "next_action": (
            "write_latent_pool_ratification_manifest_and_advance_to_multi_head_identity_decision_packet_scaffolding"
            if passed
            else "diagnose_failed_frozen_capability_without_training_on_this_challenge_then_build_independent_repair_and_new_challenge"
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
