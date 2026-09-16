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

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.cross_context_fusion_routing_constraints import evaluate_routing_constraints
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import DualEndpointEvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

from eval_n0_v02_cross_context_fusion_frozen_challenge_v0_2 import load_candidate
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
    sha256_file,
    to_device,
)
from train_n0_v02_cross_context_fusion_repair_full_scale import forward_fusion


def _active_max_abs_error(actual: torch.Tensor, expected: torch.Tensor, active: torch.Tensor) -> float:
    if not active.any():
        return 0.0
    mask = active
    while mask.ndim < actual.ndim:
        mask = mask.unsqueeze(-1)
    values = (actual - expected).abs().masked_select(mask.expand_as(actual))
    return float(values.max().detach().float().cpu()) if values.numel() else 0.0


def evaluate_constraint_challenge(
    *,
    model: Any,
    dataset: FusionDataset,
    cache: dict[str, Any],
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    family_pass: dict[str, list[float]] = defaultdict(list)
    family_semantic: dict[str, list[float]] = defaultdict(list)
    semantic_cosines: list[float] = []
    contextualized_cosines: list[float] = []
    disagreement_errors: list[float] = []
    total_violation = 0.0
    row_passes = 0
    singleton_correct = 0
    singleton_total = 0
    missing_leak = 0.0
    anchor_summary_max_error = 0.0
    anchor_token_max_error = 0.0
    gate_values: list[float] = []
    row_results: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = forward_fusion(model, batch)

            semantic_cos = F.cosine_similarity(out["fused_state"], batch["semantic_target"], dim=-1)
            contextualized_cos = F.cosine_similarity(
                out["contextualized_view_summaries"], batch["source_view_summaries"], dim=-1
            )
            active = batch["view_available"]
            contextualized_cosines.extend(
                float(value) for value in contextualized_cos[active].detach().float().cpu().tolist()
            )
            semantic_cosines.extend(float(value) for value in semantic_cos.detach().float().cpu().tolist())

            anchor_summary_max_error = max(
                anchor_summary_max_error,
                _active_max_abs_error(out["source_view_summaries"], batch["source_view_summaries"], active),
            )
            source_token_targets = [batch["semantic_tokens"], batch["structured_tokens"], batch["evidence_tokens"]]
            source_token_masks = [batch["semantic_valid_mask"], batch["structured_valid_mask"], batch["evidence_valid_mask"]]
            for actual, expected, mask in zip(out["source_view_tokens"], source_token_targets, source_token_masks):
                anchor_token_max_error = max(anchor_token_max_error, _active_max_abs_error(actual, expected, mask))

            out_norm = F.normalize(out["contextualized_view_summaries"], dim=-1)
            src_norm = F.normalize(batch["source_view_summaries"], dim=-1)
            out_sim = torch.einsum("bvd,bwd->bvw", out_norm, out_norm)
            src_sim = torch.einsum("bvd,bwd->bvw", src_norm, src_norm)
            pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
            eye = torch.eye(active.size(1), device=device, dtype=torch.bool).unsqueeze(0)
            pair_mask = pair_mask & ~eye
            for local in range(active.size(0)):
                mask = pair_mask[local]
                if mask.any():
                    disagreement_errors.append(float((out_sim[local] - src_sim[local]).abs()[mask].mean().float().cpu()))

            unavailable = ~active
            if unavailable.any():
                missing_leak = max(missing_leak, float(out["view_weights"][unavailable].max().float().cpu()))
            gate_values.extend(float(value) for value in out["cross_gate_means"].float().cpu().reshape(-1).tolist())

            indices = [int(value) for value in batch["global_index"].detach().cpu().tolist()]
            weights_cpu = out["view_weights"].detach().float().cpu().tolist()
            semantic_cpu = semantic_cos.detach().float().cpu().tolist()
            for local, global_index in enumerate(indices):
                row = rows[global_index]
                weights = [float(value) for value in weights_cpu[local]]
                availability = [bool(value) for value in row["view_available"]]
                check = evaluate_routing_constraints(weights, row["routing_constraints"], available=availability)
                passed = bool(check["passed"])
                row_passes += int(passed)
                total_violation += float(check["total_violation"])
                family = str(row["family"])
                family_pass[family].append(float(passed))
                family_semantic[family].append(float(semantic_cpu[local]))
                allowed = row["routing_constraints"].get("allowed_top_views")
                if isinstance(allowed, list) and len(allowed) == 1:
                    singleton_total += 1
                    singleton_correct += int(max(range(len(weights)), key=weights.__getitem__) == int(allowed[0]))
                row_results.append(
                    {
                        "id": row["id"],
                        "family": family,
                        "routing_weights": weights,
                        "routing_constraint_pass": passed,
                        "routing_violations": check["violations"],
                        "fused_semantic_cosine": float(semantic_cpu[local]),
                    }
                )

    family_pass_rate = {name: sum(values) / len(values) for name, values in sorted(family_pass.items())}
    family_semantic_cosine = {name: sum(values) / len(values) for name, values in sorted(family_semantic.items())}
    return {
        "rows": len(dataset),
        "routing_constraint_pass_rate": row_passes / max(len(dataset), 1),
        "family_routing_constraint_pass_rate": family_pass_rate,
        "family_min_routing_constraint_pass_rate": min(family_pass_rate.values()),
        "singleton_required_top_accuracy": singleton_correct / max(singleton_total, 1),
        "singleton_required_top_rows": singleton_total,
        "mean_routing_constraint_violation": total_violation / max(len(dataset), 1),
        "fused_semantic_cosine": sum(semantic_cosines) / max(len(semantic_cosines), 1),
        "family_fused_semantic_cosine": family_semantic_cosine,
        "family_min_fused_semantic_cosine": min(family_semantic_cosine.values()),
        "contextualized_source_cosine": sum(contextualized_cosines) / max(len(contextualized_cosines), 1),
        "source_anchor_summary_max_abs_error": anchor_summary_max_error,
        "source_anchor_token_max_abs_error": anchor_token_max_error,
        "disagreement_geometry_mae": sum(disagreement_errors) / max(len(disagreement_errors), 1),
        "missing_view_max_weight": missing_leak,
        "mean_cross_gate_activation": sum(gate_values) / max(len(gate_values), 1),
        "row_results": row_results,
    }


def gate_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(metrics["routing_constraint_pass_rate"]) >= float(gate["routing_constraint_pass_rate_min"])
        and float(metrics["family_min_routing_constraint_pass_rate"]) >= float(gate["family_min_routing_constraint_pass_rate_min"])
        and float(metrics["singleton_required_top_accuracy"]) >= float(gate["singleton_required_top_accuracy_min"])
        and float(metrics["fused_semantic_cosine"]) >= float(gate["fused_semantic_cosine_min"])
        and float(metrics["family_min_fused_semantic_cosine"]) >= float(gate["family_min_fused_semantic_cosine_min"])
        and float(metrics["source_anchor_summary_max_abs_error"]) <= float(gate["source_anchor_summary_max_abs_error_max"])
        and float(metrics["source_anchor_token_max_abs_error"]) <= float(gate["source_anchor_token_max_abs_error_max"])
        and float(metrics["disagreement_geometry_mae"]) <= float(gate["disagreement_geometry_mae_max"])
        and float(metrics["missing_view_max_weight"]) <= float(gate["missing_view_max_weight_max"])
        and float(metrics["mean_routing_constraint_violation"]) <= float(gate["mean_routing_constraint_violation_max"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ["semantic-config","semantic-checkpoint","tokenizer-dir","structured-config","structured-checkpoint","evidence-adapter","evidence-graph","fusion-config","challenge-spec","challenge","challenge-manifest","repair-comparison","v0-2-result","candidate-checkpoint","output"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("constraint confirmatory fusion challenge v0.3 requires CUDA")
    device = torch.device("cuda")

    spec_path=Path(args.challenge_spec).resolve(); challenge_path=Path(args.challenge).resolve(); manifest_path=Path(args.challenge_manifest).resolve()
    repair_comparison_path=Path(args.repair_comparison).resolve(); v02_result_path=Path(args.v0_2_result).resolve(); candidate_checkpoint=Path(args.candidate_checkpoint).resolve(); output_path=Path(args.output).resolve(); fusion_config_path=Path(args.fusion_config).resolve()
    spec=json.loads(spec_path.read_text(encoding="utf-8")); manifest=json.loads(manifest_path.read_text(encoding="utf-8")); repair=json.loads(repair_comparison_path.read_text(encoding="utf-8")); v02=json.loads(v02_result_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CONSTRAINT_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED" or spec.get("training_use") != "FORBIDDEN":
        raise SystemExit("v0.3 frozen spec drift")
    if spec.get("v0_2_result_reclassified") is not False or spec.get("v0_2_remains_immutable_failed_historical_evaluation") is not True:
        raise SystemExit("v0.2 historical result integrity rule missing")
    if manifest.get("status") != "FROZEN_UNTOUCHED_CONSTRAINT_CONFIRMATORY_NOT_EVALUATED" or manifest.get("challenge_sha256") != sha256_file(challenge_path) or manifest.get("spec_sha256") != sha256_file(spec_path):
        raise SystemExit("v0.3 frozen manifest/hash drift")
    if manifest.get("exact_target_routing_distribution_used_for_ratification") is not False:
        raise SystemExit("v0.3 must not use exact routing distributions for ratification")
    if v02.get("status") != "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_RATIFICATION" or v02.get("gradient_performed") is not False:
        raise SystemExit("expected immutable failed v0.2 result")
    if repair.get("status") != "PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE":
        raise SystemExit("repair comparison status drift")
    candidate_spec=spec["candidate_selection"]; expected_step=int(candidate_spec["repair_step"]); expected_sha=str(candidate_spec["fusion_sha256"])
    if int(repair.get("provisional_winner_repair_step",-1)) != expected_step or repair.get("winner",{}).get("fusion_sha256") != expected_sha:
        raise SystemExit("preselected unchanged candidate lineage drift")

    rows=read_jsonl(challenge_path)
    if len(rows) != int(spec["challenge"]["rows"]) or len({row["family"] for row in rows}) != int(spec["challenge"]["families"]):
        raise SystemExit("v0.3 challenge coverage drift")
    if any(row.get("training_authorized") is not False or row.get("exact_target_routing_distribution_used_for_ratification") is not False for row in rows):
        raise SystemExit("v0.3 row governance drift")

    semantic_config=load_n0_config(Path(args.semantic_config).resolve()); semantic_checkpoint=Path(args.semantic_checkpoint).resolve(); semantic_model=AliceN0V02Model(semantic_config)
    state=load_file(str(semantic_checkpoint/"alice_n0_v02.safetensors"),device="cpu"); missing,unexpected=semantic_model.load_state_dict(state,strict=False)
    if missing or unexpected or semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parent mismatch")
    for parameter in semantic_model.parameters(): parameter.requires_grad=False
    semantic_model.to(device).eval(); tokenizer=load_tokenizer(Path(args.tokenizer_dir).resolve())
    structured_checkpoint=Path(args.structured_checkpoint).resolve(); receipt=json.loads((structured_checkpoint/"receipt.json").read_text(encoding="utf-8"))
    if int(receipt.get("step",-1)) != EXPECTED_STRUCTURED_PARENT_STEP: raise SystemExit("structured parent mismatch")
    structured=StructuredStateEncoder(load_structured_config(Path(args.structured_config).resolve())); structured.load_state_dict(load_file(str(structured_checkpoint/"structured_state.safetensors"),device="cpu"),strict=True)
    adapter=EvidenceViewAdapter(expanded_adapter_config()); adapter.load_state_dict(load_file(str(Path(args.evidence_adapter).resolve()),device="cpu"),strict=True)
    graph=DualEndpointEvidenceGraphEncoder(expanded_graph_config()); graph.load_state_dict(load_file(str(Path(args.evidence_graph).resolve()),device="cpu"),strict=True)
    for module in (structured,adapter,graph):
        for parameter in module.parameters(): parameter.requires_grad=False

    print("fusion_constraint_confirmatory_v03_parent_cache_start=true",flush=True)
    cache=build_parent_cache(rows,semantic_model=semantic_model,tokenizer=tokenizer,structured=structured,evidence_adapter=adapter,evidence_graph=graph,device=device,encode_batch_size=args.encode_batch_size,raw_max_length=args.raw_max_length,field_max_length=args.field_max_length)
    dataset=FusionDataset(cache,list(range(len(rows))))
    del semantic_model,state,structured,adapter,graph; torch.cuda.empty_cache()
    model,candidate_receipt=load_candidate(candidate_checkpoint,fusion_config_path=fusion_config_path,expected_sha256=expected_sha,expected_repair_step=expected_step,device=device)
    metrics=evaluate_constraint_challenge(model=model,dataset=dataset,cache=cache,rows=rows,device=device,batch_size=args.eval_batch_size)
    passed=gate_pass(metrics,spec["ratification_gate"])
    report={
        "schema":"alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-result.v0.3",
        "status":"PASS_UNCHANGED_CANDIDATE_ELIGIBLE_FOR_FUSION_RATIFICATION" if passed else "FAIL_UNCHANGED_CANDIDATE_NEEDS_REAL_FAILURE_DRIVEN_WORK",
        "git_revision_at_evaluation":git_revision(),
        "candidate_preselected_before_challenge":True,
        "candidate_weights_changed_after_v0_2":False,
        "checkpoint_selection_performed_on_challenge":False,
        "challenge_sha256":sha256_file(challenge_path),
        "challenge_manifest_sha256":sha256_file(manifest_path),
        "challenge_spec_sha256":sha256_file(spec_path),
        "repair_comparison_sha256":sha256_file(repair_comparison_path),
        "v0_2_failed_result_sha256":sha256_file(v02_result_path),
        "routing_validation":"behavioral_constraints_not_exact_soft_distribution_similarity",
        "ratification_gate":spec["ratification_gate"],
        "decision_policy":spec["decision_policy"],
        "candidate":{"repair_step":expected_step,"total_fusion_update_count":int(candidate_receipt.get("total_fusion_update_count",-1)),"checkpoint":str(candidate_checkpoint),"fusion_sha256":expected_sha,"metrics":metrics,"ratification_gate_pass":passed},
        "challenge_rows_used_for_training":False,"gradient_performed":False,"parents_mutated":False,"private_identity_data":False,"private_identity_gradient":False,"production_promotion_authorized":False,
        "next_action":"write_fusion_ratification_manifest_and_advance_to_adaptive_multi_view_latent_pooling" if passed else "retire_v0.3_and_build_failure_driven_constraint_based_training_without_using_v0.3_rows",
    }
    output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
