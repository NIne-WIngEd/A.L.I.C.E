#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

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
    sha256_file,
)
from train_n0_v02_cross_context_fusion_repair_full_scale import evaluate


def gate_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(metrics["family_min_routing_similarity"]) >= float(gate["family_min_routing_similarity_min"])
        and float(metrics["family_macro_routing_similarity"]) >= float(gate["family_macro_routing_similarity_min"])
        and float(metrics["decisive_view_accuracy"]) >= float(gate["decisive_view_accuracy_min"])
        and float(metrics["fused_semantic_cosine"]) >= float(gate["fused_semantic_cosine_min"])
        and float(metrics["source_anchor_summary_max_abs_error"])
        <= float(gate["source_anchor_summary_max_abs_error_max"])
        and float(metrics["source_anchor_token_max_abs_error"])
        <= float(gate["source_anchor_token_max_abs_error_max"])
        and float(metrics["disagreement_geometry_mae"]) <= float(gate["disagreement_geometry_mae_max"])
        and float(metrics["missing_view_max_weight"]) <= float(gate["missing_view_max_weight_max"])
    )


def load_candidate(
    checkpoint_dir: Path,
    *,
    fusion_config_path: Path,
    expected_sha256: str,
    expected_repair_step: int,
    device: torch.device,
) -> tuple[SourceAnchoredCrossContextFusion, dict[str, Any]]:
    receipt_path = checkpoint_dir / "receipt.json"
    model_path = checkpoint_dir / "cross_context_fusion.safetensors"
    if not receipt_path.is_file() or not model_path.is_file():
        raise SystemExit(f"incomplete confirmatory fusion candidate: {checkpoint_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "TRAINED_FULL_SCALE_REPAIR_NOT_RATIFIED":
        raise SystemExit("confirmatory candidate status mismatch")
    if int(receipt.get("repair_step", -1)) != expected_repair_step:
        raise SystemExit("confirmatory candidate repair step mismatch")
    if receipt.get("full_scale_model") is not True or receipt.get("reduced_capability_pilot") is not False:
        raise SystemExit("confirmatory candidate is not full scale")
    if receipt.get("private_identity_gradient") is not False:
        raise SystemExit("confirmatory candidate crossed private gradient boundary")
    if receipt.get("fusion_sha256") != expected_sha256:
        raise SystemExit("confirmatory candidate receipt hash differs from frozen spec")
    if sha256_file(model_path) != expected_sha256:
        raise SystemExit("confirmatory candidate file hash differs from frozen spec")
    report = receipt.get("fusion_parameter_report", {})
    if report.get("explicit_source_view_anchor_channel") is not True:
        raise SystemExit("candidate lacks explicit source summary anchor")
    if report.get("explicit_source_token_anchor_channel") is not True:
        raise SystemExit("candidate lacks explicit source token anchor")
    if report.get("hard_parameter_ceiling") is not None or report.get("view_count_ceiling") is not None:
        raise SystemExit("candidate unexpectedly contains a capability ceiling")

    model = SourceAnchoredCrossContextFusion(load_fusion_config(fusion_config_path))
    model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
    model.to(device).eval()
    return model, receipt


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
    parser.add_argument("--challenge-spec", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--challenge-manifest", required=True)
    parser.add_argument("--repair-comparison", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("confirmatory full-scale fusion challenge v0.2 requires CUDA")
    device = torch.device("cuda")

    spec_path = Path(args.challenge_spec).resolve()
    challenge_path = Path(args.challenge).resolve()
    manifest_path = Path(args.challenge_manifest).resolve()
    repair_comparison_path = Path(args.repair_comparison).resolve()
    candidate_checkpoint = Path(args.candidate_checkpoint).resolve()
    fusion_config_path = Path(args.fusion_config).resolve()
    output_path = Path(args.output).resolve()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repair_comparison = json.loads(repair_comparison_path.read_text(encoding="utf-8"))

    if spec.get("status") != "FROZEN_CONFIRMATORY_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("confirmatory challenge spec status drift")
    if spec.get("training_use") != "FORBIDDEN":
        raise SystemExit("confirmatory challenge training use must be forbidden")
    if spec.get("candidate_selection", {}).get("preselected_before_challenge") is not True:
        raise SystemExit("candidate was not frozen before confirmatory challenge")
    if spec.get("candidate_selection", {}).get("challenge_may_not_select_a_different_checkpoint") is not True:
        raise SystemExit("confirmatory challenge may not select checkpoints")
    if spec.get("ratification_gate", {}).get("thresholds_are_frozen_before_results") is not True:
        raise SystemExit("confirmatory thresholds are not frozen")

    if manifest.get("status") != "FROZEN_UNTOUCHED_CONFIRMATORY_NOT_EVALUATED":
        raise SystemExit("confirmatory challenge manifest status drift")
    if manifest.get("challenge_sha256") != sha256_file(challenge_path):
        raise SystemExit("confirmatory challenge hash mismatch")
    if manifest.get("spec_sha256") != sha256_file(spec_path):
        raise SystemExit("confirmatory challenge spec hash mismatch")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("confirmatory challenge may not authorize training")
    if manifest.get("frozen_challenge_training_use_forbidden") is not True:
        raise SystemExit("confirmatory training-use prohibition missing")
    if manifest.get("results_observed_at_compile_time") is not False:
        raise SystemExit("confirmatory challenge was not frozen before results")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("confirmatory challenge crossed private boundary")

    candidate_spec = spec["candidate_selection"]
    expected_step = int(candidate_spec["repair_step"])
    expected_sha = str(candidate_spec["fusion_sha256"])
    if int(manifest.get("candidate_repair_step", -1)) != expected_step:
        raise SystemExit("manifest candidate step differs from frozen spec")
    if manifest.get("candidate_fusion_sha256") != expected_sha:
        raise SystemExit("manifest candidate hash differs from frozen spec")

    if repair_comparison.get("status") != "PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE":
        raise SystemExit("repair comparison is not eligible for confirmatory challenge")
    if int(repair_comparison.get("provisional_winner_repair_step", -1)) != expected_step:
        raise SystemExit("preselected repair winner step drift")
    winner = repair_comparison.get("winner", {})
    if winner.get("fusion_sha256") != expected_sha:
        raise SystemExit("preselected repair winner hash drift")
    if winner.get("repair_gate_pass") is not True:
        raise SystemExit("preselected repair winner did not pass repair gate")
    if repair_comparison.get("frozen_challenge_rows_used_for_training") is not False:
        raise SystemExit("repair comparison reports frozen-challenge contamination")
    if repair_comparison.get("private_identity_gradient") is not False:
        raise SystemExit("repair comparison crossed private gradient boundary")

    rows = read_jsonl(challenge_path)
    if len(rows) != int(spec["challenge"]["rows"]):
        raise SystemExit("confirmatory challenge row count drift")
    if len({row["family"] for row in rows}) != int(spec["challenge"]["families"]):
        raise SystemExit("confirmatory challenge family count drift")
    if any(row.get("training_authorized") is not False for row in rows):
        raise SystemExit("confirmatory row unexpectedly authorizes training")
    if any(row.get("frozen_challenge_training_use_forbidden") is not True for row in rows):
        raise SystemExit("confirmatory row training-use prohibition missing")

    semantic_config = load_n0_config(Path(args.semantic_config).resolve())
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    semantic_model = AliceN0V02Model(semantic_config)
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
        raise SystemExit("confirmatory challenge requires ratified structured step80")
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

    print("fusion_confirmatory_v02_parent_cache_start=true", flush=True)
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
    dataset = FusionDataset(cache, list(range(len(rows))))

    del semantic_model, semantic_state, structured, evidence_adapter, evidence_graph
    torch.cuda.empty_cache()

    model, candidate_receipt = load_candidate(
        candidate_checkpoint,
        fusion_config_path=fusion_config_path,
        expected_sha256=expected_sha,
        expected_repair_step=expected_step,
        device=device,
    )
    metrics = evaluate(
        model=model,
        dataset=dataset,
        cache=cache,
        device=device,
        batch_size=args.eval_batch_size,
    )
    gate = spec["ratification_gate"]
    passed = gate_pass(metrics, gate)
    candidate_result = {
        "repair_step": expected_step,
        "total_fusion_update_count": int(candidate_receipt.get("total_fusion_update_count", -1)),
        "checkpoint": str(candidate_checkpoint),
        "fusion_sha256": expected_sha,
        "git_revision_at_training": candidate_receipt.get("git_revision"),
        "metrics": metrics,
        "ratification_gate_pass": passed,
    }
    print("fusion_confirmatory_v02_candidate=" + json.dumps(candidate_result, sort_keys=True), flush=True)

    report = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-result.v0.2",
        "status": "PASS_PRESELECTED_CANDIDATE_ELIGIBLE_FOR_FUSION_RATIFICATION" if passed else "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_RATIFICATION",
        "git_revision_at_evaluation": git_revision(),
        "candidate_preselected_before_challenge": True,
        "checkpoint_selection_performed_on_challenge": False,
        "challenge_sha256": sha256_file(challenge_path),
        "challenge_manifest_sha256": sha256_file(manifest_path),
        "challenge_spec_sha256": sha256_file(spec_path),
        "repair_comparison_sha256": sha256_file(repair_comparison_path),
        "ratification_gate": gate,
        "decision_policy": spec["decision_policy"],
        "candidate": candidate_result,
        "ratification_candidate_repair_step": expected_step if passed else None,
        "ratification_candidate_sha256": expected_sha if passed else None,
        "challenge_rows_used_for_training": False,
        "gradient_performed": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": "write_fusion_ratification_manifest_and_advance_to_adaptive_multi_view_latent_pooling" if passed else "retire_confirmatory_challenge_v0.2_and_perform_failure_driven_analysis_without_training_on_its_rows",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
