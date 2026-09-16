#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.cross_context_fusion import CrossContextFusion
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
    capability_score,
    evaluate,
    expanded_adapter_config,
    expanded_graph_config,
    git_revision,
    load_fusion_config,
    load_structured_config,
    read_jsonl,
    sha256_file,
)


def gate_pass(metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(metrics["family_min_routing_similarity"]) >= float(gate["family_min_routing_similarity_min"])
        and float(metrics["family_macro_routing_similarity"]) >= float(gate["family_macro_routing_similarity_min"])
        and float(metrics["decisive_view_accuracy"]) >= float(gate["decisive_view_accuracy_min"])
        and float(metrics["fused_semantic_cosine"]) >= float(gate["fused_semantic_cosine_min"])
        and float(metrics["view_preservation_cosine"]) >= float(gate["view_preservation_cosine_min"])
        and float(metrics["disagreement_geometry_mae"]) <= float(gate["disagreement_geometry_mae_max"])
        and float(metrics["missing_view_max_weight"]) <= float(gate["missing_view_max_weight_max"])
    )


def load_fusion_checkpoint(
    checkpoint_dir: Path,
    *,
    fusion_config_path: Path,
    device: torch.device,
) -> tuple[CrossContextFusion, dict[str, Any]]:
    receipt_path = checkpoint_dir / "receipt.json"
    model_path = checkpoint_dir / "cross_context_fusion.safetensors"
    if not receipt_path.is_file() or not model_path.is_file():
        raise SystemExit(f"incomplete fusion checkpoint: {checkpoint_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "TRAINED_FULL_SCALE_NOT_RATIFIED":
        raise SystemExit(f"unexpected fusion checkpoint status: {checkpoint_dir}")
    if receipt.get("full_scale_model") is not True or receipt.get("reduced_capability_pilot") is not False:
        raise SystemExit(f"checkpoint is not the full-scale fusion model: {checkpoint_dir}")
    if receipt.get("private_identity_gradient") is not False:
        raise SystemExit(f"private identity gradient contamination: {checkpoint_dir}")
    if receipt.get("fusion_sha256") != sha256_file(model_path):
        raise SystemExit(f"fusion checkpoint hash mismatch: {checkpoint_dir}")

    model = CrossContextFusion(load_fusion_config(fusion_config_path))
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
    parser.add_argument("--training-comparison", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-max-length", type=int, default=96)
    parser.add_argument("--field-max-length", type=int, default=96)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("frozen full-scale fusion challenge requires CUDA")
    device = torch.device("cuda")

    spec_path = Path(args.challenge_spec).resolve()
    challenge_path = Path(args.challenge).resolve()
    manifest_path = Path(args.challenge_manifest).resolve()
    comparison_path = Path(args.training_comparison).resolve()
    training_root = Path(args.training_root).resolve()
    fusion_config_path = Path(args.fusion_config).resolve()
    output_path = Path(args.output).resolve()

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    if spec.get("status") != "FROZEN_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("frozen challenge spec status drift")
    if spec.get("training_use") != "FORBIDDEN":
        raise SystemExit("frozen challenge training-use rule missing")
    if manifest.get("status") != "FROZEN_UNTOUCHED_NOT_EVALUATED":
        raise SystemExit("frozen challenge manifest status drift")
    if manifest.get("challenge_sha256") != sha256_file(challenge_path):
        raise SystemExit("frozen challenge hash mismatch")
    if manifest.get("spec_sha256") != sha256_file(spec_path):
        raise SystemExit("frozen challenge spec hash mismatch")
    if manifest.get("training_authorized") is not False:
        raise SystemExit("frozen challenge may not authorize training")
    if manifest.get("frozen_challenge_training_use_forbidden") is not True:
        raise SystemExit("frozen challenge training-use prohibition missing")
    if manifest.get("results_observed_at_compile_time") is not False:
        raise SystemExit("challenge was not frozen before results")
    if manifest.get("private_identity_content") is not False:
        raise SystemExit("challenge crossed private boundary")
    if comparison.get("status") != "PASS_FULL_SCALE_TRAINED_NOT_RATIFIED":
        raise SystemExit("training comparison is not eligible for frozen challenge")
    if comparison.get("production_promotion_authorized") is not False:
        raise SystemExit("fusion was promoted before frozen challenge")

    candidates = [int(step) for step in spec["checkpoint_candidates"]]
    comparison_steps = {int(item["step"]) for item in comparison["checkpoints"]}
    if set(candidates) != comparison_steps:
        raise SystemExit(
            f"challenge candidate set differs from trained checkpoint set: spec={candidates} training={sorted(comparison_steps)}"
        )

    rows = read_jsonl(challenge_path)
    if len(rows) != int(spec["challenge"]["rows"]):
        raise SystemExit("challenge row count drift")
    if any(row.get("training_authorized") is not False for row in rows):
        raise SystemExit("challenge row unexpectedly authorizes training")
    if any(row.get("frozen_challenge_training_use_forbidden") is not True for row in rows):
        raise SystemExit("challenge row training-use prohibition missing")

    semantic_config = load_n0_config(Path(args.semantic_config).resolve())
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    semantic_model = AliceN0V02Model(semantic_config)
    semantic_state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"), device="cpu"
    )
    missing, unexpected = semantic_model.load_state_dict(semantic_state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic state mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic_model.parameter_report()["total_parameters"] != EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False
    semantic_model.to(device).eval()
    tokenizer = load_tokenizer(Path(args.tokenizer_dir).resolve())

    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    structured_receipt = json.loads(
        (structured_checkpoint / "receipt.json").read_text(encoding="utf-8")
    )
    if int(structured_receipt.get("step", -1)) != EXPECTED_STRUCTURED_PARENT_STEP:
        raise SystemExit("frozen challenge requires ratified structured step80")
    structured = StructuredStateEncoder(
        load_structured_config(Path(args.structured_config).resolve())
    )
    structured.load_state_dict(
        load_file(str(structured_checkpoint / "structured_state.safetensors"), device="cpu"),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    evidence_adapter_path = Path(args.evidence_adapter).resolve()
    evidence_adapter = EvidenceViewAdapter(expanded_adapter_config())
    evidence_adapter.load_state_dict(
        load_file(str(evidence_adapter_path), device="cpu"), strict=True
    )
    for parameter in evidence_adapter.parameters():
        parameter.requires_grad = False

    evidence_graph_path = Path(args.evidence_graph).resolve()
    evidence_graph = DualEndpointEvidenceGraphEncoder(expanded_graph_config())
    evidence_graph.load_state_dict(
        load_file(str(evidence_graph_path), device="cpu"), strict=True
    )
    for parameter in evidence_graph.parameters():
        parameter.requires_grad = False

    print("fusion_frozen_challenge_parent_cache_start=true", flush=True)
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

    gate = spec["ratification_gate"]
    results: list[dict[str, Any]] = []
    for step in candidates:
        checkpoint_dir = training_root / f"step-{step:08d}"
        model, receipt = load_fusion_checkpoint(
            checkpoint_dir,
            fusion_config_path=fusion_config_path,
            device=device,
        )
        metrics = evaluate(
            model=model,
            dataset=dataset,
            cache=cache,
            device=device,
            batch_size=args.eval_batch_size,
        )
        result = {
            "step": step,
            "checkpoint": str(checkpoint_dir),
            "fusion_sha256": receipt["fusion_sha256"],
            "git_revision_at_training": receipt.get("git_revision"),
            "metrics": metrics,
            "ratification_gate_pass": gate_pass(metrics, gate),
        }
        results.append(result)
        print("fusion_frozen_challenge_checkpoint=" + json.dumps(result, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    passing = [item for item in results if item["ratification_gate_pass"]]
    winner = None
    if passing:
        winner = max(
            passing,
            key=lambda item: capability_score(item["metrics"], int(item["step"])),
        )

    report = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-result.v0.1",
        "status": (
            "PASS_RATIFICATION_CANDIDATE_SELECTED"
            if winner is not None
            else "FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION"
        ),
        "git_revision_at_evaluation": git_revision(),
        "challenge_sha256": sha256_file(challenge_path),
        "challenge_manifest_sha256": sha256_file(manifest_path),
        "challenge_spec_sha256": sha256_file(spec_path),
        "training_comparison_sha256": sha256_file(comparison_path),
        "ratification_gate": gate,
        "selection_policy": spec["selection_policy"],
        "results": results,
        "ratification_candidate_step": None if winner is None else int(winner["step"]),
        "ratification_candidate": winner,
        "challenge_rows_used_for_training": False,
        "gradient_performed": False,
        "parents_mutated": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "next_action": (
            "write_fusion_ratification_manifest_and_advance_to_adaptive_multi_view_pooling"
            if winner is not None
            else "perform_failure_driven_analysis_without_training_on_frozen_challenge_rows"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
