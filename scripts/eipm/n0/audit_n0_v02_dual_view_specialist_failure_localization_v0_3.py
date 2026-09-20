#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter

import train_n0_v02_relation_repair as rr
import train_n0_v02_query_edge_cross_attention_bridge_v0_1 as old
import train_n0_v02_dual_view_specialist_v0_1 as tr
import audit_n0_v02_dual_view_specialist_failure_localization_v0_1 as loc1


SCHEMA = "alice.eipm.n0.dual-view-specialist-failure-localization.v0.3"

EXPECTED_RESULT_SHA256 = (
    "1a38c37a41de68bea0bb9bc897b86d62cbc457b8b93189c2d16731110c1694ec"
)
EXPECTED_CAUSAL_CACHE_SHA256 = (
    "5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
)
EXPECTED_FIELD_TOKEN_CACHE_SHA256 = (
    "8bdf72561bafd2bbb88b56e9202570401f76cd7a0bcf48aa0abd5d8e2400ad5c"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_LAYER_MAP_SHA256 = (
    "ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
)
EXPECTED_CHECKPOINTS = {
    40: "baae065f9869af316ac6ba82695b6076cedd223a319f36fd15cd24337e993f06",
    80: "18eb6b4026472f64d12d87c2b048e1551dfe488a6e6e3ff07785c9ec4997b07f",
    120: "5296ee7ffc1b8a5d85d70f8e14a643db088eef18a14f6b9267786baba0300664",
    160: "f7d7ff95dd59758b326b3f4b70ae85e2328fb00f18a8175f02a145a4e1d9fdf0",
    200: "bbe0e38955f5887acf24649007b8bf755aa9cdda8c95ead7afaf6e1d5bbebdd7",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qstats(values: torch.Tensor) -> dict[str, float]:
    v = values.float().flatten()
    q = torch.quantile(
        v,
        torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
    )
    return {
        "min": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "max": float(q[6]),
        "mean": float(v.mean()),
        "top1_at_0_5": float(v.gt(0.5).float().mean()),
    }


@torch.inference_mode()
def causal_specialist_probabilities(
    graph: Any,
    adapter: EvidenceViewAdapter,
    dataset: tr.DualViewQuadDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts: list[torch.Tensor] = []
    for batch in loader:
        flat = tr.to_device(tr.flatten_quad_batch(batch), device)
        out = tr.candidate_forward(graph, adapter, flat)
        parts.append(out["specialist_probability"].float().cpu())
    return torch.cat(parts, dim=0)


def preservation_receipt_summary(
    result: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    key = f"step-{step:08d}"
    receipt = result["checkpoints"][key]
    ordinary = receipt["ordinary_noop_metrics"]
    endpoint = receipt["endpoint_noop_metrics"]
    return {
        "preservation_pass": bool(receipt["preservation_pass"]),
        "ordinary_noop_top1_accuracy": float(ordinary["noop_top1_accuracy"]),
        "ordinary_mean_specialist_probability": (
            1.0 - float(ordinary["mean_noop_probability"])
        ),
        "endpoint_noop_top1_accuracy": float(endpoint["noop_top1_accuracy"]),
        "endpoint_mean_specialist_probability": (
            1.0 - float(endpoint["mean_noop_probability"])
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--training-result", required=True)
    p.add_argument("--training-root", required=True)
    p.add_argument("--causal-cache", required=True)
    p.add_argument("--field-token-cache", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=24)
    args = p.parse_args()

    result_path = Path(args.training_result).resolve()
    training_root = Path(args.training_root).resolve()
    causal_cache_path = Path(args.causal_cache).resolve()
    field_cache_path = Path(args.field_token_cache).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    layer_map = Path(args.layer_map).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    output = Path(args.output).resolve()

    for path in (
        result_path,
        causal_cache_path,
        field_cache_path,
        curriculum_path,
        tokenizer_dir / "tokenizer.json",
        layer_map,
        adapter_path,
        graph_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing fast-localization input: {path}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite fast-localization audit: {output}")

    exact = {
        result_path: EXPECTED_RESULT_SHA256,
        causal_cache_path: EXPECTED_CAUSAL_CACHE_SHA256,
        field_cache_path: EXPECTED_FIELD_TOKEN_CACHE_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        layer_map: EXPECTED_LAYER_MAP_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
    }
    for path, digest in exact.items():
        if sha(path) != digest:
            raise SystemExit(f"fast-localization lineage drift: {path.name}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != (
        "FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
    ):
        raise SystemExit("source experiment status drift")
    if result.get("eligible_checkpoint_keys") != []:
        raise SystemExit("source experiment unexpectedly eligible")
    if result.get("selected_checkpoint") is not None:
        raise SystemExit("source experiment unexpectedly selected a checkpoint")
    if result.get("causal_test_split_evaluated") is not False:
        raise SystemExit("source experiment TEST contamination")
    if result.get("frozen_challenge_evaluated") is not False:
        raise SystemExit("source experiment challenge contamination")

    checkpoint_paths: dict[int, Path] = {}
    for step, digest in EXPECTED_CHECKPOINTS.items():
        cp = (
            training_root
            / f"step-{step:08d}"
            / "dual_view_query_edge_bridge.safetensors"
        )
        if not cp.is_file():
            raise SystemExit(f"missing trained checkpoint: {cp}")
        if sha(cp) != digest:
            raise SystemExit(f"checkpoint hash drift at step {step}")
        checkpoint_paths[step] = cp

    causal_rows_all = old.read_jsonl(curriculum_path)
    causal_rows = [
        row for row in causal_rows_all
        if str(row["split"]) in {"train", "dev"}
    ]
    causal = torch.load(causal_cache_path, map_location="cpu")
    if causal["ids"] != [str(row["id"]) for row in causal_rows]:
        raise SystemExit("causal cache/curriculum order drift")

    dev_indices = [
        i for i, split in enumerate(causal["splits"])
        if str(split) == "dev"
    ]
    if len(dev_indices) != 144:
        raise SystemExit(f"causal DEV row-count drift: {len(dev_indices)}")

    tokenizer = load_tokenizer(tokenizer_dir)
    query_content = tr.build_query_content_lookup(
        causal,
        causal_rows,
        dev_indices,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    field_cache = torch.load(field_cache_path, map_location="cpu")
    causal_fields = field_cache["causal_field_tokens"]
    for index in dev_indices:
        if index not in causal_fields:
            raise SystemExit(f"missing causal field-token cache row: {index}")

    causal_dev = tr.DualViewQuadDataset(
        causal,
        "dev",
        query_content_lookup=query_content,
        field_lookup=causal_fields,
    )

    if not torch.cuda.is_available():
        raise SystemExit("v0.3 localization requires CUDA inference")
    device = torch.device("cuda")
    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    parent_state = load_file(str(graph_path), device="cpu")

    checkpoint_results: dict[str, Any] = {}
    for step in sorted(EXPECTED_CHECKPOINTS):
        print(f"checkpoint={step} phase=load", flush=True)
        graph = loc1.load_candidate(
            parent_state=parent_state,
            checkpoint_path=checkpoint_paths[step],
            layer_map=layer_map,
            device=device,
        )
        print(f"checkpoint={step} phase=counterfactual", flush=True)
        with torch.inference_mode():
            counterfactual = loc1.evaluate_counterfactual_modes(
                graph,
                adapter,
                causal_dev,
                causal,
                device=device,
                batch_size=args.eval_batch_size,
            )
        print(f"checkpoint={step} phase=source_receipt_activation", flush=True)
        source_receipt = result["checkpoints"][f"step-{step:08d}"]
        source_causal = source_receipt["causal_dev_metrics"]
        preservation = preservation_receipt_summary(result, step)
        actual = counterfactual["modes"]["actual_activation_soft_binding"]
        forced_soft = counterfactual["modes"]["forced_specialist_soft_binding"]
        actual_hard = counterfactual["modes"]["actual_activation_hard_top1_binding"]
        forced_hard = counterfactual["modes"]["forced_specialist_hard_top1_binding"]

        checkpoint_results[str(step)] = {
            "candidate_sha256": EXPECTED_CHECKPOINTS[step],
            "causal_specialist_activation_from_source_receipt": {
                "top1_accuracy": float(source_causal["specialist_top1_accuracy"]),
                "family_min_top1_accuracy": float(
                    source_causal["family_min_specialist_top1_accuracy"]
                ),
                "mean_probability": float(
                    source_causal["mean_specialist_probability"]
                ),
            },
            "preservation_from_frozen_training_receipt": preservation,
            "counterfactual": counterfactual,
            "diagnostic_deltas": {
                "forced_activation_soft_row_gain": (
                    forced_soft["row_accuracy"] - actual["row_accuracy"]
                ),
                "hard_top1_actual_activation_row_gain": (
                    actual_hard["row_accuracy"] - actual["row_accuracy"]
                ),
                "hard_top1_forced_activation_row_gain_vs_soft_forced": (
                    forced_hard["row_accuracy"] - forced_soft["row_accuracy"]
                ),
                "fully_forced_hard_top1_row_gain_vs_actual": (
                    forced_hard["row_accuracy"] - actual["row_accuracy"]
                ),
                "fully_forced_hard_top1_quad_gain_vs_actual": (
                    forced_hard["quad_accuracy"] - actual["quad_accuracy"]
                ),
                "fully_forced_hard_top1_margin_gain_vs_actual": (
                    forced_hard["mean_target_margin"]
                    - actual["mean_target_margin"]
                ),
            },
        }
        print(
            "checkpoint_summary="
            + json.dumps(
                {
                    "step": step,
                    "causal_specialist_top1": float(
                        source_causal["specialist_top1_accuracy"]
                    ),
                    "actual_row": actual["row_accuracy"],
                    "forced_soft_row": forced_soft["row_accuracy"],
                    "actual_hard_row": actual_hard["row_accuracy"],
                    "forced_hard_row": forced_hard["row_accuracy"],
                    "proposal_role_accuracy": counterfactual[
                        "relevant_edge_residual_proposal_role_accuracy"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    best_forced_hard_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k]["counterfactual"]["modes"][
            "forced_specialist_hard_top1_binding"
        ]["row_accuracy"],
    )
    best_role_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k]["counterfactual"][
            "relevant_edge_residual_proposal_role_accuracy"
        ],
    )
    best_activation_step = max(
        checkpoint_results,
        key=lambda k: checkpoint_results[k][
            "causal_specialist_activation_from_source_receipt"
        ]["top1_accuracy"],
    )

    audit = {
        "schema": SCHEMA,
        "status": "PASS_P100_INFERENCE_CAUSAL_DUAL_VIEW_SPECIALIST_FAILURE_LOCALIZATION_NO_GRADIENT",
        "source_failed_result_sha256": sha(result_path),
        "causal_cache_sha256": sha(causal_cache_path),
        "field_token_cache_sha256": sha(field_cache_path),
        "curriculum_sha256": sha(curriculum_path),
        "parent_graph_sha256": sha(graph_path),
        "parent_adapter_sha256": sha(adapter_path),
        "layer_map_sha256": sha(layer_map),
        "semantic_backbone_executed": False,
        "preservation_semantic_reencoding_performed": False,
        "gpu_inference_used": True,
        "preservation_metrics_reused_from_source_receipts": True,
        "checkpoint_results": checkpoint_results,
        "summary": {
            "best_causal_activation_step": int(best_activation_step),
            "best_causal_activation_top1": checkpoint_results[
                best_activation_step
            ]["causal_specialist_activation_from_source_receipt"]["top1_accuracy"],
            "best_forced_hard_row_step": int(best_forced_hard_step),
            "best_forced_hard_row_accuracy": checkpoint_results[
                best_forced_hard_step
            ]["counterfactual"]["modes"][
                "forced_specialist_hard_top1_binding"
            ]["row_accuracy"],
            "best_forced_hard_quad_accuracy": checkpoint_results[
                best_forced_hard_step
            ]["counterfactual"]["modes"][
                "forced_specialist_hard_top1_binding"
            ]["quad_accuracy"],
            "best_residual_role_step": int(best_role_step),
            "best_residual_role_accuracy": checkpoint_results[
                best_role_step
            ]["counterfactual"][
                "relevant_edge_residual_proposal_role_accuracy"
            ],
        },
        "interpretation_contract": {
            "activation_bottleneck": (
                "forcing specialist materially improves causal DEV under the same edge execution"
            ),
            "soft_mixture_bottleneck": (
                "model-predicted hard-top1 materially improves causal DEV over soft edge execution"
            ),
            "directional_residual_bottleneck": (
                "forced specialist plus predicted hard-top1 remains weak and/or relevant-edge residual role accuracy remains weak"
            ),
            "multiple_failures_may_coexist": True,
        },
        "optimizer_created": False,
        "gradient_performed": False,
        "gpu_required": True,
        "training_authorized": False,
        "causal_test_split_evaluated": False,
        "frozen_challenge_evaluated": False,
        "heldout_opening_authorized": False,
        "scale_authorized": False,
        "semantic_retraining_authorized": False,
        "graph_parent_retraining_authorized": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "interpret activation-vs-soft-mixture-vs-endpoint-role residual failure "
            "before any architecture change or retraining"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
