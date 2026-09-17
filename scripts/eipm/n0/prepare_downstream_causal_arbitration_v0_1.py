#!/usr/bin/env python3
"""Prepare and preflight one frozen full-stack N0 downstream arbitration run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_POLICY_VERSION = "n0_downstream_causal_arbitration_metric_policy_v0.1"
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_METRIC_DIRECTIONS = {
    "metrics.best_slot_semantic_cosine": "higher",
    "metrics.family_min_best_slot_semantic_cosine": "higher",
    "metrics.pooled_semantic_cosine": "higher",
    "metrics.family_min_pooled_semantic_cosine": "higher",
    "metrics.mean_pairwise_offdiag_slot_cosine": "lower",
    "metrics.mean_centered_slot_effective_rank": "higher",
    "metrics.mean_min_available_view_best_slot_semantic_cosine": "higher",
    "metrics.mean_disagreement_weighted_view_specialization": "higher",
    "metrics.mean_min_available_view_best_slot_attention": "higher",
    "metrics.mean_min_channel_best_slot_attention": "higher",
    "metrics.missing_view_attention_max": "lower",
    "counterfactual_metrics.counterfactual_mean_target_drop": "higher",
    "counterfactual_metrics.counterfactual_family_min_mean_target_drop": "higher",
}


class PreparationError(RuntimeError):
    pass


def load_sibling(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PreparationError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"unable to load JSON {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparationError(f"{label} must be an object")
    return value


def validate_metric_policy(path: Path, arb: Any, sha256_file: Any) -> dict[str, Any]:
    root = require_mapping(load_json(path), "metric_policy")
    if root.get("protocol_version") != METRIC_POLICY_VERSION:
        raise PreparationError(
            f"metric policy protocol_version must be {METRIC_POLICY_VERSION}"
        )
    if root.get("selected_before_arm_results") is not True:
        raise PreparationError("metric policy must be selected before arm results")
    if root.get("results_observed") is not False:
        raise PreparationError("metric policy must declare results_observed=false")

    try:
        metrics = arb.validate_metrics(root.get("metrics"))
    except arb.ProtocolError as exc:
        raise PreparationError(f"invalid metric policy: {exc}") from exc

    observed = {item["path"]: item["direction"] for item in metrics}
    if observed != EXPECTED_METRIC_DIRECTIONS:
        missing = set(EXPECTED_METRIC_DIRECTIONS) - set(observed)
        extra = set(observed) - set(EXPECTED_METRIC_DIRECTIONS)
        wrong = {
            key: (EXPECTED_METRIC_DIRECTIONS[key], observed[key])
            for key in EXPECTED_METRIC_DIRECTIONS.keys() & observed.keys()
            if EXPECTED_METRIC_DIRECTIONS[key] != observed[key]
        }
        raise PreparationError(
            "metric policy must cover the complete frozen downstream metric surface; "
            f"missing={sorted(missing)} extra={sorted(extra)} wrong_direction={wrong}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path.resolve()),
        "metrics": metrics,
    }


def artifact_spec(path: Path, *, role: str, full_eval: Any) -> dict[str, Any]:
    path = path.resolve()
    if role in full_eval.DIRECTORY_ARTIFACTS:
        digest, _rows = full_eval.sha256_directory_tree(path)
        return {
            "path": str(path),
            "artifact_type": "directory_tree",
            "sha256": digest,
        }
    if not path.is_file():
        raise PreparationError(f"{role} is not a regular file: {path}")
    return {
        "path": str(path),
        "artifact_type": "file",
        "sha256": full_eval.sha256_file(path),
    }


def prepare(
    *,
    experiment_id: str,
    source_revision: str,
    metric_policy_path: Path,
    evaluation_set: Path,
    latent_checkpoint: Path,
    latent_config: Path,
    semantic_config: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    structured_config: Path,
    structured_checkpoint: Path,
    evidence_adapter: Path,
    fusion_checkpoint: Path,
    fusion_config: Path,
    fusion_ratification: Path,
    canonical_graph: Path,
    candidate_graph: Path,
    stack_manifest_output: Path,
    arbitration_manifest_output: Path,
    preflight_output: Path,
    evaluator_path: Path | None = None,
) -> dict[str, Any]:
    if not experiment_id.strip():
        raise PreparationError("experiment_id must be non-empty")
    source_revision = source_revision.lower()
    if not SOURCE_REVISION_RE.fullmatch(source_revision):
        raise PreparationError("source_revision must be immutable lowercase 40-hex")

    outputs = [
        stack_manifest_output.resolve(),
        arbitration_manifest_output.resolve(),
        preflight_output.resolve(),
    ]
    for output in outputs:
        if output.exists():
            raise PreparationError(f"refusing to overwrite prepared output: {output}")

    full_eval = load_sibling(
        "downstream_full_stack_graph_evaluator_v01",
        "downstream_full_stack_graph_evaluator_v0_1.py",
    )
    arb = load_sibling(
        "downstream_causal_arbitration_v02",
        "downstream_causal_arbitration_v0_2.py",
    )
    policy = validate_metric_policy(
        metric_policy_path.resolve(), arb, full_eval.sha256_file
    )

    evaluation_set = evaluation_set.resolve()
    latent_checkpoint = latent_checkpoint.resolve()
    canonical_graph = canonical_graph.resolve()
    candidate_graph = candidate_graph.resolve()
    evaluator_path = (
        evaluator_path.resolve()
        if evaluator_path is not None
        else (SCRIPT_DIR / "downstream_full_stack_graph_evaluator_v0_1.py").resolve()
    )
    for label, path in (
        ("evaluation_set", evaluation_set),
        ("latent_checkpoint", latent_checkpoint),
        ("canonical_graph", canonical_graph),
        ("candidate_graph", candidate_graph),
        ("evaluator", evaluator_path),
    ):
        if not path.is_file():
            raise PreparationError(f"{label} is not a regular file: {path}")

    canonical_sha = full_eval.sha256_file(canonical_graph)
    candidate_sha = full_eval.sha256_file(candidate_graph)
    if canonical_graph == candidate_graph or canonical_sha == candidate_sha:
        raise PreparationError(
            "canonical and candidate graph checkpoints must be distinct by path and hash"
        )

    artifact_paths = {
        "latent_config": latent_config,
        "semantic_config": semantic_config,
        "semantic_checkpoint": semantic_checkpoint,
        "tokenizer_dir": tokenizer_dir,
        "structured_config": structured_config,
        "structured_checkpoint": structured_checkpoint,
        "evidence_adapter": evidence_adapter,
        "fusion_checkpoint": fusion_checkpoint,
        "fusion_config": fusion_config,
        "fusion_ratification": fusion_ratification,
    }
    stack_manifest = {
        "protocol_version": full_eval.PROTOCOL_VERSION,
        "source_revision": source_revision,
        "invariants": {
            "diagnostic_only": True,
            "training_enabled": False,
            "ratifies_repair": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "graph_checkpoint_is_only_arm_variant": True,
            "proxy_graph_transform_used": False,
        },
        "external_inputs": {
            "latent_checkpoint": {
                "sha256": full_eval.sha256_file(latent_checkpoint)
            },
            "evaluation_set": {"sha256": full_eval.sha256_file(evaluation_set)},
        },
        "artifacts": {
            role: artifact_spec(Path(path), role=role, full_eval=full_eval)
            for role, path in sorted(artifact_paths.items())
        },
    }
    full_eval.atomic_write_json(stack_manifest_output.resolve(), stack_manifest)

    full_eval.validate_stack_manifest(
        stack_manifest_output.resolve(),
        observed_source_revision=source_revision,
    )

    arbitration_manifest = {
        "protocol_version": arb.PROTOCOL_VERSION,
        "experiment_id": experiment_id,
        "source_revision": source_revision,
        "invariants": {
            "diagnostic_only": True,
            "training_enabled": False,
            "ratifies_repair": False,
            "scale_authorized": False,
            "promotion_authorized": False,
        },
        "metric_policy": {
            "path": policy["path"],
            "sha256": policy["sha256"],
            "protocol_version": METRIC_POLICY_VERSION,
            "selected_before_arm_results": True,
            "results_observed": False,
        },
        "common_inputs": {
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": full_eval.sha256_file(evaluator_path),
            },
            "latent_checkpoint": {
                "path": str(latent_checkpoint),
                "sha256": full_eval.sha256_file(latent_checkpoint),
            },
            "evaluation_config": {
                "path": str(stack_manifest_output.resolve()),
                "sha256": full_eval.sha256_file(stack_manifest_output.resolve()),
            },
            "evaluation_set": {
                "path": str(evaluation_set),
                "sha256": full_eval.sha256_file(evaluation_set),
            },
        },
        "arms": {
            "canonical": {
                "label": "canonical_graph",
                "graph": {"path": str(canonical_graph), "sha256": canonical_sha},
            },
            "candidate": {
                "label": "relation_endpoint_repair_step_00000200",
                "graph": {"path": str(candidate_graph), "sha256": candidate_sha},
            },
        },
        "evaluator_argv": [
            sys.executable,
            "{input:evaluator}",
            "--graph",
            "{graph}",
            "--latent-checkpoint",
            "{input:latent_checkpoint}",
            "--config",
            "{input:evaluation_config}",
            "--evaluation-set",
            "{input:evaluation_set}",
            "--output",
            "{output}",
        ],
        "metrics": policy["metrics"],
    }
    full_eval.atomic_write_json(
        arbitration_manifest_output.resolve(), arbitration_manifest
    )

    try:
        preflight = arb.run_protocol(
            arbitration_manifest_output.resolve(),
            preflight_output.resolve(),
            preflight_only=True,
        )
    except arb.ProtocolError as exc:
        raise PreparationError(f"outer arbitration preflight failed: {exc}") from exc

    return {
        "status": "PREPARED_AND_PREFLIGHTED",
        "source_revision": source_revision,
        "stack_manifest": {
            "path": str(stack_manifest_output.resolve()),
            "sha256": full_eval.sha256_file(stack_manifest_output.resolve()),
        },
        "arbitration_manifest": {
            "path": str(arbitration_manifest_output.resolve()),
            "sha256": full_eval.sha256_file(arbitration_manifest_output.resolve()),
        },
        "preflight_receipt": {
            "path": str(preflight_output.resolve()),
            "sha256": full_eval.sha256_file(preflight_output.resolve()),
            "common_fingerprint": preflight["common_fingerprint"],
        },
        "metric_policy": {
            "path": policy["path"],
            "sha256": policy["sha256"],
        },
        "next_command": [
            sys.executable,
            str((SCRIPT_DIR / "execute_frozen_downstream_causal_arbitration_v0_1.py").resolve()),
            "--manifest",
            str(arbitration_manifest_output.resolve()),
            "--preflight-receipt",
            str(preflight_output.resolve()),
            "--output",
            str(
                arbitration_manifest_output.resolve().with_name(
                    f"{arbitration_manifest_output.stem}.result.json"
                )
            ),
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--metric-policy", required=True, type=Path)
    parser.add_argument("--evaluation-set", required=True, type=Path)
    parser.add_argument("--latent-checkpoint", required=True, type=Path)
    parser.add_argument("--latent-config", required=True, type=Path)
    parser.add_argument("--semantic-config", required=True, type=Path)
    parser.add_argument("--semantic-checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--structured-config", required=True, type=Path)
    parser.add_argument("--structured-checkpoint", required=True, type=Path)
    parser.add_argument("--evidence-adapter", required=True, type=Path)
    parser.add_argument("--fusion-checkpoint", required=True, type=Path)
    parser.add_argument("--fusion-config", required=True, type=Path)
    parser.add_argument("--fusion-ratification", required=True, type=Path)
    parser.add_argument("--canonical-graph", required=True, type=Path)
    parser.add_argument("--candidate-graph", required=True, type=Path)
    parser.add_argument("--stack-manifest-output", required=True, type=Path)
    parser.add_argument("--arbitration-manifest-output", required=True, type=Path)
    parser.add_argument("--preflight-output", required=True, type=Path)
    parser.add_argument("--evaluator", type=Path)
    args = parser.parse_args(argv)

    full_eval = load_sibling(
        "downstream_full_stack_graph_evaluator_revision_v01",
        "downstream_full_stack_graph_evaluator_v0_1.py",
    )
    try:
        revision = full_eval.git_revision(Path(__file__).resolve().parents[3])
        result = prepare(
            experiment_id=args.experiment_id,
            source_revision=revision,
            metric_policy_path=args.metric_policy,
            evaluation_set=args.evaluation_set,
            latent_checkpoint=args.latent_checkpoint,
            latent_config=args.latent_config,
            semantic_config=args.semantic_config,
            semantic_checkpoint=args.semantic_checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            structured_config=args.structured_config,
            structured_checkpoint=args.structured_checkpoint,
            evidence_adapter=args.evidence_adapter,
            fusion_checkpoint=args.fusion_checkpoint,
            fusion_config=args.fusion_config,
            fusion_ratification=args.fusion_ratification,
            canonical_graph=args.canonical_graph,
            candidate_graph=args.candidate_graph,
            stack_manifest_output=args.stack_manifest_output,
            arbitration_manifest_output=args.arbitration_manifest_output,
            preflight_output=args.preflight_output,
            evaluator_path=args.evaluator,
        )
    except (PreparationError, full_eval.BindingError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
