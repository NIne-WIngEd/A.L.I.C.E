#!/usr/bin/env python3
"""Calibrate N0 arbitration tolerances from canonical-only full-stack repeats.

This stage never receives or inspects a candidate graph. It binds the frozen N0
stack, runs the canonical graph through the real full-stack evaluator in
separate processes, measures repeatability across the frozen comparison metric
surface, and writes the metric policy required by downstream causal arbitration.

The tolerance rule is fixed in source before calibration evidence exists:

    atol = max(ABSOLUTE_FLOOR,
               REPEATABILITY_MULTIPLIER * max_pairwise_abs_repeat_delta)
    rtol = 0.0

This is diagnostic infrastructure for the full production N0 path, not a pilot
model and not a reduced proxy.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_VERSION = "n0_downstream_causal_arbitration_metric_policy_v0.1"
CALIBRATION_VERSION = "n0_downstream_causal_arbitration_metric_calibration_v0.1"
EXPECTED_EVALUATOR_PROTOCOL = "n0_downstream_full_stack_graph_evaluator_v0.1"
EXPECTED_STATUS = "COMPLETE_DIAGNOSTIC_ONLY"

# Frozen before any candidate-arm result is observed.
ABSOLUTE_FLOOR = 1e-6
REPEATABILITY_MULTIPLIER = 2.0
RELATIVE_TOLERANCE = 0.0
DEFAULT_REPEAT_COUNT = 2


class CalibrationError(RuntimeError):
    pass


def load_sibling(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CalibrationError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"unable to load JSON {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{label} must be a non-empty string")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise CalibrationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def metric_value(payload: Any, path: str) -> float:
    current = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise CalibrationError(f"metric path not found: {path}")
        current = current[segment]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise CalibrationError(f"metric is not numeric: {path}")
    value = float(current)
    if not math.isfinite(value):
        raise CalibrationError(f"metric is not finite: {path}")
    return value


def repeat_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    if payload.get("protocol_version") != EXPECTED_EVALUATOR_PROTOCOL:
        raise CalibrationError(
            f"repeat protocol_version must be {EXPECTED_EVALUATOR_PROTOCOL}"
        )
    if payload.get("status") != EXPECTED_STATUS:
        raise CalibrationError(f"repeat status must be {EXPECTED_STATUS}")

    invariants = require_mapping(payload.get("invariants"), "invariants")
    for key, expected in {
        "diagnostic_only": True,
        "training_enabled": False,
        "gradient_performed": False,
        "parents_mutated": False,
        "graph_checkpoint_is_only_arm_variant": True,
        "proxy_graph_transform_used": False,
    }.items():
        if invariants.get(key) is not expected:
            raise CalibrationError(
                f"invariants.{key} must be {str(expected).lower()}"
            )

    stack = require_mapping(payload.get("stack_manifest"), "stack_manifest")
    graph = require_mapping(payload.get("graph_condition"), "graph_condition")
    external = require_mapping(
        payload.get("common_external_inputs"), "common_external_inputs"
    )
    latent = require_mapping(external.get("latent_checkpoint"), "latent_checkpoint")
    evaluation = require_mapping(external.get("evaluation_set"), "evaluation_set")

    return {
        "source_revision": require_string(
            payload.get("source_revision"), "source_revision"
        ),
        "stack_manifest_sha256": require_string(
            stack.get("sha256"), "stack_manifest.sha256"
        ),
        "stack_binding_fingerprint": require_string(
            stack.get("stack_binding_fingerprint"),
            "stack_manifest.stack_binding_fingerprint",
        ),
        "graph_sha256": require_string(graph.get("sha256"), "graph_condition.sha256"),
        "latent_sha256": require_string(
            latent.get("sha256"), "latent_checkpoint.sha256"
        ),
        "evaluation_sha256": require_string(
            evaluation.get("sha256"), "evaluation_set.sha256"
        ),
    }


def max_pairwise_delta(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise CalibrationError("at least two repeat values are required")
    return max(
        abs(values[i] - values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )


def derive_policy(
    repeat_paths: Sequence[Path],
    policy_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if len(repeat_paths) < 2:
        raise CalibrationError("at least two canonical repeat results are required")

    prepare = load_sibling(
        "prepare_downstream_causal_arbitration_for_calibration_v01",
        "prepare_downstream_causal_arbitration_v0_1.py",
    )

    payloads = []
    identities = []
    for path in repeat_paths:
        resolved = path.resolve()
        payload = require_mapping(load_json(resolved), str(resolved))
        payloads.append(payload)
        identities.append(repeat_identity(payload))

    reference = identities[0]
    for index, identity in enumerate(identities[1:], start=1):
        if identity != reference:
            differing = {
                key: (reference[key], identity[key])
                for key in reference
                if reference[key] != identity[key]
            }
            raise CalibrationError(
                f"canonical repeat {index} does not share the frozen identity: "
                f"{differing}"
            )

    metric_rows = []
    evidence_rows = []
    for path, direction in prepare.EXPECTED_METRIC_DIRECTIONS.items():
        values = [metric_value(payload, path) for payload in payloads]
        drift = max_pairwise_delta(values)
        atol = max(ABSOLUTE_FLOOR, REPEATABILITY_MULTIPLIER * drift)
        metric_rows.append(
            {
                "path": path,
                "direction": direction,
                "atol": atol,
                "rtol": RELATIVE_TOLERANCE,
            }
        )
        evidence_rows.append(
            {
                "path": path,
                "repeat_values": values,
                "max_pairwise_abs_delta": drift,
                "frozen_atol": atol,
                "frozen_rtol": RELATIVE_TOLERANCE,
            }
        )

    policy = {
        "protocol_version": POLICY_VERSION,
        "selected_before_arm_results": True,
        "results_observed": False,
        "calibration": {
            "protocol_version": CALIBRATION_VERSION,
            "source": "canonical_full_stack_repeatability_only",
            "repeat_count": len(payloads),
            "absolute_floor": ABSOLUTE_FLOOR,
            "repeatability_multiplier": REPEATABILITY_MULTIPLIER,
            "relative_tolerance": RELATIVE_TOLERANCE,
        },
        "metrics": metric_rows,
    }
    receipt = {
        "protocol_version": CALIBRATION_VERSION,
        "status": "CALIBRATION_COMPLETE",
        "candidate_graph_supplied": False,
        "candidate_arm_result_observed": False,
        "repeat_count": len(payloads),
        "frozen_identity": reference,
        "rule": {
            "atol": "max(absolute_floor, repeatability_multiplier * max_pairwise_abs_delta)",
            "absolute_floor": ABSOLUTE_FLOOR,
            "repeatability_multiplier": REPEATABILITY_MULTIPLIER,
            "rtol": RELATIVE_TOLERANCE,
        },
        "repeat_results": [str(path.resolve()) for path in repeat_paths],
        "metrics": evidence_rows,
        "policy_output": str(policy_output.resolve()),
    }

    atomic_write_json(policy_output, policy)
    atomic_write_json(receipt_output, receipt)
    return {
        "status": "CALIBRATION_COMPLETE",
        "policy_output": str(policy_output.resolve()),
        "receipt_output": str(receipt_output.resolve()),
        "repeat_count": len(payloads),
        "candidate_arm_result_observed": False,
    }


def build_stack_manifest(
    *,
    source_revision: str,
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
    stack_manifest_output: Path,
) -> dict[str, Any]:
    full_eval = load_sibling(
        "downstream_full_stack_graph_evaluator_calibration_v01",
        "downstream_full_stack_graph_evaluator_v0_1.py",
    )
    prepare = load_sibling(
        "prepare_downstream_causal_arbitration_stack_calibration_v01",
        "prepare_downstream_causal_arbitration_v0_1.py",
    )

    if stack_manifest_output.resolve().exists():
        raise CalibrationError(
            f"refusing to overwrite stack manifest: {stack_manifest_output.resolve()}"
        )

    artifacts = {
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
    manifest = {
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
                "sha256": full_eval.sha256_file(latent_checkpoint.resolve())
            },
            "evaluation_set": {
                "sha256": full_eval.sha256_file(evaluation_set.resolve())
            },
        },
        "artifacts": {
            role: prepare.artifact_spec(Path(path), role=role, full_eval=full_eval)
            for role, path in sorted(artifacts.items())
        },
    }
    full_eval.atomic_write_json(stack_manifest_output.resolve(), manifest)
    full_eval.validate_stack_manifest(
        stack_manifest_output.resolve(),
        observed_source_revision=source_revision,
    )
    return manifest


def run_full_calibration(
    *,
    repeat_count: int,
    canonical_graph: Path,
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
    stack_manifest_output: Path,
    repeat_output_dir: Path,
    policy_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if repeat_count < 2:
        raise CalibrationError("repeat_count must be at least 2")

    full_eval = load_sibling(
        "downstream_full_stack_graph_evaluator_runner_calibration_v01",
        "downstream_full_stack_graph_evaluator_v0_1.py",
    )
    repo_root = Path(__file__).resolve().parents[3]
    source_revision = full_eval.git_revision(repo_root)

    canonical_graph = canonical_graph.resolve()
    if not canonical_graph.is_file():
        raise CalibrationError(f"canonical graph is not a regular file: {canonical_graph}")

    build_stack_manifest(
        source_revision=source_revision,
        evaluation_set=evaluation_set,
        latent_checkpoint=latent_checkpoint,
        latent_config=latent_config,
        semantic_config=semantic_config,
        semantic_checkpoint=semantic_checkpoint,
        tokenizer_dir=tokenizer_dir,
        structured_config=structured_config,
        structured_checkpoint=structured_checkpoint,
        evidence_adapter=evidence_adapter,
        fusion_checkpoint=fusion_checkpoint,
        fusion_config=fusion_config,
        fusion_ratification=fusion_ratification,
        stack_manifest_output=stack_manifest_output,
    )

    repeat_output_dir = repeat_output_dir.resolve()
    if repeat_output_dir.exists() and any(repeat_output_dir.iterdir()):
        raise CalibrationError(
            f"repeat output directory is not empty: {repeat_output_dir}"
        )
    repeat_output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = (SCRIPT_DIR / "downstream_full_stack_graph_evaluator_v0_1.py").resolve()
    repeat_paths = []
    for index in range(1, repeat_count + 1):
        output = repeat_output_dir / f"canonical-repeat-{index:02d}.json"
        argv = [
            sys.executable,
            str(evaluator),
            "--graph",
            str(canonical_graph),
            "--latent-checkpoint",
            str(latent_checkpoint.resolve()),
            "--config",
            str(stack_manifest_output.resolve()),
            "--evaluation-set",
            str(evaluation_set.resolve()),
            "--output",
            str(output),
        ]
        completed = subprocess.run(argv, check=False)
        if completed.returncode != 0:
            raise CalibrationError(
                f"canonical repeat {index} evaluator exited with "
                f"status {completed.returncode}"
            )
        if not output.is_file():
            raise CalibrationError(
                f"canonical repeat {index} did not create output: {output}"
            )
        repeat_paths.append(output)

    result = derive_policy(repeat_paths, policy_output, receipt_output)
    return {
        **result,
        "source_revision": source_revision,
        "canonical_graph_sha256": full_eval.sha256_file(canonical_graph),
        "stack_manifest_output": str(stack_manifest_output.resolve()),
        "repeat_output_dir": str(repeat_output_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-count", type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument("--canonical-graph", required=True, type=Path)
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
    parser.add_argument("--stack-manifest-output", required=True, type=Path)
    parser.add_argument("--repeat-output-dir", required=True, type=Path)
    parser.add_argument("--policy-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        result = run_full_calibration(
            repeat_count=args.repeat_count,
            canonical_graph=args.canonical_graph,
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
            stack_manifest_output=args.stack_manifest_output,
            repeat_output_dir=args.repeat_output_dir,
            policy_output=args.policy_output,
            receipt_output=args.receipt_output,
        )
    except CalibrationError as exc:
        print(f"CALIBRATION_ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"CALIBRATION_ERROR: {exc}")
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
