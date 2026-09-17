#!/usr/bin/env python3
"""Finalize one frozen N0 downstream causal arbitration into the decision gate.

This stage performs no evaluation, training, tuning, repair ratification, promotion,
or scaling. It independently verifies the immutable preflight/result pair, raw arm
receipts, and metric comparisons before materializing the exact gate config and
running the already-committed post-arbitration decision gate.
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
PROTOCOL_VERSION = "n0_downstream_causal_arbitration_finalizer_v0.1"
ARBITRATION_PROTOCOL_VERSION = "n0_downstream_causal_arbitration_v0.2"
GATE_PROTOCOL_VERSION = "n0_post_arbitration_decision_gate_v0.1"


class FinalizationError(RuntimeError):
    pass


def load_sibling(module_name: str, filename: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise FinalizationError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"unable to load JSON {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
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


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{label} must be an object")
    return value


def resolve_recorded_path(raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise FinalizationError(f"{label} must be a non-empty path string")
    return Path(raw).expanduser().resolve()


def checked_sha(path: Path, expected: Any, label: str, sha256_file: Any) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FinalizationError(f"{label} is not a regular file: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise FinalizationError(f"{label} recorded sha256 is invalid")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise FinalizationError(
            f"{label} sha256 mismatch: expected {expected.lower()}, got {actual}"
        )
    return actual


def repo_root_for(script_path: Path) -> Path:
    return script_path.resolve().parents[3]


def verify_source_freeze(
    *, source_revision: str, source_files: Sequence[Path], repo_root: Path
) -> dict[str, str]:
    """Require the checked-out, tracked source files to match the frozen commit."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinalizationError(f"unable to resolve git HEAD: {exc}") from exc
    if head != source_revision.lower():
        raise FinalizationError(
            f"source revision drift: frozen={source_revision.lower()} checked_out={head}"
        )

    hashes: dict[str, str] = {}
    import hashlib

    for path in source_files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise FinalizationError(
                f"source file is outside repository: {resolved}"
            ) from exc
        try:
            dirty = subprocess.run(
                ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--", str(relative)],
                check=False,
            ).returncode
        except OSError as exc:
            raise FinalizationError(f"unable to verify source file {relative}: {exc}") from exc
        if dirty != 0:
            raise FinalizationError(
                f"source file differs from frozen commit {source_revision}: {relative}"
            )
        if not resolved.is_file():
            raise FinalizationError(f"frozen source file missing: {resolved}")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        hashes[str(relative).replace("\\", "/")] = digest
    return hashes


def verify_result_against_preflight(
    *,
    preflight_path: Path,
    result_path: Path,
    arb: Any,
) -> dict[str, Any]:
    preflight_path = preflight_path.resolve()
    result_path = result_path.resolve()
    preflight = require_mapping(load_json(preflight_path), "preflight receipt")
    result = require_mapping(load_json(result_path), "arbitration result")

    if preflight.get("protocol_version") != ARBITRATION_PROTOCOL_VERSION:
        raise FinalizationError("preflight protocol mismatch")
    if preflight.get("status") != "PREFLIGHT_OK":
        raise FinalizationError("preflight status must be PREFLIGHT_OK")
    if result.get("protocol_version") != ARBITRATION_PROTOCOL_VERSION:
        raise FinalizationError("arbitration result protocol mismatch")
    if result.get("status") != "COMPLETE":
        raise FinalizationError("arbitration result status must be COMPLETE")

    for key in (
        "experiment_id",
        "source_revision",
        "manifest_path",
        "manifest_sha256",
        "evaluator_argv_sha256",
        "common_inputs",
        "common_fingerprint",
        "arms",
        "metrics",
        "invariants",
    ):
        if result.get(key) != preflight.get(key):
            raise FinalizationError(f"result changed frozen preflight field: {key}")

    for key in ("ratifies_repair", "scale_authorized", "promotion_authorized", "n0_complete"):
        if preflight.get(key) is not False or result.get(key) is not False:
            raise FinalizationError(f"{key} must remain false in preflight and result")

    fingerprint = result.get("common_fingerprint")
    arm_fingerprints = require_mapping(
        result.get("arm_common_fingerprints"), "arm_common_fingerprints"
    )
    if set(arm_fingerprints) != {"canonical", "candidate"}:
        raise FinalizationError(
            "arm_common_fingerprints must contain exactly canonical and candidate"
        )
    if any(arm_fingerprints[name] != fingerprint for name in arm_fingerprints):
        raise FinalizationError("an arbitration arm changed the common input fingerprint")

    manifest_path = resolve_recorded_path(result.get("manifest_path"), "manifest_path")
    checked_sha(
        manifest_path,
        result.get("manifest_sha256"),
        "arbitration manifest",
        arb.sha256_file,
    )
    try:
        verified_manifest = arb.validate_manifest(load_json(manifest_path), manifest_path)
    except arb.ProtocolError as exc:
        raise FinalizationError(f"arbitration manifest no longer validates: {exc}") from exc
    if verified_manifest["common_fingerprint"] != fingerprint:
        raise FinalizationError("manifest common fingerprint differs from result")
    if verified_manifest["common_inputs"] != result.get("common_inputs"):
        raise FinalizationError("manifest common inputs differ from frozen result")
    if verified_manifest["arms"] != result.get("arms"):
        raise FinalizationError("manifest arm bindings differ from frozen result")
    if verified_manifest["metrics"] != result.get("metrics"):
        raise FinalizationError("manifest metric policy differs from frozen result")

    raw_outputs = require_mapping(result.get("raw_outputs"), "raw_outputs")
    if set(raw_outputs) != {"canonical", "candidate"}:
        raise FinalizationError("raw_outputs must contain exactly canonical and candidate")
    raw_payloads: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    for arm_name in ("canonical", "candidate"):
        record = require_mapping(raw_outputs[arm_name], f"raw_outputs.{arm_name}")
        path = resolve_recorded_path(record.get("path"), f"raw_outputs.{arm_name}.path")
        raw_hashes[arm_name] = checked_sha(
            path,
            record.get("sha256"),
            f"raw_outputs.{arm_name}",
            arb.sha256_file,
        )
        raw_payloads[arm_name] = load_json(path)

    comparisons = result.get("comparisons")
    metrics = result.get("metrics")
    if not isinstance(comparisons, list) or not isinstance(metrics, list):
        raise FinalizationError("result metrics/comparisons must be lists")
    if len(comparisons) != len(metrics):
        raise FinalizationError("comparison count differs from frozen metric policy")

    recomputed: list[dict[str, Any]] = []
    for index, spec in enumerate(metrics):
        try:
            canonical = arb.metric_value(raw_payloads["canonical"], spec["path"])
            candidate = arb.metric_value(raw_payloads["candidate"], spec["path"])
            expected = arb.compare_metric(canonical, candidate, spec)
        except (arb.ProtocolError, KeyError, TypeError) as exc:
            raise FinalizationError(
                f"unable to recompute comparison {index}: {exc}"
            ) from exc
        observed = comparisons[index]
        if not isinstance(observed, Mapping):
            raise FinalizationError(f"comparisons[{index}] must be an object")
        if dict(observed) != expected:
            raise FinalizationError(
                f"comparisons[{index}] does not match immutable raw arm evidence"
            )
        recomputed.append(expected)

    states = [item["result"] for item in recomputed]
    classification = (
        "HARM"
        if "HARM" in states
        else "IMPROVEMENT"
        if "IMPROVEMENT" in states
        else "EQUIVALENT"
    )
    if result.get("classification") != classification:
        raise FinalizationError(
            f"classification mismatch: expected {classification}, got {result.get('classification')}"
        )

    return {
        "preflight": preflight,
        "result": result,
        "manifest_path": manifest_path,
        "raw_hashes": raw_hashes,
        "classification": classification,
    }


def build_gate_config(
    *, result_path: Path, result_sha256: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    result = require_mapping(evidence.get("result"), "verified result")
    experiment_id = result.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise FinalizationError("experiment_id must be non-empty")
    return {
        "protocol_version": GATE_PROTOCOL_VERSION,
        "decision_id": f"{experiment_id}:post-arbitration-gate",
        "arbitration_receipt": {
            "path": str(result_path.resolve()),
            "sha256": result_sha256,
        },
        "expected_source_revision": result["source_revision"],
        "expected_arbitration_manifest_sha256": result["manifest_sha256"],
        "expected_common_fingerprint": result["common_fingerprint"],
        "invariants": {
            "ratifies_repair": False,
            "training_enabled": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "n0_ready": False,
        },
    }


def finalize(
    *,
    preflight_path: Path,
    result_path: Path,
    gate_config_output: Path,
    gate_receipt_output: Path,
    finalization_receipt_output: Path,
    enforce_source_freeze: bool = True,
) -> dict[str, Any]:
    outputs = [
        gate_config_output.resolve(),
        gate_receipt_output.resolve(),
        finalization_receipt_output.resolve(),
    ]
    if len(set(outputs)) != len(outputs):
        raise FinalizationError("finalization output paths must be distinct")
    for output in outputs:
        if output.exists():
            raise FinalizationError(f"refusing to overwrite output: {output}")

    arb = load_sibling(
        "downstream_causal_arbitration_finalize_v02",
        "downstream_causal_arbitration_v0_2.py",
    )
    gate = load_sibling(
        "post_arbitration_decision_gate_finalize_v01",
        "post_arbitration_decision_gate_v0_1.py",
    )
    evidence = verify_result_against_preflight(
        preflight_path=preflight_path,
        result_path=result_path,
        arb=arb,
    )
    result = require_mapping(evidence["result"], "verified result")
    source_revision = str(result["source_revision"]).lower()

    source_files = [
        Path(__file__).resolve(),
        (SCRIPT_DIR / "downstream_causal_arbitration_v0_2.py").resolve(),
        (SCRIPT_DIR / "post_arbitration_decision_gate_v0_1.py").resolve(),
    ]
    if enforce_source_freeze:
        source_hashes = verify_source_freeze(
            source_revision=source_revision,
            source_files=source_files,
            repo_root=repo_root_for(Path(__file__)),
        )
    else:
        source_hashes = {
            path.name: arb.sha256_file(path) for path in source_files if path.is_file()
        }

    result_sha = arb.sha256_file(result_path.resolve())
    preflight_sha = arb.sha256_file(preflight_path.resolve())
    gate_config = build_gate_config(
        result_path=result_path,
        result_sha256=result_sha,
        evidence=evidence,
    )
    atomic_write_json(gate_config_output.resolve(), gate_config)

    try:
        gate_result = gate.run_gate(
            gate_config_output.resolve(), gate_receipt_output.resolve()
        )
    except (gate.GateError, OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"post-arbitration decision gate failed: {exc}") from exc

    if gate_result.get("status") != "COMPLETE":
        raise FinalizationError("decision gate did not complete")
    if gate_result.get("classification") != evidence["classification"]:
        raise FinalizationError("decision gate classification differs from verified evidence")
    for key in (
        "candidate_graph_promoted",
        "ratifies_repair",
        "training_authorized",
        "promotion_authorized",
        "scale_authorized",
        "n0_ready",
        "n0_complete",
    ):
        if gate_result.get(key) is not False:
            raise FinalizationError(f"decision gate unexpectedly set {key}=true")

    receipt = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE",
        "source_revision": source_revision,
        "experiment_id": result["experiment_id"],
        "classification": evidence["classification"],
        "decision": gate_result["decision"],
        "final_frozen_challenge_authorized": gate_result[
            "final_frozen_challenge_authorized"
        ],
        "final_frozen_challenge_limit": gate_result["final_frozen_challenge_limit"],
        "frozen_inputs": {
            "preflight_receipt": {
                "path": str(preflight_path.resolve()),
                "sha256": preflight_sha,
            },
            "arbitration_result": {
                "path": str(result_path.resolve()),
                "sha256": result_sha,
            },
            "arbitration_manifest": {
                "path": str(evidence["manifest_path"]),
                "sha256": result["manifest_sha256"],
            },
            "raw_outputs": {
                name: {
                    "path": result["raw_outputs"][name]["path"],
                    "sha256": evidence["raw_hashes"][name],
                }
                for name in ("canonical", "candidate")
            },
        },
        "materialized_outputs": {
            "gate_config": {
                "path": str(gate_config_output.resolve()),
                "sha256": arb.sha256_file(gate_config_output.resolve()),
            },
            "gate_receipt": {
                "path": str(gate_receipt_output.resolve()),
                "sha256": arb.sha256_file(gate_receipt_output.resolve()),
            },
        },
        "source_files": source_hashes,
        "invariants": {
            "diagnostic_only": True,
            "training_enabled": False,
            "candidate_graph_promoted": False,
            "ratifies_repair": False,
            "promotion_authorized": False,
            "scale_authorized": False,
            "n0_ready": False,
            "n0_complete": False,
        },
    }
    atomic_write_json(finalization_receipt_output.resolve(), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--arbitration-result", required=True, type=Path)
    parser.add_argument("--gate-config-output", required=True, type=Path)
    parser.add_argument("--gate-receipt-output", required=True, type=Path)
    parser.add_argument("--finalization-receipt-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = finalize(
            preflight_path=args.preflight_receipt,
            result_path=args.arbitration_result,
            gate_config_output=args.gate_config_output,
            gate_receipt_output=args.gate_receipt_output,
            finalization_receipt_output=args.finalization_receipt_output,
            enforce_source_freeze=True,
        )
    except FinalizationError as exc:
        print(f"FINALIZATION_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
