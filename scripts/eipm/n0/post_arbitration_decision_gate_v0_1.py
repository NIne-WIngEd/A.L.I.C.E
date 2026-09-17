#!/usr/bin/env python3
"""Fail-closed N0 post-arbitration decision gate.

Consumes one immutable downstream-causal-arbitration v0.2 receipt. The gate does
not rerun evaluation, train, ratify, promote, scale, or mark N0 ready. Its only
positive authority is to permit one final frozen challenge when the candidate
shows downstream improvement with no degraded metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "n0_post_arbitration_decision_gate_v0.1"
ARBITRATION_PROTOCOL_VERSION = "n0_downstream_causal_arbitration_v0.2"
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLASSIFICATIONS = frozenset({"HARM", "EQUIVALENT", "IMPROVEMENT"})

class GateError(RuntimeError):
    pass

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
        raise GateError(f"{label} must be an object")
    return value

def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{label} must be a non-empty string")
    return value

def require_hash(value: Any, label: str) -> str:
    text = require_string(value, label).lower()
    if not SHA256_RE.fullmatch(text):
        raise GateError(f"{label} must be lowercase 64-hex")
    return text

def resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()

def validate_config(config: Any, config_path: Path) -> dict[str, Any]:
    root = require_mapping(config, "config")
    if root.get("protocol_version") != PROTOCOL_VERSION:
        raise GateError(f"protocol_version must be {PROTOCOL_VERSION}")
    decision_id = require_string(root.get("decision_id"), "decision_id")

    receipt_spec = require_mapping(root.get("arbitration_receipt"), "arbitration_receipt")
    receipt_path = resolve(config_path.resolve().parent, require_string(receipt_spec.get("path"), "arbitration_receipt.path"))
    expected_receipt_sha = require_hash(receipt_spec.get("sha256"), "arbitration_receipt.sha256")
    if not receipt_path.is_file():
        raise GateError(f"arbitration receipt is not a regular file: {receipt_path}")
    actual_receipt_sha = sha256_file(receipt_path)
    if actual_receipt_sha != expected_receipt_sha:
        raise GateError(f"arbitration receipt sha256 mismatch: expected {expected_receipt_sha}, got {actual_receipt_sha}")

    expected_revision = require_string(root.get("expected_source_revision"), "expected_source_revision").lower()
    if not SOURCE_REVISION_RE.fullmatch(expected_revision):
        raise GateError("expected_source_revision must be an immutable lowercase 40-hex Git commit id")
    expected_manifest_sha = require_hash(root.get("expected_arbitration_manifest_sha256"), "expected_arbitration_manifest_sha256")
    expected_common_fingerprint = require_hash(root.get("expected_common_fingerprint"), "expected_common_fingerprint")

    invariants = require_mapping(root.get("invariants"), "invariants")
    required = {
        "ratifies_repair": False,
        "training_enabled": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "n0_ready": False,
    }
    for key, value in required.items():
        if invariants.get(key) is not value:
            raise GateError(f"invariants.{key} must be {str(value).lower()}")

    return {
        "decision_id": decision_id,
        "receipt_path": receipt_path,
        "receipt_sha256": actual_receipt_sha,
        "expected_source_revision": expected_revision,
        "expected_arbitration_manifest_sha256": expected_manifest_sha,
        "expected_common_fingerprint": expected_common_fingerprint,
        "invariants": dict(invariants),
    }

def validate_receipt(receipt: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    root = require_mapping(receipt, "arbitration receipt")
    if root.get("protocol_version") != ARBITRATION_PROTOCOL_VERSION:
        raise GateError(f"arbitration receipt protocol_version must be {ARBITRATION_PROTOCOL_VERSION}")
    if root.get("status") != "COMPLETE":
        raise GateError("arbitration receipt status must be COMPLETE")

    revision = require_string(root.get("source_revision"), "arbitration receipt source_revision").lower()
    if revision != expected["expected_source_revision"]:
        raise GateError("arbitration receipt source_revision does not match frozen expectation")
    if require_hash(root.get("manifest_sha256"), "arbitration receipt manifest_sha256") != expected["expected_arbitration_manifest_sha256"]:
        raise GateError("arbitration receipt manifest_sha256 does not match frozen expectation")
    common_fingerprint = require_hash(root.get("common_fingerprint"), "arbitration receipt common_fingerprint")
    if common_fingerprint != expected["expected_common_fingerprint"]:
        raise GateError("arbitration receipt common_fingerprint does not match frozen expectation")

    for key in ("ratifies_repair", "scale_authorized", "promotion_authorized", "n0_complete"):
        if root.get(key) is not False:
            raise GateError(f"arbitration receipt {key} must be false")

    arm_fingerprints = require_mapping(root.get("arm_common_fingerprints"), "arm_common_fingerprints")
    if set(arm_fingerprints) != {"canonical", "candidate"}:
        raise GateError("arm_common_fingerprints must contain exactly canonical and candidate")
    if arm_fingerprints["canonical"] != common_fingerprint or arm_fingerprints["candidate"] != common_fingerprint:
        raise GateError("both arbitration arms must use the frozen common fingerprint")

    comparisons = root.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise GateError("arbitration receipt comparisons must be a non-empty list")
    states: list[str] = []
    for index, item in enumerate(comparisons):
        comparison = require_mapping(item, f"comparisons[{index}]")
        state = comparison.get("result")
        if state not in CLASSIFICATIONS:
            raise GateError(f"comparisons[{index}].result is invalid")
        states.append(state)

    recomputed = "HARM" if "HARM" in states else "IMPROVEMENT" if "IMPROVEMENT" in states else "EQUIVALENT"
    classification = root.get("classification")
    if classification not in CLASSIFICATIONS:
        raise GateError("arbitration receipt classification is invalid")
    if classification != recomputed:
        raise GateError(f"arbitration classification inconsistent with comparisons: expected {recomputed}, got {classification}")

    return {
        "classification": classification,
        "comparison_states": states,
        "source_revision": revision,
        "manifest_sha256": expected["expected_arbitration_manifest_sha256"],
        "common_fingerprint": common_fingerprint,
        "experiment_id": require_string(root.get("experiment_id"), "arbitration receipt experiment_id"),
    }

def decision_for(classification: str) -> tuple[str, bool, str]:
    if classification == "HARM":
        return (
            "RETAIN_CANONICAL_DOWNSTREAM_HARM",
            False,
            "Retain the canonical graph. The candidate degraded at least one frozen downstream metric.",
        )
    if classification == "EQUIVALENT":
        return (
            "RETAIN_CANONICAL_DOWNSTREAM_EQUIVALENCE",
            False,
            "Retain the canonical graph. The candidate did not produce measurable downstream improvement.",
        )
    if classification == "IMPROVEMENT":
        return (
            "AUTHORIZE_ONE_FINAL_FROZEN_CHALLENGE",
            True,
            "Authorize exactly one final frozen challenge of the same candidate frontier. This does not ratify or promote it.",
        )
    raise GateError(f"unsupported classification: {classification}")

def run_gate(config_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise GateError(f"refusing to overwrite output: {output_path}")

    config = load_json(config_path)
    verified = validate_config(config, config_path)
    receipt = load_json(verified["receipt_path"])
    evidence = validate_receipt(receipt, verified)
    decision, challenge_authorized, rationale = decision_for(evidence["classification"])

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_id": verified["decision_id"],
        "status": "COMPLETE",
        "source": {
            "arbitration_receipt_path": str(verified["receipt_path"]),
            "arbitration_receipt_sha256": verified["receipt_sha256"],
            "arbitration_protocol_version": ARBITRATION_PROTOCOL_VERSION,
            "arbitration_experiment_id": evidence["experiment_id"],
            "source_revision": evidence["source_revision"],
            "arbitration_manifest_sha256": evidence["manifest_sha256"],
            "common_fingerprint": evidence["common_fingerprint"],
        },
        "classification": evidence["classification"],
        "comparison_states": evidence["comparison_states"],
        "decision": decision,
        "final_frozen_challenge_authorized": challenge_authorized,
        "final_frozen_challenge_limit": 1 if challenge_authorized else 0,
        "canonical_remains_current_baseline": True,
        "candidate_graph_promoted": False,
        "ratifies_repair": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "n0_ready": False,
        "n0_complete": False,
        "rationale": rationale,
    }
    atomic_write_json(output_path, result)
    return result

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run_gate(args.config, args.output)
    except (GateError, OSError, json.JSONDecodeError) as exc:
        print(f"POST_ARBITRATION_GATE_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
