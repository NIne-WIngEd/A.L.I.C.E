#!/usr/bin/env python3
"""Fail-closed N0 downstream causal arbitration over one frozen frontier.

This protocol is diagnostic only. It compares a canonical evidence graph and one
candidate graph through exactly the same pinned evaluator, latent checkpoint,
evaluation configuration, and evaluation set. It contains no training,
ratification, promotion, or scaling path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_VERSION = "n0_downstream_causal_arbitration_v0.2"
SOURCE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_.:-]+)\}")
MANDATORY_COMMON_ROLES = frozenset({"evaluator", "latent_checkpoint", "evaluation_config", "evaluation_set"})
RESERVED_PLACEHOLDERS = frozenset({"graph", "output"})
FORBIDDEN_EXACT = frozenset({
    "--train", "--training", "--fit", "--finetune", "--fine-tune",
    "--backprop", "--backward", "--optimizer", "--learning-rate", "--lr",
    "--max-steps", "--epochs", "--num-epochs", "--gradient-accumulation-steps",
    "--scale", "--promote", "--ratify",
})
FORBIDDEN_PREFIXES = tuple(f"{x}=" for x in FORBIDDEN_EXACT)

class ProtocolError(RuntimeError):
    pass

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

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
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping): raise ProtocolError(f"{label} must be an object")
    return value

def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip(): raise ProtocolError(f"{label} must be a non-empty string")
    return value

def resolve(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()

def verify_file(label: str, spec: Any, base: Path) -> dict[str, str]:
    obj = require_mapping(spec, label)
    raw = require_string(obj.get("path"), f"{label}.path")
    expected = require_string(obj.get("sha256"), f"{label}.sha256").lower()
    if not SHA256_RE.fullmatch(expected): raise ProtocolError(f"{label}.sha256 must be lowercase 64-hex")
    path = resolve(base, raw)
    if not path.is_file(): raise ProtocolError(f"{label} is not a regular file: {path}")
    actual = sha256_file(path)
    if actual != expected: raise ProtocolError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}

def scan_forbidden(argv: Sequence[str]) -> None:
    for token in argv:
        low = token.lower()
        if low in FORBIDDEN_EXACT or low.startswith(FORBIDDEN_PREFIXES):
            raise ProtocolError(f"forbidden training/mutation option in evaluator argv: {token}")

def placeholders(argv: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for token in argv: found.update(PLACEHOLDER_RE.findall(token))
    return found

def render(argv: Sequence[str], values: Mapping[str, str]) -> list[str]:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values: raise ProtocolError(f"unknown evaluator placeholder: {key}")
        return values[key]
    result = [PLACEHOLDER_RE.sub(repl, token) for token in argv]
    for token in result:
        if "{" in token or "}" in token: raise ProtocolError(f"unresolved/malformed evaluator placeholder in token: {token}")
    return result

def metric_value(payload: Any, path: str) -> float:
    current = payload
    for segment in path.split("."):
        if isinstance(current, Mapping):
            if segment not in current: raise ProtocolError(f"metric path not found: {path}")
            current = current[segment]
        elif isinstance(current, list):
            try: current = current[int(segment)]
            except (ValueError, IndexError) as exc: raise ProtocolError(f"invalid list metric path: {path}") from exc
        else: raise ProtocolError(f"metric path traverses scalar: {path}")
    if isinstance(current, bool) or not isinstance(current, (int, float)): raise ProtocolError(f"metric is not numeric: {path}")
    out = float(current)
    if not math.isfinite(out): raise ProtocolError(f"metric is not finite: {path}")
    return out

def validate_metrics(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw: raise ProtocolError("metrics must be a non-empty list")
    out, seen = [], set()
    for i, item in enumerate(raw):
        obj = require_mapping(item, f"metrics[{i}]")
        path = require_string(obj.get("path"), f"metrics[{i}].path")
        if path in seen: raise ProtocolError(f"duplicate metric path: {path}")
        seen.add(path)
        direction = obj.get("direction")
        if direction not in {"higher", "lower"}: raise ProtocolError(f"metrics[{i}].direction must be higher or lower")
        try: atol, rtol = float(obj.get("atol", 0.0)), float(obj.get("rtol", 0.0))
        except (TypeError, ValueError) as exc: raise ProtocolError(f"metrics[{i}] tolerances must be numeric") from exc
        if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)): raise ProtocolError(f"metrics[{i}] tolerances must be finite and non-negative")
        out.append({"path": path, "direction": direction, "atol": atol, "rtol": rtol})
    return out

def common_fingerprint(source_revision: str, common: Mapping[str, Mapping[str, str]]) -> str:
    payload = {"source_revision": source_revision, "common_inputs": [
        {"role": role, "path": common[role]["path"], "sha256": common[role]["sha256"]}
        for role in sorted(common)
    ]}
    return sha256_bytes(canonical_json_bytes(payload))

def validate_manifest(manifest: Any, manifest_path: Path) -> dict[str, Any]:
    root = require_mapping(manifest, "manifest")
    if root.get("protocol_version") != PROTOCOL_VERSION: raise ProtocolError(f"protocol_version must be {PROTOCOL_VERSION}")
    require_string(root.get("experiment_id"), "experiment_id")
    revision = require_string(root.get("source_revision"), "source_revision").lower()
    if not SOURCE_REVISION_RE.fullmatch(revision): raise ProtocolError("source_revision must be an immutable lowercase 40-hex Git commit id")

    invariants = require_mapping(root.get("invariants"), "invariants")
    required_invariants = {"diagnostic_only": True, "training_enabled": False, "ratifies_repair": False,
                           "scale_authorized": False, "promotion_authorized": False}
    for key, value in required_invariants.items():
        if invariants.get(key) is not value: raise ProtocolError(f"invariants.{key} must be {str(value).lower()}")

    base = manifest_path.resolve().parent
    common_raw = require_mapping(root.get("common_inputs"), "common_inputs")
    roles = set(common_raw)
    missing = MANDATORY_COMMON_ROLES - roles
    if missing: raise ProtocolError(f"common_inputs missing mandatory roles: {sorted(missing)}")
    common: dict[str, dict[str, str]] = {}
    for role, spec in common_raw.items():
        if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", role): raise ProtocolError(f"invalid common input role: {role!r}")
        if role in RESERVED_PLACEHOLDERS: raise ProtocolError(f"reserved common input role: {role}")
        common[role] = verify_file(f"common_inputs.{role}", spec, base)

    arms_raw = require_mapping(root.get("arms"), "arms")
    if set(arms_raw) != {"canonical", "candidate"}: raise ProtocolError("arms must contain exactly canonical and candidate")
    arms: dict[str, dict[str, str]] = {}
    for arm_name in ("canonical", "candidate"):
        arm = require_mapping(arms_raw[arm_name], f"arms.{arm_name}")
        graph = verify_file(f"arms.{arm_name}.graph", arm.get("graph"), base)
        graph["label"] = require_string(arm.get("label"), f"arms.{arm_name}.label")
        arms[arm_name] = graph
    if arms["canonical"]["path"] == arms["candidate"]["path"] or arms["canonical"]["sha256"] == arms["candidate"]["sha256"]:
        raise ProtocolError("canonical and candidate graphs must be distinct by path and hash")

    argv = root.get("evaluator_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv): raise ProtocolError("evaluator_argv must be a non-empty string list")
    scan_forbidden(argv)
    found = placeholders(argv)
    required = RESERVED_PLACEHOLDERS | {f"input:{role}" for role in common}
    missing_placeholders = required - found
    if missing_placeholders: raise ProtocolError(f"evaluator_argv does not reference every frozen input: {sorted(missing_placeholders)}")
    unknown = found - required
    if unknown: raise ProtocolError(f"evaluator_argv contains unknown placeholders: {sorted(unknown)}")

    metrics = validate_metrics(root.get("metrics"))
    fingerprint = common_fingerprint(revision, common)
    return {
        "manifest_sha256": sha256_file(manifest_path), "source_revision": revision,
        "common_inputs": common, "common_fingerprint": fingerprint, "arms": arms,
        "evaluator_argv": list(argv), "evaluator_argv_sha256": sha256_bytes(canonical_json_bytes(argv)),
        "metrics": metrics,
    }

def arm_output(final: Path, arm: str) -> Path:
    return final.parent / f".{final.stem}.{arm}.raw.json"

def run_arm(arm: str, graph: str, output: Path, argv_template: Sequence[str], common: Mapping[str, Mapping[str, str]]) -> tuple[list[str], Any]:
    values = {f"input:{role}": spec["path"] for role, spec in common.items()}
    values.update({"graph": graph, "output": str(output)})
    argv = render(argv_template, values); scan_forbidden(argv)
    if output.exists(): raise ProtocolError(f"refusing to overwrite raw output: {output}")
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0: raise ProtocolError(f"{arm} evaluator exited with status {completed.returncode}")
    if not output.is_file(): raise ProtocolError(f"{arm} evaluator did not create output: {output}")
    return argv, load_json(output)

def compare_metric(base: float, candidate: float, spec: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = spec["atol"] + spec["rtol"] * max(abs(base), abs(candidate))
    delta = candidate - base
    gain = delta if spec["direction"] == "higher" else -delta
    state = "IMPROVEMENT" if gain > tolerance else "HARM" if gain < -tolerance else "EQUIVALENT"
    return {**spec, "canonical": base, "candidate": candidate, "candidate_minus_canonical": delta,
            "effective_tolerance": tolerance, "result": state}

def run_protocol(manifest_path: Path, output_path: Path, preflight_only: bool = False) -> dict[str, Any]:
    if output_path.exists(): raise ProtocolError(f"refusing to overwrite final output: {output_path}")
    manifest = load_json(manifest_path); verified = validate_manifest(manifest, manifest_path)
    base = {
        "protocol_version": PROTOCOL_VERSION, "experiment_id": manifest["experiment_id"],
        "source_revision": verified["source_revision"], "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": verified["manifest_sha256"], "evaluator_argv_sha256": verified["evaluator_argv_sha256"],
        "common_inputs": verified["common_inputs"], "common_fingerprint": verified["common_fingerprint"],
        "arms": verified["arms"], "metrics": verified["metrics"], "invariants": dict(manifest["invariants"]),
        "ratifies_repair": False, "scale_authorized": False, "promotion_authorized": False, "n0_complete": False,
    }
    if preflight_only:
        result = {**base, "status": "PREFLIGHT_OK", "classification": None,
                  "arm_common_fingerprints": {"canonical": verified["common_fingerprint"], "candidate": verified["common_fingerprint"]}}
        atomic_write_json(output_path, result); return result

    raw_paths = {name: arm_output(output_path, name) for name in ("canonical", "candidate")}
    commands, payloads = {}, {}
    for name in ("canonical", "candidate"):
        commands[name], payloads[name] = run_arm(name, verified["arms"][name]["path"], raw_paths[name], verified["evaluator_argv"], verified["common_inputs"])
    comparisons = []
    for spec in verified["metrics"]:
        comparisons.append(compare_metric(metric_value(payloads["canonical"], spec["path"]), metric_value(payloads["candidate"], spec["path"]), spec))
    states = [x["result"] for x in comparisons]
    classification = "HARM" if "HARM" in states else "IMPROVEMENT" if "IMPROVEMENT" in states else "EQUIVALENT"
    result = {**base, "status": "COMPLETE", "classification": classification, "comparisons": comparisons,
              "commands": commands, "raw_outputs": {name: {"path": str(raw_paths[name]), "sha256": sha256_file(raw_paths[name])} for name in raw_paths},
              "arm_common_fingerprints": {"canonical": verified["common_fingerprint"], "candidate": verified["common_fingerprint"]},
              "diagnostic_interpretation": "Descriptive evidence under one frozen downstream frontier only; no repair ratification, promotion, scaling, or training is authorized."}
    atomic_write_json(output_path, result); return result

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true"); args = parser.parse_args(argv)
    try: run_protocol(args.manifest, args.output, args.preflight_only)
    except (ProtocolError, OSError, json.JSONDecodeError) as exc:
        print(f"ARBITRATION_PROTOCOL_ERROR: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
