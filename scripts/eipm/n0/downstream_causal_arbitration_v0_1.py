#!/usr/bin/env python3
"""Diagnostic-only downstream causal arbitration for A.L.I.C.E. N0.

The protocol compares two graph candidates through one frozen downstream
frontier. It deliberately has no training, promotion, repair-ratification,
or scaling path. All mutable/frozen inputs are content-addressed and checked
before either arm is evaluated.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROTOCOL_VERSION = "n0_downstream_causal_arbitration_v0.1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Secondary fail-closed guard. The evaluator itself must still be a known
# evaluation-only implementation pinned by hash in the manifest.
FORBIDDEN_ARG_EXACT = {
    "--train",
    "--training",
    "--fit",
    "--finetune",
    "--fine-tune",
    "--backprop",
    "--backward",
    "--optimizer",
    "--learning-rate",
    "--lr",
    "--max-steps",
    "--epochs",
    "--num-epochs",
    "--gradient-accumulation-steps",
}
FORBIDDEN_ARG_PREFIXES = tuple(f"{item}=" for item in FORBIDDEN_ARG_EXACT)


class ProtocolError(RuntimeError):
    """Fail-closed protocol validation error."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return value


def _require_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ProtocolError(f"{label} must be {str(expected).lower()}")


def _resolve(base_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _verify_file_spec(label: str, spec: Any, base_dir: Path) -> Dict[str, str]:
    obj = _require_mapping(spec, label)
    raw_path = _require_nonempty_str(obj.get("path"), f"{label}.path")
    expected = _require_nonempty_str(
        obj.get("sha256"),
        f"{label}.sha256",
    ).lower()
    if not SHA256_RE.fullmatch(expected):
        raise ProtocolError(
            f"{label}.sha256 must be exactly 64 hexadecimal characters"
        )
    path = _resolve(base_dir, raw_path)
    if not path.is_file():
        raise ProtocolError(
            f"{label} does not exist or is not a regular file: {path}"
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ProtocolError(
            f"{label} sha256 mismatch: expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": actual}


def _scan_forbidden_argv(argv: Sequence[str]) -> None:
    for token in argv:
        lower = token.lower()
        if lower in FORBIDDEN_ARG_EXACT or lower.startswith(FORBIDDEN_ARG_PREFIXES):
            raise ProtocolError(
                "evaluator_argv contains forbidden training/mutation argument: "
                f"{token}"
            )


def _extract_placeholders(template: Sequence[str]) -> set[str]:
    fields: set[str] = set()
    formatter = __import__("string").Formatter()
    for token in template:
        for _, field_name, _, _ in formatter.parse(token):
            if field_name:
                fields.add(field_name)
    return fields


def _render_argv(
    template: Sequence[str],
    values: Mapping[str, str],
) -> List[str]:
    rendered: List[str] = []
    for token in template:
        try:
            rendered.append(token.format_map(values))
        except KeyError as exc:
            raise ProtocolError(
                f"unknown evaluator_argv placeholder: {exc.args[0]}"
            ) from exc
    return rendered


def _metric_value(payload: Any, dotted_path: str) -> float:
    current = payload
    for segment in dotted_path.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                raise ProtocolError(f"metric path not found: {dotted_path}")
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except ValueError as exc:
                raise ProtocolError(
                    f"list metric path segment is not an integer: {segment}"
                ) from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise ProtocolError(
                    f"metric path index out of range: {dotted_path}"
                ) from exc
        else:
            raise ProtocolError(
                "metric path traverses a scalar before completion: "
                f"{dotted_path}"
            )
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ProtocolError(f"metric path is not numeric: {dotted_path}")
    value = float(current)
    if not math.isfinite(value):
        raise ProtocolError(f"metric path is not finite: {dotted_path}")
    return value


def _compare_metric(
    baseline: float,
    candidate: float,
    direction: str,
    atol: float,
    rtol: float,
) -> Tuple[str, float, float]:
    tolerance = atol + rtol * max(abs(baseline), abs(candidate))
    raw_delta = candidate - baseline
    gain = raw_delta if direction == "higher" else -raw_delta
    if gain > tolerance:
        state = "IMPROVEMENT"
    elif gain < -tolerance:
        state = "HARM"
    else:
        state = "EQUIVALENT"
    return state, raw_delta, tolerance


def _validate_metric_specs(raw_metrics: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ProtocolError("metrics must be a non-empty list")
    seen: set[str] = set()
    metrics: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_metrics):
        obj = _require_mapping(raw, f"metrics[{index}]")
        path = _require_nonempty_str(
            obj.get("path"),
            f"metrics[{index}].path",
        )
        if path in seen:
            raise ProtocolError(f"duplicate metric path: {path}")
        seen.add(path)
        direction = obj.get("direction")
        if direction not in {"higher", "lower"}:
            raise ProtocolError(
                f"metrics[{index}].direction must be 'higher' or 'lower'"
            )
        try:
            atol = float(obj.get("atol", 0.0))
            rtol = float(obj.get("rtol", 0.0))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                f"metrics[{index}] tolerances must be numeric"
            ) from exc
        if (
            not math.isfinite(atol)
            or not math.isfinite(rtol)
            or atol < 0
            or rtol < 0
        ):
            raise ProtocolError(
                f"metrics[{index}] tolerances must be finite and non-negative"
            )
        metrics.append(
            {
                "path": path,
                "direction": direction,
                "atol": atol,
                "rtol": rtol,
            }
        )
    return metrics


def validate_manifest(manifest: Any, manifest_path: Path) -> Dict[str, Any]:
    root = _require_mapping(manifest, "manifest")
    if root.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol_version must be {PROTOCOL_VERSION!r}"
        )
    _require_nonempty_str(root.get("experiment_id"), "experiment_id")
    _require_nonempty_str(root.get("source_revision"), "source_revision")

    invariants = _require_mapping(root.get("invariants"), "invariants")
    for name, expected in {
        "diagnostic_only": True,
        "training_enabled": False,
        "ratifies_repair": False,
        "scale_authorized": False,
        "promotion_authorized": False,
    }.items():
        _require_bool(invariants.get(name), expected, f"invariants.{name}")

    base_dir = manifest_path.resolve().parent
    common_raw = _require_mapping(root.get("common_inputs"), "common_inputs")
    if not common_raw:
        raise ProtocolError("common_inputs must contain at least one pinned file")
    common: Dict[str, Dict[str, str]] = {}
    for name, spec in common_raw.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {"graph_path", "output_path"}
        ):
            raise ProtocolError(f"invalid common input name: {name!r}")
        common[name] = _verify_file_spec(
            f"common_inputs.{name}",
            spec,
            base_dir,
        )

    arms = _require_mapping(root.get("arms"), "arms")
    if set(arms.keys()) != {"canonical", "candidate"}:
        raise ProtocolError(
            "arms must contain exactly 'canonical' and 'candidate'"
        )
    verified_arms: Dict[str, Dict[str, str]] = {}
    for name in ("canonical", "candidate"):
        arm = _require_mapping(arms[name], f"arms.{name}")
        label = _require_nonempty_str(
            arm.get("label"),
            f"arms.{name}.label",
        )
        verified = _verify_file_spec(
            f"arms.{name}.graph",
            arm.get("graph"),
            base_dir,
        )
        verified["label"] = label
        verified_arms[name] = verified
    if (
        verified_arms["canonical"]["sha256"]
        == verified_arms["candidate"]["sha256"]
    ):
        raise ProtocolError(
            "canonical and candidate graph hashes are identical; "
            "arbitration would be non-informative"
        )

    argv = root.get("evaluator_argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(token, str) and token for token in argv)
    ):
        raise ProtocolError(
            "evaluator_argv must be a non-empty list of non-empty strings"
        )
    _scan_forbidden_argv(argv)
    placeholders = _extract_placeholders(argv)
    required_placeholders = {"graph_path", "output_path"}
    missing = required_placeholders - placeholders
    if missing:
        raise ProtocolError(
            "evaluator_argv missing required placeholders: "
            f"{sorted(missing)}"
        )
    allowed_placeholders = set(common) | required_placeholders
    unknown = placeholders - allowed_placeholders
    if unknown:
        raise ProtocolError(
            "evaluator_argv contains placeholders without pinned inputs: "
            f"{sorted(unknown)}"
        )

    metrics = _validate_metric_specs(root.get("metrics"))

    return {
        "manifest_sha256": sha256_file(manifest_path),
        "base_dir": str(base_dir),
        "common_inputs": common,
        "arms": verified_arms,
        "evaluator_argv": list(argv),
        "evaluator_argv_sha256": sha256_bytes(_canonical_json_bytes(argv)),
        "metrics": metrics,
    }


def _arm_output_path(final_output: Path, arm_name: str) -> Path:
    stem = final_output.stem or final_output.name
    suffix = final_output.suffix or ".json"
    return final_output.parent / f".{stem}.{arm_name}.raw{suffix}"


def _run_arm(
    arm_name: str,
    graph_path: str,
    output_path: Path,
    argv_template: Sequence[str],
    common_inputs: Mapping[str, Mapping[str, str]],
) -> Tuple[List[str], Any]:
    values = {name: spec["path"] for name, spec in common_inputs.items()}
    values.update(
        {
            "graph_path": graph_path,
            "output_path": str(output_path),
        }
    )
    argv = _render_argv(argv_template, values)
    _scan_forbidden_argv(argv)
    if output_path.exists():
        raise ProtocolError(
            f"refusing to overwrite pre-existing raw arm output: {output_path}"
        )
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise ProtocolError(
            f"{arm_name} evaluator exited with status {completed.returncode}"
        )
    if not output_path.is_file():
        raise ProtocolError(
            f"{arm_name} evaluator did not create expected output: {output_path}"
        )
    return argv, load_json(output_path)


def run_protocol(
    manifest_path: Path,
    final_output: Path,
    preflight_only: bool,
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    verified = validate_manifest(manifest, manifest_path)
    invariants = dict(manifest["invariants"])
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": manifest["experiment_id"],
        "source_revision": manifest["source_revision"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": verified["manifest_sha256"],
        "evaluator_argv_sha256": verified["evaluator_argv_sha256"],
        "invariants": invariants,
        "common_inputs": verified["common_inputs"],
        "arms": verified["arms"],
        "metrics": verified["metrics"],
    }
    if preflight_only:
        result = dict(base)
        result.update(
            {
                "status": "PREFLIGHT_OK",
                "classification": None,
                "ratifies_repair": False,
                "scale_authorized": False,
                "promotion_authorized": False,
            }
        )
        atomic_write_json(final_output, result)
        return result

    canonical_raw_path = _arm_output_path(final_output, "canonical")
    candidate_raw_path = _arm_output_path(final_output, "candidate")
    canonical_argv, canonical_raw = _run_arm(
        "canonical",
        verified["arms"]["canonical"]["path"],
        canonical_raw_path,
        verified["evaluator_argv"],
        verified["common_inputs"],
    )
    candidate_argv, candidate_raw = _run_arm(
        "candidate",
        verified["arms"]["candidate"]["path"],
        candidate_raw_path,
        verified["evaluator_argv"],
        verified["common_inputs"],
    )

    comparisons: List[Dict[str, Any]] = []
    states: List[str] = []
    for spec in verified["metrics"]:
        baseline = _metric_value(canonical_raw, spec["path"])
        candidate = _metric_value(candidate_raw, spec["path"])
        state, raw_delta, tolerance = _compare_metric(
            baseline,
            candidate,
            spec["direction"],
            spec["atol"],
            spec["rtol"],
        )
        states.append(state)
        comparisons.append(
            {
                **spec,
                "canonical": baseline,
                "candidate": candidate,
                "candidate_minus_canonical": raw_delta,
                "effective_tolerance": tolerance,
                "result": state,
            }
        )
    if "HARM" in states:
        classification = "HARM"
    elif "IMPROVEMENT" in states:
        classification = "IMPROVEMENT"
    else:
        classification = "EQUIVALENT"

    result = dict(base)
    result.update(
        {
            "status": "COMPLETE",
            "classification": classification,
            "comparisons": comparisons,
            "raw_outputs": {
                "canonical": {
                    "path": str(canonical_raw_path),
                    "sha256": sha256_file(canonical_raw_path),
                },
                "candidate": {
                    "path": str(candidate_raw_path),
                    "sha256": sha256_file(candidate_raw_path),
                },
            },
            "commands": {
                "canonical": canonical_argv,
                "candidate": candidate_argv,
            },
            "diagnostic_interpretation": (
                "Classification is descriptive evidence under the single "
                "frozen downstream frontier only. It does not ratify the "
                "endpoint repair and does not authorize promotion, scaling, "
                "or training."
            ),
            "ratifies_repair": False,
            "scale_authorized": False,
            "promotion_authorized": False,
        }
    )
    atomic_write_json(final_output, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_protocol(args.manifest, args.output, args.preflight_only)
    except (ProtocolError, OSError, json.JSONDecodeError) as exc:
        print(f"ARBITRATION_PROTOCOL_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
