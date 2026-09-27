#!/usr/bin/env python3
"""Trace every real J3 system call; diagnostic only, no gradient authority."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


PINNED = "a19f8e8893422702c138182f239064385addf91c"
DIAG_SCHEMA = "alice.eipm.n0.full-envelope-fp16-j3-diagnostic.v1"
TRACE_SCHEMA = "alice.n0.p43.full-j3-forward-attribution.v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    args, remaining = parser.parse_known_args()
    if not args.diagnostic_only or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("requires original single-process --diagnostic-only full-J3 path")
    trace_path, diagnostic_path = Path(args.trace_output), Path(args.output)
    if trace_path.resolve() == diagnostic_path.resolve():
        raise SystemExit("trace and diagnostic must use separate paths")
    if trace_path.exists() or diagnostic_path.exists():
        raise SystemExit("preserve existing diagnostic/trace roots")
    qualifier = (Path.cwd() / "scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py").resolve()
    if not qualifier.is_file():
        raise SystemExit("must run from pinned N0 source root")
    import subprocess
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("pinned N0 source must be clean")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("real CUDA device required")

    spec = importlib.util.spec_from_file_location("n0_p43_forward_qualifier", qualifier)
    if spec is None or spec.loader is None:
        raise SystemExit("qualifier import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_step = module.execute_full_envelope_joint_step
    measurements: list[dict] = []
    pair = 0

    def traced_step(*step_args, **step_kwargs):
        nonlocal pair
        pair += 1
        system = step_kwargs["system"]
        original_forward = system.forward
        call = 0

        def traced_forward(*forward_args, **forward_kwargs):
            nonlocal call
            call += 1
            device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.synchronize(device)
            before_allocated = int(torch.cuda.memory_allocated(device))
            before_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
            result = original_forward(*forward_args, **forward_kwargs)
            torch.cuda.synchronize(device)
            measurements.append({
                "pair_index": pair,
                "call_index": call,
                "task": str(forward_kwargs.get("task", "unknown")),
                "allocated_before_bytes": before_allocated,
                "allocated_after_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_before_bytes": before_reserved,
                "reserved_after_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            })
            return result

        system.forward = traced_forward
        try:
            return original_step(*step_args, **step_kwargs)
        finally:
            system.forward = original_forward

    module.execute_full_envelope_joint_step = traced_step
    sys.argv = [str(qualifier), "--output", str(diagnostic_path), "--diagnostic-only", *remaining]
    try:
        module.main()
    except BaseException:
        # The original qualifier owns its failure state. Never publish a partial PASS trace.
        raise
    if not diagnostic_path.is_file():
        raise SystemExit("original diagnostic did not write its receipt")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (diagnostic.get("schema") != DIAG_SCHEMA
            or diagnostic.get("source_revision") != PINNED
            or diagnostic.get("world_size") != 1
            or diagnostic.get("stress_pair_count") != 18
            or diagnostic.get("gradient") is not False
            or diagnostic.get("optimizer_object_created") is not False
            or diagnostic.get("gpu_training_authorized") is not False):
        raise SystemExit("original full-J3 single-process diagnostic mismatch")
    if len(set(row["pair_index"] for row in measurements)) != 18:
        raise SystemExit("incomplete per-pair model-forward attribution")
    if any(sum(row["task"] == "full_envelope" for row in measurements
               if row["pair_index"] == index) != 4 for index in range(1, 19)):
        raise SystemExit("missing primary/three-counterfactual forward calls")
    trace = {
        "schema": TRACE_SCHEMA,
        "status": "COMPLETE_FORWARD_ATTRIBUTION_NOT_P43_AUTHORITY",
        "source_revision": revision,
        "qualifier_sha256": digest(qualifier),
        "diagnostic_sha256": digest(diagnostic_path),
        "mixture_manifest_sha256": diagnostic["mixture_manifest_sha256"],
        "stress_pair_count": 18,
        "forward_calls": measurements,
        "gradient": False, "backward": False,
        "optimizer_created": False, "training_authorized": False,
        "measured_training_peak_bytes": None,
        "projection_is_authority": False,
        "private_identity_data": False,
        "final_results_observed": False,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8") as output:
        json.dump(trace, output, sort_keys=True, indent=2)
        output.write("\n")
    print(f"FORWARD_ATTRIBUTION={trace_path}", flush=True)


if __name__ == "__main__":
    main()
