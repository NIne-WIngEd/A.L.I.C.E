#!/usr/bin/env python3
"""One-identity Kaggle L4 access and per-rank capacity check; uploads no model/data."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OWNER = "mkrayanyan"
N0_REVISION = "a19f8e8893422702c138182f239064385addf91c"
PROJECTED_BYTES_PER_RANK = 17754461984  # P43 job 576210, not a new estimate.
MAX_FRACTION = 0.85  # Frozen P43 qualification contract.
REQUEST = {"schema": "alice.n0.p43.kaggle-l4-hardware-check.v1",
           "n0_revision": N0_REVISION, "accelerator": "NvidiaL4",
           "minimum_world_size": 2, "projected_bytes_per_rank": PROJECTED_BYTES_PER_RANK,
           "max_fraction": MAX_FRACTION, "private_identity_data": False,
           "model_training": False}
RUNNER_TEMPLATE = '''import hashlib
import json
import platform
from pathlib import Path

import torch

request = __REQUEST__
devices = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    total = int(props.total_memory)
    devices.append({"index": index, "name": props.name,
                    "total_memory_bytes": total,
                    "projection_fraction": request["projected_bytes_per_rank"] / total,
                    "capacity_pass": request["projected_bytes_per_rank"] <=
                    request["max_fraction"] * total})
receipt = {"schema": "alice.n0.p43.kaggle-l4-hardware-result.v1",
           "request_sha256": "__SHA__", "n0_revision": request["n0_revision"],
           "torch_version": torch.__version__, "python_version": platform.python_version(),
           "cuda_runtime": torch.version.cuda,
           "devices": devices, "gpu_training_authorized": False,
           "model_training_performed": False, "private_identity_data": False,
           "status": "CANDIDATE_HARDWARE_ONLY" if len(devices) >= 2 and
               all(d["capacity_pass"] and "L4" in d["name"].upper()
                   for d in devices[:2]) else "INSUFFICIENT_OR_UNAVAILABLE_HARDWARE"}
out = Path("/kaggle/working/probe_result.json")
out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
transfer = {"result_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "request_sha256": "__SHA__"}
Path("/kaggle/working/probe_transfer.json").write_text(
    json.dumps(transfer, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
'''
REQUEST["runner_template_sha256"] = hashlib.sha256(RUNNER_TEMPLATE.encode()).hexdigest()
REQUEST_SHA = hashlib.sha256(json.dumps(REQUEST, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
SLUG = f"rayan-n0-l4-p43-{REQUEST_SHA[:12]}"
REF = f"{OWNER}/{SLUG}"
ROOT = Path.home() / "Downloads" / f"RAYAN_N0_L4_P43_{REQUEST_SHA[:12]}"
STAGE = ROOT / "stage"
OUTPUT = ROOT / "output"
STATE = ROOT / "dispatch_state.json"
RUNNER = RUNNER_TEMPLATE.replace("__REQUEST__", repr(REQUEST)).replace("__SHA__", REQUEST_SHA)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def call(cli: str, args: list[str], label: str) -> subprocess.CompletedProcess[str]:
    p = subprocess.run([cli, *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    logs = ROOT / "cli"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{label}.out").write_text(p.stdout, encoding="utf-8")
    (logs / f"{label}.err").write_text(p.stderr, encoding="utf-8")
    write_json(logs / f"{label}.json", {"argv": args, "exit_code": p.returncode,
               "utc": datetime.now(timezone.utc).isoformat()})
    return p


def status(cli: str, number: int) -> tuple[str | None, int]:
    p = call(cli, ["kernels", "status", REF], f"status-{number:04d}")
    if p.returncode:
        return None, p.returncode
    response = (p.stdout + " " + p.stderr).upper()
    for token in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"):
        if token in response:
            return token, p.returncode
    return None, p.returncode


def stage() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    runner = STAGE / "runner.py"
    runner.write_text(RUNNER, encoding="utf-8", newline="\n")
    compile(runner.read_text(encoding="utf-8"), str(runner), "exec")
    write_json(STAGE / "kernel-metadata.json", {
        "id": REF, "title": SLUG.replace("-", " "), "code_file": "runner.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaL4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": []})
    assert sorted(p.name for p in STAGE.iterdir()) == ["kernel-metadata.json", "runner.py"]
    assert read_json(STAGE / "kernel-metadata.json")["id"] == REF


def verify() -> dict:
    results = list(OUTPUT.rglob("probe_result.json"))
    transfers = list(OUTPUT.rglob("probe_transfer.json"))
    if len(results) != 1 or len(transfers) != 1:
        raise RuntimeError("expected exactly one result and transfer receipt")
    result, transfer = read_json(results[0]), read_json(transfers[0])
    actual = hashlib.sha256(results[0].read_bytes()).hexdigest()
    if transfer.get("result_sha256") != actual or transfer.get("request_sha256") != REQUEST_SHA:
        raise RuntimeError("Kaggle output hash or request identity mismatch")
    if result.get("request_sha256") != REQUEST_SHA or result.get("n0_revision") != N0_REVISION:
        raise RuntimeError("Kaggle result identity mismatch")
    if result.get("model_training_performed") is not False or result.get("gpu_training_authorized") is not False:
        raise RuntimeError("unexpected training claim")
    devices = result.get("devices", [])
    expected = (len(devices) >= 2 and all(
        "L4" in d["name"].upper() and d["total_memory_bytes"] > 0 and
        d["capacity_pass"] is True and
        PROJECTED_BYTES_PER_RANK <= MAX_FRACTION * d["total_memory_bytes"]
        for d in devices[:2]))
    if (result.get("status") == "CANDIDATE_HARDWARE_ONLY") != expected:
        raise RuntimeError("Kaggle result hardware claim mismatch")
    return result


def main() -> int:
    stage()
    if "--prepare-only" in sys.argv:
        print(f"STAGED={STAGE} REF={REF} REQUEST_SHA={REQUEST_SHA}")
        return 0
    cli = shutil.which("kaggle.exe") or shutil.which("kaggle")
    if not cli:
        raise RuntimeError("run from the existing authenticated Kaggle CLI environment")
    if call(cli, ["--version"], "version").returncode:
        raise RuntimeError("Kaggle CLI version check failed")
    if call(cli, ["kernels", "list", "-m", "--page-size", "1"], "authenticated-list").returncode:
        raise RuntimeError("Kaggle account authentication check failed before any push intent")
    state = read_json(STATE) if STATE.exists() else {
        "request_sha256": REQUEST_SHA, "kernel_ref": REF, "push_attempted": False,
        "status_number": 0}
    if state.get("request_sha256") != REQUEST_SHA or state.get("kernel_ref") != REF:
        raise RuntimeError("existing dispatch state identity drift")
    write_json(STATE, state)
    state["status_number"] += 1
    observed, rc = status(cli, state["status_number"])
    write_json(STATE, state)
    if rc == 0 and observed is None:
        print("UNRECOGNIZED_EXISTING_KERNEL_STATUS; inspect logs and preserve identity")
        return 4
    if observed is None and not state["push_attempted"]:
        state["push_attempted"] = True
        write_json(STATE, state)  # Persist intent before exactly one remote push.
        push = call(cli, ["kernels", "push", "-p", str(STAGE), "--accelerator", "NvidiaL4"], "push")
        state["push_exit_code"] = push.returncode
        write_json(STATE, state)
        if push.returncode:
            print("PUSH_UNRESOLVED; inspect cli/push.* and reconcile the same identity; no automatic repush")
            return 3
        time.sleep(20)
    elif observed is None:
        print("PUSH_INTENT_EXISTS; reconcile exact kernel identity; no automatic repush")
        return 4
    for _ in range(40):
        state["status_number"] += 1
        observed, rc = status(cli, state["status_number"])
        write_json(STATE, state)
        if observed in ("COMPLETE", "ERROR", "CANCELLED"):
            break
        time.sleep(15)
    if observed not in ("COMPLETE", "ERROR", "CANCELLED"):
        print("STATUS_UNRESOLVED; preserve remote identity; rerun controller to resume")
        return 5
    OUTPUT.mkdir(parents=True, exist_ok=True)
    downloaded = call(cli, ["kernels", "output", REF, "-p", str(OUTPUT), "-o"], "output")
    if downloaded.returncode or observed != "COMPLETE":
        print(f"REMOTE_{observed}; preserved CLI logs and remote kernel; no retry")
        return 6
    receipt = verify()
    state["output_verified"] = True
    state["hardware_result"] = receipt["status"]
    write_json(STATE, state)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"VERIFIED_OUTPUT={OUTPUT}; remote kernel preserved for inspection")
    return 0 if receipt["status"] == "CANDIDATE_HARDWARE_ONLY" else 7


if __name__ == "__main__":
    raise SystemExit(main())
