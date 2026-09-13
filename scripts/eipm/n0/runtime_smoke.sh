#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKDIR="${ALICE_N0_WORKDIR:-$ROOT/.alice-private/n0-runtime-smoke}"
REQUIRE_CUDA="${N0_SMOKE_REQUIRE_CUDA:-1}"
MIN_CUDA_DEVICES="${N0_SMOKE_MIN_CUDA_DEVICES:-1}"
LOG_DIR="$WORKDIR/runtime-smoke-logs"

if [[ -e "$WORKDIR/corpus-smoke/corpus_receipt.json" || -d "$WORKDIR/corpus-smoke/shards" ]]; then
  cat >&2 <<EOF
Smoke runtime already contains a corpus lineage:
  $WORKDIR

Use a fresh ALICE_N0_WORKDIR for another smoke run. This prevents a second run from
silently reusing or appending to the previous smoke corpus.
EOF
  exit 2
fi

mkdir -p "$LOG_DIR"
export ALICE_N0_WORKDIR="$WORKDIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STARTED_AT="$(date -Is)"
GIT_HEAD="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
HOST="$(hostname 2>/dev/null || printf 'unknown')"

{
  echo "===== A.L.I.C.E. N0 RUNTIME SMOKE ====="
  echo "started_at=$STARTED_AT"
  echo "host=$HOST"
  echo "git_branch=$GIT_BRANCH"
  echo "git_head=$GIT_HEAD"
  echo "root=$ROOT"
  echo "workdir=$WORKDIR"
  echo "python=$PYTHON"
  echo "min_cuda_devices=$MIN_CUDA_DEVICES"
  echo
  echo "===== DISK ====="
  df -h "$WORKDIR" || true
  echo
  echo "===== GPU ====="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi not found"
  fi
} | tee "$LOG_DIR/environment.txt"

"$PYTHON" - <<'PY' | tee "$LOG_DIR/python-runtime.json"
import importlib
import json
import platform
import sys

packages = [
    "torch",
    "transformers",
    "tokenizers",
    "accelerate",
    "safetensors",
    "datasets",
    "huggingface_hub",
]
result = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {},
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_devices": [],
}
missing = []
for name in packages:
    try:
        module = importlib.import_module(name)
        result["packages"][name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        result["packages"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        missing.append(name)

try:
    import torch
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cuda_device_count"] = int(torch.cuda.device_count())
    result["cuda_devices"] = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
except Exception:
    pass

print(json.dumps(result, indent=2, sort_keys=True))
if missing:
    raise SystemExit("missing or broken N0 dependencies: " + ", ".join(missing))
PY

read -r CUDA_AVAILABLE CUDA_DEVICE_COUNT < <("$PYTHON" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0", torch.cuda.device_count())
PY
)
if [[ "$REQUIRE_CUDA" == "1" && "$CUDA_AVAILABLE" != "1" ]]; then
  cat >&2 <<EOF
CUDA is required for this runtime smoke by default, but torch.cuda.is_available() is false.
Run this inside the intended GPU allocation/runtime, or set N0_SMOKE_REQUIRE_CUDA=0
only when intentionally testing a sufficiently provisioned CPU runtime.
EOF
  exit 3
fi
if [[ "$REQUIRE_CUDA" == "1" && "$CUDA_DEVICE_COUNT" -lt "$MIN_CUDA_DEVICES" ]]; then
  echo "Runtime exposes $CUDA_DEVICE_COUNT CUDA device(s), but this smoke requires at least $MIN_CUDA_DEVICES." >&2
  exit 4
fi

run_stage() {
  local stage="$1"
  echo "===== STAGE: $stage =====" | tee "$LOG_DIR/$stage.log"
  bash "$ROOT/scripts/eipm/n0/run_stage.sh" "$stage" 2>&1 | tee -a "$LOG_DIR/$stage.log"
}

run_stage corpus-smoke
run_stage tokenizer-smoke
run_stage preflight-smoke

COMPLETED_AT="$(date -Is)"
export N0_RUNTIME_SMOKE_ROOT="$ROOT"
export N0_RUNTIME_SMOKE_WORKDIR="$WORKDIR"
export N0_RUNTIME_SMOKE_LOG_DIR="$LOG_DIR"
export N0_RUNTIME_SMOKE_STARTED_AT="$STARTED_AT"
export N0_RUNTIME_SMOKE_COMPLETED_AT="$COMPLETED_AT"
export N0_RUNTIME_SMOKE_GIT_HEAD="$GIT_HEAD"
export N0_RUNTIME_SMOKE_GIT_BRANCH="$GIT_BRANCH"
export N0_RUNTIME_SMOKE_HOST="$HOST"
export N0_RUNTIME_SMOKE_CUDA_COUNT="$CUDA_DEVICE_COUNT"

"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["N0_RUNTIME_SMOKE_ROOT"])
workdir = Path(os.environ["N0_RUNTIME_SMOKE_WORKDIR"])
log_dir = Path(os.environ["N0_RUNTIME_SMOKE_LOG_DIR"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def artifact(path: Path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

corpus_receipt = workdir / "corpus-smoke" / "corpus_receipt.json"
tokenizer_receipt = workdir / "tokenizer-smoke" / "tokenizer_receipt.json"
artifacts = []
for path in [corpus_receipt, tokenizer_receipt]:
    if path.is_file():
        artifacts.append(artifact(path))
for path in sorted(log_dir.glob("*.log")):
    artifacts.append(artifact(path))
for path in [log_dir / "environment.txt", log_dir / "python-runtime.json"]:
    if path.is_file():
        artifacts.append(artifact(path))

receipt = {
    "schema": "alice.eipm.n0.runtime-smoke-receipt.v0.2",
    "status": "PASS",
    "started_at": os.environ["N0_RUNTIME_SMOKE_STARTED_AT"],
    "completed_at": os.environ["N0_RUNTIME_SMOKE_COMPLETED_AT"],
    "host": os.environ["N0_RUNTIME_SMOKE_HOST"],
    "git_branch": os.environ["N0_RUNTIME_SMOKE_GIT_BRANCH"],
    "git_head": os.environ["N0_RUNTIME_SMOKE_GIT_HEAD"],
    "workdir": str(workdir),
    "cuda_device_count": int(os.environ["N0_RUNTIME_SMOKE_CUDA_COUNT"]),
    "artifacts": artifacts,
    "private_identity_gradient": False,
    "private_identity_data": False,
}
output = workdir / "runtime_smoke_receipt.json"
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "receipt": str(output)}, sort_keys=True))
PY

echo "===== N0 RUNTIME SMOKE PASS ====="
echo "completed_at=$COMPLETED_AT"
echo "receipt=$WORKDIR/runtime_smoke_receipt.json"
