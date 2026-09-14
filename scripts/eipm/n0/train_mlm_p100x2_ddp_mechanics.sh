#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
SMOKE_WORKDIR="${N0_SMOKE_WORKDIR:?Set N0_SMOKE_WORKDIR to the completed P100 smoke workdir}"
CONFIG="${ALICE_N0_CONFIG:-$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json}"
SOURCE_CONFIG="${ALICE_N0_SOURCE_CONFIG:-$ROOT/configs/eipm/n0/public_corpus_bootstrap_v0.1.json}"
CORPUS_DIR="$SMOKE_WORKDIR/corpus-smoke"
TOKENIZER_DIR="$SMOKE_WORKDIR/tokenizer-smoke"
OUTPUT_DIR="$SMOKE_WORKDIR/ddp-mechanics-checkpoints"
PORT="${N0_MAIN_PROCESS_PORT:-$((20000 + (${SLURM_JOB_ID:-1337} % 20000)))}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$SMOKE_WORKDIR/runtime_smoke_receipt.json" ]]; then
  echo "Missing completed runtime smoke receipt: $SMOKE_WORKDIR/runtime_smoke_receipt.json" >&2
  exit 2
fi

"$PYTHON" - "$SMOKE_WORKDIR/runtime_smoke_receipt.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads(p.read_text())
if r.get("status") != "PASS":
    raise SystemExit("runtime smoke receipt is not PASS")
if int(r.get("cuda_device_count", 0)) < 2:
    raise SystemExit("runtime smoke receipt does not prove two CUDA devices")
if r.get("private_identity_gradient") is not False or r.get("private_identity_data") is not False:
    raise SystemExit("runtime smoke receipt has unexpected private-data flags")
PY

CUDA_COUNT="$($PYTHON - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 2 ]]; then
  echo "DDP mechanics run requires at least two visible CUDA devices; found $CUDA_COUNT." >&2
  exit 3
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
for name in "${GPU_NAMES[@]:0:2}"; do
  if [[ "$name" != *"P100"* ]]; then
    echo "Expected P100 devices but observed: $name" >&2
    exit 4
  fi
done

"$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG"

if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" || ! -f "$TOKENIZER_DIR/tokenizer_receipt.json" ]]; then
  echo "Smoke tokenizer artifacts are missing." >&2
  exit 5
fi

mapfile -t SHARDS < <(find "$CORPUS_DIR/shards" -type f -name '*.jsonl' | sort)
if [[ "${#SHARDS[@]}" -eq 0 ]]; then
  echo "No smoke shards found under $CORPUS_DIR/shards." >&2
  exit 6
fi
TRAIN_ARGS=()
for shard in "${SHARDS[@]}"; do TRAIN_ARGS+=(--train "$shard"); done

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

run_train() {
  local max_steps="$1"
  shift
  accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --main_process_port "$PORT" \
    "$ROOT/scripts/eipm/n0/train_mlm.py" \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
    --tokenizer-receipt "$TOKENIZER_DIR/tokenizer_receipt.json" \
    --corpus-dir "$CORPUS_DIR" \
    --source-config "$SOURCE_CONFIG" \
    "${TRAIN_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --sequence-length 512 \
    --micro-batch-size 1 \
    --grad-accum 2 \
    --max-steps "$max_steps" \
    --warmup-steps 1 \
    --save-every 4 \
    --learning-rate 3e-4 \
    --mixed-precision fp16 \
    "$@"
}

echo "===== DDP MECHANICS LEG 1: STEPS 0 -> 4 ====="
run_train 4

STEP4="$OUTPUT_DIR/step-00000004"
if [[ ! -f "$STEP4/receipt.json" || ! -d "$STEP4/accelerator_state" ]]; then
  echo "Step-4 checkpoint/state missing." >&2
  exit 7
fi

"$PYTHON" - "$STEP4/receipt.json" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r["step"] == 4, r
assert r["world_size"] == 2, r
assert r["mixed_precision"] == "fp16", r
assert r["private_identity_gradient"] is False, r
assert int(r["tokens_seen_total"]) > 0, r
print(json.dumps({"leg1":"PASS","tokens_seen_total":r["tokens_seen_total"]}, sort_keys=True))
PY

echo "===== DDP MECHANICS LEG 2: RESUME 4 -> 8 ====="
run_train 8 --resume-from "$STEP4"

STEP8="$OUTPUT_DIR/step-00000008"
if [[ ! -f "$STEP8/receipt.json" || ! -d "$STEP8/accelerator_state" ]]; then
  echo "Step-8 checkpoint/state missing after resume." >&2
  exit 8
fi

export N0_DDP_ROOT="$ROOT"
export N0_DDP_SMOKE_WORKDIR="$SMOKE_WORKDIR"
export N0_DDP_STEP4="$STEP4"
export N0_DDP_STEP8="$STEP8"
export N0_DDP_GPU0="${GPU_NAMES[0]}"
export N0_DDP_GPU1="${GPU_NAMES[1]}"

"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

step4 = Path(os.environ["N0_DDP_STEP4"])
step8 = Path(os.environ["N0_DDP_STEP8"])
r4 = json.loads((step4 / "receipt.json").read_text())
r8 = json.loads((step8 / "receipt.json").read_text())
if r8["step"] != 8:
    raise SystemExit("resume leg did not reach step 8")
if r8["world_size"] != 2:
    raise SystemExit("resume leg world_size != 2")
if r8["resume_parent"] != str(step4):
    raise SystemExit("resume_parent does not match step-4 checkpoint")
if int(r8["tokens_seen_total"]) <= int(r4["tokens_seen_total"]):
    raise SystemExit("cumulative token counter did not increase after resume")

out = Path(os.environ["N0_DDP_SMOKE_WORKDIR"]) / "ddp_mechanics_receipt.json"
receipt = {
    "schema": "alice.eipm.n0.ddp-mechanics-receipt.v0.1",
    "status": "PASS",
    "purpose": "non_promotable_distributed_training_and_resume_mechanics_only",
    "world_size": 2,
    "gpu_devices": [os.environ["N0_DDP_GPU0"], os.environ["N0_DDP_GPU1"]],
    "mixed_precision": "fp16",
    "sequence_length": 512,
    "micro_batch_size_per_process": 1,
    "gradient_accumulation": 2,
    "first_checkpoint": {
        "step": 4,
        "receipt_sha256": sha256(step4 / "receipt.json"),
        "tokens_seen_total": r4["tokens_seen_total"],
    },
    "resumed_checkpoint": {
        "step": 8,
        "receipt_sha256": sha256(step8 / "receipt.json"),
        "tokens_seen_total": r8["tokens_seen_total"],
        "resume_parent": r8["resume_parent"],
    },
    "private_identity_data": False,
    "private_identity_gradient": False,
    "promotable_checkpoint": False,
}
out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status":"PASS","receipt":str(out)}, sort_keys=True))
PY

echo "===== 2xP100 DDP MECHANICS PASS ====="
echo "receipt=$SMOKE_WORKDIR/ddp_mechanics_receipt.json"
