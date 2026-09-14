#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
CHECKPOINT_DIR="$WORKDIR/checkpoints"
PARENT="${N0_RESUME_FROM:-$CHECKPOINT_DIR/step-00000200}"
TARGET_STEP="${N0_MAX_STEPS:-1000}"
SCHEDULER_TOTAL="${N0_SCHEDULER_TOTAL_STEPS:-10000}"
TARGET_DIR="$CHECKPOINT_DIR/step-$(printf '%08d' "$TARGET_STEP")"

export ALICE_N0_WORKDIR="$WORKDIR"
export N0_RESUME_FROM="$PARENT"
export N0_MAX_STEPS="$TARGET_STEP"
export N0_SCHEDULER_TOTAL_STEPS="$SCHEDULER_TOTAL"
export N0_WARMUP_STEPS="${N0_WARMUP_STEPS:-20}"
export N0_SAVE_EVERY="${N0_SAVE_EVERY:-200}"
export N0_SEQUENCE_LENGTH="${N0_SEQUENCE_LENGTH:-512}"
export N0_MICRO_BATCH_SIZE="${N0_MICRO_BATCH_SIZE:-1}"
export N0_GRAD_ACCUM="${N0_GRAD_ACCUM:-16}"
export N0_MIXED_PRECISION="${N0_MIXED_PRECISION:-fp16}"
export N0_MLM_EVAL_DEVICE="${N0_MLM_EVAL_DEVICE:-cuda}"
export N0_MLM_EVAL_BATCH_SIZE="${N0_MLM_EVAL_BATCH_SIZE:-4}"
export N0_MLM_EVAL_MAX_BATCHES="${N0_MLM_EVAL_MAX_BATCHES:-256}"
export N0_MLM_EVAL_SEED="${N0_MLM_EVAL_SEED:-424242}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$PARENT/receipt.json" || ! -d "$PARENT/model" || ! -d "$PARENT/accelerator_state" ]]; then
  echo "Parent checkpoint is incomplete: $PARENT" >&2
  exit 2
fi
if [[ -e "$TARGET_DIR" ]]; then
  echo "Target checkpoint already exists; refusing duplicate continuation: $TARGET_DIR" >&2
  exit 3
fi

python - "$PARENT/receipt.json" "$TARGET_STEP" "$SCHEDULER_TOTAL" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = int(sys.argv[2])
schedule = int(sys.argv[3])
step = int(receipt["step"])
if receipt.get("model_id") != "alice-n0-semantic-v0.1":
    raise SystemExit("unexpected parent model_id")
if receipt.get("private_identity_gradient") is not False:
    raise SystemExit("parent checkpoint is not a public N0 gradient lineage")
if step >= target:
    raise SystemExit(f"parent step {step} must be smaller than target {target}")
if schedule < target:
    raise SystemExit(f"scheduler horizon {schedule} must cover target {target}")
print(json.dumps({
    "parent_step": step,
    "target_step": target,
    "scheduler_total_steps_global": schedule,
    "parent_schema": receipt.get("schema"),
    "parent_tokens_seen_total": receipt.get("tokens_seen_total"),
}, sort_keys=True))
PY

echo "===== BASELINE HELD-OUT MLM EVALUATION ====="
N0_MLM_STEP_DIR="$PARENT" \
  bash "$ROOT/scripts/eipm/n0/run_stage.sh" evaluate-mlm

echo "===== CONTINUE DURABLE N0 MLM ====="
bash "$ROOT/scripts/eipm/n0/train_mlm_p100x2.sh"

if [[ ! -f "$TARGET_DIR/receipt.json" ]]; then
  echo "Expected target checkpoint receipt was not produced: $TARGET_DIR/receipt.json" >&2
  exit 4
fi

echo "===== POST-SEGMENT HELD-OUT MLM EVALUATION ====="
N0_MLM_STEP_DIR="$TARGET_DIR" \
  bash "$ROOT/scripts/eipm/n0/run_stage.sh" evaluate-mlm

echo "===== N0 CONTINUATION SEGMENT COMPLETE ====="
python - "$TARGET_DIR/receipt.json" <<'PY'
import json
import sys
from pathlib import Path
receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "step": receipt.get("step"),
    "tokens_seen_total": receipt.get("tokens_seen_total"),
    "resume_parent": receipt.get("resume_parent"),
    "scheduler_total_steps_global": receipt.get("scheduler_total_steps_global"),
    "scheduler_total_steps_internal": receipt.get("scheduler_total_steps_internal"),
    "legacy_scheduler_rebase": receipt.get("legacy_scheduler_rebase"),
    "private_identity_gradient": receipt.get("private_identity_gradient"),
}, sort_keys=True))
PY
