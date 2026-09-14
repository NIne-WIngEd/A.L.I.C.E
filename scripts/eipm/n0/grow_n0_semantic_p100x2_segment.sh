#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
CHECKPOINT_DIR="$WORKDIR/checkpoints"
PARENT="${N0_RESUME_FROM:-$CHECKPOINT_DIR/step-00001000}"
TARGET_STEP="${N0_MAX_STEPS:-2500}"
TARGET_DIR="$CHECKPOINT_DIR/step-$(printf '%08d' "$TARGET_STEP")"
SCHEDULER_TOTAL="${N0_SCHEDULER_TOTAL_STEPS:-10000}"
READINESS_DIR="$WORKDIR/teacher-probes/$(basename "$TARGET_DIR")-sol-readiness-v0.3"

export ALICE_N0_WORKDIR="$WORKDIR"
export N0_RESUME_FROM="$PARENT"
export N0_MAX_STEPS="$TARGET_STEP"
export N0_SCHEDULER_TOTAL_STEPS="$SCHEDULER_TOTAL"
export N0_WARMUP_STEPS="${N0_WARMUP_STEPS:-20}"
export N0_SAVE_EVERY="${N0_SAVE_EVERY:-500}"
export N0_SEQUENCE_LENGTH="${N0_SEQUENCE_LENGTH:-512}"
export N0_MICRO_BATCH_SIZE="${N0_MICRO_BATCH_SIZE:-1}"
export N0_GRAD_ACCUM="${N0_GRAD_ACCUM:-16}"
export N0_LEARNING_RATE="${N0_LEARNING_RATE:-3e-4}"
export N0_MIXED_PRECISION="${N0_MIXED_PRECISION:-fp16}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

bash -n "$ROOT/scripts/eipm/n0/run_n0_semantic_readiness_probe.sh"
python -m py_compile \
  "$ROOT/scripts/eipm/n0/train_mlm.py" \
  "$ROOT/scripts/eipm/n0/evaluate_mlm.py" \
  "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
  "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py"

if [[ ! -f "$PARENT/receipt.json" || ! -d "$PARENT/model" || ! -d "$PARENT/accelerator_state" ]]; then
  echo "Parent checkpoint is incomplete: $PARENT" >&2
  exit 2
fi
if [[ -e "$TARGET_DIR" ]]; then
  echo "Target checkpoint already exists; refusing duplicate semantic-growth segment: $TARGET_DIR" >&2
  exit 3
fi
if [[ -e "$READINESS_DIR" ]]; then
  echo "Target readiness output already exists; refusing ambiguous rerun: $READINESS_DIR" >&2
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
    "status": "PASS",
    "parent_step": step,
    "parent_tokens_seen_total": receipt.get("tokens_seen_total"),
    "target_step": target,
    "scheduler_total_steps_global": schedule,
    "private_identity_gradient": False,
}, sort_keys=True))
PY

echo "===== N0 SEMANTIC REPRESENTATION GROWTH ====="
bash "$ROOT/scripts/eipm/n0/train_mlm_p100x2.sh"

if [[ ! -f "$TARGET_DIR/receipt.json" || ! -d "$TARGET_DIR/model" ]]; then
  echo "Expected target checkpoint was not produced: $TARGET_DIR" >&2
  exit 4
fi

echo "===== FIXED POST-GROWTH SEMANTIC READINESS ====="
N0_MLM_STEP_DIR="$TARGET_DIR" \
N0_SEMANTIC_READINESS_DIR="$READINESS_DIR" \
  bash "$ROOT/scripts/eipm/n0/run_n0_semantic_readiness_probe.sh"

echo "===== N0 SEMANTIC GROWTH COMPARISON ====="
python - \
  "$PARENT/receipt.json" \
  "$TARGET_DIR/receipt.json" \
  "$WORKDIR/teacher-probes/step-00001000-sol-repair-v0.3/repaired-dev/dev_metrics.json" \
  "$READINESS_DIR/dev/dev_metrics.json" \
  "$WORKDIR/teacher-probes/step-00001000-sol-v0.2/mlm-dev-multimask.json" \
  "$READINESS_DIR/mlm-dev-multimask.json" <<'PY'
import json
import sys
from pathlib import Path

def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

parent = load(sys.argv[1])
target = load(sys.argv[2])
old_dev = load(sys.argv[3])
new_dev = load(sys.argv[4])
old_mlm = load(sys.argv[5])
new_mlm = load(sys.argv[6])

out = {
    "status": "PASS",
    "parent_step": parent.get("step") if parent else None,
    "target_step": target.get("step") if target else None,
    "parent_tokens_seen_total": parent.get("tokens_seen_total") if parent else None,
    "target_tokens_seen_total": target.get("tokens_seen_total") if target else None,
    "old_fixed_dev_top1": old_dev.get("top1_accuracy") if old_dev else None,
    "new_fixed_dev_top1": new_dev.get("top1_accuracy") if new_dev else None,
    "old_mlm_mean_nll": old_mlm.get("mean_masked_token_nll") if old_mlm else None,
    "new_mlm_mean_nll": new_mlm.get("mean_masked_token_nll") if new_mlm else None,
    "private_identity_gradient": False,
}
if old_dev and new_dev:
    out["fixed_dev_top1_delta"] = new_dev["top1_accuracy"] - old_dev["top1_accuracy"]
if old_mlm and new_mlm:
    out["mlm_mean_nll_delta"] = new_mlm["mean_masked_token_nll"] - old_mlm["mean_masked_token_nll"]
print(json.dumps(out, indent=2, sort_keys=True))
PY
