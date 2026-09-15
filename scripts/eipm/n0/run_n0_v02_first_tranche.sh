#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
CORPUS_DIR="${N0_V02_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
TEACHER_DIR="${N0_V02_TEACHER_DIR:-$WORKDIR/teacher-bank-v0.5}"
TEACHER_REGISTRY="$TEACHER_DIR/n0_v02_teacher_bank_v0.5.runtime.json"
TEACHER_AUDIT="$TEACHER_DIR/teacher-bank-v0.5-audit.json"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
RUN_ROOT="${N0_V02_RUN_ROOT:-$WORKDIR/first-tranche-v0.1}"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
BASELINE_ROOT="$RUN_ROOT/random-baseline"
EVAL_ROOT="$RUN_ROOT/eval"
SEED="${N0_V02_SEED:-20260914}"
MAX_STEPS="${N0_V02_MAX_STEPS:-500}"
SAVE_EVERY="${N0_V02_SAVE_EVERY:-250}"
PORT="${N0_MAIN_PROCESS_PORT:-29500}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$RUN_ROOT" && -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite an existing first-tranche run: $RUN_ROOT" >&2
  exit 2
fi
for required in \
  "$CORPUS_DIR/corpus_receipt.json" \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$TOKENIZER_DIR/tokenizer_receipt.json" \
  "$TEACHER_REGISTRY" \
  "$TEACHER_AUDIT" \
  "$CONFIG" \
  "$SOURCE_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Required N0 v0.2 artifact is missing: $required" >&2
    exit 3
  fi
done

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 2 ]]; then
  echo "First N0 v0.2 tranche requires two visible CUDA devices; found $CUDA_COUNT" >&2
  exit 4
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
for name in "${GPU_NAMES[@]:0:2}"; do
  if [[ "$name" != *"P100"* ]]; then
    echo "Expected P100 devices, observed: $name" >&2
    exit 5
  fi
done

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT" "$EVAL_ROOT"

echo "===== A.L.I.C.E. N0 V0.2 FIRST REAL MULTITASK TRANCHE ====="
date -Is
echo "repo_root=$ROOT"
echo "run_root=$RUN_ROOT"
echo "corpus_dir=$CORPUS_DIR"
echo "tokenizer_dir=$TOKENIZER_DIR"
echo "teacher_dir=$TEACHER_DIR"
echo "cuda_count=$CUDA_COUNT"
echo "gpu0=${GPU_NAMES[0]}"
echo "gpu1=${GPU_NAMES[1]}"
echo "seed=$SEED"
echo "target_step=$MAX_STEPS"
echo "save_every=$SAVE_EVERY"
echo "sequence_length=512"
echo "objective_weights=mlm:0.70,preference:0.10,rationale:0.10,contrastive:0.10"
echo "network_required=false"
echo "private_identity_gradient=false"

# One fail-fast code check inside the real learning allocation. This is not a
# separate qualification stage; it simply prevents spending the allocation on
# a syntax/import error.
python -m py_compile \
  "$ROOT/src/alice_personality/n0/v02_model.py" \
  "$ROOT/src/alice_personality/n0/v02_training.py" \
  "$ROOT/scripts/eipm/n0/export_n0_v02_initial.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_multitask.py" \
  "$ROOT/scripts/eipm/n0/compile_n0_v02_fixed_eval.py" \
  "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py"

CORE_COMPILED="$EVAL_ROOT/core-fixed-compiled.jsonl"
CORE_RECEIPT="$EVAL_ROOT/core-fixed-receipt.json"
VOICE_COMPILED="$EVAL_ROOT/voice-fixed-compiled.jsonl"
VOICE_RECEIPT="$EVAL_ROOT/voice-fixed-receipt.json"

python "$ROOT/scripts/eipm/n0/compile_n0_v02_fixed_eval.py" \
  --base "$ROOT/evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl" \
  --output "$CORE_COMPILED" \
  --manifest "$CORE_RECEIPT" \
  --order-variants 3 \
  --minimum-cases-per-competency 1 \
  --expected-competencies 43

python "$ROOT/scripts/eipm/n0/compile_n0_v02_fixed_eval.py" \
  --base "$ROOT/evaluation/eipm/n0/n0_v02_voice_readiness_base_v0.1.jsonl" \
  --output "$VOICE_COMPILED" \
  --manifest "$VOICE_RECEIPT" \
  --order-variants 3 \
  --minimum-cases-per-competency 1 \
  --expected-competencies 8

# Freeze the exact random starting point and score it before any optimizer step.
python "$ROOT/scripts/eipm/n0/export_n0_v02_initial.py" \
  --config "$CONFIG" \
  --output-dir "$BASELINE_ROOT" \
  --seed "$SEED"

python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$BASELINE_ROOT/mlm" \
  --ranker "$BASELINE_ROOT/ranker.safetensors" \
  --benchmark "$CORE_COMPILED" \
  --benchmark-receipt "$CORE_RECEIPT" \
  --output-dir "$EVAL_ROOT/step-00000000-core" \
  --batch-size 4 \
  --device cuda

python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$BASELINE_ROOT/mlm" \
  --ranker "$BASELINE_ROOT/ranker.safetensors" \
  --benchmark "$VOICE_COMPILED" \
  --benchmark-receipt "$VOICE_RECEIPT" \
  --output-dir "$EVAL_ROOT/step-00000000-voice" \
  --batch-size 4 \
  --device cuda

# This is the first permanent native N0 v0.2 weight update. No v0.1 weights
# and no private Elaina material are permitted anywhere in this invocation.
accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_multitask.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG" \
  --teacher-registry "$TEACHER_REGISTRY" \
  --teacher-audit "$TEACHER_AUDIT" \
  --output-dir "$CHECKPOINT_ROOT" \
  --sequence-length 512 \
  --mlm-micro-batch-size 1 \
  --mlm-grad-accum 8 \
  --teacher-batch-size 2 \
  --teacher-max-length 256 \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 100 \
  --scheduler-total-steps 10000 \
  --save-every "$SAVE_EVERY" \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --seed "$SEED" \
  --mixed-precision fp16

for STEP in 250 500; do
  if (( STEP > MAX_STEPS )); then
    continue
  fi
  STEP_DIR="$CHECKPOINT_ROOT/step-$(printf '%08d' "$STEP")"
  if [[ ! -f "$STEP_DIR/receipt.json" ]]; then
    echo "Expected checkpoint missing: $STEP_DIR" >&2
    exit 6
  fi

  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --benchmark "$CORE_COMPILED" \
    --benchmark-receipt "$CORE_RECEIPT" \
    --output-dir "$EVAL_ROOT/step-$(printf '%08d' "$STEP")-core" \
    --batch-size 4 \
    --device cuda

  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --benchmark "$VOICE_COMPILED" \
    --benchmark-receipt "$VOICE_RECEIPT" \
    --output-dir "$EVAL_ROOT/step-$(printf '%08d' "$STEP")-voice" \
    --batch-size 4 \
    --device cuda
done

python - "$RUN_ROOT" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_steps = int(sys.argv[2])
steps = [0] + [step for step in (250, 500) if step <= max_steps]
summary = {
    "schema": "alice.eipm.n0.v02-first-tranche-summary.v0.1",
    "status": "PASS",
    "private_identity_gradient": False,
    "steps": {},
}
for step in steps:
    key = f"step-{step:08d}"
    summary["steps"][key] = {}
    for suite in ("core", "voice"):
        path = root / "eval" / f"{key}-{suite}" / "fixed_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        summary["steps"][key][suite] = {
            "top1_accuracy": metrics["top1_accuracy"],
            "separation_rate": metrics["separation_rate"],
            "full_invariance_pass_rate": metrics["full_invariance_pass_rate"],
            "mean_margin": metrics["mean_margin"],
        }
(root / "first_tranche_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "===== N0 V0.2 FIRST REAL MULTITASK TRANCHE COMPLETE ====="
date -Is
