#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
CORPUS_DIR="${N0_V02_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
TEACHER_DIR="${N0_V02_TEACHER_DIR:-$WORKDIR/teacher-bank-v0.5}"
TEACHER_REGISTRY="$TEACHER_DIR/n0_v02_teacher_bank_v0.5.runtime.json"
TEACHER_AUDIT="$TEACHER_DIR/teacher-bank-v0.5-audit.json"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
REPAIR_CURRICULUM="$ROOT/data/eipm/n0/n0_v02_targeted_repair_v0.1.jsonl"
REPAIR_MANIFEST="$ROOT/data/eipm/n0/n0_v02_targeted_repair_v0.1.manifest.json"
FIRST_TRANCHE_ROOT="${N0_V02_FIRST_TRANCHE_ROOT:-$WORKDIR/first-tranche-v0.1}"
PARENT_DIR="$FIRST_TRANCHE_ROOT/checkpoints/step-00000250"
RUN_ROOT="${N0_V02_TARGETED_REPAIR_ROOT:-$WORKDIR/targeted-repair-v0.1}"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
PORT="${N0_MAIN_PROCESS_PORT:-29500}"
SEED="${N0_V02_REPAIR_SEED:-20260915}"
MAX_STEPS="${N0_V02_REPAIR_MAX_STEPS:-120}"
SAVE_EVERY="${N0_V02_REPAIR_SAVE_EVERY:-40}"
TOP_LAYERS="${N0_V02_REPAIR_TOP_LAYERS:-4}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$RUN_ROOT" && -n "$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing targeted repair: $RUN_ROOT" >&2
  exit 2
fi

for required in \
  "$CORPUS_DIR/corpus_receipt.json" \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$TOKENIZER_DIR/tokenizer_receipt.json" \
  "$TEACHER_REGISTRY" \
  "$TEACHER_AUDIT" \
  "$CONFIG" \
  "$SOURCE_CONFIG" \
  "$REPAIR_CURRICULUM" \
  "$REPAIR_MANIFEST" \
  "$PARENT_DIR/receipt.json" \
  "$PARENT_DIR/alice_n0_v02.safetensors" \
  "$PARENT_DIR/ranker.safetensors"
do
  if [[ ! -f "$required" ]]; then
    echo "Required targeted-repair artifact is missing: $required" >&2
    exit 3
  fi
done

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 2 ]]; then
  echo "Targeted repair requires two visible CUDA devices; found $CUDA_COUNT" >&2
  exit 4
fi
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
for name in "${GPU_NAMES[@]:0:2}"; do
  if [[ "$name" != *"P100"* ]]; then
    echo "Expected P100 devices, observed: $name" >&2
    exit 5
  fi
done

mkdir -p "$RUN_ROOT" "$CHECKPOINT_ROOT"

# Compile/import checks happen inside the same allocation immediately before
# real weight updates. They are not a separate infrastructure qualification.
python -m py_compile \
  "$ROOT/src/alice_personality/n0/v02_model.py" \
  "$ROOT/src/alice_personality/n0/v02_training.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_targeted_repair.py"

echo "===== A.L.I.C.E. N0 V0.2 FAILURE-DRIVEN TARGETED REPAIR ====="
date -Is
echo "repo_root=$ROOT"
echo "run_root=$RUN_ROOT"
echo "parent_checkpoint=$PARENT_DIR"
echo "parent_step=250"
echo "repair_curriculum=$REPAIR_CURRICULUM"
echo "repair_train_rows=20"
echo "repair_dev_rows_held_out=10"
echo "top_backbone_layers_trainable=$TOP_LAYERS"
echo "target_steps=$MAX_STEPS"
echo "save_every=$SAVE_EVERY"
echo "learning_rate=3e-5"
echo "loss_mix=public_mlm:0.15,targeted_repair:0.55,governed_replay:0.30"
echo "serving_graph_unchanged=true"
echo "parameter_growth=0"
echo "context_growth=0"
echo "runtime_adapter_required=false"
echo "private_identity_gradient=false"
echo "network_required=false"
echo "cuda_count=$CUDA_COUNT"
echo "gpu0=${GPU_NAMES[0]}"
echo "gpu1=${GPU_NAMES[1]}"

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_targeted_repair.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG" \
  --teacher-registry "$TEACHER_REGISTRY" \
  --teacher-audit "$TEACHER_AUDIT" \
  --repair-curriculum "$REPAIR_CURRICULUM" \
  --repair-manifest "$REPAIR_MANIFEST" \
  --parent-checkpoint "$PARENT_DIR" \
  --output-dir "$CHECKPOINT_ROOT" \
  --top-layers "$TOP_LAYERS" \
  --max-steps "$MAX_STEPS" \
  --save-every "$SAVE_EVERY" \
  --learning-rate 3e-5 \
  --weight-decay 0.05 \
  --warmup-steps 10 \
  --teacher-batch-size 2 \
  --teacher-max-length 256 \
  --sequence-length 512 \
  --seed "$SEED" \
  --mixed-precision fp16

echo "===== A.L.I.C.E. N0 V0.2 TARGETED REPAIR COMPLETE ====="
date -Is
