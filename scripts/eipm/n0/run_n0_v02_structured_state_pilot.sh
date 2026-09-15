#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
PREP_ROOT="${N0_V02_STRUCTURED_PREP_ROOT:-$WORKDIR/structured-state-curriculum-v0.2}"
CURRICULUM="$PREP_ROOT/structured_state_curriculum.jsonl"
CURRICULUM_MANIFEST="$PREP_ROOT/structured_state_curriculum_manifest.json"
OUT_ROOT="${N0_V02_STRUCTURED_PILOT_ROOT:-$WORKDIR/structured-state-pilot-v0.1}"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_semantic_base_ratification_v0.1.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$SEMANTIC_CHECKPOINT/receipt.json" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$RATIFICATION"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required structured-state pilot artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing structured-state pilot: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile \
  "$ROOT/src/alice_personality/n0/structured_state.py" \
  "$ROOT/src/alice_personality/n0/structured_state_objectives.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_structured_state_pilot.py"

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Structured-state pilot requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 STRUCTURED-STATE PUBLIC PILOT ====="
date -Is
echo "semantic_parent=targeted-repair-v0.1/step-00000080"
echo "semantic_core_trainable=false"
echo "structured_branch_parameters=1656064"
echo "curriculum=$CURRICULUM"
echo "max_steps=240 save_every=80"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_structured_state_pilot.py" \
  --config "$CONFIG" \
  --structured-config "$STRUCTURED_CONFIG" \
  --semantic-ratification "$RATIFICATION" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --output-dir "$OUT_ROOT" \
  --max-length 256 \
  --encode-batch-size 32 \
  --train-batch-size 64 \
  --eval-batch-size 128 \
  --max-steps 240 \
  --save-every 80 \
  --warmup-steps 20 \
  --learning-rate 0.0003 \
  --weight-decay 0.05 \
  --seed 20260915

echo "===== N0 V0.2 STRUCTURED-STATE PUBLIC PILOT COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/structured_state_pilot_comparison.json"
