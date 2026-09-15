#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
TEACHER_DIR="${N0_V02_TEACHER_DIR:-$WORKDIR/teacher-bank-v0.5}"
TEACHER_REGISTRY="$TEACHER_DIR/n0_v02_teacher_bank_v0.5.runtime.json"
TEACHER_AUDIT="$TEACHER_DIR/teacher-bank-v0.5-audit.json"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
REPAIR_CURRICULUM="$ROOT/data/eipm/n0/n0_v02_targeted_repair_v0.1.jsonl"
REPAIR_MANIFEST="$ROOT/data/eipm/n0/n0_v02_targeted_repair_v0.1.manifest.json"
TRANCHE_ROOT="${N0_V02_FIRST_TRANCHE_ROOT:-$WORKDIR/first-tranche-v0.1}"
PARENT_DIR="$TRANCHE_ROOT/checkpoints/step-00000250"
REPAIR_ROOT="${N0_V02_TARGETED_REPAIR_ROOT:-$WORKDIR/targeted-repair-v0.1}/checkpoints"
TEACHER_DEV_ROOT="${N0_V02_TEACHER_DEV_EVAL_ROOT:-$WORKDIR/teacher-dev-challenge-v0.2}"
NOVEL_ROOT="$WORKDIR/novel-cross-challenge-v0.1"
NOVEL_COMPILED="$NOVEL_ROOT/novel_cross_competency_compiled.jsonl"
NOVEL_MANIFEST="$NOVEL_ROOT/novel_cross_competency_manifest.json"
OUT_ROOT="${N0_V02_POST_REPAIR_ROOT:-$WORKDIR/post-repair-selection-v0.1}"
SELECTION="$OUT_ROOT/post_repair_selection.json"
LATENCY="$OUT_ROOT/selected_latency.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing post-repair selection: $OUT_ROOT" >&2
  exit 2
fi

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$CONFIG" \
  "$REPAIR_CURRICULUM" \
  "$REPAIR_MANIFEST" \
  "$TEACHER_REGISTRY" \
  "$TEACHER_AUDIT" \
  "$PARENT_DIR/ranker.safetensors" \
  "$PARENT_DIR/mlm/config.json" \
  "$REPAIR_ROOT/step-00000040/ranker.safetensors" \
  "$REPAIR_ROOT/step-00000080/ranker.safetensors" \
  "$REPAIR_ROOT/step-00000120/ranker.safetensors" \
  "$NOVEL_COMPILED" \
  "$NOVEL_MANIFEST" \
  "$NOVEL_ROOT/novel_cross_challenge_comparison.json" \
  "$TEACHER_DEV_ROOT/teacher_dev_challenge_comparison.json"
do
  if [[ ! -f "$required" ]]; then
    echo "Required post-repair artifact is missing: $required" >&2
    exit 3
  fi
done

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Post-repair selection requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  "$ROOT/scripts/eipm/n0/evaluate_n0_v02_teacher_dev.py" \
  "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
  "$ROOT/scripts/eipm/n0/select_n0_v02_post_repair.py" \
  "$ROOT/scripts/eipm/n0/benchmark_n0_v02_warm_latency.py"

echo "===== A.L.I.C.E. N0 V0.2 COMPACT POST-REPAIR SELECTION ====="
date -Is
echo "states=parent250,repair40,repair80,repair120"
echo "suites=repair_dev,teacher_dev,frozen_novel"
echo "new_training=false"
echo "selection_runs_once=true"
echo "latency_check=selected_state_only"
echo "private_identity_gradient=false"

run_eval () {
  local KEY="$1"
  local STEP_DIR="$2"
  local DEST="$OUT_ROOT/$KEY"
  mkdir -p "$DEST/repair-dev" "$DEST/teacher-dev" "$DEST/novel"

  python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --curriculum "$REPAIR_CURRICULUM" \
    --curriculum-manifest "$REPAIR_MANIFEST" \
    --output-dir "$DEST/repair-dev" \
    --split dev \
    --max-length 256 \
    --batch-size 8 \
    --device cuda \
    > "$DEST/repair-dev/stdout.json"

  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_teacher_dev.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --teacher-registry "$TEACHER_REGISTRY" \
    --teacher-audit "$TEACHER_AUDIT" \
    --output-dir "$DEST/teacher-dev" \
    --max-length 256 \
    --batch-size 8 \
    --order-variants 3 \
    --device cuda \
    > "$DEST/teacher-dev/stdout.json"

  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --benchmark "$NOVEL_COMPILED" \
    --benchmark-receipt "$NOVEL_MANIFEST" \
    --output-dir "$DEST/novel" \
    --max-length 512 \
    --batch-size 8 \
    --device cuda \
    > "$DEST/novel/stdout.json"
}

run_eval "parent-step250" "$PARENT_DIR"
run_eval "repair-step040" "$REPAIR_ROOT/step-00000040"
run_eval "repair-step080" "$REPAIR_ROOT/step-00000080"
run_eval "repair-step120" "$REPAIR_ROOT/step-00000120"

python "$ROOT/scripts/eipm/n0/select_n0_v02_post_repair.py" \
  --evaluation-root "$OUT_ROOT" \
  --previous-teacher-comparison "$TEACHER_DEV_ROOT/teacher_dev_challenge_comparison.json" \
  --previous-novel-comparison "$NOVEL_ROOT/novel_cross_challenge_comparison.json" \
  --output "$SELECTION" \
  > "$OUT_ROOT/selection.stdout.json"

SELECTED="$(python - "$SELECTION" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_checkpoint"])
PY
)"

case "$SELECTED" in
  parent-step250) SELECTED_DIR="$PARENT_DIR" ;;
  repair-step040) SELECTED_DIR="$REPAIR_ROOT/step-00000040" ;;
  repair-step080) SELECTED_DIR="$REPAIR_ROOT/step-00000080" ;;
  repair-step120) SELECTED_DIR="$REPAIR_ROOT/step-00000120" ;;
  *) echo "Unknown selected checkpoint: $SELECTED" >&2; exit 5 ;;
esac

python "$ROOT/scripts/eipm/n0/benchmark_n0_v02_warm_latency.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$SELECTED_DIR/mlm" \
  --ranker "$SELECTED_DIR/ranker.safetensors" \
  --sample-jsonl "$NOVEL_COMPILED" \
  --output "$LATENCY" \
  > "$OUT_ROOT/latency.stdout.json"

python - "$SELECTION" "$LATENCY" <<'PY'
import json, sys
from pathlib import Path
selection_path = Path(sys.argv[1])
latency_path = Path(sys.argv[2])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selection["latency_sanity"] = json.loads(latency_path.read_text(encoding="utf-8"))
selection["efficiency_rule"] = "latency sanity is architecture evidence only; do not create a separate optimization loop unless a measured bottleneck blocks A.L.I.C.E."
selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(selection, indent=2, sort_keys=True))
PY

echo "===== A.L.I.C.E. N0 V0.2 POST-REPAIR SELECTION COMPLETE ====="
date -Is
