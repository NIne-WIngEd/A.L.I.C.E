#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
STRUCTURED_REGRESSION_CACHE="$WORKDIR/structured-state-pilot-v0.1/semantic_cache.pt"
CURRICULUM_ROOT="${N0_V02_GRAPH_CURRICULUM_ROOT:-$WORKDIR/evidence-graph-curriculum-v0.1}"
CURRICULUM="$CURRICULUM_ROOT/evidence_graph_curriculum.jsonl"
CURRICULUM_MANIFEST="$CURRICULUM_ROOT/evidence_graph_curriculum_manifest.json"
OUT_ROOT="${N0_V02_GRAPH_PILOT_ROOT:-$WORKDIR/evidence-graph-pilot-v0.1}"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
GRAPH_CONFIG="$ROOT/configs/eipm/n0/n0_v02_evidence_graph_v0.1.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$SEMANTIC_CHECKPOINT/receipt.json" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$STRUCTURED_CHECKPOINT/receipt.json" \
  "$STRUCTURED_REGRESSION_CACHE" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$GRAPH_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required evidence-graph pilot artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing evidence-graph pilot: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile \
  "$ROOT/src/alice_personality/n0/evidence_graph.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_objectives.py" \
  "$ROOT/src/alice_personality/n0/structured_state.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_evidence_graph_pilot.py"

python - <<'PY'
import json
from pathlib import Path
p = Path("configs/eipm/n0/n0_v02_evidence_graph_v0.1.json")
cfg = json.loads(p.read_text())
training = cfg.get("training_boundary", {})
if training.get("graph_gradient_authorized") is not True:
    raise SystemExit("graph gradient pilot is not authorized in config")
if cfg.get("capacity_policy", {}).get("hard_parameter_ceiling") is not None:
    raise SystemExit("graph config unexpectedly imposes a hard parameter ceiling")
print("graph_pilot_config_gate=true")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Evidence-graph pilot requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 EVIDENCE-GRAPH CAPABILITY PILOT ====="
date -Is
echo "semantic_parent=targeted-repair-v0.1/step-00000080"
echo "structured_parent=structured-state-pilot-v0.1/step-00000080"
echo "variants=compact_graph_only,compact_joint,expanded_joint_512x2"
echo "hard_parameter_ceiling=none"
echo "selection=capability_first_efficiency_secondary"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_evidence_graph_pilot.py" \
  --config "$CONFIG" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --structured-regression-cache "$STRUCTURED_REGRESSION_CACHE" \
  --output-dir "$OUT_ROOT" \
  --max-length 128 \
  --encode-batch-size 64 \
  --train-batch-size 32 \
  --eval-batch-size 64 \
  --max-steps 240 \
  --save-every 80 \
  --warmup-steps 20 \
  --learning-rate 0.0003 \
  --weight-decay 0.05 \
  --joint-anchor-weight 0.05 \
  --seed 20260915

echo "===== N0 V0.2 EVIDENCE-GRAPH CAPABILITY PILOT COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/evidence_graph_pilot_comparison.json"
