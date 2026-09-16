#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST_PARENT="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
PARENT_ADAPTER="$SPECIALIST_PARENT/evidence_view_adapter.safetensors"
PARENT_GRAPH="$SPECIALIST_PARENT/evidence_graph.safetensors"
REPLAY_CACHE="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
REPAIR_ROOT="$WORKDIR/relation-repair-curriculum-v0.1"
REPAIR_CURRICULUM="$REPAIR_ROOT/relation_repair_curriculum.jsonl"
REPAIR_MANIFEST="$REPAIR_ROOT/relation_repair_manifest.json"
OUT_ROOT="$WORKDIR/relation-repair-v0.1"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
GRAPH_REPAIR_CONFIG="$ROOT/configs/eipm/n0/n0_v02_evidence_graph_v0.3.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$STRUCTURED_CHECKPOINT/receipt.json" \
  "$PARENT_ADAPTER" \
  "$PARENT_GRAPH" \
  "$REPLAY_CACHE" \
  "$SEMANTIC_CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$GRAPH_REPAIR_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing relation-repair artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing relation repair: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile \
  "$ROOT/src/alice_personality/n0/evidence_graph_dual_endpoint.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_relation_repair_curriculum.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_relation_repair.py"

pytest -q \
  "$ROOT/tests/eipm/test_n0_evidence_graph.py" \
  "$ROOT/tests/eipm/test_n0_evidence_graph_dual_endpoint.py" \
  "$ROOT/tests/eipm/test_n0_evidence_view_adapter.py"

mkdir -p "$REPAIR_ROOT"
python "$ROOT/scripts/eipm/n0/build_n0_v02_relation_repair_curriculum.py" \
  --output "$REPAIR_CURRICULUM" \
  --manifest "$REPAIR_MANIFEST"

python - "$REPAIR_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path("configs/eipm/n0/n0_v02_evidence_graph_v0.3.json").read_text())
if cfg.get("repair_training", {}).get("authorized") is not True:
    raise SystemExit("relation repair is not authorized")
if cfg.get("capacity_policy", {}).get("hard_parameter_ceiling") is not None:
    raise SystemExit("unexpected hard parameter ceiling")
manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest.get("frozen_relation_essential_challenge_rows_used_for_training") is not False:
    raise SystemExit("frozen challenge leak detected")
if manifest.get("training_authorized") is not True:
    raise SystemExit("repair curriculum is not training-authorized")
print("relation_repair_governance_gate=true")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Relation repair requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 DUAL-ENDPOINT RELATION REPAIR ====="
date -Is
echo "parent_specialist=specialized_expanded_640x3_graph512x2/step-00000080"
echo "adapter_trainable=false"
echo "graph_trainable_scope=relation_read_only"
echo "frozen_relation_essential_challenge_used_for_training=false"
echo "hard_parameter_ceiling=none"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_relation_repair.py" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --repair-curriculum "$REPAIR_CURRICULUM" \
  --repair-manifest "$REPAIR_MANIFEST" \
  --replay-cache "$REPLAY_CACHE" \
  --output-dir "$OUT_ROOT" \
  --max-length 128 \
  --encode-batch-size 64 \
  --train-pair-batch-size 16 \
  --replay-batch-size 32 \
  --eval-batch-size 64 \
  --max-steps 160 \
  --save-every 40 \
  --warmup-steps 10 \
  --learning-rate 0.0001 \
  --weight-decay 0.05 \
  --seed 20260916

echo "===== N0 V0.2 RELATION REPAIR COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/relation_repair_comparison.json"
