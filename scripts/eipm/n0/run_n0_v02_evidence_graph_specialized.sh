#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
CURRICULUM_ROOT="${N0_V02_GRAPH_CURRICULUM_ROOT:-$WORKDIR/evidence-graph-curriculum-v0.1}"
CURRICULUM="$CURRICULUM_ROOT/evidence_graph_curriculum.jsonl"
CURRICULUM_MANIFEST="$CURRICULUM_ROOT/evidence_graph_curriculum_manifest.json"
PRIOR_ROOT="${N0_V02_GRAPH_PILOT_ROOT:-$WORKDIR/evidence-graph-pilot-v0.1}"
GRAPH_CACHE="$PRIOR_ROOT/graph_semantic_cache.pt"
PRIOR_COMPARISON="$PRIOR_ROOT/evidence_graph_pilot_comparison.json"
OUT_ROOT="${N0_V02_GRAPH_SPECIALIST_ROOT:-$WORKDIR/evidence-graph-specialist-v0.1}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$STRUCTURED_CHECKPOINT/receipt.json" \
  "$STRUCTURED_CONFIG" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$GRAPH_CACHE" \
  "$PRIOR_COMPARISON"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required evidence specialist artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing evidence specialist run: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile \
  "$ROOT/src/alice_personality/n0/evidence_graph.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_objectives.py" \
  "$ROOT/src/alice_personality/n0/evidence_view_adapter.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_evidence_graph_specialized.py"

python -m pytest -q \
  "$ROOT/tests/eipm/test_n0_evidence_graph.py" \
  "$ROOT/tests/eipm/test_n0_evidence_graph_objectives.py" \
  "$ROOT/tests/eipm/test_n0_evidence_view_adapter.py"

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Evidence specialist pilot requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 EVIDENCE SPECIALIST CAPABILITY PILOT ====="
date -Is
echo "structured_parent=structured-state-pilot-v0.1/step-00000080 (immutable shared view)"
echo "semantic_cache_reused=$GRAPH_CACHE"
echo "variants=specialized_compact_384x2_graph256x1,specialized_expanded_640x3_graph512x2"
echo "relation_pooling=query_conditioned_no_fixed_supersession_penalty"
echo "hard_parameter_ceiling=none"
echo "selection=capability_first"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_evidence_graph_specialized.py" \
  --graph-cache "$GRAPH_CACHE" \
  --prior-comparison "$PRIOR_COMPARISON" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --structured-config "$STRUCTURED_CONFIG" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --output-dir "$OUT_ROOT" \
  --train-batch-size 32 \
  --eval-batch-size 64 \
  --max-steps 240 \
  --save-every 80 \
  --warmup-steps 20 \
  --learning-rate 0.0003 \
  --weight-decay 0.05 \
  --seed 20260915

echo "===== N0 V0.2 EVIDENCE SPECIALIST CAPABILITY PILOT COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/evidence_specialist_comparison.json"
