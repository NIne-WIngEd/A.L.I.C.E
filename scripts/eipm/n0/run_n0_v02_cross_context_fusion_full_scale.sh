#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST_PARENT="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
EVIDENCE_ADAPTER="$SPECIALIST_PARENT/evidence_view_adapter.safetensors"
EVIDENCE_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
CURRICULUM_ROOT="$WORKDIR/cross-context-fusion-curriculum-v0.2"
CURRICULUM="$CURRICULUM_ROOT/cross_context_fusion_curriculum.jsonl"
CURRICULUM_MANIFEST="$CURRICULUM_ROOT/cross_context_fusion_manifest.json"
PREP_RECEIPT="$CURRICULUM_ROOT/preparation_receipt.json"
OUT_ROOT="$WORKDIR/cross-context-fusion-training-v0.1"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_training_v0.1.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$STRUCTURED_CHECKPOINT/receipt.json" \
  "$EVIDENCE_ADAPTER" \
  "$EVIDENCE_GRAPH" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$PREP_RECEIPT" \
  "$SEMANTIC_CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$FUSION_CONFIG" \
  "$TRAINING_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing full-scale fusion artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing full-scale fusion training output: $OUT_ROOT" >&2
  exit 3
fi

python - "$PREP_RECEIPT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
receipt = json.loads(Path(sys.argv[1]).read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if receipt.get("status") != "PASS":
    raise SystemExit("fusion preparation did not pass")
if receipt.get("git_revision") != head:
    raise SystemExit(f"fusion prep receipt is stale: receipt={receipt.get('git_revision')} head={head}")
if receipt.get("full_scale_model") is not True:
    raise SystemExit("prep receipt is not for the full-scale fusion architecture")
if receipt.get("reduced_capability_pilot") is not False:
    raise SystemExit("reduced-capability fusion training is forbidden")
if receipt.get("no_accidental_capability_ceilings_gate") is not True:
    raise SystemExit("capability-limit audit gate missing")
print("fusion_same_revision_preparation_gate=true")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Full-scale fusion training requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 FULL-SCALE CROSS-CONTEXT FUSION TRAINING ====="
date -Is
echo "architecture=full_scale_frontier_multi_stream_gated_bidirectional_cross_attention"
echo "reduced_capability_pilot=false"
echo "generic_fallback_architecture=false"
echo "current_instantiated_public_views=3"
echo "view_count_ceiling=none"
echo "field_node_edge_runtime_ceilings=none"
echo "semantic_tokens_preserved=true"
echo "parents_trainable_in_this_run=false"
echo "parent_freeze_permanent=false"
echo "hard_parameter_ceiling=none"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --fusion-config "$FUSION_CONFIG" \
  --training-config "$TRAINING_CONFIG" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --prep-receipt "$PREP_RECEIPT" \
  --output-dir "$OUT_ROOT" \
  --raw-max-length 96 \
  --field-max-length 96 \
  --encode-batch-size 32

echo "===== N0 V0.2 FULL-SCALE FUSION TRAINING COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/cross_context_fusion_training_comparison.json"
