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
FUSION_PARENT="$WORKDIR/cross-context-fusion-training-v0.1/step-00000240"
REPAIR_ROOT="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3"
REPAIR_CURRICULUM="$REPAIR_ROOT/cross_context_fusion_repair_curriculum.jsonl"
REPAIR_MANIFEST="$REPAIR_ROOT/cross_context_fusion_repair_manifest.json"
PREP_RECEIPT="$REPAIR_ROOT/preparation_receipt.json"
FAILED_CHALLENGE="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1/result.json"
OUT_ROOT="$WORKDIR/cross-context-fusion-repair-training-v0.2"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_repair_training_v0.2.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
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
  "$FUSION_PARENT/cross_context_fusion.safetensors" \
  "$REPAIR_CURRICULUM" \
  "$REPAIR_MANIFEST" \
  "$PREP_RECEIPT" \
  "$FAILED_CHALLENGE" \
  "$SEMANTIC_CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$FUSION_CONFIG" \
  "$TRAINING_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing full-scale fusion repair artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing full-scale fusion repair output: $OUT_ROOT" >&2
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
    raise SystemExit("fusion repair preparation did not pass")
if receipt.get("git_revision") != head:
    raise SystemExit(f"fusion repair prep receipt is stale: receipt={receipt.get('git_revision')} head={head}")
if receipt.get("full_scale_model") is not True:
    raise SystemExit("repair prep is not for the full-scale fusion architecture")
if receipt.get("source_anchor_summary_channel") is not True:
    raise SystemExit("source summary anchor channel missing from prep")
if receipt.get("source_anchor_token_channel") is not True:
    raise SystemExit("source token anchor channel missing from prep")
if receipt.get("frozen_challenge_rows_used_for_training") is not False:
    raise SystemExit("frozen challenge contamination detected")
print("fusion_repair_same_revision_preparation_gate=true")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Full-scale fusion repair requires one visible CUDA device." >&2
  exit 4
fi

mkdir -p "$OUT_ROOT"

echo "===== N0 V0.2 FULL-SCALE SOURCE-ANCHORED FUSION REPAIR ====="
date -Is
echo "architecture=full_scale_frontier_multi_stream_gated_bidirectional_cross_attention_with_explicit_source_anchors"
echo "parent_fusion_step=240"
echo "reduced_capability_pilot=false"
echo "generic_fallback_architecture=false"
echo "source_anchor_summary_channel=true"
echo "source_anchor_token_channel=true"
echo "frozen_challenge_rows_used_for_training=false"
echo "old_frozen_challenge_retired_for_future_ratification=true"
echo "parents_trainable_in_this_run=false"
echo "parent_freeze_permanent=false"
echo "hard_parameter_ceiling=none"
echo "view_count_ceiling=none"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/train_n0_v02_cross_context_fusion_repair_full_scale.py" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --fusion-config "$FUSION_CONFIG" \
  --fusion-parent-checkpoint "$FUSION_PARENT" \
  --training-config "$TRAINING_CONFIG" \
  --curriculum "$REPAIR_CURRICULUM" \
  --curriculum-manifest "$REPAIR_MANIFEST" \
  --prep-receipt "$PREP_RECEIPT" \
  --failed-challenge-result "$FAILED_CHALLENGE" \
  --output-dir "$OUT_ROOT" \
  --raw-max-length 96 \
  --field-max-length 96 \
  --encode-batch-size 32

echo "===== N0 V0.2 FULL-SCALE SOURCE-ANCHORED FUSION REPAIR COMPLETE ====="
date -Is
echo "comparison=$OUT_ROOT/cross_context_fusion_repair_comparison.json"
