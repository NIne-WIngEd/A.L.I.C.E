#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
CHALLENGE="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2/challenge.jsonl"
V02_RESULT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2/result.json"
CANDIDATE="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
EVIDENCE_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
EVIDENCE_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
FUSION_CHECKPOINT="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_latent_pool_counterfactual_value_contrast_v0_1.py"
OUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-v0.2-counterfactual-diagnostic-v0.1"
OUTPUT="$OUT_ROOT/value_contrast_diagnostic.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in "$CHALLENGE" "$V02_RESULT" "$CANDIDATE" "$LATENT_CONFIG" "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH" "$FUSION_CHECKPOINT/cross_context_fusion.safetensors" "$FUSION_CONFIG" "$FUSION_RATIFICATION" "$DIAGNOSTIC"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing value-contrast diagnostic artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite value-contrast diagnostic: $OUTPUT" >&2
  exit 3
fi

python -m py_compile "$DIAGNOSTIC"
mkdir -p "$OUT_ROOT"

python "$DIAGNOSTIC" \
  --challenge "$CHALLENGE" \
  --v02-result "$V02_RESULT" \
  --candidate "$CANDIDATE" \
  --latent-config "$LATENT_CONFIG" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --fusion-checkpoint "$FUSION_CHECKPOINT" \
  --fusion-config "$FUSION_CONFIG" \
  --fusion-ratification "$FUSION_RATIFICATION" \
  --output "$OUTPUT"

echo "latent_pool_counterfactual_value_contrast_diagnostic_complete=true"
echo "output=$OUTPUT"
