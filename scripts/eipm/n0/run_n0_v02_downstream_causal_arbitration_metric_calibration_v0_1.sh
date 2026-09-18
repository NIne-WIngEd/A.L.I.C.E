#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUTDIR="${ALICE_N0_ARB_CAL_DIR:-$WORKDIR/downstream-causal-arbitration-metric-calibration-v0.1}"

cd "$ROOT"

# The full-stack evaluator is launched as a subprocess from the calibrator.
# Bind both import roots inside the container so alice_personality and sibling
# N0 evaluation/training helpers resolve exactly as they do in validated N0 jobs.
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

CANON="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
EVAL="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2/challenge.jsonl"
ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
FUSION="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"

STACK_MANIFEST="$OUTDIR/full_stack_manifest.json"
REPEATS="$OUTDIR/canonical-repeats"
POLICY="$OUTDIR/n0_v02_downstream_causal_arbitration_metric_policy_v0.1.json"
RECEIPT="$OUTDIR/calibration_receipt.json"

if [[ -e "$OUTDIR" ]]; then
  echo "REFUSING: calibration output directory already exists: $OUTDIR" >&2
  exit 2
fi

for path in   "$CANON"   "$LATENT"   "$EVAL"   "$ADAPTER"   "$SEMANTIC_CONFIG"   "$SEMANTIC_CHECKPOINT"   "$TOKENIZER_DIR"   "$STRUCTURED_CONFIG"   "$STRUCTURED_CHECKPOINT"   "$LATENT_CONFIG"   "$FUSION"   "$FUSION_CONFIG"   "$FUSION_RATIFICATION"
do
  if [[ ! -e "$path" ]]; then
    echo "MISSING: $path" >&2
    exit 3
  fi
done

echo "===== N0 FULL-STACK CANONICAL REPEATABILITY CALIBRATION ====="
echo "repo_head=$(git rev-parse HEAD)"
echo "workdir=$WORKDIR"
echo "outdir=$OUTDIR"
echo "canonical_graph=$CANON"
echo "candidate_graph_supplied=false"
echo "repeat_count=2"

mkdir -p "$OUTDIR"

python scripts/eipm/n0/calibrate_downstream_causal_arbitration_metric_policy_v0_1.py   --repeat-count 2   --canonical-graph "$CANON"   --evaluation-set "$EVAL"   --latent-checkpoint "$LATENT"   --latent-config "$LATENT_CONFIG"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED_CHECKPOINT"   --evidence-adapter "$ADAPTER"   --fusion-checkpoint "$FUSION"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --stack-manifest-output "$STACK_MANIFEST"   --repeat-output-dir "$REPEATS"   --policy-output "$POLICY"   --receipt-output "$RECEIPT"

echo
echo "===== CALIBRATION OUTPUT HASHES ====="
sha256sum "$STACK_MANIFEST" "$POLICY" "$RECEIPT"
find "$REPEATS" -maxdepth 1 -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum

echo
echo "===== MATERIALIZED POLICY ====="
cat "$POLICY"

echo
echo "===== CALIBRATION COMPLETE ====="
