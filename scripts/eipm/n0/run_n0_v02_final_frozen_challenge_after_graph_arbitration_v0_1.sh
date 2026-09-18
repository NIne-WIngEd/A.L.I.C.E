#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
: "${ALICE_N0_ARB_DIR:?ALICE_N0_ARB_DIR must point to the authorized arbitration output directory}"
ARB_DIR="$ALICE_N0_ARB_DIR"
OUTDIR="${ALICE_N0_FINAL_CHALLENGE_DIR:-$WORKDIR/adaptive-multi-view-latent-pool-final-frozen-challenge-v0.1}"

CHALLENGE_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2"
SPEC="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0.2.json"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2.py"
BINDER="$ROOT/scripts/eipm/n0/run_final_frozen_challenge_after_graph_arbitration_v0_1.py"

LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
EVIDENCE_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
REPAIRED_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
FUSION_CHECKPOINT="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUTDIR" ]]; then
  echo "REFUSING: final frozen challenge output directory already exists: $OUTDIR" >&2
  exit 2
fi

for required in   "$ARB_DIR/arbitration_result.json"   "$ARB_DIR/post_arbitration_gate_receipt.json"   "$ARB_DIR/finalization_receipt.json"   "$ARB_DIR/full_stack_manifest.json"   "$CHALLENGE_ROOT/challenge.jsonl"   "$CHALLENGE_ROOT/manifest.json"   "$CHALLENGE_ROOT/freeze_receipt.json"   "$SPEC" "$EVALUATOR" "$BINDER" "$LATENT" "$LATENT_CONFIG"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" "$TOKENIZER_DIR"   "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT" "$EVIDENCE_ADAPTER"   "$REPAIRED_GRAPH" "$FUSION_CHECKPOINT" "$FUSION_CONFIG" "$FUSION_RATIFICATION"
do
  if [[ ! -e "$required" ]]; then
    echo "MISSING: $required" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository worktree is not clean." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$BINDER" "$EVALUATOR"

echo "===== N0 FINAL FROZEN CHALLENGE AFTER GRAPH ARBITRATION ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "arbitration_dir=$ARB_DIR"
echo "challenge_root=$CHALLENGE_ROOT"
echo "selected_repaired_graph=$REPAIRED_GRAPH"
echo "output_dir=$OUTDIR"
echo "training_enabled=false"
echo "promotion_authorized=false"
echo "scale_authorized=false"
echo "automatic_rerun=false"

python "$BINDER"   --repo-root "$ROOT"   --arbitration-dir "$ARB_DIR"   --challenge-root "$CHALLENGE_ROOT"   --spec "$SPEC"   --evaluator "$EVALUATOR"   --latent-checkpoint "$LATENT"   --latent-config "$LATENT_CONFIG"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED_CHECKPOINT"   --evidence-adapter "$EVIDENCE_ADAPTER"   --repaired-graph "$REPAIRED_GRAPH"   --fusion-checkpoint "$FUSION_CHECKPOINT"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --output-dir "$OUTDIR"

echo
echo "===== FINAL FROZEN CHALLENGE OUTPUT HASHES ====="
sha256sum   "$OUTDIR/derived_freeze_receipt.json"   "$OUTDIR/launch_receipt.json"   "$OUTDIR/result.json"   "$OUTDIR/execution_receipt.json"

echo
echo "===== FINAL FROZEN CHALLENGE DECISION ====="
python - "$OUTDIR/execution_receipt.json" <<'PY'
import json,sys
from pathlib import Path
receipt=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={receipt.get('status')}")
print(f"model_conclusion={receipt.get('model_conclusion')}")
print(f"result_status={receipt.get('result_status')}")
print(f"gate_pass={str(receipt.get('gate_pass')).lower()}")
print(f"candidate_graph_promoted={str(receipt.get('invariants',{}).get('candidate_graph_promoted')).lower()}")
print(f"training_enabled={str(receipt.get('invariants',{}).get('training_enabled')).lower()}")
print(f"scale_authorized={str(receipt.get('invariants',{}).get('scale_authorized')).lower()}")
print(f"n0_complete={str(receipt.get('invariants',{}).get('n0_complete')).lower()}")
PY

echo
echo "===== N0 FINAL FROZEN CHALLENGE COMPLETE ====="
date -Is
