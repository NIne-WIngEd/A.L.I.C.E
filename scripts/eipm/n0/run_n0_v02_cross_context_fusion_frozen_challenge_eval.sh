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
TRAINING_ROOT="$WORKDIR/cross-context-fusion-training-v0.1"
TRAINING_COMPARISON="$TRAINING_ROOT/cross_context_fusion_training_comparison.json"
CHALLENGE_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1"
CHALLENGE="$CHALLENGE_ROOT/challenge.jsonl"
CHALLENGE_MANIFEST="$CHALLENGE_ROOT/manifest.json"
FREEZE_RECEIPT="$CHALLENGE_ROOT/freeze_receipt.json"
RESULT="$CHALLENGE_ROOT/result.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
CHALLENGE_SPEC="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.1.json"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge.py"

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
  "$TRAINING_COMPARISON" \
  "$CHALLENGE" \
  "$CHALLENGE_MANIFEST" \
  "$FREEZE_RECEIPT" \
  "$SEMANTIC_CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$FUSION_CONFIG" \
  "$CHALLENGE_SPEC" \
  "$EVALUATOR"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing frozen fusion challenge artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite existing frozen challenge result: $RESULT" >&2
  exit 3
fi

python - "$FREEZE_RECEIPT" "$CHALLENGE" "$CHALLENGE_MANIFEST" "$CHALLENGE_SPEC" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path = Path(sys.argv[1])
challenge_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
spec_path = Path(sys.argv[4])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt = json.loads(receipt_path.read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if receipt.get("status") != "FROZEN_BEFORE_EVALUATION":
    raise SystemExit("challenge freeze receipt status mismatch")
if receipt.get("git_revision") != head:
    raise SystemExit(f"challenge freeze receipt is stale: receipt={receipt.get('git_revision')} head={head}")
if receipt.get("challenge_sha256") != sha(challenge_path):
    raise SystemExit("challenge changed after freeze")
if receipt.get("manifest_sha256") != sha(manifest_path):
    raise SystemExit("challenge manifest changed after freeze")
if receipt.get("spec_sha256") != sha(spec_path):
    raise SystemExit("challenge thresholds/spec changed after freeze")
if receipt.get("thresholds_frozen_before_results") is not True:
    raise SystemExit("challenge thresholds were not frozen")
if receipt.get("results_observed") is not False:
    raise SystemExit("challenge result was observed before this run")
if receipt.get("training_authorized") is not False:
    raise SystemExit("frozen challenge unexpectedly authorizes training")
print("fusion_frozen_challenge_same_revision_gate=true")
print(f"challenge_sha256={receipt['challenge_sha256']}")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Frozen fusion challenge requires one visible CUDA device." >&2
  exit 4
fi

echo "===== N0 V0.2 FROZEN CROSS-CONTEXT FUSION CHALLENGE ====="
date -Is
echo "challenge_rows_used_for_training=false"
echo "thresholds_frozen_before_results=true"
echo "gradient_performed=false"
echo "candidate_steps=80,160,240"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$EVALUATOR" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --fusion-config "$FUSION_CONFIG" \
  --challenge-spec "$CHALLENGE_SPEC" \
  --challenge "$CHALLENGE" \
  --challenge-manifest "$CHALLENGE_MANIFEST" \
  --training-comparison "$TRAINING_COMPARISON" \
  --training-root "$TRAINING_ROOT" \
  --output "$RESULT" \
  --raw-max-length 96 \
  --field-max-length 96 \
  --encode-batch-size 32 \
  --eval-batch-size 8

echo "===== N0 V0.2 FROZEN FUSION CHALLENGE COMPLETE ====="
date -Is
echo "result=$RESULT"
