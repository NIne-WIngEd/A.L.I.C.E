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
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.2.json"
CHALLENGE_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.2"
CHALLENGE="$CHALLENGE_ROOT/challenge.jsonl"
MANIFEST="$CHALLENGE_ROOT/manifest.json"
FREEZE_RECEIPT="$CHALLENGE_ROOT/freeze_receipt.json"
RESULT="$CHALLENGE_ROOT/result.json"
REPAIR_ROOT="$WORKDIR/cross-context-fusion-repair-training-v0.2"
REPAIR_COMPARISON="$REPAIR_ROOT/cross_context_fusion_repair_comparison.json"
CANDIDATE="$REPAIR_ROOT/repair-step-00000240"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py"

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
  "$SEMANTIC_CONFIG" \
  "$STRUCTURED_CONFIG" \
  "$FUSION_CONFIG" \
  "$SPEC" \
  "$CHALLENGE" \
  "$MANIFEST" \
  "$FREEZE_RECEIPT" \
  "$REPAIR_COMPARISON" \
  "$CANDIDATE/cross_context_fusion.safetensors" \
  "$CANDIDATE/receipt.json" \
  "$EVALUATOR"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing confirmatory evaluation artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite existing confirmatory result: $RESULT" >&2
  exit 3
fi

python - "$FREEZE_RECEIPT" "$CHALLENGE" "$MANIFEST" "$SPEC" "$REPAIR_COMPARISON" "$CANDIDATE/cross_context_fusion.safetensors" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path, challenge_path, manifest_path, spec_path, comparison_path, candidate_path = map(Path, sys.argv[1:7])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if receipt.get("status") != "FROZEN_CONFIRMATORY_BEFORE_EVALUATION":
    raise SystemExit("confirmatory freeze receipt status drift")
if receipt.get("git_revision") != head:
    raise SystemExit(f"confirmatory freeze receipt is stale: receipt={receipt.get('git_revision')} head={head}")
if receipt.get("challenge_sha256") != sha(challenge_path):
    raise SystemExit("confirmatory challenge changed after freeze")
if receipt.get("manifest_sha256") != sha(manifest_path):
    raise SystemExit("confirmatory manifest changed after freeze")
if receipt.get("spec_sha256") != sha(spec_path):
    raise SystemExit("confirmatory spec changed after freeze")
if receipt.get("repair_comparison_sha256") != sha(comparison_path):
    raise SystemExit("repair comparison changed after freeze")
if receipt.get("candidate_fusion_sha256") != sha(candidate_path):
    raise SystemExit("candidate checkpoint changed after freeze")
if receipt.get("candidate_preselected_before_challenge") is not True:
    raise SystemExit("candidate preselection gate missing")
if receipt.get("checkpoint_selection_on_challenge_forbidden") is not True:
    raise SystemExit("checkpoint-selection prohibition missing")
if receipt.get("thresholds_frozen_before_results") is not True:
    raise SystemExit("threshold freeze gate missing")
if receipt.get("results_observed") is not False:
    raise SystemExit("confirmatory results were already observed")
if receipt.get("challenge_rows_used_for_training") is not False:
    raise SystemExit("confirmatory challenge contamination detected")
print("fusion_confirmatory_v02_same_revision_gate=true")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print(f"candidate_fusion_sha256={receipt['candidate_fusion_sha256']}")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Confirmatory fusion evaluation requires one visible CUDA device." >&2
  exit 4
fi

 echo "===== N0 V0.2 CONFIRMATORY FUSION CHALLENGE V0.2 ====="
date -Is
echo "candidate_preselected_before_challenge=true"
echo "candidate_repair_step=240"
echo "checkpoint_selection_on_challenge=false"
echo "challenge_rows_used_for_training=false"
echo "thresholds_frozen_before_results=true"
echo "gradient_performed=false"
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
  --challenge-spec "$SPEC" \
  --challenge "$CHALLENGE" \
  --challenge-manifest "$MANIFEST" \
  --repair-comparison "$REPAIR_COMPARISON" \
  --candidate-checkpoint "$CANDIDATE" \
  --output "$RESULT" \
  --raw-max-length 96 \
  --field-max-length 96 \
  --encode-batch-size 32 \
  --eval-batch-size 8

echo "===== N0 V0.2 CONFIRMATORY FUSION CHALLENGE V0.2 COMPLETE ====="
date -Is
echo "result=$RESULT"
