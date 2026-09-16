#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
CHALLENGE_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.1"
CHALLENGE="$CHALLENGE_ROOT/challenge.jsonl"
MANIFEST="$CHALLENGE_ROOT/manifest.json"
FREEZE_RECEIPT="$CHALLENGE_ROOT/freeze_receipt.json"
RESULT="$CHALLENGE_ROOT/result.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0.1.json"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
CANDIDATE="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
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

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" "$SPEC" "$EVALUATOR" "$LATENT_CONFIG" "$CANDIDATE" "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH" "$FUSION_CHECKPOINT/cross_context_fusion.safetensors" "$FUSION_CONFIG" "$FUSION_RATIFICATION"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing latent frozen challenge artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite latent frozen challenge result: $RESULT" >&2
  exit 3
fi

python - "$FREEZE_RECEIPT" "$CHALLENGE" "$MANIFEST" "$SPEC" "$EVALUATOR" "$CANDIDATE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
receipt_path,challenge_path,manifest_path,spec_path,evaluator_path,candidate_path=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
receipt=json.loads(receipt_path.read_text())
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if receipt.get('status')!='FROZEN_CONFIRMATORY_BEFORE_EVALUATION': raise SystemExit('latent freeze receipt status drift')
if receipt.get('git_revision')!=head: raise SystemExit(f"latent freeze receipt stale: receipt={receipt.get('git_revision')} current={head}")
for path,key in [(challenge_path,'challenge_sha256'),(manifest_path,'manifest_sha256'),(spec_path,'spec_sha256'),(evaluator_path,'evaluator_sha256'),(candidate_path,'candidate_latent_pool_sha256')]:
    if receipt.get(key)!=sha(path): raise SystemExit(f'latent frozen artifact drift: {key}')
if receipt.get('candidate_preselected_before_challenge') is not True or receipt.get('checkpoint_selection_on_challenge_forbidden') is not True: raise SystemExit('latent confirmatory selection contract drift')
if receipt.get('challenge_rows_used_for_training') is not False or receipt.get('gradient_performed') is not False: raise SystemExit('latent confirmatory governance drift')
print('adaptive_latent_pool_frozen_same_revision_gate=true')
print(f'git_revision={head}')
print(f"candidate_latent_pool_sha256={receipt['candidate_latent_pool_sha256']}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
PY

python "$EVALUATOR" \
  --challenge "$CHALLENGE" \
  --manifest "$MANIFEST" \
  --spec "$SPEC" \
  --freeze-receipt "$FREEZE_RECEIPT" \
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
  --output "$RESULT"

echo "adaptive_latent_pool_frozen_challenge_complete=true"
echo "result=$RESULT"
