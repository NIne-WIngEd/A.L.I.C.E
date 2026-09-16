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
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_latent_pool_counterfactual_value_contrast_v0_2.py"
OUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-v0.2-counterfactual-diagnostic-v0.2"
CONTRACT="$OUT_ROOT/contrast_contract.jsonl"
MANIFEST="$OUT_ROOT/contrast_manifest.json"
PREP="$OUT_ROOT/preparation_receipt.json"
OUTPUT="$OUT_ROOT/value_contrast_diagnostic.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in "$CHALLENGE" "$V02_RESULT" "$CANDIDATE" "$LATENT_CONFIG" "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH" "$FUSION_CHECKPOINT/cross_context_fusion.safetensors" "$FUSION_CONFIG" "$FUSION_RATIFICATION" "$DIAGNOSTIC" "$CONTRACT" "$MANIFEST" "$PREP"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing explicit-contract value diagnostic artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite value-contrast diagnostic v0.2: $OUTPUT" >&2
  exit 3
fi

python - "$PREP" "$CHALLENGE" "$V02_RESULT" "$CANDIDATE" "$DIAGNOSTIC" "$CONTRACT" "$MANIFEST" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
prep_path,challenge,result,candidate,diagnostic,contract,manifest=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
prep=json.loads(prep_path.read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if prep.get('status')!='PASS_SEMANTIC_CONTRACT_COMPILED_BEFORE_GPU': raise SystemExit('value diagnostic v0.2 preparation missing')
if prep.get('git_revision')!=head: raise SystemExit(f"value diagnostic prep stale: prep={prep.get('git_revision')} current={head}")
for path,key in [(challenge,'challenge_sha256'),(result,'historical_v0_2_result_sha256'),(candidate,'candidate_latent_pool_sha256'),(diagnostic,'diagnostic_sha256'),(contract,'contrast_contract_sha256'),(manifest,'contrast_manifest_sha256')]:
    if prep.get(key)!=sha(path): raise SystemExit(f'value diagnostic artifact drift: {key}')
if prep.get('query_conditioned_answer_semantics') is not True or prep.get('global_current_value_assumption') is not False: raise SystemExit('value diagnostic semantic contract drift')
if prep.get('target_and_foil_materialized_before_gpu') is not True: raise SystemExit('value diagnostic target/foil contract not materialized')
if prep.get('historical_v0_2_result_reclassified') is not False or prep.get('ratification_effect') is not False: raise SystemExit('value diagnostic governance drift')
if prep.get('gradient_performed') is not False: raise SystemExit('value diagnostic may not perform gradient')
print('latent_pool_value_contrast_v02_same_revision_gate=true')
print(f'git_revision={head}')
print('query_conditioned_answer_semantics=true')
print('global_current_value_assumption=false')
print('contrast_rows=32')
PY

python "$DIAGNOSTIC" \
  --challenge "$CHALLENGE" \
  --contrast-contract "$CONTRACT" \
  --contrast-manifest "$MANIFEST" \
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

echo "latent_pool_counterfactual_value_contrast_diagnostic_v02_complete=true"
echo "output=$OUTPUT"
