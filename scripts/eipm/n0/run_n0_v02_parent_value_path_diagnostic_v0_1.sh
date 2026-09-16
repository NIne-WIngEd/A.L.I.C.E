#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
CHALLENGE="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2/challenge.jsonl"
VALUE_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-v0.2-counterfactual-diagnostic-v0.2"
CONTRACT="$VALUE_ROOT/contrast_contract.jsonl"
MANIFEST="$VALUE_ROOT/contrast_manifest.json"
VALUE_RESULT="$VALUE_ROOT/value_contrast_diagnostic.json"
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_parent_value_path_v0_1.py"
OUT_ROOT="$WORKDIR/parent-value-path-diagnostic-v0.1"
PREP="$OUT_ROOT/preparation_receipt.json"
OUTPUT="$OUT_ROOT/result.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
SEMANTIC="$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
STRUCTURED="$STRUCTURED_CHECKPOINT/structured_state.safetensors"
EVIDENCE_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
EVIDENCE_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in "$CHALLENGE" "$CONTRACT" "$MANIFEST" "$VALUE_RESULT" "$DIAGNOSTIC" "$PREP" "$SEMANTIC_CONFIG" "$SEMANTIC" "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" "$STRUCTURED" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing parent value-path diagnostic artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite parent value-path result: $OUTPUT" >&2
  exit 3
fi

python - "$ROOT" "$PREP" "$CHALLENGE" "$CONTRACT" "$MANIFEST" "$VALUE_RESULT" "$DIAGNOSTIC" "$SEMANTIC" "$STRUCTURED" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root,prep_path,challenge,contract,manifest,value_result,diagnostic,semantic,structured,adapter,graph=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
prep=json.loads(prep_path.read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=root).strip()
if prep.get('status')!='PASS_PARENT_VALUE_PATH_DIAGNOSTIC_READY_FOR_GPU': raise SystemExit('parent value-path preparation missing')
if prep.get('git_revision')!=head: raise SystemExit(f"parent value-path prep stale: prep={prep.get('git_revision')} current={head}")
for path,key in [
 (challenge,'challenge_sha256'),(contract,'contrast_contract_sha256'),(manifest,'contrast_manifest_sha256'),
 (value_result,'value_contrast_result_sha256'),(diagnostic,'diagnostic_sha256'),
 (semantic,'semantic_checkpoint_sha256'),(structured,'structured_checkpoint_sha256'),
 (adapter,'evidence_adapter_sha256'),(graph,'evidence_graph_sha256')]:
    if prep.get(key)!=sha(path): raise SystemExit(f'parent value-path artifact drift: {key}')
if prep.get('canonical_parent_path_required') is not True or prep.get('canonical_parent_trace_parity_required') is not True: raise SystemExit('parent-path parity contract drift')
if prep.get('graph_field_weight_and_relation_bias_measured') is not True: raise SystemExit('graph selection instrumentation missing')
if prep.get('gradient_performed') is not False or prep.get('ratification_effect') is not False or prep.get('scale_authorized') is not False: raise SystemExit('parent diagnostic governance drift')
print('parent_value_path_v01_same_revision_gate=true')
print(f'git_revision={head}')
print('canonical_parent_trace_parity_required=true')
print('graph_field_weight_and_relation_bias_measured=true')
print('gradient_performed=false')
PY

python "$DIAGNOSTIC" \
  --challenge "$CHALLENGE" \
  --contrast-contract "$CONTRACT" \
  --contrast-manifest "$MANIFEST" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --output "$OUTPUT"

echo "parent_value_path_v01_complete=true"
echo "output=$OUTPUT"
