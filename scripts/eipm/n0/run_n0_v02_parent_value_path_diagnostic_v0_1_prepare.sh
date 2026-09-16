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
TEST="$ROOT/tests/eipm/test_n0_parent_value_path_diagnostic_v0_1.py"
SEMANTIC="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
STRUCTURED="$WORKDIR/structured-state-pilot-v0.1/step-00000080/structured_state.safetensors"
EVIDENCE_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
EVIDENCE_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
OUT_ROOT="$WORKDIR/parent-value-path-diagnostic-v0.1"
PREP="$OUT_ROOT/preparation_receipt.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in "$CHALLENGE" "$CONTRACT" "$MANIFEST" "$VALUE_RESULT" "$DIAGNOSTIC" "$TEST" "$SEMANTIC" "$STRUCTURED" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing parent value-path preparation artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty parent value-path preparation: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$DIAGNOSTIC"
pytest -q "$TEST"
mkdir -p "$OUT_ROOT"

python - "$ROOT" "$CHALLENGE" "$CONTRACT" "$MANIFEST" "$VALUE_RESULT" "$DIAGNOSTIC" "$SEMANTIC" "$STRUCTURED" "$EVIDENCE_ADAPTER" "$EVIDENCE_GRAPH" "$PREP" <<'PY'
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
root,challenge,contract,manifest_path,value_result,diagnostic,semantic,structured,adapter,graph,prep_path=map(Path,sys.argv[1:])

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
value=json.loads(value_result.read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=root).strip()
if manifest.get('status')!='COMPILED_DIAGNOSTIC_CONTRACT_NOT_EVALUATED': raise SystemExit('contrast manifest status drift')
if manifest.get('challenge_sha256')!=sha(challenge): raise SystemExit('challenge/contrast binding drift')
if manifest.get('contrast_contract_sha256')!=sha(contract): raise SystemExit('contrast contract hash drift')
if manifest.get('rows')!=32: raise SystemExit('parent diagnostic requires exactly 32 contrast rows')
if value.get('status')!='DIAGNOSTIC_ONLY_NO_RATIFICATION_EFFECT': raise SystemExit('expected completed value-contrast diagnostic v0.2')
if value.get('challenge_sha256')!=sha(challenge) or value.get('contrast_contract_sha256')!=sha(contract): raise SystemExit('value result lineage drift')
if value.get('interpretation_contract',{}).get('gradient_performed') is not False: raise SystemExit('value diagnostic governance drift')
receipt={
 'schema':'alice.eipm.n0.v02-parent-value-path-diagnostic-preparation.v0.1',
 'status':'PASS_PARENT_VALUE_PATH_DIAGNOSTIC_READY_FOR_GPU',
 'created_at':datetime.now(timezone.utc).isoformat(),
 'git_revision':head,
 'challenge_sha256':sha(challenge),
 'contrast_contract_sha256':sha(contract),
 'contrast_manifest_sha256':sha(manifest_path),
 'value_contrast_result_sha256':sha(value_result),
 'diagnostic_sha256':sha(diagnostic),
 'semantic_checkpoint_sha256':sha(semantic),
 'structured_checkpoint_sha256':sha(structured),
 'evidence_adapter_sha256':sha(adapter),
 'evidence_graph_sha256':sha(graph),
 'contrast_rows':32,
 'canonical_parent_path_required':True,
 'forward_hook_capture_only_no_parent_mutation':True,
 'canonical_parent_trace_parity_required':True,
 'graph_field_weight_and_relation_bias_measured':True,
 'challenge_rows_used_for_training':False,
 'gradient_performed':False,
 'ratification_effect':False,
 'latent_weights_changed':False,
 'scale_authorized':False,
 'private_identity_data':False,
 'private_identity_gradient':False,
}
prep_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2,sort_keys=True))
print('parent_value_path_v01_prepare_pass=true')
print(f'git_revision={head}')
print('contrast_rows=32')
print('canonical_parent_trace_parity_required=true')
print('graph_field_weight_and_relation_bias_measured=true')
print('gradient_performed=false')
print(f'preparation_receipt={prep_path}')
PY
