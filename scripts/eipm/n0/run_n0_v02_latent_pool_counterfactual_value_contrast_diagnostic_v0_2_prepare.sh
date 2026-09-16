#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
CHALLENGE_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2"
CHALLENGE="$CHALLENGE_ROOT/challenge.jsonl"
FREEZE_RECEIPT="$CHALLENGE_ROOT/freeze_receipt.json"
V02_RESULT="$CHALLENGE_ROOT/result.json"
CANDIDATE="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
COMPILER="$ROOT/scripts/eipm/n0/compile_n0_v02_latent_pool_value_contrast_contract_v0_2.py"
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_latent_pool_counterfactual_value_contrast_v0_2.py"
TEST="$ROOT/tests/eipm/test_n0_latent_pool_value_contrast_contract_v0_2.py"
OUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-v0.2-counterfactual-diagnostic-v0.2"
CONTRACT="$OUT_ROOT/contrast_contract.jsonl"
MANIFEST="$OUT_ROOT/contrast_manifest.json"
PREP="$OUT_ROOT/preparation_receipt.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in "$CHALLENGE" "$FREEZE_RECEIPT" "$V02_RESULT" "$CANDIDATE" "$COMPILER" "$DIAGNOSTIC" "$TEST"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing contrast diagnostic v0.2 preparation artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty contrast diagnostic v0.2 preparation: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$COMPILER" "$DIAGNOSTIC"
pytest -q "$TEST"
mkdir -p "$OUT_ROOT"
python "$COMPILER" --challenge "$CHALLENGE" --output "$CONTRACT" --manifest "$MANIFEST"

python - "$CHALLENGE" "$FREEZE_RECEIPT" "$V02_RESULT" "$CANDIDATE" "$COMPILER" "$DIAGNOSTIC" "$CONTRACT" "$MANIFEST" "$PREP" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

challenge, freeze_path, result_path, candidate, compiler, diagnostic, contract, manifest_path, prep_path = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
result = json.loads(result_path.read_text(encoding='utf-8'))
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

if result.get('status') != 'FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION':
    raise SystemExit('expected immutable failed latent challenge v0.2 result')
if result.get('gradient_performed') is not False or result.get('challenge_rows_used_for_training') is not False:
    raise SystemExit('historical v0.2 challenge governance drift')
if result.get('candidate_latent_pool_sha256') != sha(candidate):
    raise SystemExit('diagnostic candidate differs from historical v0.2 candidate')
if freeze.get('challenge_sha256') != sha(challenge):
    raise SystemExit('frozen v0.2 challenge hash drift')
if manifest.get('challenge_sha256') != sha(challenge) or manifest.get('contrast_contract_sha256') != sha(contract):
    raise SystemExit('contrast sidecar is not bound to the frozen challenge')
if manifest.get('rows') != 32:
    raise SystemExit('contrast sidecar must contain exactly 32 eligible rows')
if manifest.get('global_current_value_assumption') is not False:
    raise SystemExit('contrast sidecar contains forbidden global current-value assumption')
if manifest.get('historical_query_rows_included') is not False:
    raise SystemExit('historical rows leaked into current-value diagnostic contract')
if manifest.get('target_and_foil_materialized_before_gpu') is not True:
    raise SystemExit('target/foil semantics were not resolved before GPU')
if manifest.get('ratification_effect') is not False:
    raise SystemExit('diagnostic sidecar may not change ratification')

receipt = {
    'schema': 'alice.eipm.n0.v02-latent-pool-value-contrast-diagnostic-preparation.v0.2',
    'status': 'PASS_SEMANTIC_CONTRACT_COMPILED_BEFORE_GPU',
    'created_at': datetime.now(timezone.utc).isoformat(),
    'git_revision': head,
    'challenge_sha256': sha(challenge),
    'freeze_receipt_sha256': sha(freeze_path),
    'historical_v0_2_result_sha256': sha(result_path),
    'candidate_latent_pool_sha256': sha(candidate),
    'compiler_sha256': sha(compiler),
    'diagnostic_sha256': sha(diagnostic),
    'contrast_contract_sha256': sha(contract),
    'contrast_manifest_sha256': sha(manifest_path),
    'contrast_rows': 32,
    'contrast_families': manifest['families'],
    'query_conditioned_answer_semantics': True,
    'global_current_value_assumption': False,
    'historical_query_rows_included': False,
    'target_and_foil_materialized_before_gpu': True,
    'prior_diagnostic_job_575683_result_available': False,
    'prior_diagnostic_failure_interpretation': 'semantic_contract_design_failure_before_model_diagnostic_result',
    'historical_v0_2_result_reclassified': False,
    'challenge_rows_used_for_training': False,
    'gradient_performed': False,
    'ratification_effect': False,
    'private_identity_data': False,
    'private_identity_gradient': False,
}
prep_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2, sort_keys=True))
print('latent_pool_value_contrast_v02_prepare_pass=true')
print(f'git_revision={head}')
print('contrast_rows=32')
print('query_conditioned_answer_semantics=true')
print('global_current_value_assumption=false')
print('historical_query_rows_included=false')
print('target_and_foil_materialized_before_gpu=true')
print('gradient_performed=false')
print(f'preparation_receipt={prep_path}')
PY
