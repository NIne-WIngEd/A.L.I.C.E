#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-train}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
REPLAY_CACHE="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_evidence_selector_repair_curriculum_v0_1.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_evidence_selector_repair_v0_1.py"
RELATION_DEP="$ROOT/scripts/eipm/n0/train_n0_v02_relation_repair.py"
REPAIR_ROOT="$WORKDIR/evidence-selector-repair-v0.1"
PREP_ROOT="$REPAIR_ROOT/preparation"
CURRICULUM="$PREP_ROOT/selector_repair_curriculum.jsonl"
MANIFEST="$PREP_ROOT/selector_repair_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"
OUT_ROOT="$REPAIR_ROOT/training"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CONFIG" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$STRUCTURED_CONFIG" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" \
  "$PARENT_GRAPH" \
  "$REPLAY_CACHE" \
  "$BUILDER" \
  "$TRAINER" \
  "$RELATION_DEP"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing evidence-selector-repair artifact: $required" >&2
    exit 2
  fi
done

validate_receipt() {
  python - "$ROOT" "$PREP_RECEIPT" "$CURRICULUM" "$MANIFEST" \
    "$BUILDER" "$TRAINER" "$RELATION_DEP" "$SEMANTIC_CONFIG" \
    "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" \
    "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
    "$PARENT_ADAPTER" "$PARENT_GRAPH" "$REPLAY_CACHE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(
 root,receipt_path,curriculum,manifest,builder,trainer,relation_dep,semantic_config,
 semantic_checkpoint,tokenizer,structured_config,structured_checkpoint,parent_adapter,
 parent_graph,replay_cache
)=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if receipt.get('schema')!='alice.eipm.n0.v02-evidence-selector-repair-preparation.v0.1': raise SystemExit('selector repair prep schema drift')
if receipt.get('status')!='PASS_EVIDENCE_SELECTOR_REPAIR_READY_FOR_GPU': raise SystemExit('selector repair prep status drift')
if receipt.get('git_revision')!=head: raise SystemExit(f"selector repair prep git drift: prep={receipt.get('git_revision')} current={head}")
paths={
 'curriculum':curriculum,'manifest':manifest,'builder':builder,'trainer':trainer,
 'relation_repair_dependency':relation_dep,'semantic_config':semantic_config,
 'semantic_checkpoint':semantic_checkpoint,'tokenizer':tokenizer,
 'structured_config':structured_config,'structured_checkpoint':structured_checkpoint,
 'parent_adapter':parent_adapter,'parent_graph':parent_graph,'replay_cache':replay_cache,
}
for name,path in paths.items():
    expected=receipt.get('artifact_sha256',{}).get(name)
    observed=sha(path)
    if expected!=observed: raise SystemExit(f'selector repair prep artifact drift: {name}')
if receipt.get('frozen_latent_challenge_rows_used_for_training') is not False: raise SystemExit('frozen challenge leakage')
if receipt.get('parent_value_path_diagnostic_rows_used_for_training') is not False: raise SystemExit('diagnostic leakage')
if receipt.get('hard_parameter_ceiling') is not None: raise SystemExit('unexpected hard parameter ceiling')
if receipt.get('scale_authorized') is not False: raise SystemExit('unexpected scale authorization')
print('selector_repair_preparation_valid=true')
print(f'git_revision={head}')
PY
}

if [[ "$MODE" == "prepare" ]]; then
  if [[ -f "$PREP_RECEIPT" ]]; then
    validate_receipt
    echo "existing_selector_repair_preparation_valid=true"
    cat "$PREP_RECEIPT"
    exit 0
  fi
  if [[ -d "$PREP_ROOT" && -n "$(find "$PREP_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty selector-repair preparation without a valid receipt: $PREP_ROOT" >&2
    exit 3
  fi

  python -m py_compile "$BUILDER" "$TRAINER" "$RELATION_DEP"
  pytest -q \
    "$ROOT/tests/eipm/test_n0_evidence_view_adapter.py" \
    "$ROOT/tests/eipm/test_n0_evidence_graph.py" \
    "$ROOT/tests/eipm/test_n0_evidence_graph_dual_endpoint.py"

  mkdir -p "$PREP_ROOT"
  python "$BUILDER" --output "$CURRICULUM" --manifest "$MANIFEST"

  python - "$ROOT" "$PREP_RECEIPT" "$CURRICULUM" "$MANIFEST" \
    "$BUILDER" "$TRAINER" "$RELATION_DEP" "$SEMANTIC_CONFIG" \
    "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" \
    "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
    "$PARENT_ADAPTER" "$PARENT_GRAPH" "$REPLAY_CACHE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(
 root,receipt_path,curriculum,manifest,builder,trainer,relation_dep,semantic_config,
 semantic_checkpoint,tokenizer,structured_config,structured_checkpoint,parent_adapter,
 parent_graph,replay_cache
)=map(Path,sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
manifest_obj=json.loads(manifest.read_text(encoding='utf-8'))
if manifest_obj.get('schema')!='alice.eipm.n0.v02-evidence-selector-repair-curriculum.v0.1': raise SystemExit('selector repair manifest schema mismatch')
if manifest_obj.get('frozen_latent_challenge_rows_used_for_training') is not False: raise SystemExit('frozen challenge leakage')
if manifest_obj.get('parent_value_path_diagnostic_rows_used_for_training') is not False: raise SystemExit('parent diagnostic leakage')
paths={
 'curriculum':curriculum,'manifest':manifest,'builder':builder,'trainer':trainer,
 'relation_repair_dependency':relation_dep,'semantic_config':semantic_config,
 'semantic_checkpoint':semantic_checkpoint,'tokenizer':tokenizer,
 'structured_config':structured_config,'structured_checkpoint':structured_checkpoint,
 'parent_adapter':parent_adapter,'parent_graph':parent_graph,'replay_cache':replay_cache,
}
receipt={
 'schema':'alice.eipm.n0.v02-evidence-selector-repair-preparation.v0.1',
 'status':'PASS_EVIDENCE_SELECTOR_REPAIR_READY_FOR_GPU',
 'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'artifact_sha256':{name:sha(path) for name,path in sorted(paths.items())},
 'curriculum_rows':manifest_obj.get('rows'),
 'family_count':manifest_obj.get('family_count'),
 'trainable_scope':'evidence_view_adapter.prior_head_only_for_initial_localized_repair',
 'trainable_scope_is_permanent_architecture_limit':False,
 'evidence_graph_frozen':True,
 'semantic_parent_frozen':True,
 'structured_parent_frozen':True,
 'adapter_field_state_transform_frozen':True,
 'graph_pooled_cosine_is_optimization_target':False,
 'frozen_latent_challenge_rows_used_for_training':False,
 'parent_value_path_diagnostic_rows_used_for_training':False,
 'private_identity_data':False,
 'private_identity_gradient':False,
 'gradient_performed':False,
 'hard_parameter_ceiling':None,
 'hard_parameter_floor':None,
 'scale_authorized':False,
 'scale_policy':'no scale before signal localization; broaden selector expressivity only if held-out evidence requires it; evaluate latent capacity only after target signal reaches latent input',
 'bounded_execution_policy':'one training run, dev-only checkpoint selection, one untouched test gate, no automatic rerun or hotfix loop',
}
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
  validate_receipt
  echo "selector_repair_prepare_complete=true"
  echo "receipt=$PREP_RECEIPT"
  exit 0
fi

if [[ "$MODE" != "train" ]]; then
  echo "Usage: $0 prepare|train" >&2
  exit 64
fi

if [[ ! -f "$PREP_RECEIPT" ]]; then
  echo "Missing selector-repair preparation receipt. Run prepare first." >&2
  exit 4
fi
validate_receipt

if [[ -d "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing selector-repair training output: $OUT_ROOT" >&2
  exit 5
fi

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Evidence selector repair requires one visible CUDA device." >&2
  exit 6
fi

mkdir -p "$OUT_ROOT"
echo "===== N0 V0.2 EVIDENCE SELECTOR REPAIR v0.1 ====="
date -Is
echo "parent_adapter=specialized_expanded_640x3_graph512x2/step-00000080"
echo "parent_graph=relation-repair-v0.1/step-00000080"
echo "trainable_scope=evidence_view_adapter.prior_head_only"
echo "scope_is_permanent_architecture_limit=false"
echo "graph_trainable=false"
echo "graph_pooled_cosine_optimized=false"
echo "hard_parameter_ceiling=none"
echo "scale_authorized=false"
echo "automatic_hotfix_loop=false"

python "$TRAINER" \
  --repo-root "$ROOT" \
  --prep-receipt "$PREP_RECEIPT" \
  --curriculum "$CURRICULUM" \
  --manifest "$MANIFEST" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --replay-cache "$REPLAY_CACHE" \
  --output-dir "$OUT_ROOT" \
  --max-steps 160 \
  --save-every 40 \
  --learning-rate 0.0001 \
  --seed 20260917

echo "===== N0 V0.2 EVIDENCE SELECTOR REPAIR COMPLETE ====="
date -Is
echo "result=$OUT_ROOT/result.json"
