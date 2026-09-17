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
SELECTOR_FAILURE="$WORKDIR/evidence-selector-repair-v0.1/training/result.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_relation_endpoint_repair_curriculum_v0_2.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_relation_endpoint_repair_v0_2.py"
RELATION_DEP="$ROOT/scripts/eipm/n0/train_n0_v02_relation_repair.py"
SELECTOR_DEP="$ROOT/scripts/eipm/n0/train_n0_v02_evidence_selector_repair_v0_1.py"
GRAPH_SOURCE="$ROOT/src/alice_personality/n0/evidence_graph.py"
DUAL_GRAPH_SOURCE="$ROOT/src/alice_personality/n0/evidence_graph_dual_endpoint.py"
ADAPTER_SOURCE="$ROOT/src/alice_personality/n0/evidence_view_adapter.py"
REPAIR_ROOT="$WORKDIR/relation-endpoint-repair-v0.2"
PREP_ROOT="$REPAIR_ROOT/preparation"
CURRICULUM="$PREP_ROOT/endpoint_repair_curriculum.jsonl"
MANIFEST="$PREP_ROOT/endpoint_repair_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"
OUT_ROOT="$REPAIR_ROOT/training"
EXPECTED_SELECTOR_FAILURE_SHA256="80968a34a9bb7dbf3ffa549b1410f45001451700bbdf6404ad8ec667f89288c6"

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
  "$SELECTOR_FAILURE" \
  "$BUILDER" \
  "$TRAINER" \
  "$RELATION_DEP" \
  "$SELECTOR_DEP" \
  "$GRAPH_SOURCE" \
  "$DUAL_GRAPH_SOURCE" \
  "$ADAPTER_SOURCE"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing relation-endpoint-repair artifact: $required" >&2
    exit 2
  fi
done

validate_selector_failure() {
  python - "$SELECTOR_FAILURE" "$EXPECTED_SELECTOR_FAILURE_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]
h=hashlib.sha256(path.read_bytes()).hexdigest()
if h != expected:
    raise SystemExit(f"selector failure result hash drift: expected={expected} observed={h}")
obj=json.loads(path.read_text(encoding='utf-8'))
if obj.get('schema')!='alice.eipm.n0.v02-evidence-selector-repair-result.v0.1':
    raise SystemExit('selector failure result schema drift')
if obj.get('status')!='FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX':
    raise SystemExit('endpoint repair requires the preserved failed selector heldout result')
if obj.get('heldout_gate',{}).get('pass') is not False:
    raise SystemExit('selector failure heldout gate unexpectedly passed')
if obj.get('automatic_rerun_or_hotfix_authorized') is not False:
    raise SystemExit('selector failure anti-loop contract drift')
print('selector_failure_evidence_bound=true')
print(f'selector_failure_sha256={h}')
PY
}

validate_receipt() {
  python - "$ROOT" "$PREP_RECEIPT" "$CURRICULUM" "$MANIFEST" \
    "$BUILDER" "$TRAINER" "$RELATION_DEP" "$SELECTOR_DEP" \
    "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
    "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" \
    "$STRUCTURED_CHECKPOINT/structured_state.safetensors" "$PARENT_ADAPTER" \
    "$PARENT_GRAPH" "$REPLAY_CACHE" "$SELECTOR_FAILURE" \
    "$GRAPH_SOURCE" "$DUAL_GRAPH_SOURCE" "$ADAPTER_SOURCE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(
 root,receipt_path,curriculum,manifest,builder,trainer,relation_dep,selector_dep,
 semantic_config,semantic_checkpoint,tokenizer,structured_config,structured_checkpoint,
 parent_adapter,parent_graph,replay_cache,selector_failure,graph_source,dual_graph_source,
 adapter_source
)=map(Path,sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if receipt.get('schema')!='alice.eipm.n0.v02-relation-endpoint-repair-preparation.v0.2': raise SystemExit('endpoint repair prep schema drift')
if receipt.get('status')!='PASS_RELATION_ENDPOINT_REPAIR_READY_FOR_GPU': raise SystemExit('endpoint repair prep status drift')
if receipt.get('git_revision')!=head: raise SystemExit(f"endpoint repair prep git drift: prep={receipt.get('git_revision')} current={head}")
paths={
 'curriculum':curriculum,'manifest':manifest,'builder':builder,'trainer':trainer,
 'relation_repair_dependency':relation_dep,'selector_eval_dependency':selector_dep,
 'semantic_config':semantic_config,'semantic_checkpoint':semantic_checkpoint,'tokenizer':tokenizer,
 'structured_config':structured_config,'structured_checkpoint':structured_checkpoint,
 'parent_adapter':parent_adapter,'parent_graph':parent_graph,'replay_cache':replay_cache,
 'selector_failure_result':selector_failure,'evidence_graph_source':graph_source,
 'dual_endpoint_graph_source':dual_graph_source,'evidence_adapter_source':adapter_source,
}
for name,path in paths.items():
    if receipt.get('artifact_sha256',{}).get(name)!=sha(path):
        raise SystemExit(f'endpoint repair prep artifact drift: {name}')
if receipt.get('failed_selector_repair_adapter_used_as_parent') is not False: raise SystemExit('failed selector adapter parent drift')
if receipt.get('frozen_latent_challenge_rows_used_for_training') is not False: raise SystemExit('frozen challenge leakage')
if receipt.get('parent_value_path_diagnostic_rows_used_for_training') is not False: raise SystemExit('parent diagnostic leakage')
if receipt.get('hard_parameter_ceiling') is not None or receipt.get('hard_parameter_floor') is not None: raise SystemExit('unexpected capacity bound')
if receipt.get('scale_authorized') is not False: raise SystemExit('unexpected scale authorization')
print('relation_endpoint_repair_preparation_valid=true')
print(f'git_revision={head}')
PY
}

if [[ "$MODE" == "prepare" ]]; then
  validate_selector_failure
  if [[ -f "$PREP_RECEIPT" ]]; then
    validate_receipt
    echo "existing_relation_endpoint_repair_preparation_valid=true"
    cat "$PREP_RECEIPT"
    exit 0
  fi
  if [[ -d "$PREP_ROOT" && -n "$(find "$PREP_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty endpoint-repair preparation without a valid receipt: $PREP_ROOT" >&2
    exit 3
  fi

  python -m py_compile "$BUILDER" "$TRAINER" "$RELATION_DEP" "$SELECTOR_DEP"
  pytest -q \
    "$ROOT/tests/eipm/test_n0_evidence_graph.py" \
    "$ROOT/tests/eipm/test_n0_evidence_graph_dual_endpoint.py"

  mkdir -p "$PREP_ROOT"
  python "$BUILDER" --output "$CURRICULUM" --manifest "$MANIFEST"

  python - "$ROOT" "$PREP_RECEIPT" "$CURRICULUM" "$MANIFEST" \
    "$BUILDER" "$TRAINER" "$RELATION_DEP" "$SELECTOR_DEP" \
    "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
    "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" \
    "$STRUCTURED_CHECKPOINT/structured_state.safetensors" "$PARENT_ADAPTER" \
    "$PARENT_GRAPH" "$REPLAY_CACHE" "$SELECTOR_FAILURE" \
    "$GRAPH_SOURCE" "$DUAL_GRAPH_SOURCE" "$ADAPTER_SOURCE" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(
 root,receipt_path,curriculum,manifest,builder,trainer,relation_dep,selector_dep,
 semantic_config,semantic_checkpoint,tokenizer,structured_config,structured_checkpoint,
 parent_adapter,parent_graph,replay_cache,selector_failure,graph_source,dual_graph_source,
 adapter_source
)=map(Path,sys.argv[1:])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest_obj=json.loads(manifest.read_text(encoding='utf-8'))
if manifest_obj.get('schema')!='alice.eipm.n0.v02-relation-endpoint-repair-curriculum.v0.2': raise SystemExit('endpoint repair manifest schema mismatch')
if manifest_obj.get('prior_selector_repair_v0_1_test_rows_reused') is not False: raise SystemExit('selector heldout row leakage')
if manifest_obj.get('frozen_latent_challenge_rows_used_for_training') is not False: raise SystemExit('frozen challenge leakage')
if manifest_obj.get('parent_value_path_diagnostic_rows_used_for_training') is not False: raise SystemExit('parent diagnostic leakage')
if manifest_obj.get('graph_pooled_cosine_is_optimization_target') is not False: raise SystemExit('endpoint objective drift')
selector_obj=json.loads(selector_failure.read_text(encoding='utf-8'))
if selector_obj.get('status')!='FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX': raise SystemExit('selector failure provenance drift')
paths={
 'curriculum':curriculum,'manifest':manifest,'builder':builder,'trainer':trainer,
 'relation_repair_dependency':relation_dep,'selector_eval_dependency':selector_dep,
 'semantic_config':semantic_config,'semantic_checkpoint':semantic_checkpoint,'tokenizer':tokenizer,
 'structured_config':structured_config,'structured_checkpoint':structured_checkpoint,
 'parent_adapter':parent_adapter,'parent_graph':parent_graph,'replay_cache':replay_cache,
 'selector_failure_result':selector_failure,'evidence_graph_source':graph_source,
 'dual_endpoint_graph_source':dual_graph_source,'evidence_adapter_source':adapter_source,
}
receipt={
 'schema':'alice.eipm.n0.v02-relation-endpoint-repair-preparation.v0.2',
 'status':'PASS_RELATION_ENDPOINT_REPAIR_READY_FOR_GPU',
 'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'artifact_sha256':{name:sha(path) for name,path in sorted(paths.items())},
 'curriculum_rows':manifest_obj.get('rows'),
 'family_count':manifest_obj.get('family_count'),
 'source_role_families':manifest_obj.get('source_role_families'),
 'target_role_families':manifest_obj.get('target_role_families'),
 'causal_localization':'query_conditioned_relation_endpoint_read_after_prior_head_only_selector_repair_failed_with_source_target_role_split',
 'trainable_scope':'query_conditioned_source_target_relation_read_only',
 'trainable_scope_is_permanent_architecture_limit':False,
 'conflict_pool_frozen':True,
 'message_passing_frozen':True,
 'evidence_view_adapter_frozen':True,
 'failed_selector_repair_adapter_used_as_parent':False,
 'semantic_parent_frozen':True,
 'structured_parent_frozen':True,
 'graph_pooled_cosine_is_optimization_target':False,
 'prior_selector_repair_v0_1_test_rows_reused':False,
 'frozen_latent_challenge_rows_used_for_training':False,
 'parent_value_path_diagnostic_rows_used_for_training':False,
 'private_identity_data':False,
 'private_identity_gradient':False,
 'gradient_performed':False,
 'hard_parameter_ceiling':None,
 'hard_parameter_floor':None,
 'scale_authorized':False,
 'scale_policy':'repair localized endpoint read before any latent capacity decision; current trainable scope and step/data budgets are operating choices, not permanent architecture limits',
 'bounded_execution_policy':'one training run, dev-only checkpoint selection, one fresh untouched test gate, no automatic rerun or hotfix loop',
}
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2,sort_keys=True))
PY
  validate_receipt
  echo "relation_endpoint_repair_prepare_complete=true"
  echo "receipt=$PREP_RECEIPT"
  exit 0
fi

if [[ "$MODE" != "train" ]]; then
  echo "Usage: $0 prepare|train" >&2
  exit 64
fi

if [[ ! -f "$PREP_RECEIPT" ]]; then
  echo "Missing endpoint-repair preparation receipt. Run prepare once first." >&2
  exit 4
fi
validate_selector_failure
validate_receipt

if [[ -d "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing endpoint-repair training output: $OUT_ROOT" >&2
  exit 5
fi

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Relation endpoint repair requires one visible CUDA device." >&2
  exit 6
fi

mkdir -p "$OUT_ROOT"
echo "===== N0 V0.2 RELATION ENDPOINT-ROLE REPAIR v0.2 ====="
date -Is
echo "parent_adapter=original_specialist_step80_not_failed_selector_repair"
echo "parent_graph=relation-repair-v0.1/step-00000080"
echo "trainable_scope=query_conditioned_source_target_relation_read_only"
echo "scope_is_permanent_architecture_limit=false"
echo "conflict_pool_trainable=false"
echo "message_passing_trainable=false"
echo "adapter_trainable=false"
echo "graph_pooled_cosine_optimized=false"
echo "hard_parameter_ceiling=none"
echo "hard_parameter_floor=none"
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
  --max-steps 200 \
  --save-every 40 \
  --learning-rate 0.0001 \
  --seed 20260917

echo "===== N0 V0.2 RELATION ENDPOINT-ROLE REPAIR COMPLETE ====="
date -Is
echo "result=$OUT_ROOT/result.json"
