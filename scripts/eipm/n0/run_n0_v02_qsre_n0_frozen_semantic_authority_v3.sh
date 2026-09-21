#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
P1_EXPECTED_SHA="91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9"
ORIGINAL_PRODUCTION_REVISION="b32fabe49f206c2d71e17df5197a62b2a37c8c43"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: frozen-authority source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no
  exit 91
fi
python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PY

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

SOURCE="$WORKDIR/qsre-production-core-v1"
RECOVERY="$WORKDIR/qsre-production-core-v1-stop-tail-recovery-v1"
FAILED_FINAL="$WORKDIR/qsre-production-core-v1-final-v3"
SEM_V2="$WORKDIR/qsre-n0-closure-semantic-v2"
SEM_V2_RESULT="$SEM_V2/p2s-schema-matcher/result.json"
RUN_ROOT="$WORKDIR/qsre-n0-frozen-semantic-authority-v3"

for root in "$SOURCE" "$RECOVERY" "$FAILED_FINAL" "$SEM_V2"; do
  if [[ ! -d "$root" ]]; then
    echo "STOP: required preserved evidence root missing: $root" >&2
    exit 92
  fi
done
if [[ -e "$RUN_ROOT" ]]; then
  echo "STOP: frozen-authority evidence already exists; preserve it: $RUN_ROOT" >&2
  exit 93
fi
mkdir -p "$RUN_ROOT"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_n0_frozen_authority_training_plan_v2.json"
AUTH_PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_frozen_semantic_authority_plan_v3.json"
META="$ROOT/configs/eipm/n0/n0_v02_qsre_closure_schema_meta_v2.json"
PROD_SCHEMA_JSON="$ROOT/configs/eipm/n0/n0_v02_qsre_production_relation_schema_v1.json"
FINAL_SCHEMA_JSON="$ROOT/configs/eipm/n0/n0_v02_qsre_final_self_validation_relation_schema_v1.json"
FULL_SCALE_PACKAGE="$ROOT/configs/eipm/n0/n0_v02_final_full_scale_capability_package_v1.json"

P0="$SOURCE/p0"
PROD_CURRICULUM="$P0/qsre_production_curriculum_v1.jsonl"
PROD_CACHE="$P0/qsre_production_cache_v1.pt"
PROD_SCHEMA_CACHE="$P0/qsre_production_schema_v1.pt"

P1_ROOT="$RECOVERY/p1"
P1_RESULT="$P1_ROOT/result.json"
FAILED_P2_RESULT="$FAILED_FINAL/p2/result.json"

FROZEN="$SOURCE/final-frozen"
FINAL_GENERAL="$FROZEN/general.jsonl"
FINAL_RELATIONAL="$FROZEN/relational.jsonl"
FINAL_MANIFEST="$FROZEN/manifest.json"
FINAL_SCHEMA_CACHE="$FROZEN/final_schema_v1.pt"
FINAL_CACHE="$FROZEN/final_relational_cache_v1.pt"
FINAL_FREEZE="$FROZEN/freeze_receipt.json"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
FINAL_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_final_self_validation_contract_v1.json"

TOKENIZER="$WORKDIR/tokenizer-v0.2.1"
SEMANTIC="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
STRUCTURED="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
ADAPTER="$SPECIALIST/evidence_view_adapter.safetensors"
GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
FUSION="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"

for required in   "$PLAN" "$AUTH_PLAN" "$META" "$PROD_SCHEMA_JSON" "$FINAL_SCHEMA_JSON"   "$PROD_CURRICULUM" "$PROD_CACHE" "$PROD_SCHEMA_CACHE"   "$P1_RESULT" "$FAILED_P2_RESULT" "$SEM_V2_RESULT"   "$FINAL_CONTRACT" "$FINAL_GENERAL" "$FINAL_RELATIONAL" "$FINAL_MANIFEST"   "$FINAL_SCHEMA_CACHE" "$FINAL_CACHE" "$FINAL_FREEZE"   "$SEMANTIC_CONFIG" "$STRUCTURED_CONFIG" "$FUSION_CONFIG"   "$FUSION_RATIFICATION" "$LATENT_CONFIG" "$TOKENIZER/tokenizer.json"   "$SEMANTIC" "$STRUCTURED/structured_state.safetensors" "$ADAPTER" "$GRAPH"   "$FUSION/cross_context_fusion.safetensors" "$LATENT"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing frozen-authority prerequisite: $required" >&2
    exit 94
  fi
done

python - "$P1_RESULT" "$P1_ROOT" "$FAILED_P2_RESULT" "$SEM_V2_RESULT"   "$FINAL_FREEZE" "$ORIGINAL_PRODUCTION_REVISION" "$P1_EXPECTED_SHA"   "$PLAN" "$AUTH_PLAN" "$META" "$PROD_SCHEMA_JSON" "$FINAL_SCHEMA_JSON" \
  "$PROD_SCHEMA_CACHE" "$FINAL_SCHEMA_CACHE" "$FULL_SCALE_PACKAGE" <<'PY'
import hashlib,json,math,sys,torch
from pathlib import Path

p1_path=Path(sys.argv[1]); p1_root=Path(sys.argv[2]); failed_p2_path=Path(sys.argv[3])
sem_v2_path=Path(sys.argv[4]); freeze_path=Path(sys.argv[5])
original_revision=sys.argv[6]; expected_p1_sha=sys.argv[7]
plan_path=Path(sys.argv[8]); auth_plan_path=Path(sys.argv[9]); meta_path=Path(sys.argv[10])
prod_schema_json=Path(sys.argv[11]); final_schema_json=Path(sys.argv[12])
prod_schema_cache_path=Path(sys.argv[13]); final_schema_cache_path=Path(sys.argv[14])
full_scale_path=Path(sys.argv[15])

p1=json.loads(p1_path.read_text())
failed_p2=json.loads(failed_p2_path.read_text())
sem_v2=json.loads(sem_v2_path.read_text())
freeze=json.loads(freeze_path.read_text())
plan=json.loads(plan_path.read_text())
auth_plan=json.loads(auth_plan_path.read_text())
meta=json.loads(meta_path.read_text())

assert p1["status"]=="PASS_QSRE_PRODUCTION_P1_EXECUTOR"
assert int(p1["selected"]["step"])==50
assert p1["selected"]["checkpoint_sha256"]==expected_p1_sha
p1_ckpt=p1_root/"step-00000050"/"qsre_production_p1.pt"
digest=hashlib.sha256(p1_ckpt.read_bytes()).hexdigest()
assert digest==expected_p1_sha

assert failed_p2["status"]=="FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3"
assert failed_p2["selected"] is None
assert failed_p2["p3_authorized"] is False

assert sem_v2["schema"]=="alice.eipm.n0.qsre-closure-matcher-result.v2"
assert sem_v2["status"]=="FAIL_QSRE_CLOSURE_SCHEMA_MATCHER"
assert sem_v2["selected"] is None
assert sem_v2["p2_authorized"] is False
assert int(sem_v2["history"][-1]["step"])==1000
assert int(sem_v2["best_observed"]["step"])==150
best=sem_v2["best_observed"]["metrics"]
expected={
    "auxiliary_holdout_relation_top1":0.6666666666666666,
    "auxiliary_seen_relation_top1":0.71875,
    "production_core_single_relation_top1":0.9477272727272728,
    "heldout_factor_macro_accuracy":0.7166666666666667,
}
for key,value in expected.items():
    assert math.isclose(float(best[key]),value,rel_tol=0.0,abs_tol=1e-12),(key,best[key])
assert sem_v2["production_open_schema_descriptions_used_in_gradient"] is False
assert sem_v2["final_only_relation_descriptions_used_in_gradient"] is False
assert sem_v2["auxiliary_holdout_relation_descriptions_used_in_gradient"] is False
assert sem_v2["relation_keys_used_as_semantic_tokens"] is False

assert freeze["status"]=="FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS"
assert freeze["git_revision"]==original_revision
assert freeze["results_observed_before_freeze"] is False
assert freeze["threshold_changes_after_results_forbidden"] is True
assert freeze["private_identity_gradient"] is False

assert plan["schema"]=="alice.eipm.n0.qsre-n0-frozen-authority-training-plan.v2"
assert plan["stages"]["P2"]["failure_source"]["magnolia_job"]==575966
assert plan["operator_architecture"]["parameter_free_token_evidence_matcher"] is True
assert plan["operator_architecture"]["frozen_semantic_authority_external"] is True
assert plan["operator_architecture"]["trainable_operator_cannot_rewrite_relation_identity"] is True

assert auth_plan["schema"]=="alice.eipm.n0.qsre-frozen-semantic-authority-plan.v3"
assert auth_plan["source_failure"]["job"]==575966
assert auth_plan["source_failure"]["status"]=="FAIL_QSRE_CLOSURE_SCHEMA_MATCHER"
assert auth_plan["semantic_authority"]["gradient"] is False
assert auth_plan["semantic_authority"]["optimizer"] is False
assert auth_plan["semantic_authority"]["trainable_authority_parameters"]==0
assert auth_plan["governance"]["threshold_changes_after_results"] is False

assert meta["schema"]=="alice.eipm.n0.qsre-closure-schema-meta.v2"
assert meta["schema_text_contract"]["relation_key_in_semantic_text"] is False

prod_schema_cache=torch.load(prod_schema_cache_path,map_location="cpu")
final_schema_cache=torch.load(final_schema_cache_path,map_location="cpu")
assert prod_schema_cache["schema"]=="alice.eipm.n0.qsre-production-schema-cache.v1"
assert final_schema_cache["schema"]=="alice.eipm.n0.qsre-production-schema-cache.v1"
assert prod_schema_cache["source_schema_sha256"]==hashlib.sha256(prod_schema_json.read_bytes()).hexdigest()
assert final_schema_cache["source_schema_sha256"]==hashlib.sha256(final_schema_json.read_bytes()).hexdigest()
assert prod_schema_cache["semantic_checkpoint_sha256"]==final_schema_cache["semantic_checkpoint_sha256"]

full_scale=json.loads(full_scale_path.read_text())
assert full_scale["schema"]=="alice.eipm.n0.final-full-scale-capability-package.v1"
assert full_scale["status"]=="PRECOMMITTED_FULL_SCALE_N0_CLOSURE_PACKAGE"
assert full_scale["n0_complete"] is False
assert full_scale["closure_authority"]=="PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE"
assert full_scale["scale_policy"]["hard_parameter_ceiling"] is None
assert full_scale["scale_policy"]["runtime_relation_count_ceiling"] is None
assert full_scale["scale_policy"]["runtime_reasoning_step_ceiling"] is None
assert full_scale["scale_policy"]["current_training_steps_are_serving_ceiling"] is False
assert full_scale["architecture"]["semantic_authority_gradient"] is False
assert full_scale["architecture"]["exact_relation_sparsity_before_binder"] is False
assert full_scale["architecture"]["binder_owns_exact_structural_sparsity"] is True
assert full_scale["governance"]["automatic_rerun"] is False
assert full_scale["governance"]["threshold_change_after_results"] is False

print("PASS_N0_FROZEN_AUTHORITY_SOURCE_LINEAGE")
print("p1_checkpoint_sha256="+digest)
print("source_failed_p2_job=575958")
print("source_failed_p2s_v1_job=575962")
print("source_failed_p2s_v2_job=575966")
print("original_frozen_final_validation_reused=true")
PY

echo "===== N0 FROZEN SEMANTIC AUTHORITY V3 ====="
date -Is
echo "source_revision=$HEAD"
echo "run_root=$RUN_ROOT"
echo "source_failed_p2s_v2_job=575966"
echo "semantic_authority_gradient=false"
echo "semantic_authority_optimizer=false"
echo "learned_schema_matcher=false"
echo "token_evidence_matcher_trainable_parameters=0"
echo "runtime_candidate_subset_before_normalization=true"
echo "ordered_query_evidence_coverage=true"
echo "continuous_relation_hypotheses=true"
echo "exact_sparsity=structural_support_only"
echo "automatic_rerun=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "width_search=false"
echo "batch_size_search=false"
echo "threshold_change=false"
echo "private_identity_gradient=false"

echo "===== P2A: ZERO-GRADIENT FROZEN SEMANTIC AUTHORITY ====="
P2A="$RUN_ROOT/p2a"
mkdir -p "$P2A"
AUTH_RESULT="$P2A/result.json"
python "$ROOT/scripts/eipm/n0/qualify_n0_v02_qsre_frozen_semantic_authority_v3.py"   --plan "$AUTH_PLAN"   --meta-config "$META"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --production-cache "$PROD_CACHE"   --production-schema-cache "$PROD_SCHEMA_CACHE"   --production-curriculum "$PROD_CURRICULUM"   --production-relation-schema "$PROD_SCHEMA_JSON"   --output "$AUTH_RESULT"   --batch-size 32

python - "$AUTH_RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["schema"]=="alice.eipm.n0.qsre-frozen-semantic-authority-result.v3"
assert r["status"]=="PASS_QSRE_FROZEN_SEMANTIC_AUTHORITY", r["status"]
assert r["p2_authorized"] is True
assert r["gradient_performed"] is False
assert r["optimizer"] is False
assert r["matcher_training"] is False
assert r["trainable_authority_parameters"]==0
assert r["relation_keys_used_as_semantic_tokens"] is False
print("PASS_N0_FROZEN_SEMANTIC_AUTHORITY")
print("metrics="+json.dumps(r["metrics"],sort_keys=True))
PY

echo "===== MATERIALIZE ZERO-GRADIENT AUTHORITY CACHES ====="
PROD_AUTH="$P2A/production_authority_v3.pt"
FINAL_AUTH="$P2A/final_authority_v3.pt"
FACTOR_CACHE="$P2A/factor_schema_v3.pt"

python "$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_frozen_authority_cache_v3.py"   --mode production   --rows "$PROD_CURRICULUM"   --relation-schema "$PROD_SCHEMA_JSON"   --schema-cache "$PROD_SCHEMA_CACHE"   --meta-config "$META"   --qualification-result "$AUTH_RESULT"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$PROD_AUTH"   --factor-schema-output "$FACTOR_CACHE"   --batch-size 32

python "$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_frozen_authority_cache_v3.py"   --mode final   --rows "$FINAL_RELATIONAL"   --relation-schema "$FINAL_SCHEMA_JSON"   --schema-cache "$FINAL_SCHEMA_CACHE"   --meta-config "$META"   --qualification-result "$AUTH_RESULT"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$FINAL_AUTH"   --batch-size 32

echo "===== P2: ORDERED OPERATOR AROUND FROZEN AUTHORITY ====="
P2="$RUN_ROOT/p2"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p2_v3.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --failed-p2-result "$FAILED_P2_RESULT"   --authority-result "$AUTH_RESULT"   --authority-cache "$PROD_AUTH"   --factor-schema-cache "$FACTOR_CACHE"   --output-dir "$P2"

python - "$P2/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P2_OPERATOR", r["status"]
assert r["selected"] is not None
assert r["p3_authorized"] is True
assert r["frozen_semantic_authority"] is True
assert r["semantic_authority_gradient"] is False
assert r["schema_matcher_trainable_parameters"]==0
assert r["fixed_factor_class_heads"] is False
assert r["ordered_query_evidence_coverage"] is True
assert r["open_schema_relation_descriptions_used_in_gradient"] is False
print("PASS_N0_FROZEN_AUTHORITY_P2")
PY

echo "===== P3: EXISTING STRUCTURAL BINDER V2 ====="
P3="$RUN_ROOT/p3"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p3_v3.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --factor-schema-cache "$FACTOR_CACHE"   --authority-cache "$PROD_AUTH"   --output-dir "$P3"

python - "$P3/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P3_BINDER", r["status"]
assert r["selected"] is not None
assert r["p4_authorized"] is True
print("PASS_N0_FROZEN_AUTHORITY_P3")
PY

echo "===== P4: FULLY PREDICTED ZERO-GRADIENT QSRE ====="
P4="$RUN_ROOT/p4"
mkdir -p "$P4"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v3.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --factor-schema-cache "$FACTOR_CACHE"   --authority-cache "$PROD_AUTH"   --p3-result "$P3/result.json"   --p3-root "$P3"   --output "$P4/result.json"

echo "===== ORIGINAL FROZEN NATIVE N0 SELF-VALIDATION ====="
FINAL_RESULT="$RUN_ROOT/final-self-validation-result.json"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_final_self_validation_v3.py"   --contract "$FINAL_CONTRACT"   --manifest "$FINAL_MANIFEST"   --freeze-receipt "$FINAL_FREEZE"   --general-corpus "$FINAL_GENERAL"   --relational-corpus "$FINAL_RELATIONAL"   --relational-cache "$FINAL_CACHE"   --final-schema-cache "$FINAL_SCHEMA_CACHE"   --plan "$PLAN"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --factor-schema-cache "$FACTOR_CACHE"   --authority-cache "$PROD_AUTH"   --final-authority-cache "$FINAL_AUTH"   --p3-result "$P3/result.json"   --p3-root "$P3"   --p4-result "$P4/result.json"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED"   --evidence-adapter "$ADAPTER"   --evidence-graph "$GRAPH"   --fusion-checkpoint "$FUSION"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --latent-candidate "$LATENT"   --latent-config "$LATENT_CONFIG"   --output "$FINAL_RESULT"

python - "$FINAL_RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE", r["status"]
assert r["n0_complete"] is True
assert r["n1_authorized"] is True
assert r["external_validator"] is False
assert r["single_frozen_validation"] is True
assert r["threshold_changed_after_results"] is False
assert r["frozen_semantic_authority"] is True
assert r["semantic_authority_gradient"] is False
assert r["private_identity_gradient"] is False
print("===== N0 FROZEN-AUTHORITY OBJECTIVE REACHED =====")
print("n0_complete=true")
print("n1_authorized=true")
print("source_failed_p2s_v2_job=575966")
PY

echo "===== N0 FROZEN SEMANTIC AUTHORITY V3 COMPLETE ====="
date -Is
echo "final_result=$FINAL_RESULT"
