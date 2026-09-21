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
  echo "STOP: N0 closure-pass source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
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
CLOSURE_ROOT="$WORKDIR/qsre-n0-closure-pass-v1"

for root in "$SOURCE" "$RECOVERY" "$FAILED_FINAL"; do
  if [[ ! -d "$root" ]]; then
    echo "STOP: required preserved evidence root missing: $root" >&2
    exit 92
  fi
done
if [[ -e "$CLOSURE_ROOT" ]]; then
  echo "STOP: closure-pass evidence already exists; preserve it: $CLOSURE_ROOT" >&2
  exit 93
fi
mkdir -p "$CLOSURE_ROOT"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_n0_closure_training_plan_v1.json"
MATCHER_PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_closure_matcher_plan_v1.json"
META_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_closure_schema_meta_v1.json"

P0="$SOURCE/p0"
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

for required in \
  "$PLAN" "$MATCHER_PLAN" "$META_SCHEMA" "$PROD_CACHE" "$PROD_SCHEMA_CACHE" \
  "$P1_RESULT" "$FAILED_P2_RESULT" \
  "$FINAL_CONTRACT" "$FINAL_GENERAL" "$FINAL_RELATIONAL" "$FINAL_MANIFEST" \
  "$FINAL_SCHEMA_CACHE" "$FINAL_CACHE" "$FINAL_FREEZE" \
  "$SEMANTIC_CONFIG" "$STRUCTURED_CONFIG" "$FUSION_CONFIG" \
  "$FUSION_RATIFICATION" "$LATENT_CONFIG" "$TOKENIZER/tokenizer.json" \
  "$SEMANTIC" "$STRUCTURED/structured_state.safetensors" "$ADAPTER" "$GRAPH" \
  "$FUSION/cross_context_fusion.safetensors" "$LATENT"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing closure-pass prerequisite: $required" >&2
    exit 94
  fi
done

python - "$P1_RESULT" "$P1_ROOT" "$FAILED_P2_RESULT" "$FINAL_FREEZE" \
  "$ORIGINAL_PRODUCTION_REVISION" "$P1_EXPECTED_SHA" \
  "$PLAN" "$MATCHER_PLAN" "$META_SCHEMA" <<'PY'
import hashlib,json,sys
from pathlib import Path

p1_result=Path(sys.argv[1]); p1_root=Path(sys.argv[2]); failed=Path(sys.argv[3])
freeze_path=Path(sys.argv[4]); original_revision=sys.argv[5]; expected_p1_sha=sys.argv[6]
plan_path=Path(sys.argv[7]); matcher_plan_path=Path(sys.argv[8]); meta_path=Path(sys.argv[9])

p1=json.loads(p1_result.read_text())
p2=json.loads(failed.read_text())
freeze=json.loads(freeze_path.read_text())
plan=json.loads(plan_path.read_text())
matcher_plan=json.loads(matcher_plan_path.read_text())
meta=json.loads(meta_path.read_text())

assert p1["status"]=="PASS_QSRE_PRODUCTION_P1_EXECUTOR"
assert int(p1["selected"]["step"])==50
assert p1["selected"]["checkpoint_sha256"]==expected_p1_sha
p1_path=p1_root/"step-00000050"/"qsre_production_p1.pt"
digest=hashlib.sha256(p1_path.read_bytes()).hexdigest()
assert digest==expected_p1_sha

assert p2["status"]=="FAIL_QSRE_PRODUCTION_P2_OPERATOR_V3", p2["status"]
assert p2["selected"] is None
assert p2["p3_authorized"] is False
assert int(p2["history"][-1]["step"])==1600
assert p2["open_schema_relation_descriptions_used_in_gradient"] is False

assert freeze["status"]=="FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS"
assert freeze["git_revision"]==original_revision
assert freeze["results_observed_before_freeze"] is False
assert freeze["threshold_changes_after_results_forbidden"] is True
assert freeze["private_identity_gradient"] is False

assert plan["schema"]=="alice.eipm.n0.qsre-n0-closure-training-plan.v1"
assert plan["stages"]["P2"]["failure_source"]["magnolia_job"]==575958
assert plan["operator_architecture"]["ordered_query_evidence_coverage"] is True
assert plan["operator_architecture"]["fixed_factor_class_heads"] is False
assert plan["operator_architecture"]["matcher_frozen_in_production_p2"] is True
assert matcher_plan["schema"]=="alice.eipm.n0.qsre-closure-matcher-plan.v1"
assert matcher_plan["source_failure"]["job"]==575958

forbidden=set(meta["forbidden_semantics"])
assert {"CONFLICTS_WITH","EXEMPLIFIES","PREREQUISITE_FOR","PART_OF","ENABLES","PREVENTS"} <= forbidden
train_keys={row["key"] for row in meta["relation_train"]}
hold_keys={row["key"] for row in meta["relation_holdout"]}
assert not (forbidden & train_keys)
assert not (forbidden & hold_keys)
assert not (train_keys & hold_keys)

print("PASS_N0_CLOSURE_SOURCE_LINEAGE")
print("p1_checkpoint_sha256="+digest)
print("source_failed_job=575958")
print("source_failed_status="+p2["status"])
print("original_frozen_final_validation_reused=true")
PY

echo "===== N0 CLOSURE PASS V1 ====="
date -Is
echo "source_revision=$HEAD"
echo "closure_root=$CLOSURE_ROOT"
echo "source_failed_job=575958"
echo "p1_retrain=false"
echo "schema_matcher_relation_identity_parameters=0"
echo "schema_matcher_unseen_auxiliary_gate=true"
echo "production_open_schema_descriptions_in_gradient=false"
echo "ordered_query_evidence_coverage=true"
echo "semantic_factor_schemas=true"
echo "fixed_factor_class_heads=false"
echo "continuous_relation_hypotheses=true"
echo "exact_sparsity=structural_support_only"
echo "automatic_rerun=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "width_search=false"
echo "threshold_change=false"
echo "private_identity_gradient=false"

echo "===== P2S: BROAD SCHEMA GENERALIZATION MATCHER ====="
P2S="$CLOSURE_ROOT/p2s-schema-matcher"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v1.py" \
  --plan "$MATCHER_PLAN" \
  --meta-config "$META_SCHEMA" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC" \
  --tokenizer-dir "$TOKENIZER" \
  --production-cache "$PROD_CACHE" \
  --production-schema-cache "$PROD_SCHEMA_CACHE" \
  --output-dir "$P2S"

python - "$P2S/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_CLOSURE_SCHEMA_MATCHER", r["status"]
assert r["selected"] is not None
assert r["p2_authorized"] is True
assert r["matcher_report"]["relation_identity_parameters"]==0
assert r["production_open_schema_query_labels_used"] is False
assert r["production_open_schema_descriptions_used_in_gradient"] is False
assert r["auxiliary_holdout_relation_descriptions_used_in_gradient"] is False
print("PASS_CLOSURE_SCHEMA_MATCHER")
print("selected_step="+str(r["selected"]["step"]))
print("selected_checkpoint_sha256="+r["selected"]["checkpoint_sha256"])
PY

FACTOR_CACHE="$P2S/factor_schema_cache.pt"

echo "===== P2: ORDERED CONTINUOUS OPERATOR WITH FROZEN MATCHER ====="
P2="$CLOSURE_ROOT/p2"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p2_v3.py" \
  --plan "$PLAN" \
  --prepared-cache "$PROD_CACHE" \
  --schema-cache "$PROD_SCHEMA_CACHE" \
  --p1-result "$P1_RESULT" \
  --p1-root "$P1_ROOT" \
  --failed-p2-result "$FAILED_P2_RESULT" \
  --closure-matcher-result "$P2S/result.json" \
  --closure-matcher-root "$P2S" \
  --factor-schema-cache "$FACTOR_CACHE" \
  --output-dir "$P2"

python - "$P2/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P2_OPERATOR", r["status"]
assert r["selected"] is not None
assert r["p3_authorized"] is True
assert r["closure_schema_matcher_frozen"] is True
assert r["fixed_factor_class_heads"] is False
assert r["semantic_factor_schemas"] is True
assert r["ordered_query_evidence_coverage"] is True
assert r["open_schema_relation_descriptions_used_in_gradient"] is False
print("PASS_N0_CLOSURE_P2")
print("selected_step="+str(r["selected"]["step"]))
print("selected_checkpoint_sha256="+r["selected"]["checkpoint_sha256"])
PY

echo "===== P3: EXISTING STRUCTURAL BINDER V2 ====="
P3="$CLOSURE_ROOT/p3"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p3_v3.py" \
  --plan "$PLAN" \
  --prepared-cache "$PROD_CACHE" \
  --schema-cache "$PROD_SCHEMA_CACHE" \
  --p1-result "$P1_RESULT" \
  --p1-root "$P1_ROOT" \
  --p2-result "$P2/result.json" \
  --p2-root "$P2" \
  --factor-schema-cache "$FACTOR_CACHE" \
  --output-dir "$P3"

python - "$P3/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P3_BINDER", r["status"]
assert r["selected"] is not None
assert r["p4_authorized"] is True
print("PASS_N0_CLOSURE_P3")
print("selected_step="+str(r["selected"]["step"]))
print("selected_checkpoint_sha256="+r["selected"]["checkpoint_sha256"])
PY

echo "===== P4: FULLY PREDICTED ZERO-GRADIENT QSRE ====="
P4="$CLOSURE_ROOT/p4"
mkdir -p "$P4"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v3.py" \
  --plan "$PLAN" \
  --prepared-cache "$PROD_CACHE" \
  --schema-cache "$PROD_SCHEMA_CACHE" \
  --p1-result "$P1_RESULT" \
  --p1-root "$P1_ROOT" \
  --p2-result "$P2/result.json" \
  --p2-root "$P2" \
  --factor-schema-cache "$FACTOR_CACHE" \
  --p3-result "$P3/result.json" \
  --p3-root "$P3" \
  --output "$P4/result.json"

echo "===== ORIGINAL FROZEN NATIVE N0 SELF-VALIDATION ====="
FINAL_RESULT="$CLOSURE_ROOT/final-self-validation-result.json"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_final_self_validation_v3.py" \
  --contract "$FINAL_CONTRACT" \
  --manifest "$FINAL_MANIFEST" \
  --freeze-receipt "$FINAL_FREEZE" \
  --general-corpus "$FINAL_GENERAL" \
  --relational-corpus "$FINAL_RELATIONAL" \
  --relational-cache "$FINAL_CACHE" \
  --final-schema-cache "$FINAL_SCHEMA_CACHE" \
  --plan "$PLAN" \
  --p1-result "$P1_RESULT" \
  --p1-root "$P1_ROOT" \
  --p2-result "$P2/result.json" \
  --p2-root "$P2" \
  --factor-schema-cache "$FACTOR_CACHE" \
  --p3-result "$P3/result.json" \
  --p3-root "$P3" \
  --p4-result "$P4/result.json" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC" \
  --tokenizer-dir "$TOKENIZER" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED" \
  --evidence-adapter "$ADAPTER" \
  --evidence-graph "$GRAPH" \
  --fusion-checkpoint "$FUSION" \
  --fusion-config "$FUSION_CONFIG" \
  --fusion-ratification "$FUSION_RATIFICATION" \
  --latent-candidate "$LATENT" \
  --latent-config "$LATENT_CONFIG" \
  --output "$FINAL_RESULT"

python - "$FINAL_RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE", r["status"]
assert r["n0_complete"] is True
assert r["n1_authorized"] is True
assert r["external_validator"] is False
assert r["single_frozen_validation"] is True
assert r["threshold_changed_after_results"] is False
assert r["private_identity_gradient"] is False
print("===== N0 CLOSURE PASS OBJECTIVE REACHED =====")
print("n0_complete=true")
print("n1_authorized=true")
print("source_failed_job=575958")
PY

echo "===== N0 CLOSURE PASS V1 COMPLETE ====="
date -Is
echo "final_result=$FINAL_RESULT"
