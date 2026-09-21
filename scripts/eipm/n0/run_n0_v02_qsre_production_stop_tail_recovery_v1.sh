#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
ORIGINAL_REVISION="b32fabe49f206c2d71e17df5197a62b2a37c8c43"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: recovery source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no
  exit 91
fi
if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PY
then
  echo "STOP: one visible CUDA device is required" >&2
  exit 92
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

SOURCE="$WORKDIR/qsre-production-core-v1"
RECOVERY="$WORKDIR/qsre-production-core-v1-stop-tail-recovery-v1"
if [[ ! -d "$SOURCE" ]]; then
  echo "STOP: preserved failed Production N0 root missing: $SOURCE" >&2
  exit 93
fi
if [[ -e "$RECOVERY" ]]; then
  echo "STOP: governed recovery evidence already exists; preserve it: $RECOVERY" >&2
  exit 94
fi
mkdir -p "$RECOVERY"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_production_training_plan_v1.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
FINAL_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_final_self_validation_contract_v1.json"

P0="$SOURCE/p0"
PROD_CACHE="$P0/qsre_production_cache_v1.pt"
PROD_SCHEMA_CACHE="$P0/qsre_production_schema_v1.pt"
P0_RECEIPT="$P0/p0_runtime_receipt.json"

FROZEN="$SOURCE/final-frozen"
FINAL_GENERAL="$FROZEN/general.jsonl"
FINAL_RELATIONAL="$FROZEN/relational.jsonl"
FINAL_MANIFEST="$FROZEN/manifest.json"
FINAL_SCHEMA_CACHE="$FROZEN/final_schema_v1.pt"
FINAL_CACHE="$FROZEN/final_relational_cache_v1.pt"
FINAL_FREEZE="$FROZEN/freeze_receipt.json"

FAILED_P1_ROOT="$SOURCE/p1"
FAILED_P1_RESULT="$FAILED_P1_ROOT/result.json"

TOKENIZER="$WORKDIR/tokenizer-v0.2.1"
SEMANTIC="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
STRUCTURED="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
ADAPTER="$SPECIALIST/evidence_view_adapter.safetensors"
GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
FUSION="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"

for required in   "$PLAN" "$PROD_CACHE" "$PROD_SCHEMA_CACHE" "$P0_RECEIPT"   "$FINAL_CONTRACT" "$FINAL_GENERAL" "$FINAL_RELATIONAL" "$FINAL_MANIFEST"   "$FINAL_SCHEMA_CACHE" "$FINAL_CACHE" "$FINAL_FREEZE"   "$FAILED_P1_RESULT" "$SEMANTIC_CONFIG" "$STRUCTURED_CONFIG"   "$FUSION_CONFIG" "$FUSION_RATIFICATION" "$LATENT_CONFIG"   "$TOKENIZER/tokenizer.json" "$SEMANTIC" "$STRUCTURED/structured_state.safetensors"   "$ADAPTER" "$GRAPH" "$FUSION/cross_context_fusion.safetensors" "$LATENT"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing preserved/governed prerequisite: $required" >&2
    exit 95
  fi
done

python - "$FAILED_P1_RESULT" "$FINAL_FREEZE" "$ORIGINAL_REVISION" <<'PY'
import json,sys
failed=json.load(open(sys.argv[1]))
freeze=json.load(open(sys.argv[2]))
expected=sys.argv[3]
assert failed["status"]=="FAIL_QSRE_PRODUCTION_P1_EXECUTOR", failed["status"]
assert failed["selected"] is None
assert failed["p2_authorized"] is False
assert freeze["status"]=="FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS"
assert freeze["git_revision"]==expected, (freeze["git_revision"],expected)
assert freeze["results_observed_before_freeze"] is False
assert freeze["threshold_changes_after_results_forbidden"] is True
assert freeze["training_authorized"] is False
print("PASS_PRESERVED_FAILED_P1_AND_FROZEN_FINAL_LINEAGE")
PY

echo "===== N0 STOP-TAIL CAUSAL RECOVERY PIPELINE ====="
date -Is
echo "source_revision=$HEAD"
echo "preserved_failed_revision=$ORIGINAL_REVISION"
echo "preserved_failed_root=$SOURCE"
echo "recovery_root=$RECOVERY"
echo "causal_correction=inactive_relation_step_preserves_path_frontier"
echo "automatic_hotfix=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "width_search=false"
echo "private_identity_gradient=false"
echo "frozen_final_validation_reused=true"

echo "===== R1: ZERO-GRADIENT REQUALIFICATION OF PRESERVED P1 CHECKPOINTS ====="
REQUAL="$RECOVERY/p1-requalification.json"
P1="$RECOVERY/p1"

set +e
python "$ROOT/scripts/eipm/n0/requalify_n0_v02_qsre_production_p1_stop_tail_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --failed-p1-result "$FAILED_P1_RESULT"   --failed-p1-root "$FAILED_P1_ROOT"   --receipt "$REQUAL"   --selected-output-dir "$P1"   --device cuda
REQUAL_RC=$?
set -e

if [[ "$REQUAL_RC" -eq 0 ]]; then
  echo "p1_recovery_mode=existing_checkpoint_requalified_without_gradient"
elif [[ "$REQUAL_RC" -eq 10 ]]; then
  echo "p1_recovery_mode=single_corrected_p1_training_same_precommitted_optimizer"
  if [[ -e "$P1" ]]; then
    echo "STOP: P1 output unexpectedly exists after unsuccessful requalification" >&2
    exit 96
  fi
  python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p1_v1.py"     --plan "$PLAN"     --p0-receipt "$P0_RECEIPT"     --prepared-cache "$PROD_CACHE"     --schema-cache "$PROD_SCHEMA_CACHE"     --output-dir "$P1"
else
  echo "STOP: P1 requalification failed as infrastructure/evidence error rc=$REQUAL_RC" >&2
  exit "$REQUAL_RC"
fi

python - "$P1/result.json" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
assert result["status"]=="PASS_QSRE_PRODUCTION_P1_EXECUTOR", result["status"]
assert result["selected"] is not None
assert result["p2_authorized"] is True
print("PASS_CORRECTED_P1_EXECUTOR_FRONTIER")
print("selected_step="+str(result["selected"]["step"]))
print("selected_checkpoint_sha256="+result["selected"]["checkpoint_sha256"])
PY

echo "===== R2: NATURAL-LANGUAGE DYNAMIC OPERATOR ====="
P2="$RECOVERY/p2"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p2_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --output-dir "$P2"

echo "===== R3: ADAPTIVE SUPPORT + LEARNED PATH FOCUS ====="
P3="$RECOVERY/p3"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p3_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --output-dir "$P3"

echo "===== R4: FULLY PREDICTED QSRE; ZERO GRADIENT ====="
P4="$RECOVERY/p4"
mkdir -p "$P4"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --output "$P4/result.json"

echo "===== R5 + FINAL: SAME PRE-FROZEN NATIVE N0 OBJECTIVE SELF-VALIDATION ====="
FINAL_RESULT="$RECOVERY/final-self-validation-result.json"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_final_self_validation_v1.py"   --contract "$FINAL_CONTRACT"   --manifest "$FINAL_MANIFEST"   --freeze-receipt "$FINAL_FREEZE"   --general-corpus "$FINAL_GENERAL"   --relational-corpus "$FINAL_RELATIONAL"   --relational-cache "$FINAL_CACHE"   --final-schema-cache "$FINAL_SCHEMA_CACHE"   --plan "$PLAN"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --p4-result "$P4/result.json"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED"   --evidence-adapter "$ADAPTER"   --evidence-graph "$GRAPH"   --fusion-checkpoint "$FUSION"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --latent-candidate "$LATENT"   --latent-config "$LATENT_CONFIG"   --output "$FINAL_RESULT"

python - "$FINAL_RESULT" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
if result.get("status")!="PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE":
    raise SystemExit("N0 final native objective gate did not pass")
if result.get("n0_complete") is not True or result.get("n1_authorized") is not True:
    raise SystemExit("N0 did not authorize N1")
print("===== N0 FULL-SCALE OBJECTIVE REACHED =====")
print("n0_complete=true")
print("n1_authorized=true")
print("external_validator=false")
print("private_identity_gradient=false")
PY

echo "===== N0 STOP-TAIL CAUSAL RECOVERY COMPLETE ====="
date -Is
echo "final_result=$FINAL_RESULT"
