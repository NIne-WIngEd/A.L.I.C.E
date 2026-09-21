#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
ORIGINAL_PRODUCTION_REVISION="b32fabe49f206c2d71e17df5197a62b2a37c8c43"
P1_EXPECTED_SHA="91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: operator-v2 recovery source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
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
FAILED_RECOVERY="$WORKDIR/qsre-production-core-v1-stop-tail-recovery-v1"
RECOVERY="$WORKDIR/qsre-production-core-v1-operator-v2-recovery-v1"

if [[ ! -d "$SOURCE" || ! -d "$FAILED_RECOVERY" ]]; then
  echo "STOP: preserved Production evidence roots are missing" >&2
  exit 92
fi
if [[ -e "$RECOVERY" ]]; then
  echo "STOP: operator-v2 recovery evidence already exists; preserve it: $RECOVERY" >&2
  exit 93
fi
mkdir -p "$RECOVERY"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_production_training_plan_v2.json"
RECOVERY_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_p2_v2_recovery_contract_v1.json"

P0="$SOURCE/p0"
PROD_CACHE="$P0/qsre_production_cache_v1.pt"
PROD_SCHEMA_CACHE="$P0/qsre_production_schema_v1.pt"

P1_ROOT="$FAILED_RECOVERY/p1"
P1_RESULT="$P1_ROOT/result.json"
FAILED_P2_ROOT="$FAILED_RECOVERY/p2"
FAILED_P2_RESULT="$FAILED_P2_ROOT/result.json"

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

for required in   "$PLAN" "$RECOVERY_CONTRACT" "$PROD_CACHE" "$PROD_SCHEMA_CACHE"   "$P1_RESULT" "$FAILED_P2_RESULT"   "$FINAL_CONTRACT" "$FINAL_GENERAL" "$FINAL_RELATIONAL" "$FINAL_MANIFEST"   "$FINAL_SCHEMA_CACHE" "$FINAL_CACHE" "$FINAL_FREEZE"   "$SEMANTIC_CONFIG" "$STRUCTURED_CONFIG" "$FUSION_CONFIG"   "$FUSION_RATIFICATION" "$LATENT_CONFIG" "$TOKENIZER/tokenizer.json"   "$SEMANTIC" "$STRUCTURED/structured_state.safetensors" "$ADAPTER" "$GRAPH"   "$FUSION/cross_context_fusion.safetensors" "$LATENT"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing preserved/governed prerequisite: $required" >&2
    exit 94
  fi
done

python - "$P1_RESULT" "$P1_ROOT" "$FAILED_P2_RESULT" "$FINAL_FREEZE" "$ORIGINAL_PRODUCTION_REVISION" "$P1_EXPECTED_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path

p1_result=Path(sys.argv[1])
p1_root=Path(sys.argv[2])
p2_result=Path(sys.argv[3])
freeze_path=Path(sys.argv[4])
original_revision=sys.argv[5]
expected_p1_sha=sys.argv[6]

p1=json.loads(p1_result.read_text())
p2=json.loads(p2_result.read_text())
freeze=json.loads(freeze_path.read_text())

assert p1["status"]=="PASS_QSRE_PRODUCTION_P1_EXECUTOR", p1["status"]
assert p1["selected"] is not None
assert int(p1["selected"]["step"])==50, p1["selected"]["step"]
assert p1["selected"]["checkpoint_sha256"]==expected_p1_sha
p1_path=p1_root/"step-00000050"/"qsre_production_p1.pt"
digest=hashlib.sha256(p1_path.read_bytes()).hexdigest()
assert digest==expected_p1_sha, (digest,expected_p1_sha)
assert p1["p2_authorized"] is True

assert p2["status"]=="FAIL_QSRE_PRODUCTION_P2_OPERATOR", p2["status"]
assert p2["selected"] is None
assert p2["p3_authorized"] is False
assert p2["operator_gradient"] is True

assert freeze["status"]=="FROZEN_N0_FINAL_SELF_VALIDATION_BEFORE_P1_RESULTS"
assert freeze["git_revision"]==original_revision
assert freeze["results_observed_before_freeze"] is False
assert freeze["threshold_changes_after_results_forbidden"] is True
assert freeze["training_authorized"] is False
assert freeze["private_identity_data"] is False
assert freeze["private_identity_gradient"] is False

print("PASS_P2_V2_RECOVERY_SOURCE_LINEAGE")
print("p1_checkpoint_sha256="+digest)
print("failed_p2_status="+p2["status"])
print("frozen_final_validation_reused=true")
PY

echo "===== N0 OPEN-SCHEMA OPERATOR/BINDER V2 CAUSAL RECOVERY ====="
date -Is
echo "source_revision=$HEAD"
echo "original_production_root=$SOURCE"
echo "preserved_p2_v1_failure_root=$FAILED_RECOVERY"
echo "recovery_root=$RECOVERY"
echo "p1_retrain=false"
echo "p2_v1_retrain=false"
echo "p2_v2_gpu_runs_authorized=1"
echo "p3_v2_gpu_runs_authorized=1_if_p2_passes"
echo "automatic_rerun=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "width_search=false"
echo "open_schema_train_query_labels=false"
echo "private_identity_gradient=false"
echo "frozen_final_validation_reused=true"

echo "===== V2-P2: OPEN-SCHEMA FACTORIZED OPERATOR ====="
P2="$RECOVERY/p2"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p2_v2.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --failed-p2-result "$FAILED_P2_RESULT"   --output-dir "$P2"

python - "$P2/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P2_OPERATOR", r["status"]
assert r["selected"] is not None
assert r["p3_authorized"] is True
assert r["open_schema_training_query_labels_used"] is False
assert r["open_schema_runtime_candidates_used"] is True
print("PASS_OPERATOR_V2_FRONTIER")
print("selected_step="+str(r["selected"]["step"]))
print("selected_checkpoint_sha256="+r["selected"]["checkpoint_sha256"])
PY

echo "===== V2-P3: OPEN-SCHEMA ADAPTIVE SUPPORT + LEARNED FOCUS ====="
P3="$RECOVERY/p3"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p3_v2.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --output-dir "$P3"

python - "$P3/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P3_BINDER", r["status"]
assert r["selected"] is not None
assert r["p4_authorized"] is True
assert r["open_schema_training_query_labels_used"] is False
assert r["full_runtime_schema_candidates_used"] is True
assert r["balanced_support_and_focus_supervision"] is True
print("PASS_BINDER_V2_FRONTIER")
print("selected_step="+str(r["selected"]["step"]))
print("selected_checkpoint_sha256="+r["selected"]["checkpoint_sha256"])
PY

echo "===== V2-P4: FULLY PREDICTED QSRE; ZERO GRADIENT ====="
P4="$RECOVERY/p4"
mkdir -p "$P4"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v2.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --output "$P4/result.json"

echo "===== FINAL: SAME PRE-FROZEN NATIVE N0 OBJECTIVE SELF-VALIDATION ====="
FINAL_RESULT="$RECOVERY/final-self-validation-result.json"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_final_self_validation_v2.py"   --contract "$FINAL_CONTRACT"   --manifest "$FINAL_MANIFEST"   --freeze-receipt "$FINAL_FREEZE"   --general-corpus "$FINAL_GENERAL"   --relational-corpus "$FINAL_RELATIONAL"   --relational-cache "$FINAL_CACHE"   --final-schema-cache "$FINAL_SCHEMA_CACHE"   --plan "$PLAN"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --p4-result "$P4/result.json"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED"   --evidence-adapter "$ADAPTER"   --evidence-graph "$GRAPH"   --fusion-checkpoint "$FUSION"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --latent-candidate "$LATENT"   --latent-config "$LATENT_CONFIG"   --output "$FINAL_RESULT"

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
print("===== N0 FULL-SCALE OBJECTIVE REACHED =====")
print("n0_complete=true")
print("n1_authorized=true")
print("external_validator=false")
print("private_identity_gradient=false")
PY

echo "===== N0 OPERATOR/BINDER V2 RECOVERY COMPLETE ====="
date -Is
echo "final_result=$FINAL_RESULT"
