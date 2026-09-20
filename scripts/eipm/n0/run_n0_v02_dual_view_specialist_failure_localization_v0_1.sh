#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

DUAL_ROOT="$WORKDIR/query-edge-dual-view-late-interaction-v0.1"
TRAIN_ROOT="$DUAL_ROOT/training-v0.1"
RESULT="$TRAIN_ROOT/result.json"
FIELD_CACHE="$TRAIN_ROOT/dual_view_field_token_cache.pt"
QUAL="$DUAL_ROOT/qualification-v0.2/runtime_contract_qualification.json"

CAUSAL_CACHE="$WORKDIR/query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"
SOURCE_ROOT="$WORKDIR/query-edge-cross-attention-bridge-v0.1"
CURRICULUM="$SOURCE_ROOT/preparation/query_edge_binding_curriculum.jsonl"

LAYER_ROOT="$WORKDIR/relation-conditioned-multilayer-interface-v0.1"
MAP="$LAYER_ROOT/design/relation_conditioned_layer_map.json"

ORDINARY_SOURCE="$WORKDIR/evidence-graph-curriculum-v0.1/evidence_graph_curriculum.jsonl"
ENDPOINT_SOURCE="$WORKDIR/relation-endpoint-repair-v0.2/preparation/endpoint_repair_curriculum.jsonl"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
ORDINARY_REPLAY="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
ENDPOINT_REPLAY="$WORKDIR/relation-endpoint-repair-v0.2/training/endpoint_repair_semantic_cache.pt"

AUDIT_ROOT="$DUAL_ROOT/failure-localization-v0.1"
AUDIT="$AUDIT_ROOT/audit.json"
SCRIPT="$ROOT/scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_1.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.36.json"

EXPECTED_RESULT_SHA256="1a38c37a41de68bea0bb9bc897b86d62cbc457b8b93189c2d16731110c1694ec"
EXPECTED_QUAL_SHA256="714f06d4939abad2b816a7e3ffb5c5fa10112129ef0b520fd838f2d182286278"
EXPECTED_CAUSAL_CACHE_SHA256="5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
EXPECTED_FIELD_CACHE_SHA256="8bdf72561bafd2bbb88b56e9202570401f76cd7a0bcf48aa0abd5d8e2400ad5c"
EXPECTED_CURRICULUM_SHA256="c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
EXPECTED_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_PARENT_ADAPTER_SHA256="50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in \
  "$RESULT" "$FIELD_CACHE" "$QUAL" "$CAUSAL_CACHE" "$CURRICULUM" "$MAP" \
  "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE" "$SEMANTIC_CONFIG" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$TOKENIZER_DIR/tokenizer.json" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY" \
  "$SCRIPT" "$STATE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 2; }
done

if [[ -e "$AUDIT_ROOT" ]]; then
  echo "REFUSING: failure-localization output already exists: $AUDIT_ROOT" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  echo "$label=$actual"
  [[ "$actual" == "$expected" ]] || {
    echo "STOP: $label hash drift" >&2
    exit 5
  }
}

check_sha "$RESULT" "$EXPECTED_RESULT_SHA256" "failed_training_result_sha256"
check_sha "$QUAL" "$EXPECTED_QUAL_SHA256" "qualification_receipt_sha256"
check_sha "$CAUSAL_CACHE" "$EXPECTED_CAUSAL_CACHE_SHA256" "causal_cache_sha256"
check_sha "$FIELD_CACHE" "$EXPECTED_FIELD_CACHE_SHA256" "field_token_cache_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MAP" "$EXPECTED_MAP_SHA256" "layer_map_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"

python - "$STATE" "$RESULT" <<'PY'
import json,sys
state=json.load(open(sys.argv[1]))
result=json.load(open(sys.argv[2]))

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.36":
    raise SystemExit("v0.36 state drift")
if "COUNTERFACTUAL_ACTIVATION_MIXTURE_RESIDUAL_LOCALIZATION_AUTHORIZED_NO_GRADIENT" not in state.get("status",""):
    raise SystemExit("localization authorization missing")
if result.get("status")!="FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX":
    raise SystemExit("source result is not the governed failure")
if result.get("selected_checkpoint") is not None:
    raise SystemExit("source failure selected a checkpoint")
if result.get("eligible_checkpoint_keys")!=[]:
    raise SystemExit("source failure has eligible checkpoints")
a=state["localization_audit"]
if a["gpu_required"] is not False:
    raise SystemExit("localization unexpectedly needs GPU")
for key in ("optimizer_authorized","gradient_authorized","training_authorized","test_opening_authorized","challenge_opening_authorized"):
    if a[key] is not False:
        raise SystemExit(f"localization boundary drift: {key}")
e=state["execution_policy"]
for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_authorized",
    "gpu_training_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "scale_authorized",
    "semantic_retraining_authorized",
    "graph_parent_retraining_authorized",
    "private_identity_gradient_authorized",
    "production_promotion_authorized",
):
    if e[key] is not False:
        raise SystemExit(f"premature authorization: {key}")
print("v0_36_failure_localization_gate=PASS")
PY

python -m py_compile "$SCRIPT"
mkdir -p "$AUDIT_ROOT"

echo "===== N0 DUAL-VIEW SPECIALIST FAILURE LOCALIZATION ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "source_failed_result_sha256=$EXPECTED_RESULT_SHA256"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "training=false"
echo "test_open=false"
echo "challenge_open=false"
echo "counterfactual_actual_soft=true"
echo "counterfactual_forced_specialist_soft=true"
echo "counterfactual_actual_hard_top1=true"
echo "counterfactual_forced_specialist_hard_top1=true"
echo "hard_top1_uses_model_prediction_not_target_label=true"

python "$SCRIPT" \
  --repo-root "$ROOT" \
  --training-result "$RESULT" \
  --training-root "$TRAIN_ROOT" \
  --qualification-receipt "$QUAL" \
  --causal-cache "$CAUSAL_CACHE" \
  --field-token-cache "$FIELD_CACHE" \
  --curriculum "$CURRICULUM" \
  --ordinary-source-curriculum "$ORDINARY_SOURCE" \
  --endpoint-source-curriculum "$ENDPOINT_SOURCE" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --layer-map "$MAP" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --ordinary-replay-cache "$ORDINARY_REPLAY" \
  --endpoint-replay-cache "$ENDPOINT_REPLAY" \
  --output "$AUDIT" \
  --max-length 64 \
  --encode-batch-size 24 \
  --eval-batch-size 24

echo
echo "===== FAILURE LOCALIZATION HASH ====="
sha256sum "$AUDIT"

echo
echo "===== FAILURE LOCALIZATION SUMMARY ====="
python - "$AUDIT" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
print("status="+str(a.get("status")))
s=a["summary"]
for key in (
    "best_gate_auc_step",
    "best_gate_auc",
    "best_gate_strict_rank_separation",
    "best_forced_hard_row_step",
    "best_forced_hard_row_accuracy",
    "best_forced_hard_quad_accuracy",
    "best_proposal_role_step",
    "best_relevant_edge_residual_proposal_role_accuracy",
):
    print(f"{key}={s.get(key)}")
for step,data in a["checkpoint_results"].items():
    g=data["gate_separation"]
    c=data["counterfactual"]
    d=data["diagnostic_deltas"]
    print(
        "step="+step
        +" gate_auc="+str(g["causal_vs_preservation_auc"])
        +" strict_gate_separation="+str(g["strict_rank_separation"])
        +" actual_row="+str(c["modes"]["actual_activation_soft_binding"]["row_accuracy"])
        +" forced_soft_row="+str(c["modes"]["forced_specialist_soft_binding"]["row_accuracy"])
        +" actual_hard_row="+str(c["modes"]["actual_activation_hard_top1_binding"]["row_accuracy"])
        +" forced_hard_row="+str(c["modes"]["forced_specialist_hard_top1_binding"]["row_accuracy"])
        +" proposal_role_acc="+str(c["relevant_edge_residual_proposal_role_accuracy"])
        +" hard_gain="+str(d["hard_top1_forced_activation_row_gain_vs_soft_forced"])
    )
print("training_authorized="+str(a["training_authorized"]))
print("next_action="+str(a["next_action"]))
PY

echo "===== N0 DUAL-VIEW SPECIALIST FAILURE LOCALIZATION COMPLETE ====="
date -Is
