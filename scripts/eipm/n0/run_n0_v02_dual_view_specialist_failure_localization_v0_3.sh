#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

DUAL_ROOT="$WORKDIR/query-edge-dual-view-late-interaction-v0.1"
TRAIN_ROOT="$DUAL_ROOT/training-v0.1"
RESULT="$TRAIN_ROOT/result.json"
FIELD_CACHE="$TRAIN_ROOT/dual_view_field_token_cache.pt"
CAUSAL_CACHE="$WORKDIR/query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"
CURRICULUM="$WORKDIR/query-edge-cross-attention-bridge-v0.1/preparation/query_edge_binding_curriculum.jsonl"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
MAP="$WORKDIR/relation-conditioned-multilayer-interface-v0.1/design/relation_conditioned_layer_map.json"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"

V01_ROOT="$DUAL_ROOT/failure-localization-v0.1"
V02_ROOT="$DUAL_ROOT/failure-localization-v0.2"
AUDIT_ROOT="$DUAL_ROOT/failure-localization-v0.3"
AUDIT="$AUDIT_ROOT/audit.json"

SCRIPT="$ROOT/scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_3.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.38.json"

EXPECTED_RESULT_SHA256="1a38c37a41de68bea0bb9bc897b86d62cbc457b8b93189c2d16731110c1694ec"
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
  "$RESULT" "$FIELD_CACHE" "$CAUSAL_CACHE" "$CURRICULUM" \
  "$TOKENIZER_DIR/tokenizer.json" "$MAP" "$PARENT_ADAPTER" "$PARENT_GRAPH" \
  "$SCRIPT" "$STATE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 2; }
done

for preserved in "$V01_ROOT" "$V02_ROOT"; do
  [[ -d "$preserved" ]] || {
    echo "STOP: expected preserved interrupted localization directory missing: $preserved" >&2
    exit 3
  }
done

if [[ -e "$AUDIT_ROOT" ]]; then
  echo "REFUSING: P100 localization output already exists: $AUDIT_ROOT" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 4
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 5
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
    exit 6
  }
}

check_sha "$RESULT" "$EXPECTED_RESULT_SHA256" "failed_training_result_sha256"
check_sha "$CAUSAL_CACHE" "$EXPECTED_CAUSAL_CACHE_SHA256" "causal_cache_sha256"
check_sha "$FIELD_CACHE" "$EXPECTED_FIELD_CACHE_SHA256" "field_token_cache_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MAP" "$EXPECTED_MAP_SHA256" "layer_map_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"

python - "$STATE" "$RESULT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
r=json.load(open(sys.argv[2]))

if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.38":
    raise SystemExit("v0.38 state drift")
for key in ("v0_1","v0_2"):
    if s["prior_localization_attempts"][key]["result_file_produced"] is not False:
        raise SystemExit(f"{key} unexpectedly produced model evidence")
    if s["prior_localization_attempts"][key]["directory_must_be_preserved"] is not True:
        raise SystemExit(f"{key} preservation rule drift")
v=s["v0_3"]
if v["scientific_question_changed"] is not False:
    raise SystemExit("scientific question changed")
if v["counterfactual_modes_changed"] is not False:
    raise SystemExit("counterfactual modes changed")
if v["gpu_inference_authorized"] is not True:
    raise SystemExit("P100 inference not authorized")
for key in (
    "gpu_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "training_authorized",
    "test_opening_authorized",
    "challenge_opening_authorized",
):
    if v[key] is not False:
        raise SystemExit(f"training/heldout boundary drift: {key}")
if v["max_gpu_jobs"]!=1:
    raise SystemExit("GPU job-count drift")
if r.get("status")!="FAIL_DUAL_VIEW_SPECIALIST_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX":
    raise SystemExit("source failed result drift")
print("v0_38_p100_inference_localization_gate=PASS")
PY

python -m py_compile "$SCRIPT"
mkdir -p "$AUDIT_ROOT"

echo "===== N0 P100 INFERENCE FAILURE LOCALIZATION v0.3 ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "prior_v0_1_preserved=true"
echo "prior_v0_2_preserved=true"
echo "semantic_backbone_executed=false"
echo "preservation_semantic_reencoding=false"
echo "causal_hidden_states_reused_from_cache=true"
echo "causal_field_tokens_reused_from_cache=true"
echo "gpu_inference=true"
echo "optimizer=false"
echo "gradient=false"
echo "training=false"
echo "test_open=false"
echo "challenge_open=false"

python "$SCRIPT" \
  --repo-root "$ROOT" \
  --training-result "$RESULT" \
  --training-root "$TRAIN_ROOT" \
  --causal-cache "$CAUSAL_CACHE" \
  --field-token-cache "$FIELD_CACHE" \
  --curriculum "$CURRICULUM" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --layer-map "$MAP" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --output "$AUDIT" \
  --max-length 64 \
  --eval-batch-size 24

echo
echo "===== P100 LOCALIZATION HASH ====="
sha256sum "$AUDIT"

echo
echo "===== P100 LOCALIZATION SUMMARY ====="
python - "$AUDIT" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
print("status="+str(a.get("status")))
s=a["summary"]
for key in (
    "best_causal_activation_step",
    "best_causal_activation_top1",
    "best_forced_hard_row_step",
    "best_forced_hard_row_accuracy",
    "best_forced_hard_quad_accuracy",
    "best_residual_role_step",
    "best_residual_role_accuracy",
):
    print(f"{key}={s.get(key)}")
for step,data in a["checkpoint_results"].items():
    c=data["counterfactual"]
    print(
        "step="+step
        +" actual_row="+str(c["modes"]["actual_activation_soft_binding"]["row_accuracy"])
        +" forced_soft_row="+str(c["modes"]["forced_specialist_soft_binding"]["row_accuracy"])
        +" actual_hard_row="+str(c["modes"]["actual_activation_hard_top1_binding"]["row_accuracy"])
        +" forced_hard_row="+str(c["modes"]["forced_specialist_hard_top1_binding"]["row_accuracy"])
        +" role_acc="+str(c["relevant_edge_residual_proposal_role_accuracy"])
    )
print("gpu_inference_used="+str(a["gpu_inference_used"]))
print("gradient_performed="+str(a["gradient_performed"]))
print("training_authorized="+str(a["training_authorized"]))
print("next_action="+str(a["next_action"]))
PY

echo "===== N0 P100 INFERENCE FAILURE LOCALIZATION COMPLETE ====="
date -Is
