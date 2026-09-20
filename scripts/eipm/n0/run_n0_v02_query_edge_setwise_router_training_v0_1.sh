#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

SETWISE_ROOT="${ALICE_N0_SETWISE_ROUTER_DIR:-$WORKDIR/query-edge-setwise-router-v0.1}"
QUAL="$SETWISE_ROOT/qualification/runtime_contract_qualification.json"
TRAIN_ROOT="$SETWISE_ROOT/training-v0.1"

FAILED_ROOT="$WORKDIR/query-edge-competitive-router-v0.1/training-v0.1"
FAILED_RESULT="$FAILED_ROOT/result.json"

QUERY_EDGE_ROOT="$WORKDIR/query-edge-cross-attention-bridge-v0.1"
SOURCE_PREP="$QUERY_EDGE_ROOT/preparation"
CURRICULUM="$SOURCE_PREP/query_edge_binding_curriculum.jsonl"
MANIFEST="$SOURCE_PREP/query_edge_binding_curriculum_manifest.json"
ANCHORS="$SOURCE_PREP/preservation_train_anchors.json"

LAYER_ROOT="$WORKDIR/relation-conditioned-multilayer-interface-v0.1"
MAP="$LAYER_ROOT/design/relation_conditioned_layer_map.json"
CAL_ROOT="$LAYER_ROOT/causal-interface-study-v0.1/preservation-calibration-v0.1"
CAL="$CAL_ROOT/calibration_receipt.json"
POLICY="$CAL_ROOT/preservation_policy.json"

ORDINARY_SOURCE="$WORKDIR/evidence-graph-curriculum-v0.1/evidence_graph_curriculum.jsonl"
ENDPOINT_SOURCE="$WORKDIR/relation-endpoint-repair-v0.2/preparation/endpoint_repair_curriculum.jsonl"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
ORDINARY_REPLAY="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
ENDPOINT_REPLAY="$WORKDIR/relation-endpoint-repair-v0.2/training/endpoint_repair_semantic_cache.pt"

TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_query_edge_setwise_router_v0_1.py"
BRIDGE="$ROOT/src/alice_personality/n0/query_edge_setwise_router_bridge.py"
INTEGRATION="$ROOT/src/alice_personality/n0/evidence_graph_query_edge_setwise_router.py"
DECISION="$ROOT/configs/eipm/n0/n0_v02_setwise_edge_router_training_decision_v0_1.json"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.30.json"

EXPECTED_QUAL_SHA256="4b1591e1eec5c450a872264a1bdf1130def5a29e443c8c35e974ae05f2861666"
EXPECTED_FAILED_RESULT_SHA256="08bef659779c2e6fcfb0d60ef209d9f6c7f03b0198bcf370e097e278f5436328"
EXPECTED_CURRICULUM_SHA256="c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
EXPECTED_MANIFEST_SHA256="0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
EXPECTED_ANCHORS_SHA256="f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
EXPECTED_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_CAL_SHA256="d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b"
EXPECTED_POLICY_SHA256="fb88b9580e38a648101277815f82c3ba60c19f3bfd16dbe3029ded40b04aea56"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_PARENT_ADAPTER_SHA256="50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
EXPECTED_ORDINARY_SOURCE_SHA256="73c590699e873fef3d148fd77886e460c06f4107b21bbe9fb6e343ec0e3020eb"
EXPECTED_ENDPOINT_SOURCE_SHA256="b9a42b632c1bc1d198bdee434d433073c6e9408579bb661d8b5343ec7ea21e97"
EXPECTED_QUAL_SOURCE_REVISION="7a8bc6b63a9f32095ba3fe4ed27db5f62cf5e061"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$TRAIN_ROOT" ]]; then
  echo "REFUSING: setwise training output already exists: $TRAIN_ROOT" >&2
  echo "Preserve it as experiment evidence; do not delete to rerun." >&2
  exit 2
fi

for required in \
  "$QUAL" "$FAILED_RESULT" "$CURRICULUM" "$MANIFEST" "$ANCHORS" "$MAP" "$CAL" "$POLICY" \
  "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY" \
  "$TRAINER" "$BRIDGE" "$INTEGRATION" "$DECISION" "$STATE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

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

check_sha "$QUAL" "$EXPECTED_QUAL_SHA256" "qualification_receipt_sha256"
check_sha "$FAILED_RESULT" "$EXPECTED_FAILED_RESULT_SHA256" "source_failed_result_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "curriculum_manifest_sha256"
check_sha "$ANCHORS" "$EXPECTED_ANCHORS_SHA256" "preservation_train_anchors_sha256"
check_sha "$MAP" "$EXPECTED_MAP_SHA256" "layer_map_sha256"
check_sha "$CAL" "$EXPECTED_CAL_SHA256" "calibration_receipt_sha256"
check_sha "$POLICY" "$EXPECTED_POLICY_SHA256" "preservation_policy_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"
check_sha "$ORDINARY_SOURCE" "$EXPECTED_ORDINARY_SOURCE_SHA256" "ordinary_source_sha256"
check_sha "$ENDPOINT_SOURCE" "$EXPECTED_ENDPOINT_SOURCE_SHA256" "endpoint_source_sha256"

python - "$QUAL" "$DECISION" "$STATE" "$EXPECTED_QUAL_SOURCE_REVISION" <<'PY'
import json,sys
qual=json.load(open(sys.argv[1]))
decision=json.load(open(sys.argv[2]))
state=json.load(open(sys.argv[3]))
expected_revision=sys.argv[4]

if qual.get("status")!="PASS_SETWISE_QUERY_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT":
    raise SystemExit("setwise-router qualification did not pass")
if qual.get("git_revision")!=expected_revision:
    raise SystemExit("setwise-router qualification source revision drift")
for key in (
    "parent_parameters_exactly_unchanged",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "route_probabilities_sum_to_one",
    "explicit_parent_noop_route",
    "all_active_directed_edges_jointly_contextualized",
    "edge_set_permutation_equivariance",
    "nonlocal_competitor_context",
    "route_probability_is_actual_source_contribution_control",
    "route_probability_is_actual_target_contribution_control",
):
    if qual.get(key) is not True:
        raise SystemExit(f"qualification invariant drift: {key}")
if qual.get("independent_per_edge_route_scoring") is not False:
    raise SystemExit("independent route scoring returned")
if qual.get("proxy_dot_product_router") is not False:
    raise SystemExit("proxy router returned")
if qual.get("gradient_performed") is not False:
    raise SystemExit("qualification unexpectedly performed gradient")
if qual.get("training_authorized") is not False:
    raise SystemExit("qualification receipt self-authorized training")

if decision.get("decision")!="AUTHORIZE_ONE_BOUNDED_SETWISE_EDGE_ROUTER_EXPERIMENT_AFTER_RUNTIME_QUALIFICATION":
    raise SystemExit("training decision drift")
if decision["operating_budget"]["max_training_runs"]!=1:
    raise SystemExit("training run-count drift")
if decision["experimental_isolation"]["same_optimizer_hyperparameters"] is not True:
    raise SystemExit("optimizer isolation drift")
if decision["experimental_isolation"]["same_training_step_budget"] is not True:
    raise SystemExit("step-budget isolation drift")
if decision["experimental_isolation"]["single_causal_change"]!="replace independent edge scalar routing with setwise contextual edge arbitration":
    raise SystemExit("causal-change isolation drift")
obj=decision["training_objective"]
if obj["loss_weight_change_from_failed_competitive_run"] is not False:
    raise SystemExit("loss weights changed during architecture isolation")
for key in (
    "proxy_routing_loss",
    "irrelevant_edge_gate_penalty",
    "load_balancing_loss",
    "gradient_surgery",
    "route_probability_detachment",
):
    if obj[key] is not False:
        raise SystemExit(f"unreviewed mechanism entered objective: {key}")

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.30":
    raise SystemExit("v0.30 state drift")
e=state["execution_policy"]
if e["gradient_execution_authorized_now"] is not True:
    raise SystemExit("gradient is not authorized")
if e["gpu_execution_authorized_now"] is not True:
    raise SystemExit("GPU is not authorized")
if e["gpu_training_runs_remaining"]!=1:
    raise SystemExit("GPU run-count drift")
for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_authorized",
    "automatic_heldout_opening_authorized",
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
if state.get("n0_complete") is not False:
    raise SystemExit("N0 completion drift")

print("setwise_router_one_p100_training_gate=PASS")
PY

mkdir "$TRAIN_ROOT"

python -m py_compile "$TRAINER" "$BRIDGE" "$INTEGRATION"

echo "===== N0 SETWISE EDGE ROUTER TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "one_bounded_experiment=true"
echo "single_causal_change=setwise_contextual_edge_arbitration"
echo "trainable_scope=setwise_query_edge_bridge_only"
echo "causal_actual_runtime_route_supervision=true"
echo "preservation_parent_noop_route_supervision=true"
echo "parent_distillation=true"
echo "loss_weights_changed_from_failed_competitive_run=false"
echo "optimizer_hyperparameters_changed_from_failed_competitive_run=false"
echo "proxy_routing_loss=false"
echo "irrelevant_edge_gate_penalty=false"
echo "load_balancing_loss=false"
echo "gradient_surgery=false"
echo "preservation_dev_gradient=false"
echo "causal_test_open=false"
echo "frozen_challenge_open=false"
echo "parent_graph_trainable=false"
echo "semantic_parent_trainable=false"
echo "scale_authorized=false"

python "$TRAINER" \
  --repo-root "$ROOT" \
  --qualification-receipt "$QUAL" \
  --failed-result "$FAILED_RESULT" \
  --layer-map "$MAP" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$MANIFEST" \
  --preservation-anchors "$ANCHORS" \
  --calibration-receipt "$CAL" \
  --ordinary-source-curriculum "$ORDINARY_SOURCE" \
  --endpoint-source-curriculum "$ENDPOINT_SOURCE" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --ordinary-replay-cache "$ORDINARY_REPLAY" \
  --endpoint-replay-cache "$ENDPOINT_REPLAY" \
  --output-dir "$TRAIN_ROOT" \
  --max-length 64 \
  --encode-batch-size 24 \
  --quad-batch-size 4 \
  --preservation-batch-size 16 \
  --eval-batch-size 24 \
  --max-steps 200 \
  --save-every 40 \
  --learning-rate 0.00015 \
  --weight-decay 0.02 \
  --warmup-steps 12 \
  --margin 0.20 \
  --margin-weight 0.25 \
  --causal-route-weight 1.0 \
  --preservation-weight 1.0 \
  --noop-route-weight 1.0 \
  --seed 20260920

echo
echo "===== SETWISE EDGE ROUTER TRAINING RESULT ====="
sha256sum "$TRAIN_ROOT/result.json"
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "eligible_checkpoint_keys",
    "selected_checkpoint",
    "selected_candidate_sha256",
    "preservation_train_anchors_used_for_gradient",
    "preservation_dev_rows_used_for_gradient",
    "causal_test_split_evaluated",
    "causal_test_split_opened_after_training",
    "frozen_challenge_evaluated",
    "parent_graph_parameters_exactly_unchanged",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "scale_authorized",
    "automatic_rerun_or_hotfix_authorized",
    "n0_complete",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY

echo "===== N0 SETWISE EDGE ROUTER TRAINING COMPLETE ====="
date -Is
