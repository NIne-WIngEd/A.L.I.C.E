#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

DUAL_ROOT="${ALICE_N0_DUAL_VIEW_DIR:-$WORKDIR/query-edge-dual-view-late-interaction-v0.1}"
QUAL="$DUAL_ROOT/qualification-v0.2/runtime_contract_qualification.json"
TRAIN_ROOT="$DUAL_ROOT/training-v0.1"

CAUSAL_CACHE="$WORKDIR/query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"
SOURCE_ROOT="$WORKDIR/query-edge-cross-attention-bridge-v0.1"
CURRICULUM="$SOURCE_ROOT/preparation/query_edge_binding_curriculum.jsonl"
MANIFEST="$SOURCE_ROOT/preparation/query_edge_binding_curriculum_manifest.json"
ANCHORS="$SOURCE_ROOT/preparation/preservation_train_anchors.json"

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

TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_dual_view_specialist_v0_1.py"
DECISION="$ROOT/configs/eipm/n0/n0_v02_dual_view_training_decision_v0_1.json"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.35.json"

EXPECTED_QUAL_SHA256="714f06d4939abad2b816a7e3ffb5c5fa10112129ef0b520fd838f2d182286278"
EXPECTED_QUAL_SOURCE_REVISION="edcee7dd7a53bac158793b805a29cfaa0c1b11b0"
EXPECTED_CAUSAL_CACHE_SHA256="5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
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

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

if [[ -e "$TRAIN_ROOT" ]]; then
  echo "REFUSING: dual-view training output already exists: $TRAIN_ROOT" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 2
fi

for required in \
  "$QUAL" "$CAUSAL_CACHE" "$CURRICULUM" "$MANIFEST" "$ANCHORS" "$MAP" \
  "$CAL" "$POLICY" "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY" \
  "$TRAINER" "$DECISION" "$STATE"
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
check_sha "$CAUSAL_CACHE" "$EXPECTED_CAUSAL_CACHE_SHA256" "causal_cache_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "curriculum_manifest_sha256"
check_sha "$ANCHORS" "$EXPECTED_ANCHORS_SHA256" "preservation_anchors_sha256"
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

if qual.get("status")!="PASS_DUAL_VIEW_LATE_INTERACTION_NO_GRADIENT_RUNTIME_CONTRACT":
    raise SystemExit("dual-view qualification did not pass")
if qual.get("git_revision")!=expected_revision:
    raise SystemExit("dual-view qualification source revision drift")
if float(qual["dev_conditional_edge_all_four_top1_accuracy"])!=1.0:
    raise SystemExit("all-edge binding is not perfect")
if float(qual["dev_conditional_edge_same_relation_top1_accuracy"])!=1.0:
    raise SystemExit("same-relation binding is not perfect")
for key in (
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "parent_parameters_exactly_unchanged",
):
    if qual.get(key) is not True:
        raise SystemExit(f"qualification invariant drift: {key}")
if qual.get("training_authorized") is not False:
    raise SystemExit("qualification receipt self-authorized training")

if decision.get("decision")!="AUTHORIZE_ONE_BOUNDED_DUAL_VIEW_SPECIALIST_EXPERIMENT_AFTER_RUNTIME_QUALIFICATION":
    raise SystemExit("training decision drift")
if decision["operating_budget"]["max_training_runs"]!=1:
    raise SystemExit("run-count drift")
if decision["trainable_scope"]["binding_logit_scale"] is not False:
    raise SystemExit("binding scale unexpectedly trainable")
obj=decision["training_objective"]
for key in (
    "conditional_edge_binding_loss",
    "route_load_balancing_loss",
    "proxy_routing_loss",
    "irrelevant_edge_penalty",
    "gradient_surgery",
):
    if obj[key] is not False:
        raise SystemExit(f"closed objective mechanism entered training: {key}")

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.35":
    raise SystemExit("v0.35 state drift")
e=state["execution_policy"]
if e["gradient_authorized"] is not True or e["gpu_training_authorized"] is not True:
    raise SystemExit("training is not authorized")
if e["gpu_runs_remaining"]!=1:
    raise SystemExit("GPU run-count drift")
for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_authorized",
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
print("dual_view_one_p100_training_gate=PASS")
PY

mkdir "$TRAIN_ROOT"
python -m py_compile "$TRAINER"

echo "===== N0 DUAL-VIEW SPECIALIST TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "one_bounded_experiment=true"
echo "edge_identity_frozen=true"
echo "binding_logit_scale_trainable=false"
echo "causal_specialist_activation_supervision=true"
echo "directional_residual_field_selection_supervision=true"
echo "preservation_parent_distillation=true"
echo "preservation_noop_activation_supervision=true"
echo "conditional_edge_binding_loss=false"
echo "proxy_routing_loss=false"
echo "gradient_surgery=false"
echo "causal_test_open=false"
echo "frozen_challenge_open=false"
echo "parent_graph_trainable=false"
echo "semantic_parent_trainable=false"

python "$TRAINER" \
  --repo-root "$ROOT" \
  --qualification-receipt "$QUAL" \
  --causal-cache "$CAUSAL_CACHE" \
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
  --layer-map "$MAP" \
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
  --activation-weight 1.0 \
  --preservation-weight 1.0 \
  --noop-weight 1.0 \
  --seed 20260920

echo
echo "===== DUAL-VIEW SPECIALIST TRAINING RESULT ====="
sha256sum "$TRAIN_ROOT/result.json"
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "eligible_checkpoint_keys",
    "selected_checkpoint",
    "selected_candidate_sha256",
    "binding_logit_scale_frozen",
    "edge_identity_training_loss",
    "causal_test_split_evaluated",
    "causal_test_split_opened_after_training",
    "frozen_challenge_evaluated",
    "parent_graph_parameters_exactly_unchanged",
    "automatic_rerun_or_hotfix_authorized",
    "heldout_opening_authorized",
    "scale_authorized",
    "n0_complete",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY

echo "===== N0 DUAL-VIEW SPECIALIST TRAINING COMPLETE ====="
date -Is
