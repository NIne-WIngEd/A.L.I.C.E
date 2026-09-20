#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

SOURCE_ROOT="$WORKDIR/query-edge-cross-attention-bridge-v0.1"
SOURCE_PREP="$SOURCE_ROOT/preparation"
SOURCE_TRAIN="$SOURCE_ROOT/training-v0.1"
FAILED_RESULT="$SOURCE_TRAIN/result.json"
CURRICULUM="$SOURCE_PREP/query_edge_binding_curriculum.jsonl"
MANIFEST="$SOURCE_PREP/query_edge_binding_curriculum_manifest.json"
ANCHORS="$SOURCE_PREP/preservation_train_anchors.json"

LAYER_ROOT="$WORKDIR/relation-conditioned-multilayer-interface-v0.1"
MAP="$LAYER_ROOT/design/relation_conditioned_layer_map.json"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"

NEW_ROOT="${ALICE_N0_COMPETITIVE_ROUTER_DIR:-$WORKDIR/query-edge-competitive-router-v0.1}"
QUAL_ROOT="$NEW_ROOT/qualification"
QUAL="$QUAL_ROOT/runtime_contract_qualification.json"

BRIDGE="$ROOT/src/alice_personality/n0/query_edge_competitive_router_bridge.py"
INTEGRATION="$ROOT/src/alice_personality/n0/evidence_graph_query_edge_competitive_router.py"
QUALIFIER="$ROOT/scripts/eipm/n0/qualify_n0_v02_query_edge_competitive_router_v0_1.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.26.json"

EXPECTED_FAILED_RESULT_SHA256="99fc21394bdfbb8e3500bf0efa7c3ce836a2e071b8b11a1c733f73aa99b6fea2"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_CURRICULUM_SHA256="c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
EXPECTED_MANIFEST_SHA256="0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
EXPECTED_ANCHORS_SHA256="f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
EXPECTED_LAYER_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$NEW_ROOT" ]]; then
  echo "REFUSING: competitive-router qualification output already exists: $NEW_ROOT" >&2
  echo "Preserve existing evidence; do not delete to rerun." >&2
  exit 2
fi

for required in   "$FAILED_RESULT" "$CURRICULUM" "$MANIFEST" "$ANCHORS" "$MAP"   "$PARENT_GRAPH" "$BRIDGE" "$INTEGRATION" "$QUALIFIER" "$STATE"
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

check_sha "$FAILED_RESULT" "$EXPECTED_FAILED_RESULT_SHA256" "failed_result_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "curriculum_manifest_sha256"
check_sha "$ANCHORS" "$EXPECTED_ANCHORS_SHA256" "preservation_anchors_sha256"
check_sha "$MAP" "$EXPECTED_LAYER_MAP_SHA256" "layer_map_sha256"

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.26":
    raise SystemExit("v0.26 state drift")
if "COMPETITIVE_EDGE_ROUTER_IMPLEMENTED" not in s.get("status",""):
    raise SystemExit("competitive-router implementation is not current")
e=s["execution_policy"]
if e["cpu_no_gradient_runtime_qualification_authorized"] is not True:
    raise SystemExit("CPU competitive-router qualification not authorized")
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
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
if s.get("n0_complete") is not False:
    raise SystemExit("N0 completion drift")
print("v0_26_competitive_router_cpu_qualification_gate=PASS")
PY

python -m py_compile "$BRIDGE" "$INTEGRATION" "$QUALIFIER"

mkdir -p "$QUAL_ROOT"

python "$QUALIFIER"   --repo-root "$ROOT"   --layer-map "$MAP"   --parent-graph "$PARENT_GRAPH"   --failed-result "$FAILED_RESULT"   --curriculum "$CURRICULUM"   --curriculum-manifest "$MANIFEST"   --preservation-anchors "$ANCHORS"   --output "$QUAL"

echo
echo "===== N0 COMPETITIVE EDGE ROUTER QUALIFICATION HASH ====="
sha256sum "$QUAL"

echo
echo "===== N0 COMPETITIVE EDGE ROUTER QUALIFICATION DECISION ====="
python - "$QUAL" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "git_revision",
    "future_trainable_scope",
    "future_trainable_parameter_count",
    "parent_parameters_exactly_unchanged",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "zero_initialized_source_residual_readout",
    "zero_initialized_target_residual_readout",
    "route_probabilities_sum_to_one",
    "explicit_parent_noop_route",
    "all_active_directed_edges_compete",
    "same_relation_edges_produce_distinct_edge_queries",
    "edge_content_conditions_query_token_attention",
    "route_probability_is_actual_source_contribution_control",
    "route_probability_is_actual_target_contribution_control",
    "proxy_dot_product_router",
    "independent_per_edge_tanh_gate",
    "optimizer_created",
    "gradient_performed",
    "gpu_required",
    "training_authorized",
    "gpu_training_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "scale_authorized",
    "n0_complete",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY
