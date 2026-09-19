#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

OLD_ROOT="$WORKDIR/relation-conditioned-multilayer-interface-v0.1"
OLD_STUDY="$OLD_ROOT/causal-interface-study-v0.1"
FAILED_RESULT="$OLD_STUDY/training-v0.1/result.json"
LAYER_MAP="$OLD_ROOT/design/relation_conditioned_layer_map.json"

NEW_ROOT="${ALICE_N0_QUERY_EDGE_BRIDGE_DIR:-$WORKDIR/query-edge-cross-attention-bridge-v0.1}"
PREP_ROOT="$NEW_ROOT/preparation"
QUAL_ROOT="$NEW_ROOT/qualification"

CURRICULUM="$PREP_ROOT/query_edge_binding_curriculum.jsonl"
CURRICULUM_MANIFEST="$PREP_ROOT/query_edge_binding_curriculum_manifest.json"
ANCHORS="$PREP_ROOT/preservation_train_anchors.json"
QUAL="$QUAL_ROOT/runtime_contract_qualification.json"

ORDINARY_SOURCE="$WORKDIR/evidence-graph-curriculum-v0.1/evidence_graph_curriculum.jsonl"
ENDPOINT_SOURCE="$WORKDIR/relation-endpoint-repair-v0.2/preparation/endpoint_repair_curriculum.jsonl"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"

BUILD_CURRICULUM="$ROOT/scripts/eipm/n0/build_n0_v02_query_edge_binding_curriculum_v0_1.py"
BUILD_ANCHORS="$ROOT/scripts/eipm/n0/build_n0_v02_query_edge_preservation_train_anchors_v0_1.py"
QUALIFIER="$ROOT/scripts/eipm/n0/qualify_n0_v02_query_edge_cross_attention_bridge_v0_1.py"
BRIDGE="$ROOT/src/alice_personality/n0/query_edge_cross_attention_bridge.py"
INTEGRATION="$ROOT/src/alice_personality/n0/evidence_graph_query_edge_bridge.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.23.json"

EXPECTED_FAILED_RESULT_SHA256="6f2f3fe6ddab764d947a7c63fe94ef24d5cd77e3cc68821c8678fdb8f2ea36bd"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_LAYER_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$NEW_ROOT" ]]; then
  echo "REFUSING: query-edge study output already exists: $NEW_ROOT" >&2
  exit 2
fi

for required in   "$FAILED_RESULT" "$LAYER_MAP" "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE"   "$PARENT_GRAPH" "$BUILD_CURRICULUM" "$BUILD_ANCHORS" "$QUALIFIER"   "$BRIDGE" "$INTEGRATION" "$STATE"
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
check_sha "$LAYER_MAP" "$EXPECTED_LAYER_MAP_SHA256" "layer_map_sha256"

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.23":
    raise SystemExit("v0.23 state schema drift")
if "CPU_NO_GRADIENT_RUNTIME_QUALIFICATION_AUTHORIZED" not in s.get("status",""):
    raise SystemExit("query-edge runtime qualification is not current")
if s["source_failure"]["rerun_authorized"] is not False:
    raise SystemExit("failed multilayer experiment was reopened")
e=s["execution_policy"]
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "scale_authorized",
    "private_identity_gradient_authorized",
    "production_promotion_authorized",
):
    if e[key] is not False:
        raise SystemExit(f"execution authorization drift: {key}")
if e["cpu_no_gradient_runtime_qualification_authorized"] is not True:
    raise SystemExit("CPU no-gradient qualification is not authorized")
print("v0_23_query_edge_qualification_gate=PASS")
PY

python -m py_compile   "$BUILD_CURRICULUM" "$BUILD_ANCHORS" "$QUALIFIER" "$BRIDGE" "$INTEGRATION"

mkdir -p "$PREP_ROOT" "$QUAL_ROOT"

python "$BUILD_CURRICULUM"   --output "$CURRICULUM"   --manifest "$CURRICULUM_MANIFEST"

python "$BUILD_ANCHORS"   --ordinary-source "$ORDINARY_SOURCE"   --endpoint-source "$ENDPOINT_SOURCE"   --output "$ANCHORS"

python "$QUALIFIER"   --repo-root "$ROOT"   --layer-map "$LAYER_MAP"   --parent-graph "$PARENT_GRAPH"   --failed-result "$FAILED_RESULT"   --curriculum "$CURRICULUM"   --curriculum-manifest "$CURRICULUM_MANIFEST"   --preservation-anchors "$ANCHORS"   --output "$QUAL"

echo
echo "===== N0 QUERY-EDGE BRIDGE PREPARATION HASHES ====="
sha256sum "$CURRICULUM" "$CURRICULUM_MANIFEST" "$ANCHORS" "$QUAL"

echo
echo "===== N0 QUERY-EDGE BRIDGE QUALIFICATION DECISION ====="
python - "$QUAL" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "git_revision",
    "parent_parameters_exactly_unchanged",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "zero_initialized_specialist_gate",
    "same_relation_edges_produce_distinct_edge_queries",
    "edge_content_conditions_query_token_attention",
    "forced_antisymmetry",
    "future_trainable_scope",
    "future_trainable_parameter_count",
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
