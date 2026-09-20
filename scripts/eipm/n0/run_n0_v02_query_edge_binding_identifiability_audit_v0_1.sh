#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

SETWISE_ROOT="${ALICE_N0_SETWISE_ROUTER_DIR:-$WORKDIR/query-edge-setwise-router-v0.1}"
CACHE="$SETWISE_ROOT/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"

SOURCE_ROOT="$WORKDIR/query-edge-cross-attention-bridge-v0.1"
CURRICULUM="$SOURCE_ROOT/preparation/query_edge_binding_curriculum.jsonl"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"

AUDIT_ROOT="${ALICE_N0_BINDING_AUDIT_DIR:-$WORKDIR/query-edge-binding-identifiability-audit-v0.1}"
OUTPUT="$AUDIT_ROOT/audit.json"

SCRIPT="$ROOT/scripts/eipm/n0/audit_n0_v02_query_edge_binding_identifiability_v0_1.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.32.json"

EXPECTED_CACHE_BYTES=827461729
EXPECTED_CURRICULUM_SHA256="c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_PARENT_ADAPTER_SHA256="50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in   "$CACHE" "$CURRICULUM" "$SEMANTIC_CONFIG"   "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"   "$TOKENIZER_DIR/tokenizer.json" "$PARENT_ADAPTER" "$PARENT_GRAPH"   "$SCRIPT" "$STATE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 2; }
done

if [[ -e "$AUDIT_ROOT" ]]; then
  echo "REFUSING: identifiability-audit output already exists: $AUDIT_ROOT" >&2
  echo "Preserve it as evidence; do not delete to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

CACHE_BYTES="$(stat -c %s "$CACHE")"
echo "cache_bytes=$CACHE_BYTES"
[[ "$CACHE_BYTES" == "$EXPECTED_CACHE_BYTES" ]] || {
  echo "STOP: setwise preflight cache size drift" >&2
  exit 5
}

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

check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.32":
    raise SystemExit("v0.32 state drift")
if "ROUTER_REPAIR_LOOP_HALTED" not in s.get("status",""):
    raise SystemExit("router repair loop is not closed")
e=s["execution_policy"]
if e["exact_cpu_trainer_preflight_required_before_audit"] is not False:
    raise SystemExit("audit was incorrectly made dependent on trainer preflight")
for key in (
    "setwise_replacement_gpu_authorized",
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
print("v0_32_binding_identifiability_audit_gate=PASS")
PY

python -m py_compile "$SCRIPT"
mkdir -p "$AUDIT_ROOT"

echo "===== N0 QUERY-EDGE BINDING IDENTIFIABILITY AUDIT ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "reuses_existing_hidden_cache=true"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "causal_test_open=false"
echo "setwise_training_authorized=false"

python "$SCRIPT"   --cache "$CACHE"   --curriculum "$CURRICULUM"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --parent-adapter "$PARENT_ADAPTER"   --parent-graph "$PARENT_GRAPH"   --output "$OUTPUT"   --max-length 64   --encode-batch-size 32

echo
echo "===== IDENTIFIABILITY AUDIT HASH ====="
sha256sum "$OUTPUT"

echo
echo "===== IDENTIFIABILITY DECISION ====="
python - "$OUTPUT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("status="+str(r.get("status")))
print("localization="+str(r.get("localization")))
for k,v in r["diagnostic_comparisons"].items():
    print(f"{k}={v}")
print("best_all_edge_layer="+str(r["dev_token_level_late_interaction"]["best_all_edge_layer"]))
print("best_same_relation_layer="+str(r["dev_token_level_late_interaction"]["best_same_relation_layer"]))
print("causal_test_evaluated="+str(r.get("causal_test_evaluated")))
print("gradient_performed="+str(r.get("gradient_performed")))
print("gpu_required="+str(r.get("gpu_required")))
print("setwise_training_authorized="+str(r.get("setwise_training_authorized")))
print("next_action="+str(r.get("next_action")))
PY

echo "===== N0 QUERY-EDGE BINDING IDENTIFIABILITY AUDIT COMPLETE ====="
date -Is
