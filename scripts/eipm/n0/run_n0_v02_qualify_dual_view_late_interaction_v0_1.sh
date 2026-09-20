#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

AUDIT="$WORKDIR/query-edge-binding-identifiability-audit-v0.1/audit.json"
CACHE="$WORKDIR/query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"
CURRICULUM="$WORKDIR/query-edge-cross-attention-bridge-v0.1/preparation/query_edge_binding_curriculum.jsonl"
LAYER_MAP="$WORKDIR/relation-conditioned-multilayer-interface-v0.1/design/relation_conditioned_layer_map.json"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"

DUAL_ROOT="${ALICE_N0_DUAL_VIEW_DIR:-$WORKDIR/query-edge-dual-view-late-interaction-v0.1}"
QUAL_DIR="$DUAL_ROOT/qualification"
RESULT="$QUAL_DIR/runtime_contract_qualification.json"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.33.json"
SCRIPT="$ROOT/scripts/eipm/n0/qualify_n0_v02_dual_view_late_interaction_v0_1.py"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in \
  "$AUDIT" "$CACHE" "$CURRICULUM" "$LAYER_MAP" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$TOKENIZER_DIR/tokenizer.json" "$PARENT_ADAPTER" "$PARENT_GRAPH" \
  "$STATE" "$SCRIPT"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 2; }
done

if [[ -e "$DUAL_ROOT" ]]; then
  echo "REFUSING: dual-view qualification root already exists: $DUAL_ROOT" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.33":
    raise SystemExit("v0.33 state drift")
if "DUAL_VIEW_LATE_INTERACTION_RUNTIME_QUALIFICATION_AUTHORIZED_NO_GRADIENT" not in s.get("status",""):
    raise SystemExit("dual-view qualification authorization missing")
q=s["qualification"]
if q["optimizer_created"] is not False or q["gradient_performed"] is not False:
    raise SystemExit("qualification gradient boundary drift")
if q["gpu_required"] is not False or q["training_authorized"] is not False:
    raise SystemExit("qualification self-authorized training")
e=s["execution_policy"]
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
    "replacement_setwise_gpu_authorized",
    "setwise_training_authorized",
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
print("v0_33_dual_view_cpu_qualification_gate=PASS")
PY

python -m py_compile \
  "$ROOT/src/alice_personality/n0/query_edge_dual_view_late_interaction.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_query_edge_dual_view.py" \
  "$SCRIPT"

mkdir -p "$QUAL_DIR"

echo "===== N0 DUAL-VIEW LATE-INTERACTION QUALIFICATION ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "training_authorized=false"
echo "causal_test_open=false"
echo "setwise_replacement_gpu=false"

python "$SCRIPT" \
  --repo-root "$ROOT" \
  --audit-receipt "$AUDIT" \
  --cache "$CACHE" \
  --curriculum "$CURRICULUM" \
  --layer-map "$LAYER_MAP" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --output-dir "$QUAL_DIR" \
  --max-length 64 \
  --encode-batch-size 32 \
  --eval-batch-size 12

echo
echo "===== DUAL-VIEW QUALIFICATION HASH ====="
sha256sum "$RESULT"

echo
echo "===== DUAL-VIEW QUALIFICATION DECISION ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("status="+str(r.get("status")))
print("git_revision="+str(r.get("git_revision")))
print("dev_conditional_edge_all_four_top1_accuracy="+str(r.get("dev_conditional_edge_all_four_top1_accuracy")))
print("dev_conditional_edge_same_relation_top1_accuracy="+str(r.get("dev_conditional_edge_same_relation_top1_accuracy")))
print("edge_binding_direction_reversal_max_delta="+str(r.get("edge_binding_direction_reversal_max_delta")))
print("edge_permutation_equivariance_max_delta="+str(r.get("edge_permutation_equivariance_max_delta")))
print("route_probability_sum_max_error="+str(r.get("route_probability_sum_max_error")))
print("specialist_probability_initialization_max_delta_from_half="+str(r.get("specialist_probability_initialization_max_delta_from_half")))
print("exact_parent_field_weights="+str(r.get("exact_parent_field_weights")))
print("exact_parent_pooled_state="+str(r.get("exact_parent_pooled_state")))
print("parent_parameters_exactly_unchanged="+str(r.get("parent_parameters_exactly_unchanged")))
print("future_trainable_parameter_count="+str(r.get("future_trainable_parameter_count")))
print("gradient_performed="+str(r.get("gradient_performed")))
print("training_authorized="+str(r.get("training_authorized")))
print("next_action="+str(r.get("next_action")))
PY

echo "===== N0 DUAL-VIEW LATE-INTERACTION QUALIFICATION COMPLETE ====="
date -Is
