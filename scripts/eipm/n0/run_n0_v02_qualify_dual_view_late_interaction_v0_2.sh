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
FAILED_QUAL_DIR="$DUAL_ROOT/qualification"
QUAL_DIR="$DUAL_ROOT/qualification-v0.2"
RESULT="$QUAL_DIR/runtime_contract_qualification.json"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.34.json"
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

[[ -d "$FAILED_QUAL_DIR" ]] || {
  echo "STOP: expected preserved attempt-1 qualification directory is missing: $FAILED_QUAL_DIR" >&2
  exit 3
}

if [[ -e "$RESULT" || -d "$QUAL_DIR" ]]; then
  echo "REFUSING: qualification-v0.2 already exists" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 4
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.34":
    raise SystemExit("v0.34 state drift")
a=s["attempt_1"]
if a["model_evidence"] is not False:
    raise SystemExit("attempt 1 was incorrectly promoted to model evidence")
if a["terminal_error"]!="direction-invariance pair coverage drift: 144":
    raise SystemExit("attempt 1 failure record drift")
c=s["correction"]
if c["expected_unique_pairs"]!=72:
    raise SystemExit("unique reversal pair expectation drift")
if c["expected_dev_rows"]!=144:
    raise SystemExit("DEV row-count drift")
if c["science_criterion_unchanged"] is not True:
    raise SystemExit("science criterion was changed")
if c["architecture_unchanged"] is not True:
    raise SystemExit("architecture changed during qualification recovery")
q=s["qualification_recovery"]
if q["attempts_remaining"]!=1:
    raise SystemExit("requalification attempt-count drift")
for key in ("optimizer_created","gradient_performed","gpu_required","training_authorized"):
    if q[key] is not False:
        raise SystemExit(f"qualification recovery boundary drift: {key}")
e=s["execution_policy"]
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
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
print("v0_34_dual_view_cpu_requalification_gate=PASS")
PY

python -m py_compile "$SCRIPT"
mkdir -p "$QUAL_DIR"

echo "===== N0 DUAL-VIEW LATE-INTERACTION REQUALIFICATION v0.2 ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "attempt_1_preserved=true"
echo "single_correction=deduplicate_unique_quad_role_direction_pairs"
echo "expected_unique_direction_pairs=72"
echo "architecture_changed=false"
echo "thresholds_changed=false"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "training_authorized=false"
echo "causal_test_open=false"

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
echo "===== DUAL-VIEW REQUALIFICATION HASH ====="
sha256sum "$RESULT"

echo
echo "===== DUAL-VIEW REQUALIFICATION DECISION ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "git_revision",
    "dev_conditional_edge_all_four_top1_accuracy",
    "dev_conditional_edge_same_relation_top1_accuracy",
    "dev_query_role_all_edge_accuracy",
    "dev_relation_family_all_edge_accuracy",
    "edge_binding_direction_reversal_max_delta",
    "edge_permutation_equivariance_max_delta",
    "route_probability_sum_max_error",
    "specialist_probability_initialization_max_delta_from_half",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "parent_parameters_exactly_unchanged",
    "future_trainable_parameter_count",
    "gradient_performed",
    "training_authorized",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY

echo "===== N0 DUAL-VIEW LATE-INTERACTION REQUALIFICATION COMPLETE ====="
date -Is
