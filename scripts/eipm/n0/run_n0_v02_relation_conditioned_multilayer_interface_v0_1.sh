#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
INTERFACE_ROOT="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
STUDY_ROOT="${ALICE_N0_MULTILAYER_CAUSAL_STUDY_DIR:-$INTERFACE_ROOT/causal-interface-study-v0.1}"
PREP_ROOT="$STUDY_ROOT/preparation"
CAL_ROOT="$STUDY_ROOT/preservation-calibration-v0.1"
TRAIN_ROOT="$STUDY_ROOT/training-v0.1"

PREP="$PREP_ROOT/preparation_receipt.json"
CAL="$CAL_ROOT/calibration_receipt.json"
MAP="$INTERFACE_ROOT/design/relation_conditioned_layer_map.json"
CURRICULUM="$PREP_ROOT/relation_conditioned_multilayer_interface_curriculum.jsonl"
MANIFEST="$PREP_ROOT/relation_conditioned_multilayer_interface_curriculum_manifest.json"

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

TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py"
INTEGRATION="$ROOT/src/alice_personality/n0/evidence_graph_multilayer_interface.py"
INTERFACE="$ROOT/src/alice_personality/n0/relation_conditioned_multilayer_query.py"

EXPECTED_PREP_SHA256="54dc915f10ee58d994e387ca59ab1180f96bd0e3523780f3e75fb14d6ac30690"
EXPECTED_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_CURRICULUM_SHA256="706fb2944ae3f95f0bf93c93763c57d54a76be5068f665d9a9be1f2425dadaab"
EXPECTED_MANIFEST_SHA256="6ca37c884b9a75d3fed9ef0e1b8a19704c03d246bb2d32f33cc538546072ba72"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$TRAIN_ROOT" ]]; then
  echo "REFUSING: multilayer training output already exists: $TRAIN_ROOT" >&2
  exit 2
fi

for required in \
  "$PREP" "$CAL" "$MAP" "$CURRICULUM" "$MANIFEST" \
  "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY" \
  "$TRAINER" "$INTEGRATION" "$INTERFACE"
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

check_sha "$PREP" "$EXPECTED_PREP_SHA256" "preparation_receipt_sha256"
check_sha "$MAP" "$EXPECTED_MAP_SHA256" "layer_map_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "curriculum_manifest_sha256"

python - "$CAL" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get("schema")!="alice.eipm.n0.v02-multilayer-preservation-calibration.v0.1":
    raise SystemExit("preservation calibration schema drift")
if r.get("status")!="PASS_PARENT_ONLY_PRESERVATION_CALIBRATION":
    raise SystemExit("preservation calibration did not pass")
if r.get("candidate_result_observed") is not False:
    raise SystemExit("candidate contaminated preservation calibration")
if r.get("interface_training_gate_satisfied") is not True:
    raise SystemExit("one bounded interface experiment is not authorized")
if r.get("authorization_scope")!="one bounded relation-conditioned multilayer interface experiment only":
    raise SystemExit("training authorization scope drift")
for key in (
    "semantic_backbone_trainable",
    "structured_state_trainable",
    "parent_adapter_trainable",
    "parent_graph_trainable",
    "scale_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "private_identity_gradient",
    "production_promotion_authorized",
    "n0_complete",
):
    if r.get(key) is not False:
        raise SystemExit(f"calibration governance drift: {key}")
print("preservation_calibration_training_gate=PASS")
PY

python -m py_compile "$TRAINER" "$INTEGRATION" "$INTERFACE"

echo "===== N0 RELATION-CONDITIONED MULTILAYER CAUSAL INTERFACE TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "one_bounded_experiment=true"
echo "trainable_scope=multilayer_query_interface_plus_zero_init_signed_endpoint_read"
echo "semantic_parent_trainable=false"
echo "structured_parent_trainable=false"
echo "parent_adapter_trainable=false"
echo "parent_graph_trainable=false"
echo "preservation_rows_used_for_gradient=false"
echo "test_split_opening_authorized=false"
echo "frozen_challenge_rerun_authorized=false"
echo "scale_authorized=false"

python "$TRAINER" \
  --repo-root "$ROOT" \
  --preparation-receipt "$PREP" \
  --calibration-receipt "$CAL" \
  --layer-map "$MAP" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$MANIFEST" \
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
  --encode-batch-size 32 \
  --quad-batch-size 4 \
  --eval-batch-size 32 \
  --max-steps 160 \
  --save-every 40 \
  --learning-rate 0.0002 \
  --weight-decay 0.02 \
  --warmup-steps 10 \
  --margin 0.20 \
  --margin-weight 0.25 \
  --seed 20260919

echo
echo "===== MULTILAYER CAUSAL RESULT ====="
sha256sum "$TRAIN_ROOT/result.json"
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "eligible_checkpoint_keys",
    "selected_checkpoint",
    "selected_candidate_sha256",
    "test_split_evaluated",
    "test_split_opened_after_training",
    "frozen_challenge_evaluated",
    "preservation_rows_used_for_gradient",
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

echo "===== N0 RELATION-CONDITIONED MULTILAYER CAUSAL INTERFACE TRAINING COMPLETE ====="
date -Is
