#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

SETWISE_ROOT="${ALICE_N0_SETWISE_ROUTER_DIR:-$WORKDIR/query-edge-setwise-router-v0.1}"
QUAL="$SETWISE_ROOT/qualification/runtime_contract_qualification.json"
FAILED_GPU_DIR="$SETWISE_ROOT/training-v0.1"
PREFLIGHT_ROOT="${ALICE_N0_SETWISE_PREFLIGHT_DIR:-$SETWISE_ROOT/training-preflight-v0.2}"

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
RECOVERY="$ROOT/configs/eipm/n0/n0_v02_setwise_training_boundary_recovery_v0_1.json"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.31.json"

EXPECTED_QUAL_SHA256="4b1591e1eec5c450a872264a1bdf1130def5a29e443c8c35e974ae05f2861666"
EXPECTED_FAILED_RESULT_SHA256="08bef659779c2e6fcfb0d60ef209d9f6c7f03b0198bcf370e097e278f5436328"
EXPECTED_CURRICULUM_SHA256="c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
EXPECTED_MANIFEST_SHA256="0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
EXPECTED_ANCHORS_SHA256="f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
EXPECTED_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_CAL_SHA256="d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_PARENT_ADAPTER_SHA256="50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

[[ -d "$FAILED_GPU_DIR" ]] || {
  echo "STOP: expected preserved 575908 output directory missing: $FAILED_GPU_DIR" >&2
  exit 2
}

if [[ -e "$PREFLIGHT_ROOT" ]]; then
  echo "REFUSING: exact setwise training preflight output already exists: $PREFLIGHT_ROOT" >&2
  echo "Preserve it; do not delete to rerun." >&2
  exit 3
fi

for required in \
  "$QUAL" "$FAILED_RESULT" "$CURRICULUM" "$MANIFEST" "$ANCHORS" "$MAP" "$CAL" \
  "$ORDINARY_SOURCE" "$ENDPOINT_SOURCE" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY" \
  "$TRAINER" "$RECOVERY" "$STATE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 4; }
done

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

check_sha "$QUAL" "$EXPECTED_QUAL_SHA256" "qualification_receipt_sha256"
check_sha "$FAILED_RESULT" "$EXPECTED_FAILED_RESULT_SHA256" "source_failed_result_sha256"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA256" "curriculum_sha256"
check_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA256" "curriculum_manifest_sha256"
check_sha "$ANCHORS" "$EXPECTED_ANCHORS_SHA256" "preservation_train_anchors_sha256"
check_sha "$MAP" "$EXPECTED_MAP_SHA256" "layer_map_sha256"
check_sha "$CAL" "$EXPECTED_CAL_SHA256" "calibration_receipt_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"

python - "$QUAL" "$RECOVERY" "$STATE" "$TRAINER" <<'PY'
import json,sys
from pathlib import Path

qual=json.load(open(sys.argv[1]))
recovery=json.load(open(sys.argv[2]))
state=json.load(open(sys.argv[3]))
trainer=Path(sys.argv[4]).read_text()

if qual.get("status")!="PASS_SETWISE_QUERY_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT":
    raise SystemExit("setwise qualification status drift")
if recovery["failed_execution"]["magnolia_job_id"]!=575908:
    raise SystemExit("575908 recovery binding drift")
if recovery["classification"]["model_failure"] is not False:
    raise SystemExit("575908 incorrectly classified as model failure")
if recovery["classification"]["gradient_performed"] is not False:
    raise SystemExit("575908 recovery incorrectly records gradient")
if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.31":
    raise SystemExit("v0.31 state drift")
e=state["execution_policy"]
if e["cpu_exact_training_preflight_authorized"] is not True:
    raise SystemExit("exact CPU preflight not authorized")
for key in (
    "gradient_execution_authorized_now",
    "gpu_execution_authorized_now",
    "replacement_gpu_run_authorized",
    "automatic_rerun_authorized",
    "automatic_hotfix_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "scale_authorized",
    "private_identity_gradient_authorized",
    "production_promotion_authorized",
):
    if e[key] is not False:
        raise SystemExit(f"premature authorization: {key}")

if '"PASS_SETWISE_QUERY_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT"' not in trainer:
    raise SystemExit("trainer does not bind exact setwise qualification status")
if "PASS_COMPETITIVE_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT" in trainer:
    raise SystemExit("stale competitive PASS qualification guard remains")
if 'p.add_argument("--preflight-only", action="store_true")' not in trainer:
    raise SystemExit("trainer-native preflight mode missing")
if 'if args.preflight_only:' not in trainer:
    raise SystemExit("preflight branch missing")
if '"optimizer_created": False' not in trainer:
    raise SystemExit("preflight optimizer evidence missing")
if '"gradient_performed": False' not in trainer:
    raise SystemExit("preflight gradient evidence missing")
if trainer.index("if args.preflight_only:") > trainer.index("optimizer = torch.optim.AdamW("):
    raise SystemExit("preflight branch occurs after optimizer creation")

print("exact_setwise_trainer_preflight_static_gate=PASS")
PY

mkdir "$PREFLIGHT_ROOT"

python -m py_compile "$TRAINER"

echo "===== N0 SETWISE EXACT TRAINER CPU PREFLIGHT ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "device=cpu"
echo "uses_exact_training_executable=true"
echo "optimizer_authorized=false"
echo "gradient_authorized=false"
echo "gpu_authorized=false"
echo "replacement_gpu_run_authorized=false"

python "$TRAINER" \
  --preflight-only \
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
  --output-dir "$PREFLIGHT_ROOT" \
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

RECEIPT="$PREFLIGHT_ROOT/preflight_receipt.json"
[[ -f "$RECEIPT" ]] || {
  echo "STOP: exact trainer preflight receipt missing" >&2
  exit 7
}

echo
echo "===== EXACT TRAINER PREFLIGHT RECEIPT ====="
sha256sum "$RECEIPT"
python - "$RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if r.get("status")!="PASS_SETWISE_TRAINING_EXACT_PREFLIGHT_NO_GRADIENT":
    raise SystemExit("exact preflight did not pass")
for key in (
    "parent_parameters_exactly_unchanged",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
):
    if r.get(key) is not True:
        raise SystemExit(f"preflight invariant failed: {key}")
for key in ("optimizer_created","gradient_performed"):
    if r.get(key) is not False:
        raise SystemExit(f"preflight unexpectedly changed execution state: {key}")
if r.get("training_steps_executed") != 0:
    raise SystemExit("preflight unexpectedly executed training steps")
for key in (
    "status",
    "git_revision",
    "device",
    "trainable_scope",
    "trainable_parameters",
    "parent_parameters_exactly_unchanged",
    "exact_parent_field_weights",
    "exact_parent_pooled_state",
    "optimizer_created",
    "gradient_performed",
    "training_steps_executed",
    "causal_test_split_evaluated",
    "frozen_challenge_evaluated",
    "scale_authorized",
    "n0_complete",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY

echo "===== N0 SETWISE EXACT TRAINER CPU PREFLIGHT COMPLETE ====="
date -Is
