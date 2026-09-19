#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
INTERFACE_ROOT="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
STUDY_ROOT="${ALICE_N0_MULTILAYER_CAUSAL_STUDY_DIR:-$INTERFACE_ROOT/causal-interface-study-v0.1}"
CAL_ROOT="$STUDY_ROOT/preservation-calibration-v0.1"

PREP="$STUDY_ROOT/preparation/preparation_receipt.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_relation_conditioned_multilayer_preservation_contract_v0_1.json"
CALIBRATOR="$ROOT/scripts/eipm/n0/calibrate_n0_v02_multilayer_preservation_v0_1.py"

PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
ORDINARY_REPLAY="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
ENDPOINT_REPLAY="$WORKDIR/relation-endpoint-repair-v0.2/training/endpoint_repair_semantic_cache.pt"

POLICY="$CAL_ROOT/preservation_policy.json"
RECEIPT="$CAL_ROOT/calibration_receipt.json"

EXPECTED_PREP_SHA256="54dc915f10ee58d994e387ca59ab1180f96bd0e3523780f3e75fb14d6ac30690"
EXPECTED_CONTRACT_SHA256="65b3fa76487fef346ef227dc77ed57da504e2dcf891e32416ce60ef5a6df99dd"
EXPECTED_PARENT_ADAPTER_SHA256="50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_ORDINARY_REPLAY_SHA256="10cf39b6382e66c4eeef6a771d748e3eadedab54e6c0ffc6f900d8bfbdea5534"
EXPECTED_ENDPOINT_REPLAY_SHA256="251c965e2113d66fb56122130bf59f9740d46f9fdf4672322daacdb576c7f02c"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$CAL_ROOT" ]]; then
  echo "REFUSING: preservation calibration output already exists: $CAL_ROOT" >&2
  exit 2
fi

for required in "$PREP" "$CONTRACT" "$CALIBRATOR" "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY" "$ENDPOINT_REPLAY"; do
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
check_sha "$CONTRACT" "$EXPECTED_CONTRACT_SHA256" "preservation_contract_sha256"
check_sha "$PARENT_ADAPTER" "$EXPECTED_PARENT_ADAPTER_SHA256" "parent_adapter_sha256"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "parent_graph_sha256"
check_sha "$ORDINARY_REPLAY" "$EXPECTED_ORDINARY_REPLAY_SHA256" "ordinary_replay_sha256"
check_sha "$ENDPOINT_REPLAY" "$EXPECTED_ENDPOINT_REPLAY_SHA256" "endpoint_replay_sha256"

python -m py_compile "$CALIBRATOR"
mkdir -p "$CAL_ROOT"

python "$CALIBRATOR" \
  --repo-root "$ROOT" \
  --preparation-receipt "$PREP" \
  --preservation-contract "$CONTRACT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --parent-graph "$PARENT_GRAPH" \
  --ordinary-replay-cache "$ORDINARY_REPLAY" \
  --endpoint-replay-cache "$ENDPOINT_REPLAY" \
  --policy-output "$POLICY" \
  --receipt-output "$RECEIPT" \
  --repeat-count 2 \
  --batch-size 64

echo
echo "===== MULTILAYER PRESERVATION CALIBRATION HASHES ====="
sha256sum "$POLICY" "$RECEIPT"

echo
echo "===== MULTILAYER PRESERVATION CALIBRATION DECISION ====="
python - "$RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "git_revision",
    "candidate_supplied",
    "candidate_result_observed",
    "gradient_performed",
    "optimizer_created",
    "gpu_required",
    "interface_training_gate_satisfied",
    "authorization_scope",
    "scale_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "n0_complete",
):
    print(f"{key}={r.get(key)}")
print("metric_policy=")
for item in r["policy"]["metrics"]:
    print(
        f"  {item['path']}: baseline={item['baseline_mean']} "
        f"atol={item['atol']} direction={item['direction']}"
    )
PY
