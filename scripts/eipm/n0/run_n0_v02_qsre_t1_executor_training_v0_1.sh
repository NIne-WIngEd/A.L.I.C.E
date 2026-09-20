#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$PWD}}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

PREPROOT="$WORKDIR/qsre-t1-executor-v0.1/pretraining-v0.2"
PREPARED="$PREPROOT/qsre_t1_real_cache_v0.1.pt"
PREP_RECEIPT="$PREPROOT/qsre_t1_real_cache_v0.1.receipt.json"
CURRICULUM="$PREPROOT/qsre_t1_curriculum_v0.2.jsonl"
CURRICULUM_RECEIPT="$PREPROOT/qsre_t1_curriculum_v0.2.receipt.json"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.48.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t1_training_contract_v0_2.json"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_qsre_t1_executor_v0_1.py"

OUTROOT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1"

EXPECTED_SOURCE_SHA="5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
EXPECTED_CURRICULUM_SHA="155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063"
EXPECTED_CURRICULUM_RECEIPT_SHA="55f7bc40a5d5f1ff5c134e1f4ad00c1788748f317d7c0b118114c111e794b60d"
EXPECTED_PREPARED_SHA="03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564"
EXPECTED_PREP_RECEIPT_SHA="b92ddc17749f03230f3feff1d70942ec2fa5d893638574d8c81ee2957608e9db"
AUTHORIZED_ANCESTOR="31b89390a2d6e6677092a0f4107fa0e9aacbcdc4"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

for required in   "$PREPARED" "$PREP_RECEIPT" "$CURRICULUM" "$CURRICULUM_RECEIPT"   "$STATE" "$CONTRACT" "$TRAINER"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: T1 training output already exists: $OUTROOT" >&2
  echo "One-shot run means preserve it; do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

git merge-base --is-ancestor "$AUTHORIZED_ANCESTOR" HEAD || {
  echo "STOP: current code is not descended from authorized corrected frontier" >&2
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

check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA" "curriculum_sha256"
check_sha "$CURRICULUM_RECEIPT" "$EXPECTED_CURRICULUM_RECEIPT_SHA" "curriculum_receipt_sha256"
check_sha "$PREPARED" "$EXPECTED_PREPARED_SHA" "prepared_cache_sha256"
check_sha "$PREP_RECEIPT" "$EXPECTED_PREP_RECEIPT_SHA" "preparation_receipt_sha256"

python - "$STATE" "$CONTRACT" "$PREP_RECEIPT" <<'PY'
import json,sys

state=json.load(open(sys.argv[1]))
contract=json.load(open(sys.argv[2]))
receipt=json.load(open(sys.argv[3]))

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.48":
    raise SystemExit("v0.48 state drift")

policy=state["execution_policy"]
required_true=(
    "t1_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
)
for key in required_true:
    if policy[key] is not True:
        raise SystemExit(f"v0.48 missing authorization: {key}")

if policy["max_gpu_training_runs"]!=1:
    raise SystemExit("v0.48 GPU run-count drift")

for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "semantic_retraining_authorized",
    "graph_parent_retraining_authorized",
    "learned_operator_training_authorized",
    "learned_support_training_authorized",
    "private_identity_gradient_authorized",
):
    if policy[key] is not False:
        raise SystemExit(f"v0.48 boundary drift: {key}")

if contract.get("schema")!="alice.eipm.n0.qsre-t1-training-contract.v0.2":
    raise SystemExit("T1 v0.2 training contract drift")
if contract.get("gpu_training_authorized") is not True:
    raise SystemExit("T1 v0.2 GPU authorization missing")
if contract.get("max_gpu_runs")!=1:
    raise SystemExit("T1 v0.2 run-count drift")
if contract.get("automatic_rerun") is not False:
    raise SystemExit("T1 v0.2 automatic rerun drift")
if contract.get("automatic_hotfix") is not False:
    raise SystemExit("T1 v0.2 automatic hotfix drift")
if contract.get("causal_test_open") is not False:
    raise SystemExit("T1 v0.2 TEST opened")
if contract.get("frozen_challenge_open") is not False:
    raise SystemExit("T1 v0.2 challenge opened")

if receipt.get("status")!="PASS_QSRE_T1_REAL_CACHE_PREPARATION":
    raise SystemExit("real-cache preparation receipt did not pass")
if receipt.get("source_cache_sha256")!=contract["source_cache_sha256"]:
    raise SystemExit("source-cache lineage mismatch")
if receipt.get("curriculum_sha256")!=contract["expected_curriculum_sha256"]:
    raise SystemExit("curriculum lineage mismatch")
if receipt.get("prepared_cache_sha256")!=contract["expected_prepared_cache_sha256"]:
    raise SystemExit("prepared-cache lineage mismatch")
if receipt.get("paired_representation_identity") is not True:
    raise SystemExit("causal-pair representation identity failed")
if receipt.get("train_dev_representation_pool_isolated") is not True:
    raise SystemExit("TRAIN/DEV representation isolation failed")
if receipt.get("field_support_all_zero") is not True:
    raise SystemExit("direct field support leaked into T1")
if receipt.get("edge_support_membership_only") is not True:
    raise SystemExit("edge support membership contract failed")
if receipt.get("test_present") is not False:
    raise SystemExit("TEST entered prepared cache")
if receipt.get("private_identity_data") is not False:
    raise SystemExit("private identity entered prepared cache")

print("v0_48_qsre_t1_one_shot_gpu_gate=PASS")
PY

echo "===== QSRE T1 ONE-SHOT TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "optimizer=true"
echo "gradient=true"
echo "gpu=true"
echo "max_gpu_runs=1"
echo "automatic_rerun=false"
echo "test_open=false"
echo "challenge_open=false"
echo "learned_operator=false"
echo "learned_support=false"
echo "private_identity_gradient=false"

python "$TRAINER"   --contract "$CONTRACT"   --prepared-cache "$PREPARED"   --output-dir "$OUTROOT"

RESULT="$OUTROOT/result.json"
[[ -f "$RESULT" ]] || {
  echo "STOP: trainer returned without result.json" >&2
  exit 7
}

echo
echo "===== QSRE T1 RESULT HASH ====="
sha256sum "$RESULT"

echo
echo "===== QSRE T1 RESULT SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("status="+str(r.get("status")))
print("selected_checkpoint_step="+str(r.get("selected_checkpoint_step")))
print("prepared_cache_sha256="+str(r.get("prepared_cache_sha256")))
print("curriculum_sha256="+str(r.get("curriculum_sha256")))
print("causal_test_opened="+str(r.get("causal_test_opened")))
print("frozen_challenge_opened="+str(r.get("frozen_challenge_opened")))
print("automatic_rerun_authorized="+str(r.get("automatic_rerun_authorized")))
if r.get("selected_checkpoint_step") is not None:
    key=f"step-{int(r['selected_checkpoint_step']):08d}"
    print("selected_dev="+json.dumps(r["checkpoints"][key]["dev"],sort_keys=True))
else:
    keys=sorted(k for k in r["checkpoints"] if k!="step-00000000")
    if keys:
        key=keys[-1]
        print("final_dev="+json.dumps(r["checkpoints"][key]["dev"],sort_keys=True))
PY

echo "===== QSRE T1 ONE-SHOT TRAINING COMPLETE ====="
date -Is
