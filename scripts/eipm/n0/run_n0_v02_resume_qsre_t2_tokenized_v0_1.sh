#!/usr/bin/env bash
set -euo pipefail

: "${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT must be set}"
: "${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR must be set}"

export PYTHONPATH="$ALICE_N0_REPO_ROOT/src:$ALICE_N0_REPO_ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

T1_ROOT="$ALICE_N0_WORKDIR/qsre-t1-executor-v0.1"
T1_PREPARED="$T1_ROOT/pretraining-v0.2/qsre_t1_real_cache_v0.1.pt"
T1_CHECKPOINT="$T1_ROOT/training-v0.1/step-00000050/qsre_t1_executor.pt"

T2_ROOT="$ALICE_N0_WORKDIR/qsre-t2-operator-v0.1"
CURRICULUM="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.jsonl"
CURRICULUM_RECEIPT="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.receipt.json"
PREPARED="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.pt"
PREPARED_RECEIPT="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.receipt.json"

TOKENIZER_DIR="$ALICE_N0_WORKDIR/tokenizer-v0.2.1"
PREP="$ALICE_N0_REPO_ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_t2_tokenized_v0_1.py"
STATE="$ALICE_N0_REPO_ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.51.json"

EXPECTED_CURRICULUM_SHA="40e2d608eaa59ad13e2f9b37a2f3da5d6c20101a28ab76434c11be3d36cb5590"
EXPECTED_T1_PREPARED_SHA="03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564"
EXPECTED_T1_CHECKPOINT_SHA="483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"

cd "$ALICE_N0_REPO_ROOT"

for required in   "$CURRICULUM" "$CURRICULUM_RECEIPT"   "$T1_PREPARED" "$T1_CHECKPOINT"   "$TOKENIZER_DIR/tokenizer.json" "$PREP" "$STATE"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$PREPARED" || -e "$PREPARED_RECEIPT" ]]; then
  echo "STOP: T2 tokenized preparation evidence already exists." >&2
  echo "Do not overwrite or delete it." >&2
  ls -l "$PREPARED" "$PREPARED_RECEIPT" 2>/dev/null || true
  exit 3
fi

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
    echo "STOP: $label drift" >&2
    exit 5
  }
}

check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA" "t2_curriculum_sha256"
check_sha "$T1_PREPARED" "$EXPECTED_T1_PREPARED_SHA" "t1_prepared_cache_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_CHECKPOINT_SHA" "t1_checkpoint_sha256"

python3 - "$CURRICULUM_RECEIPT" "$STATE" "$EXPECTED_CURRICULUM_SHA" <<'PY'
import json,sys
receipt=json.load(open(sys.argv[1]))
state=json.load(open(sys.argv[2]))
expected=sys.argv[3]

if receipt.get("schema")!="alice.eipm.n0.qsre-t2-curriculum-result.v0.1":
    raise SystemExit("T2 curriculum receipt schema drift")
if receipt.get("status")!="PASS_QSRE_T2_PUBLIC_OPERATOR_CURRICULUM_CONTRACT":
    raise SystemExit("T2 curriculum receipt did not pass")
if receipt.get("sha256")!=expected:
    raise SystemExit("T2 curriculum receipt hash mismatch")
if receipt.get("rows")!=1008:
    raise SystemExit("T2 curriculum row-count drift")
if receipt.get("split_counts")!={"train":720,"dev":288}:
    raise SystemExit("T2 curriculum split-count drift")
if receipt.get("test_split_present") is not False:
    raise SystemExit("T2 curriculum unexpectedly contains TEST")
if receipt.get("private_identity_data") is not False:
    raise SystemExit("T2 curriculum unexpectedly contains private identity data")
if receipt.get("gradient_authorized") is not False:
    raise SystemExit("T2 curriculum receipt unexpectedly authorizes gradient")
if receipt.get("gpu_training_authorized") is not False:
    raise SystemExit("T2 curriculum receipt unexpectedly authorizes GPU training")

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.51":
    raise SystemExit("v0.51 state drift")
policy=state["execution_policy"]
if policy["cpu_tokenized_preparation_resume_authorized"] is not True:
    raise SystemExit("CPU resume is not authorized")
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
    "automatic_rerun_authorized",
    "automatic_hotfix_chain_authorized",
    "t1_rerun_authorized",
    "causal_test_opening_authorized",
    "frozen_challenge_opening_authorized",
    "private_identity_gradient_authorized",
    "production_promotion_authorized",
):
    if policy[key] is not False:
        raise SystemExit(f"premature authorization: {key}")

print("qsre_t2_partial_evidence_resume_gate=PASS")
PY

python3 - <<'PY'
import alice_personality
from alice_personality.n0.curriculum_data import load_tokenizer
print("alice_personality_import=PASS")
print("load_tokenizer_import=PASS")
PY

python3 "$PREP"   --curriculum "$CURRICULUM"   --t1-prepared-cache "$T1_PREPARED"   --t1-checkpoint "$T1_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --output "$PREPARED"   --receipt "$PREPARED_RECEIPT"   --max-length 96

echo
echo "===== QSRE T2 RESUMED PREPARATION HASHES ====="
sha256sum "$CURRICULUM" "$CURRICULUM_RECEIPT" "$PREPARED" "$PREPARED_RECEIPT"

echo
echo "===== QSRE T2 RESUMED PREPARATION RECEIPT ====="
cat "$PREPARED_RECEIPT"
