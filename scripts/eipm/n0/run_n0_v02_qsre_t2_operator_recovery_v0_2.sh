#!/usr/bin/env bash
set -euo pipefail

echo "STOP: superseded by N0 state v0.55 CPU/GPU reality-gap and data-validity audit. No T2 GPU recovery is currently authorized." >&2
exit 90

ROOT="$${ALICE_N0_REPO_ROOT:-$${ALICE_REPO_ROOT:-$PWD}}"
WORKDIR="$${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.54.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_training_recovery_contract_v0_2.json"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_qsre_t2_operator_v0_1.py"

T2_ROOT="$WORKDIR/qsre-t2-operator-v0.1"
FAILED_ROOT="$T2_ROOT/training-v0.1"
OUTROOT="$T2_ROOT/training-v0.2-evaluator-totality-recovery"

CURRICULUM="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.jsonl"
CURRICULUM_RECEIPT="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.receipt.json"
PREPARED="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.pt"
PREP_RECEIPT="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.receipt.json"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
T1_CHECKPOINT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1/step-00000050/qsre_t1_executor.pt"

EXPECTED_CURRICULUM_SHA="40e2d608eaa59ad13e2f9b37a2f3da5d6c20101a28ab76434c11be3d36cb5590"
EXPECTED_CURRICULUM_RECEIPT_SHA="26719ccff1ad8b08140a495a9cfefa8696270b70e3efd5d0a9b32e46393a2165"
EXPECTED_PREPARED_SHA="774722d57ccaacc7dba8e53ff3aec66bc4b4b5c6efd20db6907ac5a70654e781"
EXPECTED_PREP_RECEIPT_SHA="52e1e0599ecda2992dd2476305c377a4645a7ad079fc6e7ea660684fa35a5967"
EXPECTED_SEMANTIC_SHA="6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"
EXPECTED_T1_SHA="483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"
AUTHORIZED_RECOVERY_CORE="157153b124d665880de237299d8dd8c81403d45c"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0$${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in \
  "$STATE" "$CONTRACT" "$TRAINER" \
  "$CURRICULUM" "$CURRICULUM_RECEIPT" \
  "$PREPARED" "$PREP_RECEIPT" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" \
  "$TOKENIZER_DIR/tokenizer.json" "$T1_CHECKPOINT"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

[[ -d "$FAILED_ROOT" ]] || {
  echo "STOP: failed T2 root missing; preservation boundary cannot be verified" >&2
  exit 3
}

[[ ! -e "$FAILED_ROOT/result.json" ]] || {
  echo "STOP: failed T2 root unexpectedly contains result.json" >&2
  exit 4
}

for preserved in \
  "$FAILED_ROOT/hidden-cache-v0.1.pt" \
  "$FAILED_ROOT/best-observed/qsre_t2_operator.pt"
do
  [[ -f "$preserved" ]] || {
    echo "STOP: expected partial evidence missing: $preserved" >&2
    exit 5
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: recovery output already exists: $OUTROOT" >&2
  echo "Preserve it. Do not delete it to rerun." >&2
  exit 6
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 7
fi

git merge-base --is-ancestor "$AUTHORIZED_RECOVERY_CORE" HEAD || {
  echo "STOP: current code is not descended from the qualified recovery core" >&2
  exit 8
}

for path in \
  configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.54.json \
  configs/eipm/n0/n0_v02_qsre_t2_training_recovery_contract_v0_2.json \
  scripts/eipm/n0/train_n0_v02_qsre_t2_operator_v0_1.py \
  src/alice_personality/n0/qsre_t2_operator.py \
  src/alice_personality/n0/qsre_t1_executor.py
do
  git diff --quiet "$AUTHORIZED_RECOVERY_CORE" -- "$path" || {
    echo "STOP: qualified recovery scientific source changed: $path" >&2
    exit 9
  }
done

check_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  echo "$label=$actual"
  [[ "$actual" == "$expected" ]] || {
    echo "STOP: $label hash drift" >&2
    exit 10
  }
}

echo "===== PRESERVED FAILED-RUN EVIDENCE ====="
sha256sum \
  "$FAILED_ROOT/hidden-cache-v0.1.pt" \
  "$FAILED_ROOT/best-observed/qsre_t2_operator.pt"

check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA" "t2_curriculum_sha256"
check_sha "$CURRICULUM_RECEIPT" "$EXPECTED_CURRICULUM_RECEIPT_SHA" "t2_curriculum_receipt_sha256"
check_sha "$PREPARED" "$EXPECTED_PREPARED_SHA" "t2_prepared_cache_sha256"
check_sha "$PREP_RECEIPT" "$EXPECTED_PREP_RECEIPT_SHA" "t2_preparation_receipt_sha256"
check_sha "$SEMANTIC_CHECKPOINT" "$EXPECTED_SEMANTIC_SHA" "semantic_checkpoint_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_SHA" "t1_checkpoint_sha256"

python3 - "$STATE" "$CONTRACT" <<'PY'
import json,sys

state=json.load(open(sys.argv[1]))
contract=json.load(open(sys.argv[2]))

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.54":
    raise SystemExit("v0.54 state drift")

q=state["totality_qualification"]
if q["workflow_run_id"]!=35533710899 or q["conclusion"]!="SUCCESS":
    raise SystemExit("evaluator totality qualification missing")

policy=state["execution_policy"]
for key in (
    "t2_replacement_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
):
    if policy[key] is not True:
        raise SystemExit(f"missing recovery authorization: {key}")
if policy["replacement_gpu_runs_authorized"]!=1:
    raise SystemExit("replacement GPU run-count drift")

for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_chain_authorized",
    "t1_rerun_authorized",
    "semantic_backbone_gradient_authorized",
    "t1_executor_gradient_authorized",
    "learned_support_training_authorized",
    "causal_test_opening_authorized",
    "frozen_challenge_opening_authorized",
    "private_identity_gradient_authorized",
    "production_promotion_authorized",
):
    if policy[key] is not False:
        raise SystemExit(f"premature recovery authorization: {key}")

if contract.get("schema")!="alice.eipm.n0.qsre-t2-training-recovery-contract.v0.2":
    raise SystemExit("recovery contract schema drift")
if contract["failed_job"]["job_id"]!=575933:
    raise SystemExit("failed-job lineage drift")
if contract["failed_job"]["model_gate_conclusion"]!="UNDETERMINED":
    raise SystemExit("failed run was incorrectly promoted to a model conclusion")
if contract.get("gpu_training_authorized") is not True:
    raise SystemExit("recovery GPU authorization missing")
if contract.get("max_gpu_runs")!=1:
    raise SystemExit("recovery max-run drift")
if contract.get("replacement_gpu_runs_authorized")!=1:
    raise SystemExit("replacement run-count drift")
if contract.get("preserve_failed_output_root") is not True:
    raise SystemExit("failed-root preservation drift")
if contract.get("restart_from_seed") is not True:
    raise SystemExit("recovery must restart from seed")
if contract.get("reuse_partial_optimizer_state") is not False:
    raise SystemExit("partial optimizer state must not be reused")

for key in (
    "automatic_rerun",
    "automatic_hotfix",
    "causal_test_open",
    "frozen_challenge_open",
    "semantic_backbone_gradient",
    "t1_executor_gradient",
    "learned_support",
    "private_identity_gradient",
):
    if contract.get(key) is not False:
        raise SystemExit(f"recovery boundary drift: {key}")

print("v0_54_qsre_t2_replacement_gpu_gate=PASS")
PY

echo "===== QSRE T2 EVALUATOR-RECOVERY TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "failed_job=575933"
echo "classification=EVALUATION_HARNESS_TOTALITY_FAILURE_NOT_MODEL_GATE_FAILURE"
echo "fresh_restart_from_seed=true"
echo "optimizer=true"
echo "gradient=true"
echo "gpu=true"
echo "replacement_gpu_runs_authorized=1"
echo "automatic_rerun=false"
echo "automatic_hotfix=false"
echo "semantic_backbone_gradient=false"
echo "t1_executor_gradient=false"
echo "learned_operator=true"
echo "learned_support=false"
echo "test_open=false"
echo "challenge_open=false"
echo "private_identity_gradient=false"

python3 "$TRAINER" \
  --contract "$CONTRACT" \
  --prepared-cache "$PREPARED" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --t1-checkpoint "$T1_CHECKPOINT" \
  --output-dir "$OUTROOT"

RESULT="$OUTROOT/result.json"
[[ -f "$RESULT" ]] || {
  echo "STOP: recovery trainer returned without result.json" >&2
  exit 11
}

echo
echo "===== QSRE T2 RECOVERY RESULT HASH ====="
sha256sum "$RESULT"

echo
echo "===== QSRE T2 RECOVERY RESULT SUMMARY ====="

python3 - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("status="+str(r.get("status")))
print("hidden_cache_sha256="+str(r.get("hidden_cache_sha256")))
print("selected_checkpoint_step="+str(r.get("selected_checkpoint_step")))
print("selected_checkpoint_sha256="+str(r.get("selected_checkpoint_sha256")))
print("best_observed_step="+str(r.get("best_observed_step")))
print("causal_test_opened="+str(r.get("causal_test_opened")))
print("frozen_challenge_opened="+str(r.get("frozen_challenge_opened")))
print("support_learning_opened="+str(r.get("support_learning_opened")))
print("automatic_rerun_authorized="+str(r.get("automatic_rerun_authorized")))

step=r.get("selected_checkpoint_step")
if step is not None:
    key=f"step-{int(step):08d}"
    print("selected_dev="+json.dumps(r["checkpoints"][key]["dev"],sort_keys=True))
else:
    keys=sorted(k for k in r["checkpoints"] if k!="step-00000000")
    if keys:
        key=keys[-1]
        print("final_dev="+json.dumps(r["checkpoints"][key]["dev"],sort_keys=True))
PY

echo "===== QSRE T2 EVALUATOR-RECOVERY COMPLETE ====="
date -Is
