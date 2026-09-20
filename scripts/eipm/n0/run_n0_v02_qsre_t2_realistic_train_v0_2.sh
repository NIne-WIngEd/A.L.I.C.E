#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.57.json"
STATIC_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_2.json"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_qsre_t2_operator_v0_1.py"
CHALLENGE_EVAL="$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_t2_operator_challenge_v0_2.py"

T2_ROOT="$WORKDIR/qsre-t2-operator-v0.2"
CURRICULUM="$T2_ROOT/qsre_t2_operator_curriculum_v0.2.jsonl"
PREPARED="$T2_ROOT/qsre_t2_tokenized_preparation_v0.2.pt"
PREP_RECEIPT="$T2_ROOT/qsre_t2_tokenized_preparation_v0.2.receipt.json"
AUTHORITY="$T2_ROOT/preparation_authority_v0.2.json"
CHALLENGE="$T2_ROOT/qsre_t2_locked_operator_challenge_v0.2.jsonl"
RUNTIME_CONTRACT="$T2_ROOT/resolved_training_contract_v0.2.json"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
T1_CHECKPOINT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1/step-00000050/qsre_t1_executor.pt"

OUTROOT="$T2_ROOT/training-v0.2-realistic"
RESULT="$OUTROOT/result.json"
CHALLENGE_RESULT="$OUTROOT/locked_operator_challenge_result_v0.2.json"
FINAL_SUMMARY="$OUTROOT/t2_v02_final_summary.json"

EXPECTED_CURRICULUM_SHA="5aa8f0aa3210466964f38b14081c71c1d3c58f4405247067c76a7cdfe115963c"
EXPECTED_CHALLENGE_SHA="197cebe5ed4c59455839db5b2822582a03021c92a94670e1aede568278c1f205"
EXPECTED_SEMANTIC_SHA="6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"
EXPECTED_T1_SHA="483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in \
  "$STATE" "$STATIC_CONTRACT" "$TRAINER" "$CHALLENGE_EVAL" \
  "$CURRICULUM" "$PREPARED" "$PREP_RECEIPT" "$AUTHORITY" "$CHALLENGE" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" "$TOKENIZER_DIR/tokenizer.json" "$T1_CHECKPOINT"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: realistic T2 training evidence already exists: $OUTROOT" >&2
  echo "Preserve it. Do not delete it to rerun." >&2
  exit 3
fi

if [[ -e "$RUNTIME_CONTRACT" ]]; then
  echo "REFUSING: resolved runtime contract already exists: $RUNTIME_CONTRACT" >&2
  exit 4
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

python3 - "$STATE" "$STATIC_CONTRACT" "$AUTHORITY" "$PREP_RECEIPT" <<'PY'
import json,sys
state=json.load(open(sys.argv[1]))
contract=json.load(open(sys.argv[2]))
authority=json.load(open(sys.argv[3]))
receipt=json.load(open(sys.argv[4]))

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.57":
    raise SystemExit("v0.57 state drift")
p=state["execution_policy"]
if p.get("t2_v02_gpu_training_authorized_after_exact_preparation_receipt") is not True:
    raise SystemExit("T2 v0.2 GPU authorization missing")
if p.get("max_t2_v02_gpu_runs")!=1:
    raise SystemExit("T2 v0.2 one-shot count drift")
for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_chain_authorized",
    "learning_rate_search_authorized",
    "step_search_authorized",
    "model_size_search_authorized",
    "semantic_backbone_gradient_authorized",
    "t1_executor_gradient_authorized",
    "learned_support_training_authorized",
    "causal_test_opening_authorized",
    "private_identity_gradient_authorized",
):
    if p.get(key) is not False:
        raise SystemExit(f"state boundary drift: {key}")

if contract.get("schema")!="alice.eipm.n0.qsre-t2-training-contract.v0.2":
    raise SystemExit("T2 v0.2 contract drift")
if contract.get("gpu_training_authorized") is not True or contract.get("max_gpu_runs")!=1:
    raise SystemExit("T2 v0.2 contract authorization drift")
if contract.get("expected_prepared_cache_sha256")!="BOUND_BY_PREPARATION_AUTHORITY":
    raise SystemExit("static contract prepared-cache binding drift")

if authority.get("schema")!="alice.eipm.n0.qsre-t2-v02-preparation-authority.v0.2":
    raise SystemExit("preparation authority schema drift")
if authority.get("status")!="PASS_QSRE_T2_V02_PREPARATION_AUTHORITY":
    raise SystemExit("preparation authority did not pass")
if authority.get("gpu_training_authorized_by_this_receipt") is not True:
    raise SystemExit("preparation authority did not open dependent GPU boundary")
if authority.get("max_gpu_runs")!=1:
    raise SystemExit("preparation authority run-count drift")
if authority.get("optimizer") is not False or authority.get("gradient") is not False or authority.get("gpu") is not False:
    raise SystemExit("preparation authority crossed the training boundary")

if receipt.get("status")!="PASS_QSRE_T2_TOKENIZED_PREPARATION":
    raise SystemExit("tokenized preparation did not pass")
if receipt.get("prepared_cache_sha256")!=authority.get("prepared_cache_sha256"):
    raise SystemExit("prepared-cache authority/receipt mismatch")
if receipt.get("curriculum_sha256")!=authority.get("curriculum_sha256"):
    raise SystemExit("curriculum authority/receipt mismatch")

print("v0_57_t2_v02_dependent_gpu_gate=PASS")
print("preparation_source_revision="+str(authority.get("source_revision")))
print("prepared_cache_sha256="+str(authority.get("prepared_cache_sha256")))
PY

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

check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA" "t2_v02_curriculum_sha256"
check_sha "$CHALLENGE" "$EXPECTED_CHALLENGE_SHA" "locked_challenge_sha256"
check_sha "$SEMANTIC_CHECKPOINT" "$EXPECTED_SEMANTIC_SHA" "semantic_checkpoint_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_SHA" "t1_checkpoint_sha256"

# Resolve only the post-preparation content hash. All scientific settings are
# copied byte-for-structure from the frozen repository contract.
python3 - "$STATIC_CONTRACT" "$AUTHORITY" "$RUNTIME_CONTRACT" <<'PY'
import hashlib,json,sys
from pathlib import Path

static=Path(sys.argv[1])
authority_path=Path(sys.argv[2])
out=Path(sys.argv[3])
contract=json.loads(static.read_text())
authority=json.loads(authority_path.read_text())

if contract["expected_prepared_cache_sha256"]!="BOUND_BY_PREPARATION_AUTHORITY":
    raise SystemExit("static contract no longer expects authority binding")

resolved=dict(contract)
resolved["expected_prepared_cache_sha256"]=authority["prepared_cache_sha256"]
resolved["preparation_authority_sha256"]=hashlib.sha256(authority_path.read_bytes()).hexdigest()
resolved["static_contract_sha256"]=hashlib.sha256(static.read_bytes()).hexdigest()
resolved["status"]="AUTHORIZED_EXACTLY_ONE_REALISTIC_QSRE_T2_OPERATOR_P100_RUN_EXACT_PREPARATION_BOUND"

# Prove the only scientific-contract mutation is the prepared-cache binding.
for key,value in contract.items():
    if key in ("expected_prepared_cache_sha256","status"):
        continue
    if resolved[key]!=value:
        raise SystemExit(f"unexpected resolved-contract mutation: {key}")

out.write_text(json.dumps(resolved,indent=2,sort_keys=True)+"\n")
print("resolved_training_contract_sha256="+hashlib.sha256(out.read_bytes()).hexdigest())
PY

echo "===== QSRE T2 v0.2 REALISTIC ONE-SHOT TRAINING ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "optimizer=true"
echo "gradient=true"
echo "gpu=true"
echo "max_gpu_runs=1"
echo "automatic_rerun=false"
echo "automatic_hotfix=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "model_size_search=false"
echo "semantic_backbone_gradient=false"
echo "t1_executor_gradient=false"
echo "learned_support=false"
echo "test_open=false"
echo "private_identity_gradient=false"

python3 "$TRAINER" \
  --contract "$RUNTIME_CONTRACT" \
  --prepared-cache "$PREPARED" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --t1-checkpoint "$T1_CHECKPOINT" \
  --output-dir "$OUTROOT"

[[ -f "$RESULT" ]] || {
  echo "STOP: realistic T2 trainer returned without result.json" >&2
  exit 7
}

STATUS="$(python3 - "$RESULT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"

echo "training_status=$STATUS"

if [[ "$STATUS" == "PASS_QSRE_T2_OPERATOR_DEV_CONTRACT" ]]; then
  SELECTED_CHECKPOINT="$(python3 - "$RESULT" "$OUTROOT" <<'PY'
import json,sys
from pathlib import Path
r=json.load(open(sys.argv[1]))
step=r.get("selected_checkpoint_step")
if step is None:
    raise SystemExit("PASS result has no selected checkpoint")
p=Path(sys.argv[2])/f"step-{int(step):08d}"/"qsre_t2_operator.pt"
if not p.is_file():
    raise SystemExit("selected operator checkpoint missing")
print(p)
PY
)"
  echo "operator_only_locked_challenge_opened=true"
  python3 "$CHALLENGE_EVAL" \
    --contract "$RUNTIME_CONTRACT" \
    --challenge "$CHALLENGE" \
    --semantic-config "$SEMANTIC_CONFIG" \
    --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --operator-checkpoint "$SELECTED_CHECKPOINT" \
    --output "$CHALLENGE_RESULT"
else
  echo "operator_only_locked_challenge_opened=false"
fi

python3 - "$RESULT" "$CHALLENGE_RESULT" "$RUNTIME_CONTRACT" "$FINAL_SUMMARY" <<'PY'
import hashlib,json,sys
from pathlib import Path

result_path=Path(sys.argv[1])
challenge_path=Path(sys.argv[2])
contract_path=Path(sys.argv[3])
out=Path(sys.argv[4])
result=json.loads(result_path.read_text())

challenge=None
if challenge_path.exists():
    challenge=json.loads(challenge_path.read_text())

payload={
  "schema":"alice.eipm.n0.qsre-t2-v02-final-summary.v0.2",
  "status":"COMPLETE_QSRE_T2_V02_CAUSAL_EXPERIMENT",
  "training_result_status":result["status"],
  "training_result_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest(),
  "resolved_contract_sha256":hashlib.sha256(contract_path.read_bytes()).hexdigest(),
  "selected_checkpoint_step":result.get("selected_checkpoint_step"),
  "selected_checkpoint_sha256":result.get("selected_checkpoint_sha256"),
  "best_observed_step":result.get("best_observed_step"),
  "best_observed_checkpoint_sha256":result.get("best_observed_checkpoint_sha256"),
  "operator_only_locked_challenge_opened":challenge is not None,
  "operator_only_locked_challenge_result":challenge,
  "causal_test_opened":False,
  "frozen_challenge_opened":False,
  "learned_support_opened":False,
  "semantic_backbone_gradient":False,
  "t1_executor_gradient":False,
  "private_identity_gradient":False,
  "automatic_rerun_authorized":False,
  "automatic_hotfix_authorized":False,
  "t3_authorized":False,
  "production_promotion_authorized":False
}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY

echo
echo "===== QSRE T2 v0.2 OUTPUT HASHES ====="
sha256sum "$RUNTIME_CONTRACT" "$RESULT" "$FINAL_SUMMARY"
if [[ -f "$CHALLENGE_RESULT" ]]; then
  sha256sum "$CHALLENGE_RESULT"
fi

echo "===== QSRE T2 v0.2 REALISTIC ONE-SHOT COMPLETE ====="
date -Is
