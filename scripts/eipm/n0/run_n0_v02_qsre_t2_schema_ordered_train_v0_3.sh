#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.58.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_3.json"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_qsre_t2_schema_ordered_v0_3.py"
CHALLENGE_EVAL="$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_t2_schema_ordered_challenge_v0_3.py"
RELATION_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_relation_schema_v0_1.json"

PREPARED="$WORKDIR/qsre-t2-operator-v0.2/qsre_t2_tokenized_preparation_v0.2.pt"
CHALLENGE="$WORKDIR/qsre-t2-operator-v0.2/qsre_t2_locked_operator_challenge_v0.2.jsonl"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
T1_CHECKPOINT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1/step-00000050/qsre_t1_executor.pt"

RUNTIME_ROOT="$WORKDIR/qsre-t2-operator-v0.3/runtime-qualification-v0.1"
RUNTIME_RECEIPT="$RUNTIME_ROOT/receipt.json"
OUTROOT="$WORKDIR/qsre-t2-operator-v0.3/training-v0.3-schema-ordered"
RESULT="$OUTROOT/result.json"
CHALLENGE_RESULT="$OUTROOT/locked_operator_challenge_result_v0.3.json"
FINAL_SUMMARY="$OUTROOT/t2_v03_final_summary.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

for required in   "$STATE" "$CONTRACT" "$TRAINER" "$CHALLENGE_EVAL" "$RELATION_SCHEMA"   "$PREPARED" "$CHALLENGE" "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT"   "$TOKENIZER_DIR/tokenizer.json" "$T1_CHECKPOINT" "$RUNTIME_RECEIPT"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: T2 v0.3 training evidence already exists: $OUTROOT" >&2
  echo "Preserve it. Do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python3 - "$STATE" "$CONTRACT" "$RUNTIME_RECEIPT" <<'PY'
import hashlib,json,sys
from pathlib import Path

state_path,contract_path,receipt_path=map(Path,sys.argv[1:])
state=json.loads(state_path.read_text())
contract=json.loads(contract_path.read_text())
receipt=json.loads(receipt_path.read_text())

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.58":
    raise SystemExit("v0.58 state drift")
p=state["execution_policy"]
if p.get("t2_v03_gpu_training_authorized_after_static_and_cpu_runtime_pass") is not True:
    raise SystemExit("T2 v0.3 dependent GPU authorization missing")
if p.get("max_t2_v03_gpu_runs")!=1:
    raise SystemExit("T2 v0.3 one-shot count drift")
for key in (
    "t2_v02_rerun_authorized",
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

if contract.get("schema")!="alice.eipm.n0.qsre-t2-training-contract.v0.3":
    raise SystemExit("T2 v0.3 contract drift")
if contract.get("operator_architecture")!="schema_grounded_ordered_relation_v0.3":
    raise SystemExit("T2 v0.3 operator architecture drift")
if contract.get("gpu_training_authorized") is not True or contract.get("max_gpu_runs")!=1:
    raise SystemExit("T2 v0.3 contract authorization drift")

if receipt.get("schema")!="alice.eipm.n0.qsre-t2-v03-real-artifact-runtime-qualification.v0.1":
    raise SystemExit("runtime receipt schema drift")
if receipt.get("status")!="PASS_QSRE_T2_V03_REAL_ARTIFACT_RUNTIME_QUALIFICATION":
    raise SystemExit("real-artifact runtime qualification did not pass")
if receipt.get("training_authorized_by_this_receipt") is not True:
    raise SystemExit("runtime receipt did not open dependent GPU boundary")
if receipt.get("max_gpu_runs")!=1:
    raise SystemExit("runtime receipt run-count drift")
for key in ("optimizer","gradient","gpu","semantic_backbone_gradient","t1_executor_gradient","learned_support","test_open","private_identity_data","automatic_rerun","automatic_hotfix"):
    if receipt.get(key) is not False:
        raise SystemExit(f"runtime boundary drift: {key}")

contract_sha=hashlib.sha256(contract_path.read_bytes()).hexdigest()
if receipt.get("contract_sha256")!=contract_sha:
    raise SystemExit("runtime receipt/contract hash mismatch")

print("v0_58_t2_v03_dependent_gpu_gate=PASS")
print("runtime_receipt_sha256="+hashlib.sha256(receipt_path.read_bytes()).hexdigest())
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
    exit 5
  }
}

EXPECTED_PREPARED_SHA="$(python3 - "$CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["expected_prepared_cache_sha256"])
PY
)"
EXPECTED_CHALLENGE_SHA="$(python3 - "$CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["locked_operator_challenge"]["expected_sha256"])
PY
)"
EXPECTED_RELATION_SCHEMA_SHA="$(python3 - "$CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["relation_schema_sha256"])
PY
)"
EXPECTED_SEMANTIC_SHA="$(python3 - "$CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["semantic_checkpoint_sha256"])
PY
)"
EXPECTED_T1_SHA="$(python3 - "$CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["t1_checkpoint_sha256"])
PY
)"

check_sha "$PREPARED" "$EXPECTED_PREPARED_SHA" "prepared_cache_sha256"
check_sha "$CHALLENGE" "$EXPECTED_CHALLENGE_SHA" "locked_challenge_sha256"
check_sha "$RELATION_SCHEMA" "$EXPECTED_RELATION_SCHEMA_SHA" "relation_schema_sha256"
check_sha "$SEMANTIC_CHECKPOINT" "$EXPECTED_SEMANTIC_SHA" "semantic_checkpoint_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_SHA" "t1_checkpoint_sha256"

echo "===== QSRE T2 v0.3 SCHEMA-GROUNDED ORDERED ONE-SHOT ====="
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

python3 "$TRAINER"   --contract "$CONTRACT"   --prepared-cache "$PREPARED"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --t1-checkpoint "$T1_CHECKPOINT"   --relation-schema "$RELATION_SCHEMA"   --output-dir "$OUTROOT"

[[ -f "$RESULT" ]] || {
  echo "STOP: T2 v0.3 trainer returned without result.json" >&2
  exit 6
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
  python3 "$CHALLENGE_EVAL"     --contract "$CONTRACT"     --challenge "$CHALLENGE"     --semantic-config "$SEMANTIC_CONFIG"     --semantic-checkpoint "$SEMANTIC_CHECKPOINT"     --tokenizer-dir "$TOKENIZER_DIR"     --operator-checkpoint "$SELECTED_CHECKPOINT"     --relation-schema "$RELATION_SCHEMA"     --output "$CHALLENGE_RESULT"
else
  echo "operator_only_locked_challenge_opened=false"
fi

python3 - "$RESULT" "$CHALLENGE_RESULT" "$CONTRACT" "$RUNTIME_RECEIPT" "$FINAL_SUMMARY" <<'PY'
import hashlib,json,sys
from pathlib import Path

result_path=Path(sys.argv[1])
challenge_path=Path(sys.argv[2])
contract_path=Path(sys.argv[3])
runtime_receipt=Path(sys.argv[4])
out=Path(sys.argv[5])

result=json.loads(result_path.read_text())
challenge=json.loads(challenge_path.read_text()) if challenge_path.exists() else None

payload={
  "schema":"alice.eipm.n0.qsre-t2-v03-final-summary.v0.3",
  "status":"COMPLETE_QSRE_T2_V03_SCHEMA_GROUNDED_ORDERED_CAUSAL_EXPERIMENT",
  "training_result_status":result["status"],
  "training_result_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest(),
  "contract_sha256":hashlib.sha256(contract_path.read_bytes()).hexdigest(),
  "runtime_receipt_sha256":hashlib.sha256(runtime_receipt.read_bytes()).hexdigest(),
  "relation_schema_sha256":result.get("relation_schema_sha256"),
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
echo "===== QSRE T2 v0.3 OUTPUT HASHES ====="
sha256sum "$CONTRACT" "$RUNTIME_RECEIPT" "$RESULT" "$FINAL_SUMMARY"
if [[ -f "$CHALLENGE_RESULT" ]]; then
  sha256sum "$CHALLENGE_RESULT"
fi

echo "===== QSRE T2 v0.3 SCHEMA-GROUNDED ORDERED ONE-SHOT COMPLETE ====="
date -Is
