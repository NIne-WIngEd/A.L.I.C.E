#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.57.json"
CURRICULUM_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_curriculum_contract_v0_2.json"
BUILD="$ROOT/scripts/eipm/n0/build_n0_v02_qsre_t2_operator_curriculum_v0_2.py"
BUILD_REALITY="$ROOT/scripts/eipm/n0/build_n0_v02_qsre_operator_reality_corpus_v0_1.py"
BUILD_CHALLENGE="$ROOT/scripts/eipm/n0/build_n0_v02_qsre_t2_operator_challenge_v0_2.py"
SHORTCUT="$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_t2_shortcuts_v0_1.py"
PREP="$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_t2_tokenized_v0_1.py"

T1_ROOT="$WORKDIR/qsre-t1-executor-v0.1"
T1_CURRICULUM="$T1_ROOT/pretraining-v0.2/qsre_t1_curriculum_v0.2.jsonl"
T1_PREPARED="$T1_ROOT/pretraining-v0.2/qsre_t1_real_cache_v0.1.pt"
T1_CHECKPOINT="$T1_ROOT/training-v0.1/step-00000050/qsre_t1_executor.pt"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"

OUTROOT="$WORKDIR/qsre-t2-operator-v0.2"
CURRICULUM="$OUTROOT/qsre_t2_operator_curriculum_v0.2.jsonl"
REALITY="$OUTROOT/qsre_operator_reality_corpus_v0.1.jsonl"
CHALLENGE="$OUTROOT/qsre_t2_locked_operator_challenge_v0.2.jsonl"
SHORTCUT_RESULT="$OUTROOT/qsre_t2_shortcut_audit_v0.2.json"
PREPARED="$OUTROOT/qsre_t2_tokenized_preparation_v0.2.pt"
PREP_RECEIPT="$OUTROOT/qsre_t2_tokenized_preparation_v0.2.receipt.json"
AUTHORITY="$OUTROOT/preparation_authority_v0.2.json"

EXPECTED_T1_CURRICULUM_SHA="155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063"
EXPECTED_T1_PREPARED_SHA="03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564"
EXPECTED_T1_CHECKPOINT_SHA="483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"
EXPECTED_CURRICULUM_SHA="5aa8f0aa3210466964f38b14081c71c1d3c58f4405247067c76a7cdfe115963c"
EXPECTED_REALITY_SHA="7652b5c0fc33c4d563678255c24666b2efc3fa943bbca9ddbd5dddac5f36b0e9"
EXPECTED_CHALLENGE_SHA="197cebe5ed4c59455839db5b2822582a03021c92a94670e1aede568278c1f205"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=0

for required in \
  "$STATE" "$CURRICULUM_CONTRACT" "$BUILD" "$BUILD_REALITY" "$BUILD_CHALLENGE" \
  "$SHORTCUT" "$PREP" "$T1_CURRICULUM" "$T1_PREPARED" "$T1_CHECKPOINT" \
  "$TOKENIZER_DIR/tokenizer.json"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: T2 v0.2 output root already exists: $OUTROOT" >&2
  echo "Preserve it as evidence. Do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python3 - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.57":
    raise SystemExit("v0.57 state drift")
p=s["execution_policy"]
if p.get("t2_v02_cpu_preparation_authorized") is not True:
    raise SystemExit("T2 v0.2 CPU preparation not authorized")
if p.get("t2_v02_gpu_training_authorized_after_exact_preparation_receipt") is not True:
    raise SystemExit("dependent GPU boundary not pre-authorized")
for key in (
    "automatic_rerun_authorized",
    "automatic_hotfix_chain_authorized",
    "learning_rate_search_authorized",
    "step_search_authorized",
    "model_size_search_authorized",
    "semantic_backbone_gradient_authorized",
    "t1_executor_gradient_authorized",
    "learned_support_training_authorized",
    "private_identity_gradient_authorized",
):
    if p.get(key) is not False:
        raise SystemExit(f"boundary drift: {key}")
print("v0_57_t2_v02_cpu_preparation_gate=PASS")
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

check_sha "$T1_CURRICULUM" "$EXPECTED_T1_CURRICULUM_SHA" "t1_curriculum_sha256"
check_sha "$T1_PREPARED" "$EXPECTED_T1_PREPARED_SHA" "t1_prepared_cache_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_CHECKPOINT_SHA" "t1_checkpoint_sha256"

mkdir -p "$OUTROOT"

python3 "$BUILD" \
  --contract "$CURRICULUM_CONTRACT" \
  --t1-curriculum "$T1_CURRICULUM" \
  --output "$CURRICULUM"
check_sha "$CURRICULUM" "$EXPECTED_CURRICULUM_SHA" "t2_v02_curriculum_sha256"

python3 "$BUILD_REALITY" --output "$REALITY"
check_sha "$REALITY" "$EXPECTED_REALITY_SHA" "reality_corpus_sha256"

python3 "$BUILD_CHALLENGE" --output "$CHALLENGE"
check_sha "$CHALLENGE" "$EXPECTED_CHALLENGE_SHA" "locked_challenge_sha256"

python3 "$SHORTCUT" \
  --t2-curriculum "$CURRICULUM" \
  --reality-corpus "$REALITY" \
  --output "$SHORTCUT_RESULT"

python3 "$PREP" \
  --curriculum "$CURRICULUM" \
  --t1-prepared-cache "$T1_PREPARED" \
  --t1-checkpoint "$T1_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output "$PREPARED" \
  --receipt "$PREP_RECEIPT" \
  --max-length 128

python3 - "$ROOT" "$CURRICULUM" "$REALITY" "$CHALLENGE" "$SHORTCUT_RESULT" "$PREPARED" "$PREP_RECEIPT" "$AUTHORITY" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path

root,curr,reality,challenge,shortcut,prepared,prep_receipt,out=map(Path,sys.argv[1:])

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

pr=json.loads(prep_receipt.read_text())
sc=json.loads(shortcut.read_text())
if pr.get("status")!="PASS_QSRE_T2_TOKENIZED_PREPARATION":
    raise SystemExit("tokenized preparation did not pass")
if pr.get("rows")!={"train":720,"dev":288}:
    raise SystemExit("prepared row-count drift")
if pr.get("private_identity_data") is not False:
    raise SystemExit("private identity entered preparation")
if pr.get("optimizer") is not False or pr.get("gradient") is not False or pr.get("gpu") is not False:
    raise SystemExit("preparation crossed training boundary")

payload={
  "schema":"alice.eipm.n0.qsre-t2-v02-preparation-authority.v0.2",
  "status":"PASS_QSRE_T2_V02_PREPARATION_AUTHORITY",
  "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
  "curriculum_sha256":sha(curr),
  "reality_corpus_sha256":sha(reality),
  "locked_challenge_sha256":sha(challenge),
  "shortcut_audit_sha256":sha(shortcut),
  "prepared_cache_sha256":sha(prepared),
  "preparation_receipt_sha256":sha(prep_receipt),
  "rows":{"train":720,"dev":288,"locked_challenge":60},
  "shortcut_summary":sc["summary"],
  "optimizer":False,
  "gradient":False,
  "gpu":False,
  "semantic_backbone_gradient":False,
  "t1_executor_gradient":False,
  "private_identity_data":False,
  "gpu_training_authorized_by_this_receipt":True,
  "max_gpu_runs":1,
  "automatic_rerun":False,
  "automatic_hotfix":False
}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY

echo
echo "===== QSRE T2 v0.2 PREPARATION HASHES ====="
sha256sum "$CURRICULUM" "$REALITY" "$CHALLENGE" "$SHORTCUT_RESULT" "$PREPARED" "$PREP_RECEIPT" "$AUTHORITY"

echo
echo "===== QSRE T2 v0.2 PREPARATION AUTHORITY ====="
cat "$AUTHORITY"
