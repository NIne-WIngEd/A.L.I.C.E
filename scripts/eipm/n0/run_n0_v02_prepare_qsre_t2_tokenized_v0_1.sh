#!/usr/bin/env bash
set -euo pipefail

: "${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT must be set}"
: "${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR must be set}"

export PYTHONPATH="$ALICE_N0_REPO_ROOT/src:$ALICE_N0_REPO_ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

T1_ROOT="$ALICE_N0_WORKDIR/qsre-t1-executor-v0.1"
T1_CURRICULUM="$T1_ROOT/pretraining-v0.2/qsre_t1_curriculum_v0.2.jsonl"
T1_PREPARED="$T1_ROOT/pretraining-v0.2/qsre_t1_real_cache_v0.1.pt"
T1_CHECKPOINT="$T1_ROOT/training-v0.1/step-00000050/qsre_t1_executor.pt"

T2_ROOT="$ALICE_N0_WORKDIR/qsre-t2-operator-v0.1"
CURRICULUM="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.jsonl"
CURRICULUM_RECEIPT="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.receipt.json"
PREPARED="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.pt"
PREPARED_RECEIPT="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.receipt.json"

TOKENIZER_DIR="$ALICE_N0_WORKDIR/tokenizer-v0.2.1"

test "$(sha256sum "$T1_CURRICULUM" | awk '{print $1}')" =   "155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063" || {
  echo "STOP: T1 curriculum drift"
  exit 2
}
test "$(sha256sum "$T1_PREPARED" | awk '{print $1}')" =   "03a45033dc41a51896f8b514c44e784ecabaa5837488a272306914979ab6b564" || {
  echo "STOP: T1 prepared cache drift"
  exit 3
}
test "$(sha256sum "$T1_CHECKPOINT" | awk '{print $1}')" =   "483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb" || {
  echo "STOP: selected T1 checkpoint drift"
  exit 4
}
test -d "$TOKENIZER_DIR" || {
  echo "STOP: tokenizer directory missing: $TOKENIZER_DIR"
  exit 5
}
test ! -e "$T2_ROOT" || {
  echo "STOP: T2 preparation root already exists"
  echo "Preserve existing evidence; do not delete/recreate it."
  exit 6
}

mkdir -p "$T2_ROOT"

python3 "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/build_n0_v02_qsre_t2_operator_curriculum_v0_1.py"   --contract "$ALICE_N0_REPO_ROOT/configs/eipm/n0/n0_v02_qsre_t2_curriculum_contract_v0_1.json"   --t1-curriculum "$T1_CURRICULUM"   --output "$CURRICULUM"

python3 "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/evaluate_n0_v02_qsre_t2_operator_curriculum_v0_1.py"   --curriculum "$CURRICULUM"   --output "$CURRICULUM_RECEIPT"

python3 "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_t2_tokenized_v0_1.py"   --curriculum "$CURRICULUM"   --t1-prepared-cache "$T1_PREPARED"   --t1-checkpoint "$T1_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --output "$PREPARED"   --receipt "$PREPARED_RECEIPT"   --max-length 96

echo "===== QSRE T2 PREPARATION HASHES ====="
sha256sum "$CURRICULUM" "$CURRICULUM_RECEIPT" "$PREPARED" "$PREPARED_RECEIPT"
echo "===== QSRE T2 PREPARATION RECEIPT ====="
cat "$PREPARED_RECEIPT"
