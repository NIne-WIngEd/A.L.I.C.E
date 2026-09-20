#!/usr/bin/env bash
set -euo pipefail

: "${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT must be set to the host repository path before entering uDocker}"
: "${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR must be set to the host N0 work directory before entering uDocker}"

T1_ROOT="$ALICE_N0_WORKDIR/qsre-t1-executor-v0.1"
RESULT="$T1_ROOT/training-v0.1/result.json"
OUTPUT="$T1_ROOT/postrun-evidence-v0.1"

test -f "$RESULT" || {
  echo "STOP: governed T1 result missing: $RESULT"
  exit 2
}

test ! -e "$OUTPUT" || {
  echo "STOP: T1 postrun evidence output already exists"
  echo "Preserve it and inspect existing evidence; do not delete/recreate it."
  exit 3
}

python3 \
  "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/capture_n0_v02_qsre_t1_postrun_evidence_v0_1.py" \
  --result "$RESULT" \
  --output "$OUTPUT"
