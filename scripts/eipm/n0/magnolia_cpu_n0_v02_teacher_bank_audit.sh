#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT="${N0_V02_TEACHER_AUDIT_OUT:-$WORKDIR/teacher-bank-v0.4a-audit.json}"

export RAYAN_UDOCKER_NVIDIA=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2 TEACHER BANK AUDIT ====="
date -Is
echo "repo_root=$ROOT"
echo "registry=$ROOT/training/eipm/n0/n0_v02_teacher_bank_v0.4a.json"
echo "output=$OUT"
echo "gpu_requested=false"
echo "private_identity_gradient=false"
echo "model_training_performed=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python scripts/eipm/n0/audit_n0_v02_teacher_bank.py \
    --registry training/eipm/n0/n0_v02_teacher_bank_v0.4a.json \
    --fixed-eval evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl \
    --output '$OUT'"

echo "===== N0 V0.2 TEACHER BANK AUDIT COMPLETE ====="
date -Is
