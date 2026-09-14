#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT="${N0_V02_TEACHER_AUDIT_OUT:-$WORKDIR/teacher-bank-v0.4b-audit.json}"

export RAYAN_UDOCKER_NVIDIA=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2 VOICE-FIRST TEACHER BANK AUDIT ====="
date -Is
echo "repo_root=$ROOT"
echo "registry=$ROOT/training/eipm/n0/n0_v02_teacher_bank_v0.4b.json"
echo "output=$OUT"
echo "gpu_requested=false"
echo "network_required=false"
echo "private_identity_gradient=false"
echo "model_training_performed=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python scripts/eipm/n0/audit_n0_v02_teacher_bank.py \
    --registry training/eipm/n0/n0_v02_teacher_bank_v0.4b.json \
    --output '$OUT'"

echo "===== N0 V0.2 VOICE-FIRST TEACHER BANK AUDIT COMPLETE ====="
date -Is
