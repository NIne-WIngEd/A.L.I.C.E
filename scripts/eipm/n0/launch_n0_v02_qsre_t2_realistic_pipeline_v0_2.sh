#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
if [[ ! -f "$ROOT/scripts/eipm/n0/magnolia_cpu_n0_v02_qsre_t2_realistic_prepare_v0_2.sbatch" ]]; then
  echo "Run from the A.L.I.C.E repository root." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 3
fi

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

PREP_JOB="$(sbatch --parsable   "$ROOT/scripts/eipm/n0/magnolia_cpu_n0_v02_qsre_t2_realistic_prepare_v0_2.sbatch")"

TRAIN_JOB="$(sbatch --parsable   --dependency="afterok:$PREP_JOB"   "$ROOT/scripts/eipm/n0/magnolia_p100_n0_v02_qsre_t2_realistic_train_v0_2.sbatch")"

cat <<EOF
status=SUBMITTED_QSRE_T2_V02_PIPELINE
prep_job_id=$PREP_JOB
train_job_id=$TRAIN_JOB
dependency=afterok:$PREP_JOB
manual_intermediate_gate=false
gpu_job_will_not_start_if_preparation_fails=true
automatic_rerun=false
automatic_hotfix=false
EOF
