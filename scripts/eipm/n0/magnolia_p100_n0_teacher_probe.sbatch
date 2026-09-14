#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-teach
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-teach-%j.out
#SBATCH --error=rayan-n0-teach-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_teacher_probe.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
export N0_MLM_STEP_DIR="${N0_MLM_STEP_DIR:-$ALICE_N0_WORKDIR/checkpoints/step-00001000}"
export N0_MLM_EVAL_MASK_REPEATS="${N0_MLM_EVAL_MASK_REPEATS:-8}"
export N0_PROBE_EPOCHS="${N0_PROBE_EPOCHS:-30}"
export N0_PROBE_BATCH_SIZE="${N0_PROBE_BATCH_SIZE:-4}"
export N0_PROBE_LR="${N0_PROBE_LR:-5e-4}"

mkdir -p "$ALICE_N0_WORKDIR"

{
  echo "===== MAGNOLIA SINGLE-P100 N0 TEACHER DIAGNOSTIC ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "checkpoint=$N0_MLM_STEP_DIR"
  echo "mask_repeats=$N0_MLM_EVAL_MASK_REPEATS"
  echo "probe_epochs=$N0_PROBE_EPOCHS"
  echo "probe_batch_size=$N0_PROBE_BATCH_SIZE"
  echo "probe_lr=$N0_PROBE_LR"
  echo "private_identity_gradient=false"
  echo "container=${RAYAN_N0_CONTAINER:-rayan-n0-base}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi || true
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/p100-teacher-probe-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_teacher_probe.sh"

echo "===== MAGNOLIA SINGLE-P100 N0 TEACHER DIAGNOSTIC COMPLETE ====="
date -Is
