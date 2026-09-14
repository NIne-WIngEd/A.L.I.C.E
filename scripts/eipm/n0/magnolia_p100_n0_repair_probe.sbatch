#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-repair
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-repair-%j.out
#SBATCH --error=rayan-n0-repair-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_repair_probe.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
export N0_MLM_STEP_DIR="${N0_MLM_STEP_DIR:-$ALICE_N0_WORKDIR/checkpoints/step-00001000}"
export N0_REPAIR_EPOCHS="${N0_REPAIR_EPOCHS:-12}"
export N0_REPAIR_BATCH_SIZE="${N0_REPAIR_BATCH_SIZE:-4}"
export N0_REPAIR_LR="${N0_REPAIR_LR:-2e-4}"

{
  echo "===== MAGNOLIA SINGLE-P100 N0 FAILURE-DRIVEN REPAIR ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "checkpoint=$N0_MLM_STEP_DIR"
  echo "repair_epochs=$N0_REPAIR_EPOCHS"
  echo "repair_batch_size=$N0_REPAIR_BATCH_SIZE"
  echo "repair_lr=$N0_REPAIR_LR"
  echo "backbone_frozen=true"
  echo "private_identity_gradient=false"
  echo "container=${RAYAN_N0_CONTAINER:-rayan-n0-base}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi || true
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/p100-repair-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_repair_probe.sh"

echo "===== MAGNOLIA SINGLE-P100 N0 FAILURE-DRIVEN REPAIR COMPLETE ====="
date -Is
