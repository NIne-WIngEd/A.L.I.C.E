#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-sem-grow
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=rayan-n0-sem-grow-%j.out
#SBATCH --error=rayan-n0-sem-grow-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/grow_n0_semantic_p100x2_segment.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
export N0_RESUME_FROM="${N0_RESUME_FROM:-$ALICE_N0_WORKDIR/checkpoints/step-00001000}"
export N0_SEQUENCE_LENGTH="${N0_SEQUENCE_LENGTH:-512}"
export N0_MICRO_BATCH_SIZE="${N0_MICRO_BATCH_SIZE:-1}"
export N0_GRAD_ACCUM="${N0_GRAD_ACCUM:-16}"
export N0_MAX_STEPS="${N0_MAX_STEPS:-2500}"
export N0_WARMUP_STEPS="${N0_WARMUP_STEPS:-20}"
export N0_SCHEDULER_TOTAL_STEPS="${N0_SCHEDULER_TOTAL_STEPS:-10000}"
export N0_SAVE_EVERY="${N0_SAVE_EVERY:-500}"
export N0_LEARNING_RATE="${N0_LEARNING_RATE:-3e-4}"
export N0_MIXED_PRECISION="${N0_MIXED_PRECISION:-fp16}"

mkdir -p "$ALICE_N0_WORKDIR"

{
  echo "===== MAGNOLIA P100X2 N0 SEMANTIC GROWTH ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "resume_from=$N0_RESUME_FROM"
  echo "target_step=$N0_MAX_STEPS"
  echo "scheduler_total_steps=$N0_SCHEDULER_TOTAL_STEPS"
  echo "save_every=$N0_SAVE_EVERY"
  echo "sequence_length=$N0_SEQUENCE_LENGTH"
  echo "private_identity_gradient=false"
  echo "purpose=public_semantic_representation_growth_then_fixed_readiness_probe"
  nvidia-smi || true
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/p100x2-sem-grow-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/grow_n0_semantic_p100x2_segment.sh"

echo "===== MAGNOLIA P100X2 N0 SEMANTIC GROWTH COMPLETE ====="
date -Is
