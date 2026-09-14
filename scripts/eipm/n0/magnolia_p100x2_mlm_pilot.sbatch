#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-p100-pilot
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=rayan-n0-p100-pilot-%j.out
#SBATCH --error=rayan-n0-p100-pilot-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/train_mlm_p100x2.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
export N0_SEQUENCE_LENGTH="${N0_SEQUENCE_LENGTH:-512}"
export N0_MICRO_BATCH_SIZE="${N0_MICRO_BATCH_SIZE:-1}"
export N0_GRAD_ACCUM="${N0_GRAD_ACCUM:-16}"
export N0_MAX_STEPS="${N0_MAX_STEPS:-200}"
export N0_WARMUP_STEPS="${N0_WARMUP_STEPS:-20}"
export N0_SAVE_EVERY="${N0_SAVE_EVERY:-100}"
export N0_MIXED_PRECISION="${N0_MIXED_PRECISION:-fp16}"

mkdir -p "$ALICE_N0_WORKDIR"

{
  echo "===== MAGNOLIA P100X2 N0 MLM — FIRST DURABLE TRAINING SEGMENT ====="
  date -Is
  echo "lineage=real_public_n0"
  echo "qualification_only=false"
  echo "purpose=durable_semantic_backbone_training"
  echo "resume_policy=continue_from_checkpoint_do_not_restart"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "job_name=${SLURM_JOB_NAME:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "checkpoint_dir=$ALICE_N0_WORKDIR/checkpoints"
  echo "container=${RAYAN_N0_CONTAINER:-rayan-n0-base}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "sequence_length=$N0_SEQUENCE_LENGTH"
  echo "micro_batch_size=$N0_MICRO_BATCH_SIZE"
  echo "grad_accum=$N0_GRAD_ACCUM"
  echo "max_steps=$N0_MAX_STEPS"
  echo "warmup_steps=$N0_WARMUP_STEPS"
  echo "save_every=$N0_SAVE_EVERY"
  echo "mixed_precision=$N0_MIXED_PRECISION"
  nvidia-smi || true
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/p100x2-pilot-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/train_mlm_p100x2.sh"

echo "===== FIRST DURABLE PUBLIC N0 MLM SEGMENT COMPLETE ====="
date -Is
