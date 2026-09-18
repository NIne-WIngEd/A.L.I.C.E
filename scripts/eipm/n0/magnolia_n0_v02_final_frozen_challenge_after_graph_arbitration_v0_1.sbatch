#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-final
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=rayan-n0-final-%j.out
#SBATCH --error=rayan-n0-final-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
RUNNER="$ROOT/scripts/eipm/n0/run_n0_v02_final_frozen_challenge_after_graph_arbitration_v0_1.sh"

if [[ ! -f "$RUNNER" ]]; then
  echo "Submit this job from the repository root." >&2
  echo "SLURM_SUBMIT_DIR=$ROOT" >&2
  exit 2
fi

: "${ALICE_N0_ARB_DIR:?ALICE_N0_ARB_DIR must be exported when submitting this job}"

cd "$ROOT"
export ALICE_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_FINAL_CHALLENGE_DIR="${ALICE_N0_FINAL_CHALLENGE_DIR:-$ALICE_N0_WORKDIR/adaptive-multi-view-latent-pool-final-frozen-challenge-v0.1}"

{
  echo "===== MAGNOLIA N0 FINAL FROZEN CHALLENGE ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "arbitration_dir=$ALICE_N0_ARB_DIR"
  echo "output_dir=$ALICE_N0_FINAL_CHALLENGE_DIR"
  echo "container=${RAYAN_N0_CONTAINER:-rayan-n0-base}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo
  nvidia-smi
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [[ "$GPU_NAME" != *"P100"* ]]; then
  echo "Expected P100 route, observed GPU: $GPU_NAME" >&2
  exit 3
fi

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" "$RUNNER"

echo "===== MAGNOLIA N0 FINAL FROZEN CHALLENGE COMPLETE ====="
date -Is
