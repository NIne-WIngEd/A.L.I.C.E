#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-arbcal
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=rayan-n0-arbcal-%j.out
#SBATCH --error=rayan-n0-arbcal-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_downstream_causal_arbitration_metric_calibration_v0_1.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  echo "SLURM_SUBMIT_DIR=$ROOT" >&2
  exit 2
fi

cd "$ROOT"

export ALICE_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

{
  echo "===== MAGNOLIA N0 ARBITRATION CALIBRATION ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
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

# Run the entire calibration stage inside one validated user-space container.
# Do not proxy individual Python calls through udocker.
bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_downstream_causal_arbitration_metric_calibration_v0_1.sh"

echo "===== MAGNOLIA N0 ARBITRATION CALIBRATION COMPLETE ====="
date -Is
