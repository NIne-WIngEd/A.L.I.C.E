#!/usr/bin/env bash
#SBATCH --job-name=alice-n0-smoke
#SBATCH --partition=suliaoma
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=alice-n0-smoke-%j.out
#SBATCH --error=alice-n0-smoke-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/runtime_smoke.sh" ]]; then
  echo "Submit this job from the A.L.I.C.E. repository root." >&2
  echo "SLURM_SUBMIT_DIR=$ROOT" >&2
  exit 2
fi

cd "$ROOT"

export PYTHON="${ALICE_N0_PYTHON:-${PYTHON:-python}}"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/alice-n0/runtime-smoke-${SLURM_JOB_ID}}"
export N0_SMOKE_REQUIRE_CUDA=1

mkdir -p "$ALICE_N0_WORKDIR"

{
  echo "===== MAGNOLIA A100 N0 RUNTIME SMOKE ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "job_name=${SLURM_JOB_NAME:-unknown}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "python=$PYTHON"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo
  which "$PYTHON" || true
  "$PYTHON" --version || true
  echo
  nvidia-smi || true
  echo
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/magnolia_job_environment.txt"

bash "$ROOT/scripts/eipm/n0/runtime_smoke.sh"

echo "===== MAGNOLIA A100 N0 SMOKE COMPLETE ====="
date -Is
echo "receipt=$ALICE_N0_WORKDIR/runtime_smoke_receipt.json"
