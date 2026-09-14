#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-p100-smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-p100-smoke-%j.out
#SBATCH --error=rayan-n0-p100-smoke-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/runtime_smoke.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  echo "SLURM_SUBMIT_DIR=$ROOT" >&2
  exit 2
fi

cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/p100x2-runtime-smoke-${SLURM_JOB_ID}}"
export N0_SMOKE_REQUIRE_CUDA=1
export N0_SMOKE_MIN_CUDA_DEVICES=2

mkdir -p "$ALICE_N0_WORKDIR"

{
  echo "===== MAGNOLIA P100X2 N0 RUNTIME SMOKE ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "job_name=${SLURM_JOB_NAME:-unknown}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "container=${RAYAN_N0_CONTAINER:-rayan-n0-base}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo
  nvidia-smi || true
  echo
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/magnolia_job_environment.txt"

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [[ "${#GPU_NAMES[@]}" -lt 2 ]]; then
  echo "Expected 2 visible GPUs, found ${#GPU_NAMES[@]}." >&2
  exit 3
fi
for name in "${GPU_NAMES[@]:0:2}"; do
  if [[ "$name" != *"P100"* ]]; then
    echo "Expected P100 route, observed GPU: $name" >&2
    exit 4
  fi
done

# Run the complete smoke inside one validated user-space container session.
# Do not proxy individual Python calls through udocker: that path was proven
# fragile on Magnolia and can recurse through /usr/bin/env or destabilize MKL.
bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/runtime_smoke.sh"

echo "===== MAGNOLIA P100X2 N0 SMOKE COMPLETE ====="
date -Is
echo "receipt=$ALICE_N0_WORKDIR/runtime_smoke_receipt.json"
