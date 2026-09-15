#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-v02-first
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=rayan-n0-v02-first-%j.out
#SBATCH --error=rayan-n0-v02-first-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_first_tranche.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export RAYAN_UDOCKER_NVIDIA=1
export N0_MAIN_PROCESS_PORT="${N0_MAIN_PROCESS_PORT:-$((20000 + (${SLURM_JOB_ID:-1337} % 20000)))}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

{
  echo "===== MAGNOLIA P100X2 N0 V0.2 FIRST REAL TRANCHE ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "main_process_port=$N0_MAIN_PROCESS_PORT"
  echo "gpu_request=2xP100"
  echo "time_limit=04:00:00"
  echo "network_required=false"
  echo "private_identity_gradient=false"
  nvidia-smi || true
  df -h "$HOME/rayan-compute/rayan-n0" || true
} | tee "$HOME/rayan-compute/rayan-n0/n0-v02-first-${SLURM_JOB_ID:-unknown}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_v02_first_tranche.sh"

echo "===== MAGNOLIA P100X2 N0 V0.2 FIRST TRANCHE JOB COMPLETE ====="
date -Is
