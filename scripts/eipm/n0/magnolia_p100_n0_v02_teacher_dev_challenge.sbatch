#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-v02-dev
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-v02-dev-%j.out
#SBATCH --error=rayan-n0-v02-dev-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_teacher_dev_challenge.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export RAYAN_UDOCKER_NVIDIA=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

ENV_DIR="$HOME/rayan-compute/rayan-n0"
mkdir -p "$ENV_DIR"
{
  echo "===== MAGNOLIA P100 N0 V0.2 TEACHER-DEV CHALLENGE ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "gpu_request=1xP100"
  echo "eval_only=true"
  echo "private_identity_gradient=false"
  nvidia-smi || true
} | tee "$ENV_DIR/n0-v02-dev-${SLURM_JOB_ID:-unknown}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_v02_teacher_dev_challenge.sh"

echo "===== MAGNOLIA P100 N0 V0.2 TEACHER-DEV CHALLENGE JOB COMPLETE ====="
date -Is
