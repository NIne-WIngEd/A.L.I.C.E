#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-v02-repair
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-v02-repair-%j.out
#SBATCH --error=rayan-n0-v02-repair-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_targeted_repair.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export RAYAN_UDOCKER_NVIDIA=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

ENV_DIR="$HOME/rayan-compute/rayan-n0"
mkdir -p "$ENV_DIR"
{
  echo "===== MAGNOLIA P100X2 N0 V0.2 TARGETED REPAIR ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "gpu_request=2xP100"
  echo "model_training=true"
  echo "parent_checkpoint=step-00000250"
  echo "parameter_growth=0"
  echo "context_growth=0"
  echo "private_identity_gradient=false"
  nvidia-smi || true
} | tee "$ENV_DIR/n0-v02-repair-${SLURM_JOB_ID:-unknown}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_v02_targeted_repair.sh"

echo "===== MAGNOLIA P100X2 N0 V0.2 TARGETED REPAIR JOB COMPLETE ====="
date -Is
