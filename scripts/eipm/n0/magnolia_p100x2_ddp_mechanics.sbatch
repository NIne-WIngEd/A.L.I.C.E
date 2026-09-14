#!/usr/bin/env bash
#SBATCH --job-name=alice-n0-p100-ddp
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=alice-n0-p100-ddp-%j.out
#SBATCH --error=alice-n0-p100-ddp-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/train_mlm_p100x2_ddp_mechanics.sh" ]]; then
  echo "Submit this job from the A.L.I.C.E. repository root." >&2
  exit 2
fi
cd "$ROOT"

export PYTHON="${ALICE_N0_PYTHON:-${PYTHON:-python}}"
export N0_SMOKE_WORKDIR="${N0_SMOKE_WORKDIR:?Set N0_SMOKE_WORKDIR to the completed runtime smoke workdir before sbatch}"

{
  echo "===== MAGNOLIA P100X2 N0 DDP MECHANICS ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "smoke_workdir=$N0_SMOKE_WORKDIR"
  echo "python=$PYTHON"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi || true
} | tee "$N0_SMOKE_WORKDIR/ddp-mechanics-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/train_mlm_p100x2_ddp_mechanics.sh"

echo "===== MAGNOLIA P100X2 N0 DDP MECHANICS COMPLETE ====="
date -Is
