#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-qsre-t2q
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-qsre-t2q-%j.out
#SBATCH --error=rayan-n0-qsre-t2q-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_qsre_t2_schema_ordered_runtime_v0_3.sh" ]]; then
  echo "Submit from the A.L.I.C.E repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export RAYAN_UDOCKER_NVIDIA=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA QSRE T2 v0.3 REAL-ARTIFACT RUNTIME QUALIFICATION ====="
date -Is
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "gpu_required=false"

RAYAN_UDOCKER_NVIDIA=0 bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_qsre_t2_schema_ordered_runtime_v0_3.sh"

echo "===== MAGNOLIA QSRE T2 v0.3 RUNTIME QUALIFICATION COMPLETE ====="
date -Is
