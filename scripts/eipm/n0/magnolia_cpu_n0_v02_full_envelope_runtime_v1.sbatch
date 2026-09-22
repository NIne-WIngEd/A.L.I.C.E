#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-full-cpu
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=rayan-n0-full-cpu-%j.out
#SBATCH --error=rayan-n0-full-cpu-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh" ]]; then
  echo "Submit from the A.L.I.C.E repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_EXPECTED_REVISION="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
export RAYAN_UDOCKER_NVIDIA=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION ====="
date -Is
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "gpu_required=false"

RAYAN_UDOCKER_NVIDIA=0 bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_full_envelope_cpu_runtime_v1.sh"

echo "===== MAGNOLIA N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION COMPLETE ====="
date -Is
