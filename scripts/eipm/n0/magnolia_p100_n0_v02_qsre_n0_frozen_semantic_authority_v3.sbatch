#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-auth-v3
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=rayan-n0-auth-v3-%j.out
#SBATCH --error=rayan-n0-auth-v3-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit this job from the A.L.I.C.E repo root}"
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
: "${ALICE_N0_EXPECTED_REVISION:?export ALICE_N0_EXPECTED_REVISION before sbatch}"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 FROZEN SEMANTIC AUTHORITY V3 ====="
date -Is
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "expected_revision=$ALICE_N0_EXPECTED_REVISION"
nvidia-smi

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh"

echo "===== MAGNOLIA N0 FROZEN SEMANTIC AUTHORITY V3 FINISHED ====="
date -Is
