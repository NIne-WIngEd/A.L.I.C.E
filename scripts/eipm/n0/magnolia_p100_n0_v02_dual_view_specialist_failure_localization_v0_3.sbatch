#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-dvloc
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=rayan-n0-dvloc-%j.out
#SBATCH --error=rayan-n0-dvloc-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit from the A.L.I.C.E repo root}"
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 DUAL-VIEW FAILURE LOCALIZATION ====="
date -Is
nvidia-smi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *P100* ]] || {
  echo "Expected P100, got $GPU_NAME" >&2
  exit 3
}

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_dual_view_specialist_failure_localization_v0_3.sh"

echo "===== MAGNOLIA N0 DUAL-VIEW FAILURE LOCALIZATION COMPLETE ====="
date -Is
