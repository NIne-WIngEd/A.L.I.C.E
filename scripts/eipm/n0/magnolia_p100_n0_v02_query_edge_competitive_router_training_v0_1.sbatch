#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-qecr
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=rayan-n0-qecr-%j.out
#SBATCH --error=rayan-n0-qecr-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit from the A.L.I.C.E repo root}"
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_COMPETITIVE_ROUTER_DIR="${ALICE_N0_COMPETITIVE_ROUTER_DIR:-$ALICE_N0_WORKDIR/query-edge-competitive-router-v0.1}"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 COMPETITIVE EDGE ROUTER ====="
date -Is
nvidia-smi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *P100* ]] || {
  echo "Expected P100, got $GPU_NAME" >&2
  exit 3
}

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_query_edge_competitive_router_training_v0_1.sh"

echo "===== MAGNOLIA N0 COMPETITIVE EDGE ROUTER COMPLETE ====="
date -Is
