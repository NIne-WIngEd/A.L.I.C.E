#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-mlqif
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rayan-n0-mlqif-%j.out
#SBATCH --error=rayan-n0-mlqif-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit from the A.L.I.C.E repo root}"
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_MULTILAYER_INTERFACE_DIR="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$ALICE_N0_WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
export ALICE_N0_MULTILAYER_CAUSAL_STUDY_DIR="${ALICE_N0_MULTILAYER_CAUSAL_STUDY_DIR:-$ALICE_N0_MULTILAYER_INTERFACE_DIR/causal-interface-study-v0.1}"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 MULTILAYER CAUSAL INTERFACE ====="
date -Is
nvidia-smi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *P100* ]] || {
  echo "Expected P100, got $GPU_NAME" >&2
  exit 3
}

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_v02_relation_conditioned_multilayer_interface_v0_1.sh"

echo "===== MAGNOLIA N0 MULTILAYER CAUSAL INTERFACE COMPLETE ====="
date -Is
