#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-fusion
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=rayan-n0-fusion-%j.out
#SBATCH --error=rayan-n0-fusion-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit this job from the A.L.I.C.E repo root}"
cd "$ROOT"

export ALICE_N0_REPO_ROOT="$ROOT"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_n0_v02_cross_context_fusion_full_scale.sh"
