#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-role-resid
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:40:00
#SBATCH --output=rayan-n0-role-resid-%j.out
#SBATCH --error=rayan-n0-role-resid-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?submit this job from the A.L.I.C.E repo root}"
cd "$ROOT"
export ALICE_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_RELATION_SEMANTIC_ROLE_RESIDUAL_DIR="${ALICE_N0_RELATION_SEMANTIC_ROLE_RESIDUAL_DIR:-$ALICE_N0_WORKDIR/relation-semantic-role-residual-v0.2}"
export RAYAN_UDOCKER_NVIDIA=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

echo "===== MAGNOLIA N0 RELATION-SEMANTIC ROLE RESIDUAL ====="
date -Is
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "repo_root=$ROOT"
echo "workdir=$ALICE_N0_WORKDIR"
echo "output_dir=$ALICE_N0_RELATION_SEMANTIC_ROLE_RESIDUAL_DIR"
nvidia-smi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [[ "$GPU_NAME" != *"P100"* ]]; then
  echo "Expected P100 route, observed GPU: $GPU_NAME" >&2
  exit 3
fi

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh"   "$ROOT/scripts/eipm/n0/run_n0_v02_relation_semantic_role_residual_v0_2.sh"

echo "===== MAGNOLIA N0 RELATION-SEMANTIC ROLE RESIDUAL COMPLETE ====="
date -Is
