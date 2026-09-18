#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-qrr
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:45:00
#SBATCH --output=rayan-n0-qrr-%j.out
#SBATCH --error=rayan-n0-qrr-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repo root}"
cd "$ROOT"
export ALICE_REPO_ROOT="$ROOT"
export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
export ALICE_N0_QUERY_RELATION_ROLE_ROUTER_DIR="${ALICE_N0_QUERY_RELATION_ROLE_ROUTER_DIR:-$ALICE_N0_WORKDIR/query-relation-role-router-v0.3}"
export RAYAN_UDOCKER_NVIDIA=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
echo "===== MAGNOLIA N0 QUERY-RELATION ROLE ROUTER =====";date -Is;nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
[[ "$GPU_NAME" == *P100* ]] || { echo "Expected P100, got $GPU_NAME" >&2; exit 3; }
bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" "$ROOT/scripts/eipm/n0/run_n0_v02_query_relation_role_router_v0_3.sh"
echo "===== MAGNOLIA N0 QUERY-RELATION ROLE ROUTER COMPLETE =====";date -Is
