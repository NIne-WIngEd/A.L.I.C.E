#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-corpus
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=rayan-n0-corpus-%j.out
#SBATCH --error=rayan-n0-corpus-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$ROOT/scripts/eipm/n0/run_stage.sh" ]]; then
  echo "Submit this job from the repository root." >&2
  exit 2
fi
cd "$ROOT"

export ALICE_N0_WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
export HF_HOME="${HF_HOME:-$HOME/rayan-compute/rayan-n0/hf-cache}"
export RAYAN_UDOCKER_NVIDIA=0
export N0_BOOTSTRAP_CHARS_PER_SOURCE="${N0_BOOTSTRAP_CHARS_PER_SOURCE:-100000000}"
export N0_SHARD_MB="${N0_SHARD_MB:-128}"

mkdir -p "$ALICE_N0_WORKDIR" "$HF_HOME"

{
  echo "===== MAGNOLIA CPU N0 CORPUS BOOTSTRAP ====="
  date -Is
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "job_name=${SLURM_JOB_NAME:-unknown}"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node=${SLURMD_NODENAME:-$(hostname)}"
  echo "repo_root=$ROOT"
  echo "workdir=$ALICE_N0_WORKDIR"
  echo "hf_home=$HF_HOME"
  echo "chars_per_source=$N0_BOOTSTRAP_CHARS_PER_SOURCE"
  echo "shard_mb=$N0_SHARD_MB"
  echo "cuda_required=false"
  df -h "$ALICE_N0_WORKDIR" || true
} | tee "$ALICE_N0_WORKDIR/corpus-bootstrap-${SLURM_JOB_ID}.environment.txt"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_stage.sh" corpus-bootstrap

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" \
  "$ROOT/scripts/eipm/n0/run_stage.sh" verify-corpus

echo "===== MAGNOLIA CPU N0 CORPUS BOOTSTRAP COMPLETE ====="
date -Is
echo "receipt=$ALICE_N0_WORKDIR/corpus/corpus_receipt.json"
