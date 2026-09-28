#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-joint-v2
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=10:00:00

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
export ALICE_N0_REPO_ROOT="$ROOT"
export RAYAN_UDOCKER_NVIDIA=1
export N0_MAIN_PROCESS_PORT="$((20000 + (${SLURM_JOB_ID:-1337} % 20000)))"
bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  'bash "$ALICE_N0_REPO_ROOT/scripts/eipm/n0/run_n0_v02_measured_joint_route_v2.sh"'
