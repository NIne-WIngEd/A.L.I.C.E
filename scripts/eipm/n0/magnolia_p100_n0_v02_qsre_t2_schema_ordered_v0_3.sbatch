#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-qsre-t2s-frozen
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:p100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00
#SBATCH --output=rayan-n0-qsre-t2s-frozen-%j.out
#SBATCH --error=rayan-n0-qsre-t2s-frozen-%j.err

set -euo pipefail

echo "STOP: direct T2 v0.3 P100 execution is revoked by N0 state v0.59." >&2
echo "No GPU work is authorized until production QSRE architecture closure completes." >&2
exit 90
