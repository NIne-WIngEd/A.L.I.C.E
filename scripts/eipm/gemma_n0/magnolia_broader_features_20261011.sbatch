#!/usr/bin/env bash
#SBATCH --job-name=rayan-broader-features
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
umask 077
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]]
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$COMPUTE_ROOT" == /homes/01/mxrayan/rayan-compute && "$(stat -c %U:%a "$COMPUTE_ROOT")" == mxrayan:700 ]]
export PERSONALITY_REPO_ROOT="$COMPUTE_ROOT/rayan-personality/code/rayan-personality-333fcd0c8239fc81a6459b13d53bbe592b33fda6"
NAMESPACE_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/n1/enter_private_network_namespace.py"
AUDIT_HELPER="$COMPUTE_ROOT/jobs/rayan-broader-features-20261011.py"
[[ "$(sha256sum "$AUDIT_HELPER" | cut -d ' ' -f 1)" == b4b509131a8e05204fa579658ea723cb05f5bcdc54e7988221703ee1e7226ed7 ]]
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store" UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)" PERSONALITY_HOST_UID="$(id -u)" PERSONALITY_HOST_GID="$(id -g)"
export COMPUTE_ROOT NAMESPACE_HELPER AUDIT_HELPER UDOCKER
bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' rayan-recovery "$NAMESPACE_HELPER" -- /bin/bash -c '
  set -euo pipefail
  exec "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
    --env="PERSONALITY_REPO_ROOT=$PERSONALITY_REPO_ROOT" --env="SLURM_JOB_ID=$SLURM_JOB_ID" --env="SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE" \
    --env="PERSONALITY_HOST_NETNS=$PERSONALITY_HOST_NETNS" --env="PERSONALITY_HOST_UID=$PERSONALITY_HOST_UID" \
    --env="PERSONALITY_HOST_GID=$PERSONALITY_HOST_GID" --env="PERSONALITY_ISOLATED_NETNS=$PERSONALITY_ISOLATED_NETNS" \
    --entrypoint=python rayan-n0-base -I "$AUDIT_HELPER"
'
