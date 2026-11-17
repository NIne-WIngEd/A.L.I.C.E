#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-profile
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
umask 077
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]]
COMPUTE_ROOT="$(realpath -e /homes/01/mxrayan/rayan-compute)"
[[ "$COMPUTE_ROOT" == /homes/01/mxrayan/rayan-compute && "$(stat -c %U:%a "$COMPUTE_ROOT")" == mxrayan:700 ]]
export PERSONALITY_REPO_ROOT="$COMPUTE_ROOT/rayan-personality/code/rayan-personality-333fcd0c8239fc81a6459b13d53bbe592b33fda6"
NAMESPACE_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/n1/enter_private_network_namespace.py"
AUDIT_HELPER="$COMPUTE_ROOT/jobs/rayan-connected-timeout-profile-20261008.py"
[[ "$(sha256sum "$AUDIT_HELPER" | cut -d ' ' -f 1)" == 655b2cd0a133ff80ef0d03c20d2e2facc7252e3e5621b034d40f3d22622b3e6e ]]
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store" UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)" PERSONALITY_HOST_UID="$(id -u)" PERSONALITY_HOST_GID="$(id -g)"
export COMPUTE_ROOT NAMESPACE_HELPER AUDIT_HELPER UDOCKER
for threads in 1 16; do
  export PROFILE_THREADS="$threads"
  export PERSONALITY_PROFILE_ROOT="$COMPUTE_ROOT/rayan-personality/private-compile/rayan-timeout-profile-$SLURM_JOB_ID-threads-$(printf '%02d' "$threads")"
  bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' rayan-profile "$NAMESPACE_HELPER" -- /bin/bash -c '
    set -euo pipefail
    exec "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
      --env="PERSONALITY_REPO_ROOT=$PERSONALITY_REPO_ROOT" --env="PERSONALITY_PROFILE_ROOT=$PERSONALITY_PROFILE_ROOT" \
      --env="SLURM_JOB_ID=$SLURM_JOB_ID" --env="PERSONALITY_HOST_NETNS=$PERSONALITY_HOST_NETNS" \
      --env="PERSONALITY_HOST_UID=$PERSONALITY_HOST_UID" --env="PERSONALITY_HOST_GID=$PERSONALITY_HOST_GID" \
      --env="PERSONALITY_ISOLATED_NETNS=$PERSONALITY_ISOLATED_NETNS" \
      --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" --env="HF_HUB_DISABLE_TELEMETRY=1" \
      --entrypoint=python rayan-n0-base -I "$AUDIT_HELPER" "$PROFILE_THREADS"
  '
done
