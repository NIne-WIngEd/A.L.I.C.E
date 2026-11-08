#!/usr/bin/env bash
#SBATCH --job-name=rayan-connected-personality
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?exact clean public-code checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?externally tested exact commit}"
: "${PRIVATE_PACKAGE_PATH:?exact curated source archive}"
[[ $# == 0 && "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
if [[ "${SLURM_RESTART_COUNT:-0}" != 0 ]]; then
  echo "state=INTERRUPTED_RESTART_REFUSED job=$SLURM_JOB_ID explicit_pinned_continuation_required=true" >&2
  exit 2
fi
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]] || exit 2
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$COMPUTE_ROOT" == /homes/01/mxrayan/rayan-compute && "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan && "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
PERSONALITY_REPO_ROOT="$(realpath -e "$PERSONALITY_REPO_ROOT")"
case "$PERSONALITY_REPO_ROOT" in "$COMPUTE_ROOT"/*) ;; *) exit 2 ;; esac
export GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
[[ "$(git -c core.fsmonitor=false -c core.hooksPath=/dev/null --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" ]]
[[ -z "$(git -c core.fsmonitor=false -c core.hooksPath=/dev/null --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]]
# Private input stays an uninterpreted string until the actual P2 guard.
[[ "$PRIVATE_PACKAGE_PATH" == /* && "$PRIVATE_PACKAGE_PATH" != *'/../'* && "$PRIVATE_PACKAGE_PATH" != *'/./'* ]] || exit 2
case "$PRIVATE_PACKAGE_PATH" in "$PERSONALITY_REPO_ROOT"/*) exit 2 ;; esac
for name in PERSONALITY_CONSTRUCTION_PLAN PERSONALITY_CONSTRUCTION_PLAN_SHA256 PERSONALITY_CONSTRUCTION_HELPER_SHA256; do
  [[ -n "${!name:-}" ]] || exit 2
done
[[ "$PERSONALITY_CONSTRUCTION_PLAN_SHA256" =~ ^[0-9a-f]{64}$ && "$PERSONALITY_CONSTRUCTION_HELPER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
export PERSONALITY_CONSTRUCTION_PLAN PERSONALITY_CONSTRUCTION_PLAN_SHA256 PERSONALITY_CONSTRUCTION_HELPER_SHA256
for directory in "$COMPUTE_ROOT/rayan-personality" "$COMPUTE_ROOT/rayan-personality/private-compile"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    [[ ! -L "$directory" && -d "$directory" ]] || exit 2
  else
    mkdir -m 700 "$directory"
  fi
  [[ "$(realpath -e "$directory")" == "$directory" && "$(stat -c %U "$directory")" == mxrayan && "$(stat -c %a "$directory")" == 700 ]] || exit 2
done
PERSONALITY_RUN_ROOT="$COMPUTE_ROOT/rayan-personality/private-compile/rayan-connected-personality-${SLURM_JOB_ID}"
if [[ -e "$PERSONALITY_RUN_ROOT" || -L "$PERSONALITY_RUN_ROOT" ]]; then
  echo "state=OCCUPIED_OUTPUT_REFUSED job=$SLURM_JOB_ID original_evidence_preserved=true" >&2
  exit 2
fi
mkdir -m 700 "$PERSONALITY_RUN_ROOT"
NAMESPACE_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/n1/enter_private_network_namespace.py"
AUDIT_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/gemma_n0/construct_personality.py"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
[[ -f "$NAMESPACE_HELPER" && -f "$AUDIT_HELPER" && -x "$UDOCKER" ]] || exit 2
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store" UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
[[ -d "$UDOCKER_DIR" && -d "$UDOCKER_TMP" ]] || exit 2
export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)"
export PERSONALITY_HOST_UID="$(id -u)" PERSONALITY_HOST_GID="$(id -g)"
export COMPUTE_ROOT PERSONALITY_REPO_ROOT PERSONALITY_EXPECTED_REVISION PERSONALITY_RUN_ROOT PRIVATE_PACKAGE_PATH
export NAMESPACE_HELPER AUDIT_HELPER UDOCKER PERSONALITY_CONSTRUCTION_HELPER_SHA256
echo "job=$SLURM_JOB_ID source_commit=$PERSONALITY_EXPECTED_REVISION stage=isolated_connected_personality_construction"
# Existing original-owner USER+NET and one existing P2; no setup/install/fallback.
if bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' \
    rayan-personality-host-python "$NAMESPACE_HELPER" -- /bin/bash -c '
      set -euo pipefail
      umask 077
      [[ "$(id -u)" == "$PERSONALITY_HOST_UID" && "$(id -g)" == "$PERSONALITY_HOST_GID" ]]
      exec "$UDOCKER" run \
        --env="PERSONALITY_CONSTRUCTION_PLAN=$PERSONALITY_CONSTRUCTION_PLAN" \
        --env="PERSONALITY_CONSTRUCTION_PLAN_SHA256=$PERSONALITY_CONSTRUCTION_PLAN_SHA256" \
        --env="SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" \
        --env="SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE" \
        --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
        --env="PERSONALITY_CONSTRUCTION_HELPER_SHA256=$PERSONALITY_CONSTRUCTION_HELPER_SHA256" \
        --env="COMPUTE_ROOT=$COMPUTE_ROOT" --env="PERSONALITY_REPO_ROOT=$PERSONALITY_REPO_ROOT" \
        --env="PERSONALITY_EXPECTED_REVISION=$PERSONALITY_EXPECTED_REVISION" \
        --env="PERSONALITY_RUN_ROOT=$PERSONALITY_RUN_ROOT" --env="PRIVATE_PACKAGE_PATH=$PRIVATE_PACKAGE_PATH" \
        --env="PERSONALITY_HOST_NETNS=$PERSONALITY_HOST_NETNS" \
        --env="PERSONALITY_HOST_UID=$PERSONALITY_HOST_UID" --env="PERSONALITY_HOST_GID=$PERSONALITY_HOST_GID" \
        --env="PERSONALITY_ISOLATED_NETNS=$PERSONALITY_ISOLATED_NETNS" \
        --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
        --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
        --entrypoint=/bin/bash rayan-n0-base -c '\''
          set -euo pipefail
          umask 077
          exec python -I "$1" --p2-connected-construction-stage
        '\'' rayan-personality-retained-source "$AUDIT_HELPER"
    '; then
  exit 0
else
  status=$?
  echo "state=FAILED_UNQUALIFIED isolated_retained_source_exit=$status" >&2
  exit "$status"
fi
