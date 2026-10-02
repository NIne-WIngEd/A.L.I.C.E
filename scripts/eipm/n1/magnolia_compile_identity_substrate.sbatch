#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-n1-compile
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Pinned candidate source compilation only: no source promotion, model, or gradient.
# Private path values remain STRINGS until fresh namespace + actual P2 checks.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set the separate clean public-code checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?set its externally tested 40-hex commit}"
: "${PRIVATE_PACKAGE_PATH:?set the exact private curated frontier archive path}"
[[ $# == 0 && "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]] || exit 2

COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan && "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
PERSONALITY_REPO_ROOT="$(realpath -e "$PERSONALITY_REPO_ROOT")"
case "$PERSONALITY_REPO_ROOT" in "$COMPUTE_ROOT"/*) ;; *) exit 2 ;; esac
export GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
[[ "$(git -c core.fsmonitor=false -c core.hooksPath=/dev/null --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" ]]
[[ -z "$(git -c core.fsmonitor=false -c core.hooksPath=/dev/null --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]]
# Lexical checks only. Do NOT realpath/stat/hash/open PRIVATE_PACKAGE_PATH here.
[[ "$PRIVATE_PACKAGE_PATH" == /* && "$PRIVATE_PACKAGE_PATH" != *'/../'* && "$PRIVATE_PACKAGE_PATH" != *'/./'* ]]
case "$PRIVATE_PACKAGE_PATH" in "$PERSONALITY_REPO_ROOT"/*) exit 2 ;; esac
RAW_PATH="${PERSONALITY_RAW_LINEAGE_PATH:-}"
RAW_SHA="${PERSONALITY_RAW_LINEAGE_SHA256:-}"
RAW_SOURCE="${PERSONALITY_RAW_SOURCE_PATH:-}"
if [[ -n "$RAW_SOURCE" ]]; then
  [[ -z "$RAW_PATH" && -z "$RAW_SHA" && "$RAW_SOURCE" == /* && "$RAW_SOURCE" != *'/../'* && "$RAW_SOURCE" != *'/./'* ]] || exit 2
  case "$RAW_SOURCE" in "$PERSONALITY_REPO_ROOT"/*) exit 2 ;; esac
fi
if [[ -n "$RAW_PATH" || -n "$RAW_SHA" ]]; then
  [[ "$RAW_PATH" == /* && "$RAW_SHA" =~ ^[0-9a-f]{64}$ && "$RAW_PATH" != *'/../'* && "$RAW_PATH" != *'/./'* ]] || exit 2
  case "$RAW_PATH" in "$PERSONALITY_REPO_ROOT"/*) exit 2 ;; esac
fi

for directory in "$COMPUTE_ROOT/rayan-personality" "$COMPUTE_ROOT/rayan-personality/private-compile"; do
  if [[ -e "$directory" || -L "$directory" ]]; then
    [[ ! -L "$directory" && -d "$directory" ]] || exit 2
  else
    mkdir -m 700 "$directory"
  fi
  [[ "$(realpath -e "$directory")" == "$directory" && "$(stat -c %U "$directory")" == mxrayan && "$(stat -c %a "$directory")" == 700 ]] || exit 2
done
PERSONALITY_RUN_ROOT="$COMPUTE_ROOT/rayan-personality/private-compile/rayan-identity-compile-${SLURM_JOB_ID}"
[[ ! -e "$PERSONALITY_RUN_ROOT" && ! -L "$PERSONALITY_RUN_ROOT" ]] || exit 2
mkdir -m 700 "$PERSONALITY_RUN_ROOT"
NAMESPACE_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/n1/enter_private_network_namespace.py"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
[[ -f "$NAMESPACE_HELPER" && -x "$UDOCKER" ]] || exit 2
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
[[ -d "$UDOCKER_DIR" && -d "$UDOCKER_TMP" ]] || exit 2
export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)"
export PERSONALITY_HOST_UID="$(id -u)"
export PERSONALITY_HOST_GID="$(id -g)"
export COMPUTE_ROOT PERSONALITY_REPO_ROOT PERSONALITY_EXPECTED_REVISION PERSONALITY_RUN_ROOT
export PRIVATE_PACKAGE_PATH NAMESPACE_HELPER UDOCKER
export PERSONALITY_RAW_LINEAGE_PATH="$RAW_PATH"
export PERSONALITY_RAW_LINEAGE_SHA256="$RAW_SHA"
export PERSONALITY_RAW_SOURCE_PATH="$RAW_SOURCE"
export GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0

echo "job=$SLURM_JOB_ID source_commit=$PERSONALITY_EXPECTED_REVISION stage=isolated_candidate_compilation"
# Legacy host module is ONLY the stdlib namespace helper. Existing P2 is reused
# unchanged: no setup, pull, execution-mode change, dependency install or fallback.
if bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' \
    rayan-personality-host-python "$NAMESPACE_HELPER" -- /bin/bash -c '
      set -euo pipefail
      umask 077
      [[ "$(id -u)" == "$PERSONALITY_HOST_UID" && "$(id -g)" == "$PERSONALITY_HOST_GID" ]]
      exec "$UDOCKER" run \
        --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
        --workdir="$PERSONALITY_REPO_ROOT" \
        --env="COMPUTE_ROOT=$COMPUTE_ROOT" \
        --env="PERSONALITY_REPO_ROOT=$PERSONALITY_REPO_ROOT" \
        --env="PERSONALITY_EXPECTED_REVISION=$PERSONALITY_EXPECTED_REVISION" \
        --env="PERSONALITY_RUN_ROOT=$PERSONALITY_RUN_ROOT" \
        --env="PRIVATE_PACKAGE_PATH=$PRIVATE_PACKAGE_PATH" \
        --env="PERSONALITY_RAW_LINEAGE_PATH=$PERSONALITY_RAW_LINEAGE_PATH" \
        --env="PERSONALITY_RAW_LINEAGE_SHA256=$PERSONALITY_RAW_LINEAGE_SHA256" \
        --env="PERSONALITY_RAW_SOURCE_PATH=$PERSONALITY_RAW_SOURCE_PATH" \
        --env="PERSONALITY_HOST_NETNS=$PERSONALITY_HOST_NETNS" \
        --env="PERSONALITY_HOST_UID=$PERSONALITY_HOST_UID" \
        --env="PERSONALITY_HOST_GID=$PERSONALITY_HOST_GID" \
        --env="PERSONALITY_ISOLATED_NETNS=$PERSONALITY_ISOLATED_NETNS" \
        --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
        --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
        --entrypoint=/bin/bash rayan-n0-base -c '\''
          set -euo pipefail
          umask 077
          exec python -I "$1" --p2-compile-stage
        '\'' rayan-personality-compiler "$NAMESPACE_HELPER"
    '; then
  exit 0
else
  status=$?
  echo "state=FAILED_UNQUALIFIED isolated_compilation_exit=$status" >&2
  exit "$status"
fi
