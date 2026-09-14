#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

BASE="${RAYAN_COMPUTE_ROOT:-$HOME/rayan-compute}"
ROOT="${ALICE_N0_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
UDOCKER="${RAYAN_UDOCKER:-$BASE/tools/udocker-1.3.17/udocker/udocker}"
CONTAINER="${RAYAN_N0_CONTAINER:-rayan-n0-base}"

export UDOCKER_DIR="${UDOCKER_DIR:-$BASE/udocker-store}"
export UDOCKER_TMP="${UDOCKER_TMP:-$BASE/udocker-tmp}"

if [[ ! -x "$UDOCKER" ]]; then
  echo "udocker executable not found or not executable: $UDOCKER" >&2
  exit 70
fi

mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

# NVIDIA binding must be refreshed on the allocated compute node. Magnolia's
# login/runtime stack is CentOS 7/glibc 2.17, while the validated N0 user-space
# runtime lives inside the Debian 12 udocker container.
"$UDOCKER" setup --nvidia --force "$CONTAINER"
"$UDOCKER" setup "$CONTAINER"

ARGS=(
  run
  --volume="$BASE:$BASE"
  --workdir="$ROOT"
  --env="PYTHON=python"
)

for name in \
  CUDA_VISIBLE_DEVICES \
  ALICE_N0_WORKDIR \
  N0_SMOKE_WORKDIR \
  N0_SMOKE_REQUIRE_CUDA \
  N0_SMOKE_MIN_CUDA_DEVICES \
  N0_SEQUENCE_LENGTH \
  N0_MICRO_BATCH_SIZE \
  N0_GRAD_ACCUM \
  N0_MAX_STEPS \
  N0_WARMUP_STEPS \
  N0_SAVE_EVERY \
  N0_MIXED_PRECISION \
  HF_HOME
  do
    if [[ -n "${!name:-}" ]]; then
      ARGS+=(--env="$name=${!name}")
    fi
  done

ARGS+=(--entrypoint=/bin/bash "$CONTAINER")

exec "$UDOCKER" "${ARGS[@]}" "$@"
