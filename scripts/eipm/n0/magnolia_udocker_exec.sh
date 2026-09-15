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
NVIDIA_MODE="${RAYAN_UDOCKER_NVIDIA:-auto}"

export UDOCKER_DIR="${UDOCKER_DIR:-$BASE/udocker-store}"
export UDOCKER_TMP="${UDOCKER_TMP:-$BASE/udocker-tmp}"

if [[ ! -x "$UDOCKER" ]]; then
  echo "udocker executable not found or not executable: $UDOCKER" >&2
  exit 70
fi

mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

# Magnolia's host stack is CentOS 7/glibc 2.17, while the validated N0
# user-space runtime lives inside the Debian 12 udocker container. GPU jobs
# must refresh NVIDIA binding on the allocated compute node. CPU preprocessing
# jobs intentionally skip that step so they do not require a GPU allocation.
case "$NVIDIA_MODE" in
  1|true|yes|on)
    "$UDOCKER" setup --nvidia --force "$CONTAINER"
    ;;
  0|false|no|off)
    ;;
  auto)
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      "$UDOCKER" setup --nvidia --force "$CONTAINER"
    fi
    ;;
  *)
    echo "RAYAN_UDOCKER_NVIDIA must be auto, 1/true/yes/on, or 0/false/no/off" >&2
    exit 65
    ;;
esac
"$UDOCKER" setup "$CONTAINER"

ARGS=(
  run
  --volume="$BASE:$BASE"
  --workdir="$ROOT"
  --env="PYTHON=python"
)

# Forward only the stage controls that are intentionally part of the N0
# runtime contract. In particular, v0.2 workdir/checkpoint overrides must cross
# the udocker boundary rather than silently falling back to container-local
# defaults. The normal v0.2 default remains derived from the mounted repo root,
# never container $HOME (/root).
for name in \
  CUDA_VISIBLE_DEVICES \
  PYTHONPATH \
  ALICE_N0_WORKDIR \
  ALICE_N0_V02_WORKDIR \
  N0_SMOKE_WORKDIR \
  N0_SMOKE_REQUIRE_CUDA \
  N0_SMOKE_MIN_CUDA_DEVICES \
  N0_BOOTSTRAP_CHARS_PER_SOURCE \
  N0_SHARD_MB \
  N0_SEQUENCE_LENGTH \
  N0_MICRO_BATCH_SIZE \
  N0_GRAD_ACCUM \
  N0_MAX_STEPS \
  N0_WARMUP_STEPS \
  N0_SCHEDULER_TOTAL_STEPS \
  N0_SAVE_EVERY \
  N0_LEARNING_RATE \
  N0_MIXED_PRECISION \
  N0_RESUME_FROM \
  N0_MAIN_PROCESS_PORT \
  N0_V02_CORPUS_DIR \
  N0_V02_TOKENIZER_DIR \
  N0_V02_TEACHER_DIR \
  N0_V02_RUN_ROOT \
  N0_V02_FIRST_TRANCHE_ROOT \
  N0_V02_TEACHER_DEV_EVAL_ROOT \
  N0_V02_SEED \
  N0_V02_MAX_STEPS \
  N0_V02_SAVE_EVERY \
  HF_HOME \
  HF_HUB_OFFLINE \
  TRANSFORMERS_OFFLINE \
  TOKENIZERS_PARALLELISM
  do
    if [[ -n "${!name:-}" ]]; then
      ARGS+=(--env="$name=${!name}")
    fi
  done

ARGS+=(--entrypoint=/bin/bash "$CONTAINER")

exec "$UDOCKER" "${ARGS[@]}" "$@"
