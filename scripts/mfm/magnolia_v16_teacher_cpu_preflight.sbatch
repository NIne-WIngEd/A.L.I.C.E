#!/usr/bin/env bash
#SBATCH --job-name=rayan-mfm-v16-teacher-cpu
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# This job processes an externally pinned, admitted v1.6 teacher corpus. It
# performs no GPU training. Unlike the public diagnostic, it never opens a
# source or manifest before entering a loopback-only network namespace.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?submit this script with sbatch}"
: "${MFM_REPO_ROOT:?absolute MFM checkout required}"
: "${MFM_REPO_COMMIT:?externally pinned 40-character MFM commit required}"
: "${MFM_FOUNDATION_ROOT:?absolute foundation checkout required}"
: "${MFM_FOUNDATION_COMMIT:?externally pinned 40-character foundation commit required}"
: "${MFM_TEACHER_MANIFEST:?absolute private v1.6 teacher manifest required}"
: "${MFM_TEACHER_MANIFEST_SHA256:?externally pinned manifest SHA-256 required}"
: "${MFM_OWNER_AUTH_REF:?owner authorization reference required}"
: "${MFM_PREPARED_BASE_DIR:?absolute MFM role base required}"
: "${MFM_PREPARED_BASE_RECEIPT:?absolute role base receipt required}"
: "${MFM_PREPARED_BASE_RECEIPT_FILE_SHA256:?externally pinned receipt file SHA-256 required}"
: "${MFM_PREFLIGHT_RECEIPT:?new absolute CPU preflight receipt required}"

COMPUTE_ROOT="$(realpath -e "${RAYAN_COMPUTE_ROOT:-$HOME/rayan-compute}")"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
CONTAINER="${MFM_CPU_CONTAINER:-rayan-n0-base}"
[[ -x "$UDOCKER" ]] || { echo 'local udocker unavailable' >&2; exit 2; }
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
export UDOCKER COMPUTE_ROOT CONTAINER
mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

for value in "$MFM_TEACHER_MANIFEST_SHA256" \
             "$MFM_PREPARED_BASE_RECEIPT_FILE_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid external SHA-256 pin' >&2; exit 2; }
done
for value in "$MFM_REPO_COMMIT" "$MFM_FOUNDATION_COMMIT"; do
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid external commit pin' >&2; exit 2; }
done

# Resolution here touches only the code checkout. Private manifest paths are
# not resolved, hashed, or opened until the --inside-namespace invocation.
MFM_REPO_ROOT="$(realpath -e "$MFM_REPO_ROOT")"
case "$MFM_REPO_ROOT" in
  "$COMPUTE_ROOT"/*) ;;
  *) echo 'MFM checkout is outside compute root' >&2; exit 2 ;;
esac
SELF="$MFM_REPO_ROOT/scripts/mfm/magnolia_v16_teacher_cpu_preflight.sbatch"
[[ -f "$SELF" ]] || { echo 'teacher CPU script missing from checkout' >&2; exit 2; }

if [[ "${1:-}" == --inside-namespace ]]; then
  : "${MFM_HOST_NETNS:?outer host namespace pin was not supplied}"
else
  MFM_HOST_NETNS="$(readlink /proc/self/ns/net)"
fi
MFM_HOST_UID="$(id -u)"
MFM_HOST_GID="$(id -g)"
[[ "$MFM_HOST_UID" != 0 ]] || { echo 'root host UID is not supported' >&2; exit 2; }
export MFM_HOST_NETNS MFM_HOST_UID MFM_HOST_GID

within_root() {
  local selected
  selected="$(realpath -e "$1")"
  case "$selected" in
    "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;;
    *) echo 'required input is outside compute root' >&2; return 2 ;;
  esac
}

p2_loopback_check() {
  [[ "$(id -u)" == "$MFM_HOST_UID" && "$(id -g)" == "$MFM_HOST_GID" ]] || {
    echo 'namespace changed the caller UID/GID before udocker' >&2
    return 1
  }
  "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
    --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" \
    --entrypoint=/bin/bash "$CONTAINER" -c '
      set -euo pipefail
      python - <<"PY"
import os
import socket

interfaces = sorted(name for _, name in socket.if_nameindex())
namespace = os.readlink("/proc/self/ns/net")
if interfaces != ["lo"] or namespace == os.environ["MFM_HOST_NETNS"]:
    raise SystemExit("private MFM CPU run has no loopback-only P2 namespace")
print("private_p2_netns=" + namespace + " interfaces=lo")
PY
    '
}
export -f p2_loopback_check

if [[ "${1:-}" == --inside-namespace ]]; then
  # This is a fresh unshare child. Verify the actual P2 process before any
  # private manifest/source bytes are opened. The trainer repeats this guard.
  p2_loopback_check

  [[ "$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD)" == "$MFM_REPO_COMMIT" &&
     -z "$(cd "$MFM_REPO_ROOT" && git status --porcelain)" ]] || {
    echo 'MFM checkout differs from exact clean commit' >&2; exit 3;
  }
  MFM_FOUNDATION_ROOT="$(within_root "$MFM_FOUNDATION_ROOT")"
  [[ "$(cd "$MFM_FOUNDATION_ROOT" && git rev-parse HEAD)" == "$MFM_FOUNDATION_COMMIT" &&
     -z "$(cd "$MFM_FOUNDATION_ROOT" && git status --porcelain)" ]] || {
    echo 'foundation checkout differs from exact clean commit' >&2; exit 3;
  }
  MFM_TEACHER_MANIFEST="$(within_root "$MFM_TEACHER_MANIFEST")"
  MFM_PREPARED_BASE_DIR="$(within_root "$MFM_PREPARED_BASE_DIR")"
  MFM_PREPARED_BASE_RECEIPT="$(within_root "$MFM_PREPARED_BASE_RECEIPT")"
  [[ -f "$MFM_TEACHER_MANIFEST" && -d "$MFM_PREPARED_BASE_DIR" &&
     -f "$MFM_PREPARED_BASE_RECEIPT" ]] || {
    echo 'private corpus or verified base input is missing' >&2; exit 3;
  }
  for pair in "$MFM_TEACHER_MANIFEST_SHA256:$MFM_TEACHER_MANIFEST" \
              "$MFM_PREPARED_BASE_RECEIPT_FILE_SHA256:$MFM_PREPARED_BASE_RECEIPT"; do
    printf '%s  %s\n' "${pair%%:*}" "${pair#*:}" | sha256sum -c - >/dev/null
  done
  RECEIPT_PARENT="$(within_root "$(dirname "$MFM_PREFLIGHT_RECEIPT")")"
  RECEIPT_NAME="$(basename "$MFM_PREFLIGHT_RECEIPT")"
  [[ "$RECEIPT_NAME" != . && "$RECEIPT_NAME" != .. &&
     "$RECEIPT_NAME" != */* ]] || { echo 'invalid receipt name' >&2; exit 3; }
  MFM_PREFLIGHT_RECEIPT="$RECEIPT_PARENT/$RECEIPT_NAME"
  [[ ! -e "$MFM_PREFLIGHT_RECEIPT" && ! -L "$MFM_PREFLIGHT_RECEIPT" ]] || {
    echo 'CPU preflight receipt already exists' >&2; exit 3;
  }

  PILLOW="$COMPUTE_ROOT/mfm/wheels/pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  VISION="$COMPUTE_ROOT/mfm/wheels/torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl"
  printf '%s  %s\n' \
    106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f \
    "$PILLOW" | sha256sum -c - >/dev/null
  printf '%s  %s\n' \
    6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627 \
    "$VISION" | sha256sum -c - >/dev/null
  PROCESSOR_DEPS="$COMPUTE_ROOT/mfm/runtime/teacher-processor-deps-${SLURM_JOB_ID}"
  mkdir -p "$COMPUTE_ROOT/mfm/runtime"
  [[ ! -e "$PROCESSOR_DEPS" && ! -L "$PROCESSOR_DEPS" ]] || {
    echo 'job runtime already exists' >&2; exit 3;
  }

  "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
    --entrypoint=/bin/bash "$CONTAINER" -c '
      set -euo pipefail
      python -m pip install --isolated --no-index --no-deps --no-cache-dir \
        --no-compile --target "$1" "$2" "$3"
    ' mfm-teacher-install "$PROCESSOR_DEPS" "$PILLOW" "$VISION"

  SOURCE_BUDGET="${MFM_MAX_SOURCE_TOKENS:-32768}"
  TARGET_BUDGET="${MFM_MAX_TARGET_TOKENS:-8192}"
  SPECIALIST_WIDTH="${MFM_SPECIALIST_WIDTH:-768}"
  SPECIALIST_LAYERS="${MFM_SPECIALIST_LAYERS:-6}"
  SPECIALIST_HEADS="${MFM_SPECIALIST_HEADS:-12}"
  echo "job=$SLURM_JOB_ID node=$(hostname) MFM=$MFM_REPO_COMMIT foundation=$MFM_FOUNDATION_COMMIT"
  echo "teacher_cpu_preflight=$MFM_PREFLIGHT_RECEIPT"
  "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
    --workdir="$MFM_REPO_ROOT" \
    --env="PYTHONPATH=$PROCESSOR_DEPS:$MFM_REPO_ROOT/src:$MFM_FOUNDATION_ROOT/src" \
    --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
    --env="HF_HUB_DISABLE_TELEMETRY=1" \
    --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
    --entrypoint=/bin/bash "$CONTAINER" -c '
      set -euo pipefail
      python - "$4" "$5" <<"PY"
import sys
from pathlib import Path
import PIL
import torch
import torchvision
import transformers
from transformers import AutoProcessor

assert PIL.__version__ == "11.3.0", PIL.__version__
assert Path(PIL.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torchvision.__version__ == "0.22.1+cu118", torchvision.__version__
assert Path(torchvision.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
assert transformers.__version__ == "5.17.0", transformers.__version__
processor = AutoProcessor.from_pretrained(
    sys.argv[1], trust_remote_code=False, local_files_only=True)
assert type(processor).__name__ == "Gemma4UnifiedProcessor"
print("processor_runtime=" + transformers.__version__)
PY
      COMMON=(--teacher-fit --teacher-training-manifest "$1" \
        --input-sha256 "$2" --owner-authorization-ref "$3" \
        --max-source-tokens "$8" --max-target-tokens "$9" \
        --specialist-width "${10}" --specialist-layers "${11}" \
        --specialist-heads "${12}")
      python -m scripts.mfm.train_v16_formation_specialist data-preflight \
        "${COMMON[@]}"
      python -m scripts.mfm.train_v16_formation_specialist processor-preflight \
        "${COMMON[@]}" --prepared-base-dir "$4" \
        --prepared-base-receipt "$6" --preflight-receipt "$7"
    ' mfm-v16-teacher "$MFM_TEACHER_MANIFEST" \
      "$MFM_TEACHER_MANIFEST_SHA256" "$MFM_OWNER_AUTH_REF" \
      "$MFM_PREPARED_BASE_DIR" "$PROCESSOR_DEPS" \
      "$MFM_PREPARED_BASE_RECEIPT" "$MFM_PREFLIGHT_RECEIPT" \
      "$SOURCE_BUDGET" "$TARGET_BUDGET" "$SPECIALIST_WIDTH" \
      "$SPECIALIST_LAYERS" "$SPECIALIST_HEADS"
  exit 0
fi

[[ $# == 0 ]] || { echo 'unexpected job argument' >&2; exit 2; }
command -v unshare >/dev/null || { echo 'unshare unavailable' >&2; exit 2; }
"$UDOCKER" setup "$CONTAINER" >/dev/null

# No data is opened by either capability check. Select the first supported
# route, then execute the *entire* private flow once under that route. A data
# or processor failure cannot trigger fallback to host networking.
if unshare --help | grep -Fq -- '--map-current-user' && \
   unshare --user --map-current-user --net -- bash -c \
     'set -euo pipefail; p2_loopback_check'; then
  NAMESPACE_ARGS=(--user --map-current-user --net)
elif unshare --net -- bash -c \
    'set -euo pipefail; p2_loopback_check'; then
  NAMESPACE_ARGS=(--net)
else
  echo 'no supported loopback-only network namespace; private CPU pass blocked' >&2
  exit 3
fi
unshare "${NAMESPACE_ARGS[@]}" -- bash "$SELF" --inside-namespace
