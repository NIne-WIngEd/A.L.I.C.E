#!/usr/bin/env bash
#SBATCH --job-name=rayan-mfm-v16-cpu
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Public 21-case processor diagnostic only. No model weights are loaded and
# this does not admit a full-role corpus or train the formation specialist.
set -euo pipefail
umask 077

: "${MFM_REPO_ROOT:?set absolute MFM checkout under RAYAN_COMPUTE_ROOT}"
: "${MFM_FOUNDATION_ROOT:?set absolute pinned foundation checkout under RAYAN_COMPUTE_ROOT}"
: "${SLURM_JOB_ID:?submit this script with sbatch}"

COMPUTE_ROOT="$(realpath -e "${RAYAN_COMPUTE_ROOT:-$HOME/rayan-compute}")"
MFM_REPO_ROOT="$(realpath -e "$MFM_REPO_ROOT")"
MFM_FOUNDATION_ROOT="$(realpath -e "$MFM_FOUNDATION_ROOT")"
for path in "$MFM_REPO_ROOT" "$MFM_FOUNDATION_ROOT"; do
  case "$path" in
    "$COMPUTE_ROOT"/*) ;;
    *) echo "checkout is outside the mounted compute root: $path" >&2; exit 2 ;;
  esac
done

SOURCE_HASH=8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22
SOURCE="$MFM_REPO_ROOT/benchmarks/mfm/v16_authoring_seed.jsonl"
read -r actual_hash _ < <(sha256sum "$SOURCE")
[[ "$actual_hash" == "$SOURCE_HASH" ]] || {
  echo "v1.6 diagnostic fixture differs from pinned bytes" >&2; exit 3;
}

CHECKPOINT_ROOT="$COMPUTE_ROOT/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08"
CLONE="$CHECKPOINT_ROOT/mfm-role-clone-576486"
CLONE_RECEIPT="$COMPUTE_ROOT/mfm/receipts/mfm-clone-576486.json"
PREFLIGHT="$COMPUTE_ROOT/mfm/receipts/mfm-v16-public-processor-${SLURM_JOB_ID}.json"
[[ -d "$CLONE" && -r "$CLONE_RECEIPT" && ! -e "$PREFLIGHT" ]] || {
  echo "clone/receipt missing or job-specific preflight already exists" >&2; exit 4;
}
[[ -f "$MFM_REPO_ROOT/scripts/mfm/train_v16_formation_specialist.py" &&
   -f "$MFM_FOUNDATION_ROOT/src/alice_foundation/gemma4_v1.py" ]] || {
  echo "MFM or foundation checkout is incomplete" >&2; exit 5;
}

UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
CONTAINER="${MFM_CPU_CONTAINER:-rayan-n0-base}"
[[ -x "$UDOCKER" ]] || { echo "udocker is unavailable" >&2; exit 6; }
PILLOW_WHEEL_NAME=pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
PILLOW_WHEEL="${MFM_PILLOW_WHEEL:-$COMPUTE_ROOT/mfm/wheels/$PILLOW_WHEEL_NAME}"
[[ -f "$PILLOW_WHEEL" ]] || {
  echo "stage the pinned Pillow wheel on the login node first: $PILLOW_WHEEL" >&2; exit 7;
}
PILLOW_WHEEL="$(realpath -e "$PILLOW_WHEEL")"
case "$PILLOW_WHEEL" in
  "$COMPUTE_ROOT"/*) ;;
  *) echo "Pillow wheel is outside the mounted compute root: $PILLOW_WHEEL" >&2; exit 7 ;;
esac
[[ "${PILLOW_WHEEL##*/}" == "$PILLOW_WHEEL_NAME" ]] || {
  echo "unexpected Pillow wheel filename" >&2; exit 7;
}
read -r pillow_hash _ < <(sha256sum "$PILLOW_WHEEL")
[[ "$pillow_hash" == 106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f ]] || {
  echo "Pillow wheel differs from pinned bytes" >&2; exit 7;
}
VISION_WHEEL_NAME='torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl'
VISION_WHEEL="${MFM_TORCHVISION_WHEEL:-$COMPUTE_ROOT/mfm/wheels/$VISION_WHEEL_NAME}"
[[ -f "$VISION_WHEEL" ]] || {
  echo "stage the matching TorchVision wheel on the login node first: $VISION_WHEEL" >&2; exit 7;
}
VISION_WHEEL="$(realpath -e "$VISION_WHEEL")"
case "$VISION_WHEEL" in
  "$COMPUTE_ROOT"/*) ;;
  *) echo "TorchVision wheel is outside the mounted compute root: $VISION_WHEEL" >&2; exit 7 ;;
esac
[[ "${VISION_WHEEL##*/}" == "$VISION_WHEEL_NAME" ]] || {
  echo "unexpected TorchVision wheel filename" >&2; exit 7;
}
read -r vision_hash _ < <(sha256sum "$VISION_WHEEL")
[[ "$vision_hash" == 6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627 ]] || {
  echo "TorchVision wheel differs from pinned bytes" >&2; exit 7;
}
PROCESSOR_DEPS="$COMPUTE_ROOT/mfm/runtime/processor-deps-${SLURM_JOB_ID}"
[[ ! -e "$PROCESSOR_DEPS" ]] || { echo "job-specific processor runtime already exists" >&2; exit 7; }
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

echo "job=$SLURM_JOB_ID node=$(hostname) MFM=$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD) foundation=$(cd "$MFM_FOUNDATION_ROOT" && git rev-parse HEAD)"
echo "diagnostic_preflight=$PREFLIGHT"
# Magnolia's host Python/glibc is not the validated scientific runtime.
# Install only into this MFM job's mounted directory; never change the shared
# N0 container. Both wheels are staged on login and verified above. --no-deps
# prevents pip from replacing the container's Torch 2.7.1+cu118.
"$UDOCKER" setup "$CONTAINER"
"$UDOCKER" run \
  --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --entrypoint=/bin/bash "$CONTAINER" -c '
    set -euo pipefail
    python -m pip install --isolated --no-index --no-deps --no-cache-dir \
      --no-compile --target "$1" "$2" "$3"
  ' mfm-v16-install "$PROCESSOR_DEPS" "$PILLOW_WHEEL" "$VISION_WHEEL"
"$UDOCKER" run \
  --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --workdir="$MFM_REPO_ROOT" \
  --env="PYTHONPATH=$PROCESSOR_DEPS:$MFM_REPO_ROOT/src:$MFM_FOUNDATION_ROOT/src" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --entrypoint=/bin/bash "$CONTAINER" -c '
    set -euo pipefail
    # A real local processor load must pass before the trainer rehashes 24 GB.
    python - "$3" "$6" <<'\''PY'\''
import sys
from pathlib import Path
import PIL
from PIL import Image, _imaging
import numpy
import torch
import torchvision
from torchvision.transforms.v2 import functional as tvF
import transformers
from transformers import AutoProcessor
import alice_foundation.gemma4_v1
import cognitive_kernel.formation_learning_v16

assert PIL.__version__ == "11.3.0", PIL.__version__
assert Path(PIL.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), PIL.__file__
assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torchvision.__version__ == "0.22.1+cu118", torchvision.__version__
assert Path(torchvision.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), torchvision.__file__
assert transformers.__version__ == "5.17.0", transformers.__version__
processor = AutoProcessor.from_pretrained(
    sys.argv[1], trust_remote_code=False, local_files_only=True)
assert all(hasattr(processor, name) for name in (
    "apply_chat_template", "tokenizer", "image_token", "audio_token", "video_token"))
print("runtime", torch.__version__, transformers.__version__, "NumPy", numpy.__version__,
      "Pillow", PIL.__version__, "TorchVision", torchvision.__version__,
      "processor", type(processor).__name__)
PY
    exec python -m scripts.mfm.train_v16_formation_specialist processor-preflight \
      --public-synthetic-curriculum "$1" --input-sha256 "$2" \
      --owner-authorization-ref owner-directed-mfm-v16-synthetic \
      --prepared-base-dir "$3" --prepared-base-receipt "$4" \
      --preflight-receipt "$5"
  ' mfm-v16 "$SOURCE" "$SOURCE_HASH" "$CLONE" "$CLONE_RECEIPT" "$PREFLIGHT" "$PROCESSOR_DEPS"
