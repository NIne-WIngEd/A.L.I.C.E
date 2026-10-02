#!/bin/bash
#SBATCH --job-name=rayan-mfm-full-capacity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
root=/homes/01/mxrayan/rayan-compute
: "${MFM_REPO_ROOT:?}" "${MFM_REPO_COMMIT:?}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
[[ "$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD)" == "$MFM_REPO_COMMIT" && -z "$(cd "$MFM_REPO_ROOT" && git status --porcelain)" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
container=rayan-mfm-cache-gpu-v2-576613
udocker="$root/tools/udocker-1.3.17/udocker/udocker"
exec 9>"$root/rayan-mfm/jobs/dedicated-gpu-runtime.lock"
flock -n 9
container_id="$(readlink "$UDOCKER_DIR/containers/$container")"
[[ "$container_id" =~ ^[0-9a-f-]{36}$ ]]
target_root="$(realpath -e "$UDOCKER_DIR/containers/$container_id/ROOT")"
[[ "$target_root" == "$UDOCKER_DIR/containers/$container_id/ROOT" && "$(stat -c %u "$target_root")" == 1905 ]]
mkdir -p "$target_root/homes/01/mxrayan/rayan-compute"
"$udocker" setup --nvidia --force "$container"
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
python3 -I "$MFM_REPO_ROOT/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
root="$1"
repo="$2"
"$root/tools/udocker-1.3.17/udocker/udocker" run --volume="$root:$root" --workdir="$repo" \
  --env="PYTHONPATH=$root/rayan-mfm/runtime/feature-deps-576610:$repo/src" \
  --env="OMP_NUM_THREADS=8" --env="MKL_NUM_THREADS=8" \
  --env="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" \
  --entrypoint=/bin/bash rayan-mfm-cache-gpu-v2-576613 -c '\''
set -euo pipefail
python - <<"PY"
import os, socket, torch
assert sorted(n for _, n in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
assert torch.__version__ == "2.7.1+cu118" and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (6, 0)
PY
python -m scripts.mfm.probe_v16_cached_capacity \
  --preflight-receipt "$1/rayan-mfm/receipts/rayan-mfm-processor-576606.json" \
  --preflight-sha256 f2a8d521a567d88833a485604dfd259e232efb20adda1334d4365c55f043230c \
  --output-dir "$1/rayan-mfm/runs/rayan-mfm-synthetic-capacity-$2"
'\'' synthetic-capacity "$root" "$SLURM_JOB_ID"
' capacity-wrapper "$root" "$MFM_REPO_ROOT"
