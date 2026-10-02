#!/usr/bin/env bash
#SBATCH --job-name=rayan-mfm-feature-export
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Same-user allocated CPU export. No model/data work on the login node, no
# modification of the shared container, no network fallback or target access
# by the representation backbone. All inputs are externally pinned.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit with sbatch}"
: "${MFM_REPO_ROOT:?}"
: "${MFM_REPO_COMMIT:?}"
: "${MFM_FOUNDATION_ROOT:?}"
: "${MFM_FOUNDATION_COMMIT:?}"
: "${MFM_TEACHER_MANIFEST:?}"
: "${MFM_TEACHER_MANIFEST_SHA256:?}"
: "${MFM_OWNER_AUTH_REF:?}"
: "${MFM_PREPARED_BASE_DIR:?}"
: "${MFM_PREPARED_BASE_RECEIPT:?}"
: "${MFM_PREPARED_BASE_RECEIPT_FILE_SHA256:?}"
: "${MFM_PREFLIGHT_RECEIPT:?}"
: "${MFM_PREFLIGHT_RECEIPT_FILE_SHA256:?}"
: "${MFM_FEATURE_OUTPUT:?}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
COMPUTE_ROOT="$(realpath -e /homes/01/mxrayan/rayan-compute)"
[[ "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]]
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
CONTAINER=rayan-n0-base
MFM_REPO_ROOT="$(realpath -e "$MFM_REPO_ROOT")"
case "$MFM_REPO_ROOT" in "$COMPUTE_ROOT"/*) ;; *) exit 2 ;; esac
[[ "$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD)" == "$MFM_REPO_COMMIT" &&
   -z "$(cd "$MFM_REPO_ROOT" && git status --porcelain)" ]]
export COMPUTE_ROOT UDOCKER CONTAINER MFM_REPO_ROOT

if [[ "${1:-}" != --inside-namespace ]]; then
  [[ $# == 0 ]]
  export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)"
  export MFM_HOST_UID="$(id -u)" MFM_HOST_GID="$(id -g)"
  module load python/3.11.5
  exec python3 -I "$MFM_REPO_ROOT/scripts/mfm/enter_private_network_namespace.py" \
    -- /bin/bash "$MFM_REPO_ROOT/scripts/mfm/magnolia_v16_frozen_feature_export.sbatch" --inside-namespace
fi
[[ $# == 1 ]]
[[ "$(id -u)" == "$MFM_HOST_UID" && "$(id -g)" == "$MFM_HOST_GID" ]]
# Establish the actual container namespace before opening private paths.
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" --entrypoint=/bin/bash "$CONTAINER" -c '
python - <<"PY"
import os, socket
assert sorted(name for _, name in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
print("private_p2_loopback_verified", flush=True)
PY
'
within_root() {
  local selected
  selected="$(realpath -e "$1")"
  case "$selected" in "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;; *) return 2 ;; esac
}
MFM_FOUNDATION_ROOT="$(within_root "$MFM_FOUNDATION_ROOT")"
[[ "$(cd "$MFM_FOUNDATION_ROOT" && git rev-parse HEAD)" == "$MFM_FOUNDATION_COMMIT" &&
   -z "$(cd "$MFM_FOUNDATION_ROOT" && git status --porcelain)" ]]
MFM_TEACHER_MANIFEST="$(within_root "$MFM_TEACHER_MANIFEST")"
MFM_PREPARED_BASE_DIR="$(within_root "$MFM_PREPARED_BASE_DIR")"
MFM_PREPARED_BASE_RECEIPT="$(within_root "$MFM_PREPARED_BASE_RECEIPT")"
MFM_PREFLIGHT_RECEIPT="$(within_root "$MFM_PREFLIGHT_RECEIPT")"
for pair in "$MFM_TEACHER_MANIFEST_SHA256:$MFM_TEACHER_MANIFEST" \
            "$MFM_PREPARED_BASE_RECEIPT_FILE_SHA256:$MFM_PREPARED_BASE_RECEIPT" \
            "$MFM_PREFLIGHT_RECEIPT_FILE_SHA256:$MFM_PREFLIGHT_RECEIPT"; do
  [[ "${pair%%:*}" =~ ^[0-9a-f]{64}$ ]]
  printf '%s  %s\n' "${pair%%:*}" "${pair#*:}" | sha256sum -c - >/dev/null
done
output_parent="$(within_root "$(dirname "$MFM_FEATURE_OUTPUT")")"
output_name="$(basename "$MFM_FEATURE_OUTPUT")"
[[ "$output_name" =~ ^rayan-mfm-features-[a-zA-Z0-9-]+$ ]]
[[ "$MFM_FEATURE_OUTPUT" == "$output_parent/$output_name" && ! -L "$MFM_FEATURE_OUTPUT" ]]
if [[ "${MFM_FEATURE_RESUME:-0}" == 1 ]]; then
  [[ -d "$MFM_FEATURE_OUTPUT" && ! -e "$MFM_FEATURE_OUTPUT/bank.json" ]]
else
  [[ "${MFM_FEATURE_RESUME:-0}" == 0 && ! -e "$MFM_FEATURE_OUTPUT" ]]
fi
PILLOW="$COMPUTE_ROOT/mfm/wheels/pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
VISION="$COMPUTE_ROOT/mfm/wheels/torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl"
printf '%s  %s\n' 106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f "$PILLOW" | sha256sum -c - >/dev/null
printf '%s  %s\n' 6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627 "$VISION" | sha256sum -c - >/dev/null
RUNTIME="$COMPUTE_ROOT/rayan-mfm/runtime/feature-deps-${SLURM_JOB_ID}"
mkdir -p "$(dirname "$RUNTIME")"
[[ ! -e "$RUNTIME" && ! -L "$RUNTIME" ]]
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --entrypoint=/bin/bash "$CONTAINER" -c '
set -euo pipefail
python -m pip install --isolated --no-index --no-deps --no-cache-dir --no-compile --target "$1" "$2" "$3"
' feature-deps "$RUNTIME" "$PILLOW" "$VISION"
echo "rayan_feature_export_job=$SLURM_JOB_ID node=$(hostname) code=$MFM_REPO_COMMIT"
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$MFM_REPO_ROOT" \
  --env="PYTHONPATH=$RUNTIME:$MFM_REPO_ROOT/src:$MFM_FOUNDATION_ROOT/src" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
  --env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --entrypoint=/bin/bash "$CONTAINER" -c '
set -euo pipefail
umask 077
python - <<"PY"
import torch, torchvision, transformers, PIL
assert torch.__version__ == "2.7.1+cu118"
assert torchvision.__version__ == "0.22.1+cu118"
assert transformers.__version__ == "5.17.0"
assert PIL.__version__ == "11.3.0"
print("exact_feature_runtime_verified", flush=True)
PY
command=(python -m scripts.mfm.export_v16_frozen_features \
  --teacher-training-manifest "$1" --input-sha256 "$2" \
  --owner-authorization-ref "$3" --prepared-base-dir "$4" \
  --prepared-base-receipt "$5" --preflight-receipt "$6" --output-dir "$7")
if [[ "$8" == 1 ]]; then command+=(--resume); fi
"${command[@]}"
' feature-export "$MFM_TEACHER_MANIFEST" "$MFM_TEACHER_MANIFEST_SHA256" \
  "$MFM_OWNER_AUTH_REF" "$MFM_PREPARED_BASE_DIR" "$MFM_PREPARED_BASE_RECEIPT" \
  "$MFM_PREFLIGHT_RECEIPT" "$MFM_FEATURE_OUTPUT" "${MFM_FEATURE_RESUME:-0}"
