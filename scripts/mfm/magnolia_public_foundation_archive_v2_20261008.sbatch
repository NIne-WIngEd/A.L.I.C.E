#!/bin/bash
#SBATCH --job-name=rayan-mfm-full-bank-archive
#SBATCH --partition=node
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
: "${MFM_CODE_COMMIT:?exact reviewed archive code commit required}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
root=/homes/01/mxrayan/rayan-compute
repo="$root/rayan-mfm/code/rayan-mfm-full-bank-archive-v2-20261008"
container=rayan-mfm-source-cpu-20261002
[[ "$(cd "$repo" && git rev-parse HEAD)" == "$MFM_CODE_COMMIT" && -z "$(cd "$repo" && git status --porcelain)" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
exec 9>"$root/rayan-mfm/jobs/dedicated-source-runtime.lock"
flock -n 9
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
/usr/bin/time -v python3 -I "$repo/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
root="$1"
repo="$2"
container="$3"
"$root/tools/udocker-1.3.17/udocker/udocker" run --volume="$root:$root" --workdir="$repo" \
  --env="PYTHONPATH=$root/rayan-mfm/runtime/feature-deps-576610:$repo/src:$root/mfm/alice-gemma4-base/src" \
  --env="OMP_NUM_THREADS=4" --env="MKL_NUM_THREADS=4" \
  --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" \
  --env="MFM_PIPELINE_JOB=$SLURM_JOB_ID" --entrypoint=/bin/bash "$container" -c '\''
set -euo pipefail
umask 077
python - <<"PY"
import os, socket, torch
assert sorted(n for _, n in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
assert torch.__version__ == "2.7.1+cu118"
assert torch.get_num_threads() == 4
print("private_CPU_archive_runtime_verified", flush=True)
PY
root="$1"
job="$MFM_PIPELINE_JOB"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
python -m scripts.mfm.archive_public_foundation_inputs \
  --census-dir "$root/rayan-mfm/runs/rayan-mfm-public-census-576995" \
  --census-sha256 0885f9a52b8ea967127a5a6b7708331163b214d3dea81ed598d24c57de6df27d \
  --feature-bank "$root/rayan-mfm/features/rayan-mfm-public-e2e-20261007" \
  --feature-bank-sha256 6b70f4ed4bcde0f702de6057d9f2aece86228ea96711483469fa943fab42af4a \
  --legal-material-archive "$root/rayan-mfm/inputs/mfm-e2e-material-and-census-576995.zip" \
  --legal-material-archive-sha256 857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512 \
  --output-dir "$root/rayan-mfm/runs/rayan-mfm-full-bank-archive-$job" \
  | tee "$root/rayan-mfm/receipts/rayan-mfm-full-bank-archive-$job.stdout"
'\'' full-bank-archive "$root"
' archive-wrapper "$root" "$repo" "$container"
