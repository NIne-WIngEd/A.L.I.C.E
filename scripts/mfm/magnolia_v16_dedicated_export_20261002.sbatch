#!/bin/bash
#SBATCH --job-name=rayan-mfm-source-resume
#SBATCH --partition=node
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
umask 077
root=/homes/01/mxrayan/rayan-compute
: "${MFM_REPO_ROOT:?}" "${MFM_REPO_COMMIT:?}" "${MFM_CONTAINER:?}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
[[ "$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD)" == "$MFM_REPO_COMMIT" && -z "$(cd "$MFM_REPO_ROOT" && git status --porcelain)" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
udocker="$root/tools/udocker-1.3.17/udocker/udocker"
if [[ "${MFM_CREATE_CONTAINER:-0}" == 1 ]]; then
  [[ "$MFM_CONTAINER" =~ ^rayan-mfm-source-cpu-[a-zA-Z0-9-]+$ && ! -e "$UDOCKER_DIR/containers/$MFM_CONTAINER" ]]
  container_id="$("$udocker" create --name="$MFM_CONTAINER" python:3.11-slim-bookworm)"
  [[ "$container_id" =~ ^[0-9a-f-]{36}$ ]]
  source_root="$(realpath -e "$UDOCKER_DIR/containers/rayan-n0-base/ROOT")"
  target_root="$(realpath -e "$UDOCKER_DIR/containers/$container_id/ROOT")"
  [[ "$target_root" == "$UDOCKER_DIR/containers/$container_id/ROOT" && "$source_root" != "$target_root" && "$(stat -c %u "$target_root")" == 1905 ]]
  tar -C "$source_root" --one-file-system --exclude="__pycache__" --exclude="usr/lib/modules" -cf - usr | tar -C "$target_root" -xf -
  # Keep real bind-path ancestors in this private runtime. Another job cannot
  # remove this container's mountpoint while a long source forward is running.
  mkdir -p "$target_root/homes/01/mxrayan/rayan-compute"
  "$udocker" setup --execmode=P2 "$MFM_CONTAINER"
else
  [[ "${MFM_CREATE_CONTAINER:-0}" == 0 && -d "$UDOCKER_DIR/containers/$MFM_CONTAINER/ROOT" ]]
fi
# One dedicated CPU container; only sequential producer jobs may use it.
exec 9>"$root/rayan-mfm/jobs/dedicated-source-runtime.lock"
flock -n 9
export MFM_FOUNDATION_ROOT="$root/mfm/alice-gemma4-base"
export MFM_FOUNDATION_COMMIT=3e1328410dac52ba18be243cb9f7144e7a3964d1
export MFM_TEACHER_MANIFEST="$root/rayan-mfm/inputs/rayan-mfm-v16-recovered-441-20261002/owner-teacher-manifest.json"
export MFM_TEACHER_MANIFEST_SHA256=0268c134596326bb592e0afaf3fdab8626670a345e813b2148a11eb68dc93676
export MFM_OWNER_AUTH_REF=owner-directed-mfm-v16-synthetic
export MFM_PREPARED_BASE_DIR="$root/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/mfm-role-clone-576486"
export MFM_PREPARED_BASE_RECEIPT="$root/mfm/receipts/mfm-clone-576486.json"
export MFM_PREPARED_BASE_RECEIPT_FILE_SHA256=cd6170e52e2ee739788ae1bd85c60cbb68ea29c3602c7ec18d6b690d69e8d526
export MFM_PREFLIGHT_RECEIPT="$root/rayan-mfm/receipts/rayan-mfm-processor-576606.json"
export MFM_PREFLIGHT_RECEIPT_FILE_SHA256=0551adb3a89cdcf5b73bfbd0c43dda6baacca30daea5acd26e1e090acf5b79e7
export MFM_FEATURE_OUTPUT="$root/rayan-mfm/features/rayan-mfm-features-441-eee3b30cf"
export MFM_FEATURE_RESUME=1
if [[ -n "${MFM_PARENT_JOB:-}" ]]; then
  [[ "$MFM_PARENT_JOB" =~ ^[0-9]+$ ]]
  state="$(sacct -j "$MFM_PARENT_JOB" -X -n -P --format=State | head -n 1 | cut -d '|' -f 1)"
  if [[ "$state" == COMPLETED ]]; then
    [[ -f "$MFM_FEATURE_OUTPUT/bank.json" && ! -L "$MFM_FEATURE_OUTPUT/bank.json" ]] || exit 3
    echo existing_completion_artifact_skip_duplicate_export
    exit 0
  fi
  [[ "$state" == TIMEOUT ]] || { echo "upstream_${state}_stop_without_retry"; exit 3; }
fi
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
# Keep the runtime lock in this parent: the namespace helper closes inherited
# nonstandard FDs, so exec would accidentally release it before exporting.
python3 -I "$MFM_REPO_ROOT/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
python3 - "$1" "$2" <<"PY"
from pathlib import Path
import os, re, socket, sys
assert sorted(n for _, n in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
bank = Path(sys.argv[1])
assert bank.parent.resolve() == Path("/homes/01/mxrayan/rayan-compute/rayan-mfm/features")
assert not bank.is_symlink() and bank.is_dir()
for path in (bank / "cases").iterdir():
    assert not path.is_symlink() and path.is_dir() and path.stat().st_uid == 1905
    if re.fullmatch(r"\.staging-[0-9a-f]{32}", path.name):
        preserved = bank.parent / ("rayan-mfm-interrupted-stages-" + sys.argv[2])
        preserved.mkdir(mode=0o700, exist_ok=True)
        assert not preserved.is_symlink()
        path.rename(preserved / path.name)
    else:
        assert re.fullmatch(r"[0-9a-f]{64}", path.name), "unknown bank member"
print("private_existing_bank_custody_verified", flush=True)
PY
exec /bin/bash "$3/scripts/mfm/magnolia_v16_frozen_feature_export.sbatch" --inside-namespace
' dedicated-source "$MFM_FEATURE_OUTPUT" "$SLURM_JOB_ID" "$MFM_REPO_ROOT"
