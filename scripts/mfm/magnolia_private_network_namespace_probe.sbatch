#!/usr/bin/env bash
#SBATCH --job-name=rayan-mfm-netns-probe
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

# No MFM corpus, private source, checkpoint, processor, or model weight is read.
# This only checks whether Magnolia can run the existing P2 container inside a
# loopback-only network namespace. A successful probe is not a corpus receipt.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?submit this probe with sbatch}"
: "${MFM_REPO_ROOT:?export the absolute MFM checkout when submitting the probe}"
COMPUTE_ROOT="$(realpath -e "${RAYAN_COMPUTE_ROOT:-$HOME/rayan-compute}")"
MFM_REPO_ROOT="$(realpath -e "$MFM_REPO_ROOT")"
case "$MFM_REPO_ROOT" in
  "$COMPUTE_ROOT"/*) ;;
  *) echo 'MFM checkout is outside compute root' >&2; exit 2 ;;
esac
NAMESPACE_HELPER="$MFM_REPO_ROOT/scripts/mfm/enter_private_network_namespace.py"
[[ -f "$NAMESPACE_HELPER" ]] || { echo 'namespace helper missing' >&2; exit 2; }
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
CONTAINER="${MFM_CPU_CONTAINER:-rayan-n0-base}"
[[ -x "$UDOCKER" ]] || { echo 'udocker unavailable' >&2; exit 2; }

export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
export UDOCKER COMPUTE_ROOT CONTAINER
mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

# The image must already be present in the local udocker store. Do not pull it.
# setup is the local P2 preparation step from the successful public CPU job.
"$UDOCKER" setup "$CONTAINER" >/dev/null

echo "job=$SLURM_JOB_ID node=$(hostname) container=$CONTAINER"
echo "host_netns=$(readlink /proc/self/ns/net)"
echo "unshare_version=$(unshare --version)"
echo "MFM=$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD)"
MFM_HOST_NETNS="$(readlink /proc/self/ns/net)"
MFM_HOST_UID="$(id -u)"
MFM_HOST_GID="$(id -g)"
[[ "$MFM_HOST_UID" != 0 ]] || { echo 'root host UID is not supported' >&2; exit 2; }
echo "host_uid=$MFM_HOST_UID host_gid=$MFM_HOST_GID"
export MFM_HOST_NETNS MFM_HOST_UID MFM_HOST_GID

# The helper identity-maps this user before exec. The P2 container must also
# pass the inner check; merely creating a namespace is not sufficient.
probe_inner() {
  # udocker must be invoked as the original nonroot user. P2 may emulate
  # root *inside* its container, which is a different boundary.
  [[ "$(id -u)" == "$MFM_HOST_UID" && "$(id -g)" == "$MFM_HOST_GID" ]] || {
    echo 'namespace changed the caller UID/GID before udocker' >&2
    return 1
  }
  echo "namespace_uid=$(id -u) namespace_gid=$(id -g)"
  "$UDOCKER" run \
    --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
    --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" \
    --entrypoint=/bin/bash "$CONTAINER" -c '
      set -euo pipefail
      python - <<"PY"
import os
import socket

interfaces = sorted(name for _, name in socket.if_nameindex())
namespace = os.readlink("/proc/self/ns/net")
print("p2_netns=" + namespace)
print("p2_interfaces=" + ",".join(interfaces))
if interfaces != ["lo"] or namespace == os.environ["MFM_HOST_NETNS"]:
    raise SystemExit("P2 container is not loopback-only")
PY
    '
}
export -f probe_inner

if bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' \
    mfm-host-python "$NAMESPACE_HELPER" -- /bin/bash -c \
    'set -euo pipefail; probe_inner'; then
  exit 0
fi

echo 'No supported loopback-only P2 namespace; private MFM CPU pass is blocked.' >&2
exit 3
