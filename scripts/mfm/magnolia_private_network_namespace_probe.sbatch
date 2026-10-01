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
COMPUTE_ROOT="$(realpath -e "${RAYAN_COMPUTE_ROOT:-$HOME/rayan-compute}")"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
CONTAINER="${MFM_CPU_CONTAINER:-rayan-n0-base}"
[[ -x "$UDOCKER" ]] || { echo 'udocker unavailable' >&2; exit 2; }
command -v unshare >/dev/null || { echo 'unshare unavailable' >&2; exit 2; }

export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
export UDOCKER COMPUTE_ROOT CONTAINER
mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

# The image must already be present in the local udocker store. Do not pull it.
# setup is the local P2 preparation step from the successful public CPU job.
"$UDOCKER" setup "$CONTAINER" >/dev/null

echo "job=$SLURM_JOB_ID node=$(hostname) container=$CONTAINER"
echo "host_netns=$(readlink /proc/self/ns/net)"
MFM_HOST_NETNS="$(readlink /proc/self/ns/net)"
export MFM_HOST_NETNS

# Try rootless user+network namespaces first. Some installations allow a
# direct network namespace instead. Both candidates must pass the *inner*
# Python check after udocker P2 starts; merely succeeding at unshare is not
# sufficient. No fallback runs a corpus with host networking.
probe_inner() {
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

if unshare --user --map-root-user --net -- bash -c \
    'set -euo pipefail; probe_inner'; then
  exit 0
fi

if unshare --net -- bash -c \
    'set -euo pipefail; probe_inner'; then
  exit 0
fi

echo 'No supported loopback-only P2 namespace; private MFM CPU pass is blocked.' >&2
exit 3
