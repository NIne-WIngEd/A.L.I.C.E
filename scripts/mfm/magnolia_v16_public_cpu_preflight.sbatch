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
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
mkdir -p "$UDOCKER_DIR" "$UDOCKER_TMP"

echo "job=$SLURM_JOB_ID node=$(hostname) MFM=$(cd "$MFM_REPO_ROOT" && git rev-parse HEAD) foundation=$(cd "$MFM_FOUNDATION_ROOT" && git rev-parse HEAD)"
echo "diagnostic_preflight=$PREFLIGHT"
# Magnolia's host Python/glibc is not the validated scientific runtime.
# Keep the whole processor stage inside one already established CPU container.
"$UDOCKER" setup "$CONTAINER"
"$UDOCKER" run \
  --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --workdir="$MFM_REPO_ROOT" \
  --env="PYTHONPATH=$MFM_REPO_ROOT/src:$MFM_FOUNDATION_ROOT/src" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --entrypoint=/bin/bash "$CONTAINER" -c '
    set -euo pipefail
    python -c '\''import torch, transformers; from transformers import AutoProcessor; import alice_foundation.gemma4_v1, cognitive_kernel.formation_learning_v16; assert transformers.__version__ == "5.17.0", transformers.__version__; print("runtime", torch.__version__, transformers.__version__)'\''
    exec python -m scripts.mfm.train_v16_formation_specialist processor-preflight \
      --public-synthetic-curriculum "$1" --input-sha256 "$2" \
      --owner-authorization-ref owner-directed-mfm-v16-synthetic \
      --prepared-base-dir "$3" --prepared-base-receipt "$4" \
      --preflight-receipt "$5"
  ' mfm-v16 "$SOURCE" "$SOURCE_HASH" "$CLONE" "$CLONE_RECEIPT" "$PREFLIGHT"
