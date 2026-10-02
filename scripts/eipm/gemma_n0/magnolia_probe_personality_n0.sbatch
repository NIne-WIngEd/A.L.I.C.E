#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-n0-probe
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Public frozen-source forward attempt only. The 64 GB allocation is a bounded
# CPU placement attempt, not a prior measured fit or personality qualification.
# No private source, gradient, publisher generation or network-isolation claim.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?submit this public probe through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set the separate exact clean personality checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?set its externally pinned tested commit}"
: "${PERSONALITY_PREPARED_RECEIPT:?set the existing personality preparation receipt}"
: "${PERSONALITY_PREPARED_RECEIPT_SHA256:?pin the preparation file bytes externally}"
[[ "$(id -un)" == mxrayan && "$(id -u)" != 0 ]] || {
  echo 'run under the original nonroot Rayan account' >&2; exit 2;
}
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ &&
   "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ &&
   "$PERSONALITY_PREPARED_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo 'invalid job, code or receipt pin' >&2; exit 2;
}

COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ -d "$COMPUTE_ROOT" && "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan &&
   "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || {
  echo 'compute root must remain owner-only' >&2; exit 2;
}
within_root() {
  local selected
  [[ "$1" == /* && ! -L "$1" ]] || return 2
  selected="$(realpath -e "$1")"
  case "$selected" in
    "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;;
    *) echo 'required path resolves outside the owner compute root' >&2; return 2 ;;
  esac
}
PERSONALITY_REPO_ROOT="$(within_root "$PERSONALITY_REPO_ROOT")"
PERSONALITY_PREPARED_RECEIPT="$(within_root "$PERSONALITY_PREPARED_RECEIPT")"
[[ -d "$PERSONALITY_REPO_ROOT" && -f "$PERSONALITY_PREPARED_RECEIPT" ]]
[[ "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]] || {
  echo 'personality code differs from the exact clean pin' >&2; exit 3;
}
PLAN="$(within_root "$PERSONALITY_REPO_ROOT/evaluation/eipm/gemma_n0/public_probe_plan_v1.json")"
PLAN_SHA256=ee90660581c471adb53a5541880917d4f4951ad51181063e377c2dcf1795a123
printf '%s  %s\n' "$PLAN_SHA256" "$PLAN" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$PERSONALITY_PREPARED_RECEIPT_SHA256" \
  "$PERSONALITY_PREPARED_RECEIPT" | sha256sum -c - >/dev/null

# Create one child at a time, checking existing ancestors before mkdir. Never
# follow a rayan-personality/runs symlink into an unrelated output directory.
ROLE_ROOT="$COMPUTE_ROOT/rayan-personality"
[[ ! -L "$ROLE_ROOT" ]] || exit 4
if [[ ! -e "$ROLE_ROOT" ]]; then mkdir -m 700 "$ROLE_ROOT"; fi
ROLE_ROOT="$(within_root "$ROLE_ROOT")"
[[ -d "$ROLE_ROOT" && "$(stat -c %U "$ROLE_ROOT")" == mxrayan &&
   "$(stat -c %a "$ROLE_ROOT")" == 700 ]] || exit 4
RUN_PARENT="$ROLE_ROOT/runs"
[[ ! -L "$RUN_PARENT" ]] || exit 4
if [[ ! -e "$RUN_PARENT" ]]; then mkdir -m 700 "$RUN_PARENT"; fi
RUN_PARENT="$(within_root "$RUN_PARENT")"
[[ -d "$RUN_PARENT" && "$(stat -c %U "$RUN_PARENT")" == mxrayan &&
   "$(stat -c %a "$RUN_PARENT")" == 700 ]] || exit 4
RUN_ROOT="$RUN_PARENT/public-forward-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || {
  echo 'preserve existing public forward evidence' >&2; exit 4;
}
mkdir -m 700 "$RUN_ROOT"

# The publisher launcher is a recorded symlink to maincmd.py in this same
# owner root. Resolve this one executable explicitly; data paths stay strict.
UDOCKER="$(within_root "$(realpath -e "$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker")")"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$(within_root "$COMPUTE_ROOT/udocker-store")"
export UDOCKER_TMP="$(within_root "$COMPUTE_ROOT/udocker-tmp")"
[[ -d "$UDOCKER_DIR" && -d "$UDOCKER_TMP" ]] || exit 5
PILLOW="$(within_root "$COMPUTE_ROOT/mfm/wheels/pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")"
VISION="$(within_root "$COMPUTE_ROOT/mfm/wheels/torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl")"
printf '%s  %s\n' \
  106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f \
  "$PILLOW" | sha256sum -c - >/dev/null
printf '%s  %s\n' \
  6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627 \
  "$VISION" | sha256sum -c - >/dev/null

echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo "public_forward_run=$RUN_ROOT"
echo 'device=cpu batch_size=1 private_source=false private_gradient=false network_isolation_claim=false n0_approved=false'

# Reuse the existing P2 container without setup-mode changes or GPU binding.
# The complete scientific stage, including offline job-local overlays and all
# frozen public forwards, stays inside this one udocker invocation.
"$UDOCKER" run \
  --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --workdir="$PERSONALITY_REPO_ROOT" \
  --env="PYTHONPATH=$PERSONALITY_REPO_ROOT" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="TOKENIZERS_PARALLELISM=false" \
  --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    # Check small metadata and every path before the admission verifier reads
    # checkpoint or interface files. Package metadata imports load no tensors.
    python - "$2" "$3" "$9" <<'\''PY'\''
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

receipt_path, expected_digest, root = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
payload = receipt_path.read_bytes()
if sha256(payload).hexdigest() != expected_digest:
    raise RuntimeError("preparation receipt file differs from the external pin")
prepared = json.loads(payload)
def bounded(raw):
    candidate = Path(raw)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise RuntimeError("prepared input is not an absolute nonsymlink path")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError("prepared input resolves outside owner compute root")
    return resolved
bounded(prepared["snapshot_path"])
bounded(prepared["clone_receipt_path"])
for contract in prepared["interfaces"].values():
    if contract is not None:
        bounded(contract["path"])
expected_runtime = {"backend": "transformers", "dtype": "bfloat16",
                    "torch_version": "2.7.1+cu118", "transformers_version": "5.17.0"}
if prepared["runtime"] != expected_runtime or version("torch") != "2.7.1+cu118" \
        or version("transformers") != "5.17.0":
    raise RuntimeError("prepared runtime differs from the Magnolia exact-source pin")
PY
    DEPS="$6/processor-deps"
    [[ ! -e "$DEPS" && ! -L "$DEPS" ]]
    python -m pip install --isolated --no-index --no-deps --no-cache-dir \
      --no-compile --target "$DEPS" "$7" "$8"
    export PYTHONPATH="$DEPS:$1"
    python - "$DEPS" <<'\''PY'\''
from pathlib import Path
import sys
import PIL
import torch
import torchvision
import transformers

overlay = Path(sys.argv[1]).resolve()
assert PIL.__version__ == "11.3.0", PIL.__version__
assert Path(PIL.__file__).resolve().is_relative_to(overlay)
assert torch.__version__ == "2.7.1+cu118", torch.__version__
assert torchvision.__version__ == "0.22.1+cu118", torchvision.__version__
assert Path(torchvision.__file__).resolve().is_relative_to(overlay)
assert transformers.__version__ == "5.17.0", transformers.__version__
print("public_cpu_runtime=" + torch.__version__ + "/" + transformers.__version__)
PY
    python "$1/scripts/eipm/gemma_n0/probe_prepared.py" "$2" \
      "$6/public-forward.json" --public-plan "$4" --device cpu --batch-size 1
  ' rayan-personality-public-probe "$PERSONALITY_REPO_ROOT" \
  "$PERSONALITY_PREPARED_RECEIPT" "$PERSONALITY_PREPARED_RECEIPT_SHA256" \
  "$PLAN" "$PLAN_SHA256" "$RUN_ROOT" "$PILLOW" "$VISION" "$COMPUTE_ROOT"

[[ "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]] || {
  echo 'code changed during the public forward diagnostic' >&2; exit 6;
}
printf '%s  %s\n' "$PLAN_SHA256" "$PLAN" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$PERSONALITY_PREPARED_RECEIPT_SHA256" \
  "$PERSONALITY_PREPARED_RECEIPT" | sha256sum -c - >/dev/null
echo "public_forward_receipt=$RUN_ROOT/public-forward.json"
echo 'public_forward_completed=true qualification=unqualified n0_approved=false'
