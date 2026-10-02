#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-public-semantics
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Initial measured public semantic-readout experiment. Counts, steps and width
# are predeclared operating choices, never full personality capability limits.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set exact clean personality checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?pin tested code}"
: "${PERSONALITY_PREPARED_RECEIPT:?set full-layer personality preparation}"
: "${PERSONALITY_PREPARED_RECEIPT_SHA256:?externally pin preparation file}"
: "${PERSONALITY_SOURCE_ADMISSION:?set public FewRel admission}"
: "${PERSONALITY_SOURCE_ADMISSION_SHA256:?externally pin admission file}"
[[ "$(id -un)" == mxrayan && "$(id -u)" != 0 &&
   "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ &&
   "$PERSONALITY_PREPARED_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$PERSONALITY_SOURCE_ADMISSION_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ -d "$COMPUTE_ROOT" && "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan &&
   "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
within_root() {
  local selected
  [[ "$1" == /* && ! -L "$1" ]] || return 2
  selected="$(realpath -e "$1")"
  [[ "$selected" == "$1" ]] || return 2
  case "$selected" in "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;; *) return 2 ;; esac
}
PERSONALITY_REPO_ROOT="$(within_root "$PERSONALITY_REPO_ROOT")"
PERSONALITY_PREPARED_RECEIPT="$(within_root "$PERSONALITY_PREPARED_RECEIPT")"
PERSONALITY_SOURCE_ADMISSION="$(within_root "$PERSONALITY_SOURCE_ADMISSION")"
git_personality() {
  git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" "$@"
}
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 3
printf '%s  %s\n' "$PERSONALITY_PREPARED_RECEIPT_SHA256" "$PERSONALITY_PREPARED_RECEIPT" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$PERSONALITY_SOURCE_ADMISSION_SHA256" "$PERSONALITY_SOURCE_ADMISSION" | sha256sum -c - >/dev/null
ROLE_ROOT="$COMPUTE_ROOT/rayan-personality"
RUN_PARENT="$ROLE_ROOT/runs"
for directory in "$ROLE_ROOT" "$RUN_PARENT"; do
  [[ ! -L "$directory" ]] || exit 4
  if [[ ! -e "$directory" ]]; then mkdir -m 700 "$directory"; fi
  [[ "$(within_root "$directory")" == "$directory" && -d "$directory" &&
     "$(stat -c %U "$directory")" == mxrayan && "$(stat -c %a "$directory")" == 700 ]] || exit 4
done
RUN_ROOT="$RUN_PARENT/public-semantics-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || exit 4
mkdir -m 700 "$RUN_ROOT"
UDOCKER="$(within_root "$(realpath -e "$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker")")"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$(within_root "$COMPUTE_ROOT/udocker-store")"
export UDOCKER_TMP="$(within_root "$COMPUTE_ROOT/udocker-tmp")"
PILLOW="$(within_root "$COMPUTE_ROOT/mfm/wheels/pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")"
VISION="$(within_root "$COMPUTE_ROOT/mfm/wheels/torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl")"
printf '%s  %s\n' 106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f "$PILLOW" | sha256sum -c - >/dev/null
printf '%s  %s\n' 6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627 "$VISION" | sha256sum -c - >/dev/null
echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo "public_semantic_run=$RUN_ROOT"
echo 'public_train_dev=true final_payload_opened=false private_identity=false upstream_weight_edits=false n0_approved=false'
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="TOKENIZERS_PARALLELISM=false" \
  --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    python - "$2" "$3" "$4" "$5" "$6" <<'\''PY'\''
from hashlib import sha256
import json
from pathlib import Path
import sys
prepared_path, prepared_sha, admission_path, admission_sha, root = sys.argv[1:]
root = Path(root)
def bounded(raw):
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink() or path != path.resolve(strict=True) or not path.is_relative_to(root):
        raise RuntimeError("source metadata redirects outside the owner compute root")
    return path
def read(raw, expected):
    path = bounded(raw)
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise RuntimeError("bound metadata differs from the external pin")
    return json.loads(data)
prepared = read(prepared_path, prepared_sha)
admission = read(admission_path, admission_sha)
bounded(prepared["snapshot_path"])
for binding in prepared["interfaces"].values():
    bounded(binding["path"])
for item in admission["files"]:
    bounded(item["path"])
if admission["private_identity_data"] is not False or admission["statistics"]["final_payload_opened"] is not False:
    raise RuntimeError("only receipted public TRAIN/DEV is eligible")
PY
    python -m pip install --no-index --no-deps --target "$7/overlay" "$8" "$9" >/dev/null
    export PYTHONPATH="$7/overlay:$1"
    # Tiny public mechanics exercise actual existing Torch/runtime compatibility
    # before source-model loading; they are not semantic performance evidence.
    python -m unittest discover -s "$1/tests/personality" -p "test_gemma_n0_semantic_*.py" -q
    python "$1/scripts/eipm/gemma_n0/run_public_semantic_experiment.py" plan \
      --source-admission "$4" --preparation "$2" --output "$7/precommitted-plan.json" \
      --train-rows-per-family 1 --dev-rows-per-family 2 --style-rows-per-dev-family 1 \
      --width 64 --steps 256 --seed 20261002 --learning-rate 0.0003 --weight-decay 0.01
    python "$1/scripts/eipm/gemma_n0/run_public_semantic_experiment.py" run \
      --plan "$7/precommitted-plan.json" --output-directory "$7/evidence" --cache-directory "$7/public-feature-cache"
  ' rayan-public-semantics "$PERSONALITY_REPO_ROOT" "$PERSONALITY_PREPARED_RECEIPT" \
  "$PERSONALITY_PREPARED_RECEIPT_SHA256" "$PERSONALITY_SOURCE_ADMISSION" \
  "$PERSONALITY_SOURCE_ADMISSION_SHA256" "$COMPUTE_ROOT" "$RUN_ROOT" "$PILLOW" "$VISION"
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 6
echo "public_semantic_evidence=$RUN_ROOT/evidence"
echo 'public_semantic_experiment_completed=true qualification=unqualified n0_approved=false'
