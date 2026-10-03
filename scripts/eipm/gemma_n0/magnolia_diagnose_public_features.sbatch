#!/usr/bin/env bash
#SBATCH --job-name=rayan-public-feature-diagnostics
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

# First-party readout diagnostics on closed PUBLIC bytes. No publisher model,
# tokenizer, source snapshot, training, dependency installation or qualification.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set exact clean diagnostic checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?pin reviewed code}"
: "${PERSONALITY_PUBLIC_PACKAGE:?set the actually verified closed package}"
: "${PERSONALITY_PUBLIC_MANIFEST_SHA256:?external manifest byte pin}"
: "${PERSONALITY_PUBLIC_CLOSURE_SHA256:?external closure byte pin}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$SLURM_JOB_ID" =~ ^[0-9]+$ &&
   "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ &&
   "$PERSONALITY_PUBLIC_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$PERSONALITY_PUBLIC_CLOSURE_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$COMPUTE_ROOT" == /homes/01/mxrayan/rayan-compute &&
   "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan && "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
within_root() {
  local selected
  [[ "$1" == /* && ! -L "$1" ]] || return 2
  selected="$(realpath -e "$1")"
  [[ "$selected" == "$1" ]] || return 2
  case "$selected" in "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;; *) return 2 ;; esac
}
PERSONALITY_REPO_ROOT="$(within_root "$PERSONALITY_REPO_ROOT")"
PERSONALITY_PUBLIC_PACKAGE="$(within_root "$PERSONALITY_PUBLIC_PACKAGE")"
[[ -d "$PERSONALITY_PUBLIC_PACKAGE" ]] || exit 2
git_personality() {
  git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" "$@"
}
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 3
SEMANTIC_EXPORT="$(within_root "$COMPUTE_ROOT/rayan-personality/runs/public-semantics-576600/evidence/semantic-readout.pt")"
EXPORT_SHA256=fbbf42eff20bc8fe9a7c514243febf0ee047b9d56df8ac72e229648b2ae9b2b5
printf '%s  %s\n' "$EXPORT_SHA256" "$SEMANTIC_EXPORT" | sha256sum -c - >/dev/null
RUN_PARENT="$(within_root "$COMPUTE_ROOT/rayan-personality/runs")"
[[ "$(stat -c %U "$RUN_PARENT")" == mxrayan && "$(stat -c %a "$RUN_PARENT")" == 700 ]] || exit 4
RUN_ROOT="$RUN_PARENT/public-feature-diagnostics-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || exit 4
mkdir -m 700 "$RUN_ROOT"
UDOCKER="$(within_root "$(realpath -e "$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker")")"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$(within_root "$COMPUTE_ROOT/udocker-store")"
export UDOCKER_TMP="$(within_root "$COMPUTE_ROOT/udocker-tmp")"
echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo "public_feature_diagnostics_run=$RUN_ROOT"
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
  --env="PYTHONPATH=$PERSONALITY_REPO_ROOT" --env="PYTHONDONTWRITEBYTECODE=1" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="TOKENIZERS_PARALLELISM=false" \
  --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    python "$1/scripts/eipm/gemma_n0/diagnose_public_features.py" \
      --package-directory "$2" --manifest-sha256 "$3" --closure-sha256 "$4" \
      --semantic-export "$5" --export-sha256 "$6" --output-directory "$7/evidence" --cpu-threads "$8"
  ' rayan-public-feature-diagnostics "$PERSONALITY_REPO_ROOT" "$PERSONALITY_PUBLIC_PACKAGE" \
  "$PERSONALITY_PUBLIC_MANIFEST_SHA256" "$PERSONALITY_PUBLIC_CLOSURE_SHA256" \
  "$SEMANTIC_EXPORT" "$EXPORT_SHA256" "$RUN_ROOT" "${SLURM_CPUS_PER_TASK:-4}"
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 6
echo "public_feature_diagnostics_evidence=$RUN_ROOT/evidence/diagnostics.json"
echo 'public_feature_diagnostics_completed=true qualification=unqualified n0_approved=false training_run=false'
