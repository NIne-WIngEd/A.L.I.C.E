#!/usr/bin/env bash
#SBATCH --job-name=rayan-public-feature-handoff
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Close actual historical 576600 public bytes with one existing P2 CPU stage.
# No model inference, training, installation, original-run writes or approval.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set exact clean handoff checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?pin reviewed new launcher code}"
: "${PERSONALITY_PUBLIC_OVERLAY_SHA256:?external canonical original-overlay byte pin}"
[[ "$(id -un)" == mxrayan && "$(id -u)" != 0 && "$SLURM_JOB_ID" =~ ^[0-9]+$ &&
   "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ &&
   "$PERSONALITY_PUBLIC_OVERLAY_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
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
git_personality() {
  git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" "$@"
}
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 3
ORIGINAL_RUN="$(within_root "$COMPUTE_ROOT/rayan-personality/runs/public-semantics-576600")"
EXPERIMENT="$(within_root "$ORIGINAL_RUN/evidence/experiment.json")"
PLAN="$(within_root "$ORIGINAL_RUN/precommitted-plan.json")"
CACHE="$(within_root "$ORIGINAL_RUN/public-feature-cache")"
OVERLAY="$(within_root "$ORIGINAL_RUN/overlay")"
EXPERIMENT_SHA256=12b583048ddf343e5724b92678b43bc7e247b46aa5f016653f9825e7da0d49df
PLAN_SHA256=6b565c38f969ceee2143f28fc29f760ec84c6f2c8cdc53dc29fd953ac3f93270
printf '%s  %s\n' "$EXPERIMENT_SHA256" "$EXPERIMENT" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$PLAN_SHA256" "$PLAN" | sha256sum -c - >/dev/null
RUN_PARENT="$(within_root "$COMPUTE_ROOT/rayan-personality/runs")"
[[ "$(stat -c %U "$RUN_PARENT")" == mxrayan && "$(stat -c %a "$RUN_PARENT")" == 700 ]] || exit 4
RUN_ROOT="$RUN_PARENT/public-feature-handoff-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || exit 4
mkdir -m 700 "$RUN_ROOT"
UDOCKER="$(within_root "$(realpath -e "$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker")")"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$(within_root "$COMPUTE_ROOT/udocker-store")"
export UDOCKER_TMP="$(within_root "$COMPUTE_ROOT/udocker-tmp")"
# Existing rayan-n0-base container is the previously established P2 runtime;
# this launcher never creates, sets up, pulls or changes a container.
echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo "public_feature_handoff_run=$RUN_ROOT"
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
  --env="PYTHONPATH=$OVERLAY:$PERSONALITY_REPO_ROOT" --env="PYTHONDONTWRITEBYTECODE=1" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --env="OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" --env="TOKENIZERS_PARALLELISM=false" \
  --env="SLURM_JOB_ID=$SLURM_JOB_ID" --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    python "$1/scripts/eipm/gemma_n0/close_public_feature_handoff.py" close \
      --experiment "$2" --experiment-sha256 "$3" --plan "$4" --plan-sha256 "$5" \
      --cache-directory "$6" --compute-root "$7" --overlay "$8" --overlay-sha256 "$9" \
      --run-root "${10}" --code-revision "${11}" --host-uid "${12}" --host-gid "${13}"
  ' rayan-public-feature-handoff "$PERSONALITY_REPO_ROOT" "$EXPERIMENT" "$EXPERIMENT_SHA256" \
  "$PLAN" "$PLAN_SHA256" "$CACHE" "$COMPUTE_ROOT" "$OVERLAY" \
  "$PERSONALITY_PUBLIC_OVERLAY_SHA256" "$RUN_ROOT" "$PERSONALITY_EXPECTED_REVISION" "$(id -u)" "$(id -g)"
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 6
echo "public_feature_handoff_evidence=$RUN_ROOT/handoff-run.json"
echo 'public_feature_handoff_completed=true qualification=unqualified n0_approved=false training_run=false'
