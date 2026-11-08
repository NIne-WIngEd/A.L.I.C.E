#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-public-source
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Only the four public TRAIN/DEV artifacts below are admitted. No source model,
# private identity payload, FINAL artifact, gradient or qualification is used.
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?submit source admission through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set the clean personality checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?pin the exact tested commit}"
[[ "$(id -un)" == mxrayan && "$(id -u)" != 0 &&
   "$SLURM_JOB_ID" =~ ^[0-9]+$ &&
   "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ -d "$COMPUTE_ROOT" && "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan &&
   "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
within_root() {
  local selected
  [[ "$1" == /* && ! -L "$1" ]] || return 2
  selected="$(realpath -e "$1")"
  [[ "$selected" == "$1" ]] || return 2
  case "$selected" in
    "$COMPUTE_ROOT"/*) printf '%s\n' "$selected" ;;
    *) return 2 ;;
  esac
}
PERSONALITY_REPO_ROOT="$(within_root "$PERSONALITY_REPO_ROOT")"
git_personality() {
  git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" "$@"
}
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 3
PUBLIC_ROOT="$(within_root "$COMPUTE_ROOT/rayan-n0/n0-v02/full-public-mixture-5e29f7f-v1/fewrel")"
ROWS="$(within_root "$PUBLIC_ROOT/train_dev_rows.jsonl")"
BANK="$(within_root "$PUBLIC_ROOT/train_dev_bank.json")"
MANIFEST="$(within_root "$PUBLIC_ROOT/manifest.json")"
AUDIT="$(within_root "$PUBLIC_ROOT/audit.json")"
for file in "$ROWS" "$BANK" "$MANIFEST" "$AUDIT"; do
  [[ -f "$file" && "$(stat -c %U "$file")" == mxrayan ]] || exit 3
done
ROLE_ROOT="$COMPUTE_ROOT/rayan-personality"
RUN_PARENT="$ROLE_ROOT/runs"
for directory in "$ROLE_ROOT" "$RUN_PARENT"; do
  [[ ! -L "$directory" ]] || exit 4
  if [[ ! -e "$directory" ]]; then mkdir -m 700 "$directory"; fi
  [[ "$(within_root "$directory")" == "$directory" &&
     -d "$directory" && "$(stat -c %U "$directory")" == mxrayan &&
     "$(stat -c %a "$directory")" == 700 ]] || exit 4
done
RUN_ROOT="$RUN_PARENT/public-fewrel-admission-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || exit 4
mkdir -m 700 "$RUN_ROOT"
UDOCKER="$(within_root "$(realpath -e "$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker")")"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$(within_root "$COMPUTE_ROOT/udocker-store")"
export UDOCKER_TMP="$(within_root "$COMPUTE_ROOT/udocker-tmp")"
echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo 'public_train_dev=true private_source=false final_payload_opened=false private_gradient=false n0_approved=false'
"$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --workdir="$PERSONALITY_REPO_ROOT" --env="PYTHONPATH=$PERSONALITY_REPO_ROOT" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    python "$1/scripts/eipm/gemma_n0/admit_public_fewrel.py" admit \
      --rows "$2" --bank "$3" --manifest "$4" --audit "$5" \
      --output "$6/public-fewrel-admission.json" --declared-public-train-dev
    python "$1/scripts/eipm/gemma_n0/admit_public_fewrel.py" verify \
      "$6/public-fewrel-admission.json"
  ' rayan-public-source "$PERSONALITY_REPO_ROOT" "$ROWS" "$BANK" "$MANIFEST" "$AUDIT" "$RUN_ROOT"
[[ "$(git_personality rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" &&
   -z "$(git_personality status --porcelain)" ]] || exit 6
echo "source_admission_receipt=$RUN_ROOT/public-fewrel-admission.json"
echo 'public_source_admission_completed=true qualification=unqualified n0_approved=false'
