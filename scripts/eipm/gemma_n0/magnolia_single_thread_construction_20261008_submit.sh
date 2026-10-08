#!/usr/bin/env bash
set -euo pipefail
umask 077
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]]
root="$(realpath -e /homes/01/mxrayan/rayan-compute)"
[[ "$root" == /homes/01/mxrayan/rayan-compute && "$(stat -c %U:%a "$root")" == mxrayan:700 ]]
export PERSONALITY_EXPECTED_REVISION=333fcd0c8239fc81a6459b13d53bbe592b33fda6
export PERSONALITY_REPO_ROOT="$root/rayan-personality/code/rayan-personality-$PERSONALITY_EXPECTED_REVISION"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
[[ "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" ]]
[[ -z "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]]
export PRIVATE_PACKAGE_PATH="$root/rayan-personality/private-inputs/ALICE_EIPM_CURATED_FRONTIER_v2-3867ff04.zip"
export PERSONALITY_CONSTRUCTION_PLAN="$root/rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json"
export PERSONALITY_CONSTRUCTION_PLAN_SHA256=09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b
export PERSONALITY_CONSTRUCTION_HELPER_SHA256=5a368f31609467d6b4b19e9e6735ab3f32c6b969b27ef9f11fe033e9478c74d9
logs="$root/rayan-personality/slurm"
[[ -d "$logs" && ! -L "$logs" && "$(realpath -e "$logs")" == "$logs" && "$(stat -c %U:%a "$logs")" == mxrayan:700 ]]
wrapper="$PERSONALITY_REPO_ROOT/scripts/eipm/gemma_n0/magnolia_connected_construction.sbatch"
[[ "$(sha256sum "$wrapper" | cut -d ' ' -f 1)" == 92d116374e23fa193091399df5a4d2e370cc97d89816a9d23836bf39530925d0 ]]
[[ "$(sha256sum "$PERSONALITY_REPO_ROOT/scripts/eipm/gemma_n0/construct_personality.py" | cut -d ' ' -f 1)" == "$PERSONALITY_CONSTRUCTION_HELPER_SHA256" ]]
bash -n "$wrapper"
submission="$root/jobs/rayan-connected-single-thread-20261008-submission.receipt"
[[ ! -e "$submission" && ! -L "$submission" ]]
# Preserve raw submission output before parsing; never retry blindly.
sbatch --parsable --no-requeue --open-mode=append --cpus-per-task=1 --export=ALL --output="$logs/rayan-connected-personality-%j.out" --error="$logs/rayan-connected-personality-%j.err" "$wrapper" > "$submission"
job="$(cat "$submission")"
[[ "$job" =~ ^[0-9]+$ ]]
printf 'owner=mxrayan connected_personality_job=%s code_revision=%s\n' "$job" "$PERSONALITY_EXPECTED_REVISION"
