#!/usr/bin/env bash
#SBATCH --job-name=rayan-cumulative-learning-r2
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
set -euo pipefail
umask 077
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 && "$(id -g)" == 100 ]]
COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$COMPUTE_ROOT" == /homes/01/mxrayan/rayan-compute && "$(stat -c %U:%a "$COMPUTE_ROOT")" == mxrayan:700 ]]
export PERSONALITY_REPO_ROOT="$COMPUTE_ROOT/rayan-personality/code/rayan-personality-333fcd0c8239fc81a6459b13d53bbe592b33fda6"
NAMESPACE_HELPER="$PERSONALITY_REPO_ROOT/scripts/eipm/n1/enter_private_network_namespace.py"
AUDIT_HELPER="$COMPUTE_ROOT/jobs/rayan-cumulative-learning-r2-20261013.py"
[[ "$(sha256sum "$AUDIT_HELPER" | cut -d ' ' -f 1)" == 08924aa781c66111ea753e8c731532cefdc376790d484f07961a5045aa9b2a48 ]]
export PERSONALITY_CUMULATIVE_PLAN_SHA256=c40303c2823d84c9246f92873713859485c2ab17460801f386f8d2ab051032fa
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-conditional_successor_report.py" | cut -d ' ' -f 1)" == 983ad3dabd1ee3ec7217c237e7abe8c31e1f87553a69a231756172791c0a91d4 ]]
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-connected_voice_references.py" | cut -d ' ' -f 1)" == 7d928048ad3a59caa35ce2292eb2b9fd400ec11797c9e6309e5e4af36ac8db70 ]]
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-cumulative_bindings.py" | cut -d ' ' -f 1)" == 0e09cc22db00ca8c3573836791df395c6dbe10f3ca0b5b88f03102d55923df20 ]]
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-cumulative_successor.py" | cut -d ' ' -f 1)" == db8db63f14f0164cd6e00f06f654d04f033a64594f0342c5f16b954d9578ba34 ]]
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-governed_response.py" | cut -d ' ' -f 1)" == bc69b2fa38eb416229e51d1dc36c84dbb4cd84924141a603ed1b7deac7ebe8b2 ]]
[[ "$(sha256sum "$COMPUTE_ROOT/jobs/rayan-20261013-r2-n0_semantic_contrasts.py" | cut -d ' ' -f 1)" == c6a0e51ab687aa905f0ca4f10a1ad9b3e6c268cb6a587a093835ab220756f2b4 ]]
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store" UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
export PERSONALITY_HOST_NETNS="$(readlink /proc/self/ns/net)" PERSONALITY_HOST_UID="$(id -u)" PERSONALITY_HOST_GID="$(id -g)"
export COMPUTE_ROOT NAMESPACE_HELPER AUDIT_HELPER UDOCKER
bash -lc 'set -euo pipefail; module load python/3.11.5; exec python3 -I "$@"' rayan-recovery "$NAMESPACE_HELPER" -- /bin/bash -c '
  set -euo pipefail
  exec "$UDOCKER" run --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" --workdir="$PERSONALITY_REPO_ROOT" \
    --env="PERSONALITY_REPO_ROOT=$PERSONALITY_REPO_ROOT" --env="SLURM_JOB_ID=$SLURM_JOB_ID" --env="SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE" --env="SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" --env="PERSONALITY_CUMULATIVE_PLAN_SHA256=$PERSONALITY_CUMULATIVE_PLAN_SHA256" \
    --env="PERSONALITY_HOST_NETNS=$PERSONALITY_HOST_NETNS" --env="PERSONALITY_HOST_UID=$PERSONALITY_HOST_UID" \
    --env="PERSONALITY_HOST_GID=$PERSONALITY_HOST_GID" --env="PERSONALITY_ISOLATED_NETNS=$PERSONALITY_ISOLATED_NETNS" \
    --entrypoint=python rayan-n0-base -I "$AUDIT_HELPER"
'
