#!/usr/bin/env bash
#SBATCH --job-name=rayan-personality-n0-prep
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Public publisher custody and role-interface preparation only. No model
# forward, private corpus, gradients, tensor mutation or behavioral approval.
set -euo pipefail
umask 077

: "${SLURM_JOB_ID:?submit through Slurm}"
: "${PERSONALITY_REPO_ROOT:?set the separate clean personality checkout}"
: "${PERSONALITY_EXPECTED_REVISION:?set the exact tested commit}"
[[ "$(id -un)" == mxrayan ]] || { echo "run under Rayan's account" >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$PERSONALITY_EXPECTED_REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2

COMPUTE_ROOT="$(realpath -e "$HOME/rayan-compute")"
[[ "$(stat -c %U "$COMPUTE_ROOT")" == mxrayan && "$(stat -c %a "$COMPUTE_ROOT")" == 700 ]] || exit 2
PERSONALITY_REPO_ROOT="$(realpath -e "$PERSONALITY_REPO_ROOT")"
case "$PERSONALITY_REPO_ROOT" in "$COMPUTE_ROOT"/*) ;; *) exit 2 ;; esac
# Magnolia's login Git predates -C. Explicit work-tree arguments preserve the
# exact clean source check without upgrading or mutating the shared runtime.
[[ "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" rev-parse HEAD)" == "$PERSONALITY_EXPECTED_REVISION" ]]
[[ -z "$(git --git-dir="$PERSONALITY_REPO_ROOT/.git" --work-tree="$PERSONALITY_REPO_ROOT" status --porcelain)" ]]

SOURCE="$(realpath -e "$COMPUTE_ROOT/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/source")"
case "$SOURCE" in "$COMPUTE_ROOT"/*) ;; *) exit 3 ;; esac
[[ -d "$SOURCE" && -f "$SOURCE/model.safetensors" ]] || exit 3
# Check every existing ancestor before creating our own namespace.
PERSONALITY_ROOT="$COMPUTE_ROOT/rayan-personality"
if [[ -e "$PERSONALITY_ROOT" || -L "$PERSONALITY_ROOT" ]]; then
  [[ ! -L "$PERSONALITY_ROOT" && -d "$PERSONALITY_ROOT" ]]
else
  mkdir -m 700 "$PERSONALITY_ROOT"
fi
[[ "$(realpath -e "$PERSONALITY_ROOT")" == "$PERSONALITY_ROOT" ]]
[[ "$(stat -c %U "$PERSONALITY_ROOT")" == mxrayan && "$(stat -c %a "$PERSONALITY_ROOT")" == 700 ]]
RUN_PARENT="$PERSONALITY_ROOT/runs"
if [[ -e "$RUN_PARENT" || -L "$RUN_PARENT" ]]; then
  [[ ! -L "$RUN_PARENT" && -d "$RUN_PARENT" ]]
else
  mkdir -m 700 "$RUN_PARENT"
fi
[[ "$(realpath -e "$RUN_PARENT")" == "$RUN_PARENT" ]]
[[ "$(stat -c %U "$RUN_PARENT")" == mxrayan && "$(stat -c %a "$RUN_PARENT")" == 700 ]]
RUN_ROOT="$RUN_PARENT/preparation-${SLURM_JOB_ID}"
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "preserve occupied preparation evidence" >&2; exit 4; }
mkdir -m 700 "$RUN_ROOT"

UDOCKER="$COMPUTE_ROOT/tools/udocker-1.3.17/udocker/udocker"
[[ -x "$UDOCKER" ]] || exit 5
export UDOCKER_DIR="$COMPUTE_ROOT/udocker-store"
export UDOCKER_TMP="$COMPUTE_ROOT/udocker-tmp"
[[ -d "$UDOCKER_DIR" && -d "$UDOCKER_TMP" ]] || exit 5

echo "owner=$(id -un) job=$SLURM_JOB_ID node=$(hostname) source_commit=$PERSONALITY_EXPECTED_REVISION"
echo "run_root=$RUN_ROOT"
# Reuse the already configured rayan-n0-base runtime. Do not pull an image,
# install packages, change its execution mode, or execute scientific Python
# directly in Magnolia's old host runtime.
"$UDOCKER" run \
  --volume="$COMPUTE_ROOT:$COMPUTE_ROOT" \
  --workdir="$PERSONALITY_REPO_ROOT" \
  --env="PYTHONPATH=$PERSONALITY_REPO_ROOT" \
  --env="HF_HUB_OFFLINE=1" --env="TRANSFORMERS_OFFLINE=1" \
  --env="HF_HUB_DISABLE_TELEMETRY=1" --env="DO_NOT_TRACK=1" \
  --env="OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}" \
  --entrypoint=/bin/bash rayan-n0-base -c '
    set -euo pipefail
    umask 077
    exec python - "$1" "$2" "$3" <<'\''PY'\''
from importlib.metadata import version
import json
from pathlib import Path
import sys
from src.alice_foundation import gemma4_v1 as foundation
from src.alice_personality.gemma_n0.preparation import prepare, verify_prepared

source, run_root, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
runtime = {"backend": "transformers", "dtype": "bfloat16",
           "torch_version": version("torch"), "transformers_version": version("transformers")}
if runtime["torch_version"] != "2.7.1+cu118" or runtime["transformers_version"] != "5.17.0":
    raise RuntimeError("existing scientific runtime differs from the observed Magnolia pin")
source_receipt = foundation.seal_source(source, run_root / "source.json")
clone = run_root / "personality-role-source"
clone_receipt = run_root / "personality-clone.json"
foundation.clone_verified_source(source, clone, clone_receipt, role="personality")
prepared_path = run_root / "personality-prepared.json"
contract_path = (Path.cwd() / "docs/eipm/GEMMA_PERSONALITY_FEATURE_INTERFACE_v1.md").resolve(strict=True)
prepared = prepare(clone_receipt, prepared_path, snapshot=clone, runtime=runtime,
                   input_contract={"contract_id": "personality-gemma-n0-source-input-v1", "path": contract_path},
                   output_contract={"contract_id": "personality-gemma-n0-token-features-v1", "path": contract_path})
verified = verify_prepared(prepared_path, snapshot=clone, expected_runtime=runtime)
assert prepared == verified
print(json.dumps({"owner": "mxrayan", "source_commit": commit,
                  "source_receipt_sha256": source_receipt["receipt_sha256"],
                  "prepared_receipt_sha256": verified["receipt_sha256"],
                  "prepared_receipt_path": str(prepared_path),
                  "state": verified["state"], "runtime": runtime,
                  "model_forward_performed": False, "private_gradient": False,
                  "n0_approved": False}, sort_keys=True))
PY
  ' rayan-personality-prep "$SOURCE" "$RUN_ROOT" "$PERSONALITY_EXPECTED_REVISION"
