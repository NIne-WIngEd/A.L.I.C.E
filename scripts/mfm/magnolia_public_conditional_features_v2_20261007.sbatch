#!/bin/bash
#SBATCH --job-name=rayan-mfm-public-features-pilot-v2
#SBATCH --partition=node
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
: "${MFM_CODE_COMMIT:?frozen public preparation code commit required}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
root=/homes/01/mxrayan/rayan-compute
repo="$root/rayan-mfm/code/rayan-mfm-public-features-pilot-v2-20261007"
container=rayan-mfm-source-cpu-20261002
[[ "$(cd "$repo" && git rev-parse HEAD)" == "$MFM_CODE_COMMIT" && -z "$(cd "$repo" && git status --porcelain)" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
udocker="$root/tools/udocker-1.3.17/udocker/udocker"
exec 9>"$root/rayan-mfm/jobs/dedicated-source-runtime.lock"
flock -n 9
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
/usr/bin/time -v python3 -I "$repo/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
root="$1"
repo="$2"
container="$3"
"$root/tools/udocker-1.3.17/udocker/udocker" run --volume="$root:$root" --workdir="$repo" \
  --env="PYTHONPATH=$root/rayan-mfm/runtime/feature-deps-576610:$repo/src:$root/mfm/alice-gemma4-base/src" \
  --env="OMP_NUM_THREADS=12" --env="MKL_NUM_THREADS=12" \
  --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" \
  --env="MFM_PIPELINE_JOB=$SLURM_JOB_ID" --entrypoint=/bin/bash "$container" -c '\''
set -euo pipefail
umask 077
python - <<"PY"
import os, socket, torch
assert sorted(n for _, n in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
assert torch.__version__ == "2.7.1+cu118"
assert torch.get_num_threads() == 12
print("private_CPU_cached_runtime_verified", flush=True)
PY
root="$1"
job="$MFM_PIPELINE_JOB"
# Corrected fixture lifecycle; runtime tests must pass before the first
# measured public source-feature pass: at most30min before the next
# complete input. Preserve a partial ordered bank for same-code verified resume.
# The dedicated actual-census receipt is separate from mock unittest stdout.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
census_sha="$(python - "$root" <<"PY"
from pathlib import Path
import json,sys
path=Path(sys.argv[1])/"rayan-mfm/receipts/rayan-mfm-public-census-576995.stdout"
terminals=[]
for line in path.read_text().splitlines():
    try:
        record=json.loads(line)
    except ValueError:
        continue
    if isinstance(record,dict) and "public_conditional_census_sha256" in record:
        terminals.append(record)
assert len(terminals)==1 and terminals[0]["rows"]==40868 and terminals[0]["qualified_for_product"] is False
assert terminals[0]["public_conditional_census_sha256"]=="0885f9a52b8ea967127a5a6b7708331163b214d3dea81ed598d24c57de6df27d"
print(terminals[0]["public_conditional_census_sha256"])
PY
)"
python -m unittest tests.governance.test_mfm_public_conditional_features tests.governance.test_mfm_public_conditional_preflight tests.governance.test_mfm_public_conditional_material tests.governance.test_mfm_caller_input tests.governance.test_mfm_reference_catalog tests.governance.test_mfm_sequence_curriculum
printf "public_source_feature_runtime_checks_passed\n"
python -m scripts.mfm.export_public_conditional_features \
  --census-dir "$root/rayan-mfm/runs/rayan-mfm-public-census-576995" \
  --census-sha256 "$census_sha" \
  --prepared-base-dir "$root/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/mfm-role-clone-576486" \
  --prepared-base-receipt "$root/mfm/receipts/mfm-clone-576486.json" \
  --stop-after-seconds 1800 \
  --output-dir "$root/rayan-mfm/features/rayan-mfm-public-e2e-20261007" \
  | tee "$root/rayan-mfm/receipts/rayan-mfm-public-features-$job.stdout"
'\'' public-features "$root"
' pipeline-wrapper "$root" "$repo" "$container"
