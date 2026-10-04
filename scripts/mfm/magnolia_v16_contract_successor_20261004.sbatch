#!/bin/bash
#SBATCH --job-name=rayan-mfm-source-view
#SBATCH --partition=node
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
umask 077
: "${MFM_EXPORT_JOB_IDS:?exact externally recorded producer job IDs required}"
: "${MFM_CODE_COMMIT:?frozen analysis code commit required}"
[[ "$MFM_EXPORT_JOB_IDS" =~ ^[0-9]+(:[0-9]+)*$ ]]
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
root=/homes/01/mxrayan/rayan-compute
repo="$root/rayan-mfm/code/rayan-mfm-contract-20261004"
container=rayan-mfm-source-cpu-20261002
[[ "$(cd "$repo" && git rev-parse HEAD)" == "$MFM_CODE_COMMIT" && -z "$(cd "$repo" && git status --porcelain)" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
udocker="$root/tools/udocker-1.3.17/udocker/udocker"
exec 9>"$root/rayan-mfm/jobs/dedicated-source-runtime.lock"
flock -n 9
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
python3 -I "$repo/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
root="$1"
repo="$2"
container="$3"
"$root/tools/udocker-1.3.17/udocker/udocker" run --volume="$root:$root" --workdir="$repo" \
  --env="PYTHONPATH=$root/rayan-mfm/runtime/feature-deps-576610:$repo/src:$root/mfm/alice-gemma4-base/src" \
  --env="OMP_NUM_THREADS=12" --env="MKL_NUM_THREADS=12" \
  --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" --env="MFM_EXPORT_JOB_IDS=$MFM_EXPORT_JOB_IDS" \
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
# Pin the bank from the successful, separately stored producer stdout, never
# from a bank self-digest supplied by the consumer. Source/corpus/model/code
# bytes are then fully verified again by the trainer before gradients.
bank_sha="$(python - "$root" <<"PY"
from pathlib import Path
import json, os, re, sys
digests = set()
for job in os.environ["MFM_EXPORT_JOB_IDS"].split(":"):
    path = Path(sys.argv[1]) / "rayan-mfm/slurm" / ("rayan-mfm-features-" + job + ".out")
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "feature_bank_sha256" in record:
            assert record.get("train_cases") == 432 and record.get("development_cases") == 9
            assert record.get("qualified_for_product") is False and record.get("final_payloads_opened") is False
            assert re.fullmatch(r"[0-9a-f]{64}", record["feature_bank_sha256"])
            digests.add(record["feature_bank_sha256"])
assert len(digests) == 1, "missing or conflicting externally published feature bank"
print(next(iter(digests)))
PY
)"
common=(--teacher-training-manifest "$root/rayan-mfm/inputs/rayan-mfm-v16-recovered-441-20261002/owner-teacher-manifest.json"
  --input-sha256 0268c134596326bb592e0afaf3fdab8626670a345e813b2148a11eb68dc93676
  --owner-authorization-ref owner-directed-mfm-v16-synthetic
  --preflight-receipt "$root/rayan-mfm/receipts/rayan-mfm-processor-576606.json"
  --feature-bank "$root/rayan-mfm/features/rayan-mfm-features-441-eee3b30cf"
  --feature-bank-sha256 "$bank_sha" --device cpu)
fit="$root/rayan-mfm/runs/rayan-mfm-fit-576637"
component_sha=12537c467518773bfe7fd2abcd9a7e9eaedee80b623ee827ce0ec8e640dbe166
python -m unittest tests.governance.test_mfm_v16_contract_successor tests.governance.test_mfm_v16_decision_interface tests.governance.test_mfm_v16_specialist_assessment tests.governance.test_mfm_v16_fit_analysis tests.governance.test_mfm_v16_cached_diagnostic tests.governance.test_mfm_v16_frozen_features
python -m scripts.mfm.prepare_v16_contract_successor "${common[@]}" --component-dir "$fit" \
  --component-sha256 "$component_sha" \
  --prepared-base-dir "$root/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/mfm-role-clone-576486" \
  --prepared-base-receipt "$root/mfm/receipts/mfm-clone-576486.json" \
  --output-dir "$root/rayan-mfm/features/rayan-mfm-contract-view-$job"
'\'' cache-learning "$root"
' pipeline-wrapper "$root" "$repo" "$container"
