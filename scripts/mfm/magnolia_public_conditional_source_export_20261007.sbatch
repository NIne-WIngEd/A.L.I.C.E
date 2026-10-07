#!/bin/bash
#SBATCH --job-name=rayan-mfm-full-source-export
#SBATCH --partition=node
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --dependency=afterok:577006
set -euo pipefail
umask 077
: "${MFM_CODE_COMMIT:?frozen public preparation code commit required}"
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
root=/homes/01/mxrayan/rayan-compute
repo="$root/rayan-mfm/code/rayan-mfm-full-source-export-20261007"
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
# Complete the entire4,862-source bank. Scheduler blocks continue the same
# immutable roster and exact verified bank; no source/model reduction.
# The completed parent supplies the unchanged actual91-test/runtime evidence.
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
# Dependency afterok:577006 protects against blind retries after failure.
# Verify the dedicated actual exporter terminal, distinct from test stdout.
python - "$root" "$job" <<"PY"
from pathlib import Path
from hashlib import sha256
import json,sys
from scripts.mfm import export_public_conditional_features as feature_worker
root=Path(sys.argv[1])/"rayan-mfm"
path=root/"receipts/rayan-mfm-public-features-577006.stdout"
raw=path.read_bytes()
records=[json.loads(line) for line in raw.decode().splitlines() if line.startswith("{")]
terminals=[row for row in records if "complete" in row and "public_conditional_feature_export_sha256" in row]
assert len(terminals)==1
terminal=terminals[0]
assert terminal["total"]==4862 and 0 < terminal["completed"] <=4862
assert terminal["complete"] is (terminal["completed"]==4862)
assert terminal["qualified_for_product"] is False
assert terminal["public_conditional_feature_export_sha256"]=="ef32550b82f7d92cc516b34b4de7e8101927e104d9003792deb1c11662106717"
bank_root=root/"features/rayan-mfm-public-e2e-20261007"
progress=feature_worker._read_sealed(feature_worker._within(bank_root,"progress.json"),feature_worker.PROGRESS_SCHEMA)
assert progress["record_sha256"]==terminal["public_conditional_feature_progress_sha256"]
assert progress["export_sha256"]==terminal["public_conditional_feature_export_sha256"]
assert progress["completed"]==terminal["completed"] and progress["total"]==terminal["total"]
assert progress["complete"] is terminal["complete"]
record={"schema":"mfm-full-source-export-parent-custody-v1","parent_job":577006,
        "parent_cli_receipt_file_sha256":sha256(raw).hexdigest(),"parent_terminal":terminal,
        "same_complete_source_roster":True,"source_export_only":True,"qualified_for_product":False}
destination=root/"receipts"/("rayan-mfm-full-source-parent-577006-for-"+sys.argv[2]+".json")
with destination.open("x",encoding="utf-8") as stream:
    destination.chmod(0o600)
    json.dump(record,stream,sort_keys=True,indent=2)
print(json.dumps({"full_source_parent_receipt_sha256":sha256(destination.read_bytes()).hexdigest()}),flush=True)
PY
python -m scripts.mfm.export_public_conditional_features \
  --census-dir "$root/rayan-mfm/runs/rayan-mfm-public-census-576995" \
  --census-sha256 "$census_sha" \
  --prepared-base-dir "$root/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/mfm-role-clone-576486" \
  --prepared-base-receipt "$root/mfm/receipts/mfm-clone-576486.json" \
  --resume \
  --stop-after-seconds 28800 \
  --output-dir "$root/rayan-mfm/features/rayan-mfm-public-e2e-20261007" \
  | tee "$root/rayan-mfm/receipts/rayan-mfm-public-features-$job.stdout"
'\'' full-source-export "$root"
' pipeline-wrapper "$root" "$repo" "$container"
