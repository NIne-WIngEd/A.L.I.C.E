#!/bin/bash
#SBATCH --job-name=rayan-mfm-cached-learning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
umask 077
: "${MFM_EXPORT_JOB_IDS:?exact externally recorded producer job IDs required}"
[[ "$MFM_EXPORT_JOB_IDS" =~ ^[0-9]+(:[0-9]+)*$ ]]
[[ "$(id -un)" == mxrayan && "$(id -u)" == 1905 ]]
root=/homes/01/mxrayan/rayan-compute
repo="$root/rayan-mfm/code/rayan-mfm-7f9621bdd"
container=rayan-mfm-cache-gpu-v2-576613
[[ "$(cd "$repo" && git rev-parse HEAD)" == 7f9621bddfe0b1136b776a9850e2f0e5c8f4842a && -z "$(cd "$repo" && git status --porcelain)" ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
export UDOCKER_DIR="$root/udocker-store" UDOCKER_TMP="$root/udocker-tmp"
udocker="$root/tools/udocker-1.3.17/udocker/udocker"
"$udocker" setup --nvidia --force "$container"
export MFM_HOST_NETNS="$(readlink /proc/self/ns/net)" MFM_HOST_UID=1905 MFM_HOST_GID=100
module load python/3.11.5
python3 -I "$repo/scripts/mfm/enter_private_network_namespace.py" -- /bin/bash -c '
set -euo pipefail
root="$1"
repo="$2"
container="$3"
"$root/tools/udocker-1.3.17/udocker/udocker" run --volume="$root:$root" --workdir="$repo" \
  --env="PYTHONPATH=$root/rayan-mfm/runtime/feature-deps-576610:$repo/src" \
  --env="OMP_NUM_THREADS=8" --env="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env="MFM_HOST_NETNS=$MFM_HOST_NETNS" --env="MFM_EXPORT_JOB_IDS=$MFM_EXPORT_JOB_IDS" \
  --env="MFM_PIPELINE_JOB=$SLURM_JOB_ID" --entrypoint=/bin/bash "$container" -c '\''
set -euo pipefail
umask 077
python - <<"PY"
import os, socket, torch
assert sorted(n for _, n in socket.if_nameindex()) == ["lo"]
assert os.readlink("/proc/self/ns/net") != os.environ["MFM_HOST_NETNS"]
assert torch.__version__ == "2.7.1+cu118" and torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (6, 0)
print("private_P100_runtime_verified", flush=True)
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
  --feature-bank-sha256 "$bank_sha" --device cuda:0)
probe="$root/rayan-mfm/runs/rayan-mfm-probe-$job"
fit="$root/rayan-mfm/runs/rayan-mfm-fit-$job"
python -m scripts.mfm.train_v16_cached_specialist "${common[@]}" --output-dir "$probe" --probe-only
python - "$probe" <<"PY"
from pathlib import Path
import sys
from scripts.mfm import train_v16_cached_specialist as route
d = route.training.shared._read_sealed(Path(sys.argv[1]) / "component.json", route.ARTIFACT_SCHEMA)
assert d["probe_only"] is True and d["result"]["backward_and_restart_passed"] is True
assert d["result"]["checkpoint_reload_executed"] is True and d["result"]["optimizer_steps"] == 3
PY
python -m scripts.mfm.train_v16_cached_specialist "${common[@]}" --output-dir "$fit" --epochs 2
component_sha="$(python - "$fit" <<"PY"
from pathlib import Path
import sys
from scripts.mfm import train_v16_cached_specialist as route
d = route.training.shared._read_sealed(Path(sys.argv[1]) / "component.json", route.ARTIFACT_SCHEMA)
assert d["probe_only"] is False and d["result"]["all_train_epochs_completed"] is True
print(d["record_sha256"])
PY
)"
python -m scripts.mfm.run_v16_cached_diagnostic "${common[@]}" --component-dir "$fit" \
  --component-sha256 "$component_sha" --output-dir "$root/rayan-mfm/runs/rayan-mfm-generated-$job" --max-new-tokens 2048
'\'' cache-learning "$root"
' pipeline-wrapper "$root" "$repo" "$container"
