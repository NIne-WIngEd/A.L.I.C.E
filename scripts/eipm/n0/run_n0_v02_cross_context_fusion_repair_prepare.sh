#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
BASE_ROOT="$WORKDIR/cross-context-fusion-curriculum-v0.2"
BASE_CURRICULUM="$BASE_ROOT/cross_context_fusion_curriculum.jsonl"
REPAIR_ROOT="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3"
REPAIR_CURRICULUM="$REPAIR_ROOT/cross_context_fusion_repair_curriculum.jsonl"
REPAIR_MANIFEST="$REPAIR_ROOT/cross_context_fusion_repair_manifest.json"
PREP_RECEIPT="$REPAIR_ROOT/preparation_receipt.json"
FAILED_CHALLENGE="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1/result.json"
FUSION_PARENT="$WORKDIR/cross-context-fusion-training-v0.1/step-00000240/cross_context_fusion.safetensors"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_repair_training_v0.2.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$BASE_CURRICULUM" \
  "$FAILED_CHALLENGE" \
  "$FUSION_PARENT" \
  "$TRAINING_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing fusion repair prerequisite: $required" >&2
    exit 2
  fi
done

python -m py_compile \
  "$ROOT/src/alice_personality/n0/cross_context_fusion_anchored.py" \
  "$ROOT/src/alice_personality/n0/cross_context_fusion_repair_objectives.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_repair_curriculum_v0_3.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_cross_context_fusion_repair_full_scale.py"

pytest -q \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_objectives.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_anchored.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_repair_objectives.py" \
  "$ROOT/tests/eipm/test_n0_no_accidental_capability_ceilings.py"

rm -rf "$REPAIR_ROOT"
mkdir -p "$REPAIR_ROOT"
python "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_repair_curriculum_v0_3.py" \
  --base-curriculum "$BASE_CURRICULUM" \
  --output "$REPAIR_CURRICULUM" \
  --manifest "$REPAIR_MANIFEST"

python - "$ROOT" "$REPAIR_MANIFEST" "$FAILED_CHALLENGE" "$FUSION_PARENT" "$TRAINING_CONFIG" "$PREP_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
failed_path = Path(sys.argv[3])
parent_path = Path(sys.argv[4])
config_path = Path(sys.argv[5])
receipt_path = Path(sys.argv[6])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = json.loads(manifest_path.read_text())
failed = json.loads(failed_path.read_text())
config = json.loads(config_path.read_text())
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-repair-curriculum.v0.3":
    raise SystemExit("repair curriculum schema mismatch")
if manifest.get("frozen_challenge_rows_reused") is not False:
    raise SystemExit("frozen challenge row contamination detected")
if manifest.get("frozen_challenge_text_templates_reused") is not False:
    raise SystemExit("frozen challenge template contamination detected")
if manifest.get("frozen_challenge_entities_reused") is not False:
    raise SystemExit("frozen challenge entity contamination detected")
if manifest.get("private_identity_content") is not False:
    raise SystemExit("repair curriculum crossed private identity boundary")
if failed.get("status") != "FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION":
    raise SystemExit("expected failed first frozen challenge")
if failed.get("challenge_rows_used_for_training") is not False:
    raise SystemExit("failed challenge reports training contamination")
if sha(parent_path) != config["fusion_parent"]["sha256"]:
    raise SystemExit("step240 fusion parent hash mismatch")
if config["architecture"].get("hard_parameter_ceiling") is not None:
    raise SystemExit("repair architecture contains a hard parameter ceiling")
if config["architecture"].get("view_count_ceiling") is not None:
    raise SystemExit("repair architecture contains a view-count ceiling")

receipt = {
    "schema": "alice.eipm.n0.v02-cross-context-fusion-repair-preparation.v0.2",
    "status": "PASS",
    "git_revision": head,
    "repair_curriculum_sha256": manifest["compiled_sha256"],
    "repair_manifest_sha256": sha(manifest_path),
    "failed_challenge_result_sha256": sha(failed_path),
    "fusion_parent_sha256": sha(parent_path),
    "fusion_parent_step": 240,
    "full_scale_model": True,
    "source_anchor_summary_channel": True,
    "source_anchor_token_channel": True,
    "frozen_challenge_rows_used_for_training": False,
    "frozen_challenge_templates_used_for_training": False,
    "row_count_is_training_tranche_not_cap": True,
    "hard_parameter_ceiling": None,
    "view_count_ceiling": None,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "gradient_performed": False,
    "gpu_required": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "cross_context_fusion_repair_prepare_pass=true"
echo "repair_curriculum=$REPAIR_CURRICULUM"
echo "repair_manifest=$REPAIR_MANIFEST"
echo "preparation_receipt=$PREP_RECEIPT"
echo "full_scale_model=true"
echo "source_anchor_summary_channel=true"
echo "source_anchor_token_channel=true"
echo "frozen_challenge_rows_used_for_training=false"
echo "hard_parameter_ceiling=none"
echo "view_count_ceiling=none"
echo "gradient_performed=false"
echo "gpu_required=false"
