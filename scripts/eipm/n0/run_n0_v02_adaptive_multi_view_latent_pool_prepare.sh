#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.1.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_training_v0.1.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
FUSION_CHECKPOINT="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240/cross_context_fusion.safetensors"
CURRICULUM_ROOT="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3"
CURRICULUM="$CURRICULUM_ROOT/cross_context_fusion_repair_curriculum.jsonl"
CURRICULUM_MANIFEST="$CURRICULUM_ROOT/cross_context_fusion_repair_manifest.json"
CONF3_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.3"
CONF3_RESULT="$CONF3_ROOT/result.json"
CONF3_CHALLENGE="$CONF3_ROOT/challenge.jsonl"
PREP_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-prep-v0.1"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$LATENT_CONFIG" \
  "$TRAINING_CONFIG" \
  "$FUSION_RATIFICATION" \
  "$FUSION_CHECKPOINT" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$CONF3_RESULT" \
  "$CONF3_CHALLENGE"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing latent-pool prerequisite: $required" >&2
    exit 2
  fi
done

python -m py_compile \
  "$ROOT/src/alice_personality/n0/adaptive_multi_view_latent_pool.py" \
  "$ROOT/src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_full_scale.py"

pytest -q \
  "$ROOT/tests/eipm/test_n0_adaptive_multi_view_latent_pool.py" \
  "$ROOT/tests/eipm/test_n0_adaptive_multi_view_latent_pool_objectives.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_anchored.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_routing_constraints.py" \
  "$ROOT/tests/eipm/test_n0_no_accidental_capability_ceilings.py"

rm -rf "$PREP_ROOT"
mkdir -p "$PREP_ROOT"

python - "$ROOT" "$LATENT_CONFIG" "$TRAINING_CONFIG" "$FUSION_RATIFICATION" "$FUSION_CHECKPOINT" "$CURRICULUM" "$CURRICULUM_MANIFEST" "$CONF3_RESULT" "$CONF3_CHALLENGE" "$PREP_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from alice_personality.n0.adaptive_multi_view_latent_pool import (
    AdaptiveMultiViewLatentPool,
    AdaptiveMultiViewLatentPoolConfig,
)

(
    root,
    latent_config_path,
    training_config_path,
    fusion_ratification_path,
    fusion_checkpoint_path,
    curriculum_path,
    curriculum_manifest_path,
    conf3_result_path,
    conf3_challenge_path,
    receipt_path,
) = map(Path, sys.argv[1:11])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
latent_cfg = json.loads(latent_config_path.read_text(encoding="utf-8"))
training_cfg = json.loads(training_config_path.read_text(encoding="utf-8"))
ratification = json.loads(fusion_ratification_path.read_text(encoding="utf-8"))
manifest = json.loads(curriculum_manifest_path.read_text(encoding="utf-8"))
result = json.loads(conf3_result_path.read_text(encoding="utf-8"))

if ratification.get("status") != "RATIFIED_PUBLIC_N0_FUSION_BASE_NOT_FULL_EIPM_PROMOTION":
    raise SystemExit("fusion is not ratified for latent-pool parent use")
if sha(fusion_checkpoint_path) != ratification["selected_fusion"]["sha256"]:
    raise SystemExit("ratified fusion checkpoint hash mismatch")
if result.get("status") != "PASS_UNCHANGED_CANDIDATE_ELIGIBLE_FOR_FUSION_RATIFICATION":
    raise SystemExit("v0.3 fusion confirmatory challenge did not pass")
if result.get("candidate", {}).get("ratification_gate_pass") is not True:
    raise SystemExit("v0.3 candidate gate pass missing")
if result.get("candidate_weights_changed_after_v0_2") is not False:
    raise SystemExit("fusion candidate changed after v0.2")
if result.get("challenge_rows_used_for_training") is not False or result.get("gradient_performed") is not False:
    raise SystemExit("v0.3 evaluation lineage contamination")
if result.get("candidate", {}).get("fusion_sha256") != ratification["selected_fusion"]["sha256"]:
    raise SystemExit("v0.3 candidate differs from ratified fusion")
if result.get("challenge_sha256") != sha(conf3_challenge_path):
    raise SystemExit("v0.3 challenge file hash drift")

if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-repair-curriculum.v0.3":
    raise SystemExit("latent source curriculum manifest mismatch")
if manifest.get("compiled_sha256") != sha(curriculum_path):
    raise SystemExit("latent source curriculum hash mismatch")
if int(manifest.get("rows", -1)) != 512 or int(manifest.get("train_rows", -1)) != 384 or int(manifest.get("dev_rows", -1)) != 128:
    raise SystemExit("latent source curriculum row/split drift")
if manifest.get("private_identity_content") is not False:
    raise SystemExit("latent source curriculum crossed private boundary")
if training_cfg["training_data"].get("existing_target_view_distribution_ignored_as_supervision") is not True:
    raise SystemExit("latent training must ignore old exact routing distributions")
if jsonl_ids(curriculum_path).intersection(jsonl_ids(conf3_challenge_path)):
    raise SystemExit("v0.3 frozen challenge rows overlap latent training source")

arch = latent_cfg["architecture"]
if arch.get("hard_parameter_ceiling") is not None or arch.get("slot_count_ceiling") is not None or arch.get("view_count_ceiling") is not None:
    raise SystemExit("latent-pool config contains accidental hard capacity ceiling")
config = AdaptiveMultiViewLatentPoolConfig(
    semantic_size=int(arch["semantic_size"]),
    latent_size=int(arch["latent_size"]),
    num_slots=int(arch["current_instantiated_slots"]),
    num_layers=int(arch["latent_layers"]),
    num_heads=int(arch["num_heads"]),
    feedforward_size=int(arch["feedforward_size"]),
    dropout=float(arch["dropout"]),
    num_views=int(arch["current_instantiated_view_count"]),
)
model = AdaptiveMultiViewLatentPool(config)
report = model.parameter_report()
if report.get("hard_parameter_ceiling") is not None:
    raise SystemExit("latent model reports hard parameter ceiling")

receipt = {
    "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-preparation.v0.1",
    "status": "PASS",
    "git_revision": head,
    "latent_config_sha256": sha(latent_config_path),
    "training_config_sha256": sha(training_config_path),
    "fusion_ratification_sha256": sha(fusion_ratification_path),
    "fusion_sha256": sha(fusion_checkpoint_path),
    "training_curriculum_sha256": sha(curriculum_path),
    "training_curriculum_manifest_sha256": sha(curriculum_manifest_path),
    "confirmatory_v0_3_result_sha256": sha(conf3_result_path),
    "confirmatory_v0_3_challenge_sha256": sha(conf3_challenge_path),
    "latent_parameter_report": report,
    "training_rows": 384,
    "dev_rows": 128,
    "exact_routing_percentage_supervision": False,
    "fixed_slot_trait_labels": False,
    "frozen_challenge_v0_3_rows_used_for_training": False,
    "public_latent_pool_gradient_authorized_after_prep": True,
    "parent_fusion_trainable": False,
    "parent_fusion_freeze_is_stage_control_not_permanent": True,
    "hard_parameter_ceiling": None,
    "slot_count_ceiling": None,
    "view_count_ceiling": None,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "gradient_performed": False,
    "gpu_required": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "adaptive_latent_pool_prepare_pass=true"
echo "preparation_receipt=$PREP_RECEIPT"
echo "fusion_ratified=true"
echo "training_rows=384 dev_rows=128"
echo "exact_routing_percentage_supervision=false"
echo "fixed_slot_trait_labels=false"
echo "frozen_challenge_v0_3_rows_used_for_training=false"
echo "public_latent_pool_gradient_authorized_after_prep=true"
echo "private_identity_gradient=false"
echo "gradient_performed=false"
echo "gpu_required=false"
