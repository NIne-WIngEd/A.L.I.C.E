#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
FAILED_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.1"
FAILED_COMPARISON="$FAILED_ROOT/adaptive_multi_view_latent_pool_comparison.json"
PARENT_CACHE="$FAILED_ROOT/parent_cache.pt"
FUSION_CACHE="$FAILED_ROOT/ratified_fusion_cache.pt"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_training_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
LATENT_MODEL="$ROOT/src/alice_personality/n0/adaptive_multi_view_latent_pool_v0_2.py"
LATENT_OBJECTIVES="$ROOT/src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives_v0_2.py"
TRAINER_BASE="$ROOT/scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale.py"
PREP_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-prep-v0.2"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$FAILED_COMPARISON" \
  "$PARENT_CACHE" \
  "$FUSION_CACHE" \
  "$LATENT_CONFIG" \
  "$TRAINING_CONFIG" \
  "$FUSION_RATIFICATION" \
  "$LATENT_MODEL" \
  "$LATENT_OBJECTIVES" \
  "$TRAINER_BASE" \
  "$TRAINER"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing competitive latent-pool v0.2 prerequisite: $required" >&2
    exit 2
  fi
done

python -m py_compile \
  "$LATENT_MODEL" \
  "$LATENT_OBJECTIVES" \
  "$TRAINER_BASE" \
  "$TRAINER"

pytest -q \
  "$ROOT/tests/eipm/test_n0_adaptive_multi_view_latent_pool_v0_2.py" \
  "$ROOT/tests/eipm/test_n0_adaptive_multi_view_latent_pool_objectives_v0_2.py" \
  "$ROOT/tests/eipm/test_n0_no_accidental_capability_ceilings.py"

rm -rf "$PREP_ROOT"
mkdir -p "$PREP_ROOT"

python - "$ROOT" "$FAILED_COMPARISON" "$PARENT_CACHE" "$FUSION_CACHE" "$LATENT_CONFIG" "$TRAINING_CONFIG" "$FUSION_RATIFICATION" "$LATENT_MODEL" "$LATENT_OBJECTIVES" "$TRAINER" "$PREP_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
)
from train_n0_v02_adaptive_multi_view_latent_pool_v0_2_full_scale import load_config

(
    root,
    failed_path,
    parent_path,
    fusion_cache_path,
    latent_path,
    training_path,
    ratification_path,
    latent_model_path,
    latent_objectives_path,
    trainer_path,
    receipt_path,
) = map(Path, sys.argv[1:12])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

failed = json.loads(failed_path.read_text(encoding="utf-8"))
training = json.loads(training_path.read_text(encoding="utf-8"))
ratification = json.loads(ratification_path.read_text(encoding="utf-8"))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

if failed.get("status") != "FAIL_NO_LATENT_POOL_CHECKPOINT_CLEARED_TRAINING_DEV_GATE":
    raise SystemExit("expected failed latent-pool v0.1 comparison")
if failed.get("winner") is not None:
    raise SystemExit("failed latent-pool v0.1 unexpectedly reports a winner")
checkpoints = failed.get("checkpoints", [])
if len(checkpoints) != 3 or [int(item["step"]) for item in checkpoints] != [120, 240, 360]:
    raise SystemExit("failed latent-pool v0.1 checkpoint lineage drift")
if any(item.get("training_gate_pass") is not False for item in checkpoints):
    raise SystemExit("v0.1 failure lineage unexpectedly contains an eligible checkpoint")
if any(float(item["dev_metrics"]["mean_max_offdiag_slot_cosine"]) < 0.99 for item in checkpoints):
    raise SystemExit("v0.1 failure signature no longer matches slot collapse diagnosis")

lineages = [item["lineage"] for item in checkpoints]
parent_sha = sha(parent_path)
fusion_cache_sha = sha(fusion_cache_path)
if any(item.get("parent_cache_sha256") != parent_sha for item in lineages):
    raise SystemExit("parent cache does not match failed-run lineage")
if any(item.get("fusion_cache_sha256") != fusion_cache_sha for item in lineages):
    raise SystemExit("fusion cache does not match failed-run lineage")

if ratification.get("status") != "RATIFIED_PUBLIC_N0_FUSION_BASE_NOT_FULL_EIPM_PROMOTION":
    raise SystemExit("fusion is not ratified")
ratified_fusion_sha = ratification["selected_fusion"]["sha256"]
if any(item.get("fusion_sha256") != ratified_fusion_sha for item in lineages):
    raise SystemExit("cached latent-pool parent was not built from ratified fusion")

if training.get("failed_predecessor", {}).get("failed_latent_weights_reused") is not False:
    raise SystemExit("v0.2 must fresh-initialize latent weights")
if training.get("capacity_policy", {}).get("hard_parameter_ceiling") is not None:
    raise SystemExit("v0.2 training config contains a hard parameter ceiling")
if training.get("capacity_policy", {}).get("slot_count_ceiling") is not None:
    raise SystemExit("v0.2 training config contains a slot ceiling")
policy = training.get("objective_policy", {})
if policy.get("old_unnormalized_smoothmax_forbidden") is not True:
    raise SystemExit("v0.2 objective correction gate missing")
if policy.get("available_source_view_semantic_recoverability") is not True:
    raise SystemExit("v0.2 source-view semantic coverage gate missing")
if policy.get("consensus_views_not_forced_into_artificial_distinctions") is not True:
    raise SystemExit("v0.2 consensus-safe specialization gate missing")

model = AdaptiveMultiViewLatentPoolV02(load_config(latent_path)).eval()
report = model.parameter_report()
if report.get("competitive_cross_attention") is not True:
    raise SystemExit("competitive cross-attention missing")
if report.get("hard_parameter_ceiling") is not None or report.get("slot_count_ceiling") is not None:
    raise SystemExit("latent v0.2 parameter report contains a hard capability ceiling")

parent = torch.load(parent_path, map_location="cpu", weights_only=False)
fusion = torch.load(fusion_cache_path, map_location="cpu", weights_only=False)
if len(parent.get("ids", [])) != 512:
    raise SystemExit("parent cache row count drift")
if len(fusion.get("fusion_view_weights", [])) != 512:
    raise SystemExit("fusion cache row count drift")
if sum(1 for value in parent.get("splits", []) if value == "train") != 384:
    raise SystemExit("parent cache train split drift")
if sum(1 for value in parent.get("splits", []) if value == "dev") != 128:
    raise SystemExit("parent cache dev split drift")

receipt = {
    "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-preparation.v0.2.1",
    "status": "PASS",
    "git_revision": head,
    "failed_v0_1_job": 575670,
    "failed_v0_1_status": failed["status"],
    "failed_v0_1_latent_weights_reused": False,
    "prior_v0_2_infrastructure_failure_job": 575673,
    "prior_v0_2_infrastructure_failure_stage": "random_baseline_before_any_optimizer_step",
    "prior_v0_2_infrastructure_failure_gradient_performed": False,
    "mixed_precision_semantic_boundaries_fp32": True,
    "failure_diagnosis": "objective_rewarded_duplicate_target_slots_and_independent_cross_attention_permitted_uncompetitive_slot_binding",
    "parent_cache_sha256": parent_sha,
    "fusion_cache_sha256": fusion_cache_sha,
    "ratified_fusion_sha256": ratified_fusion_sha,
    "fusion_ratification_sha256": sha(ratification_path),
    "latent_config_sha256": sha(latent_path),
    "training_config_sha256": sha(training_path),
    "latent_model_sha256": sha(latent_model_path),
    "latent_objectives_sha256": sha(latent_objectives_path),
    "trainer_sha256": sha(trainer_path),
    "latent_parameter_report": report,
    "training_rows": 384,
    "dev_rows": 128,
    "competitive_cross_attention": True,
    "old_unnormalized_smoothmax_forbidden": True,
    "duplicate_target_slot_reward_forbidden": True,
    "available_source_view_semantic_recoverability": True,
    "consensus_views_not_forced_into_artificial_distinctions": True,
    "exact_routing_percentage_supervision": False,
    "fixed_slot_trait_labels": False,
    "frozen_challenge_rows_used_for_training": False,
    "parent_fusion_freeze_is_stage_control_not_permanent": True,
    "hard_parameter_ceiling": None,
    "slot_count_ceiling": None,
    "view_count_ceiling": None,
    "public_latent_pool_v0_2_gradient_authorized_after_prep": True,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "gradient_performed": False,
    "gpu_required": False
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "adaptive_latent_pool_v0_2_prepare_pass=true"
echo "preparation_receipt=$PREP_RECEIPT"
echo "failed_v0_1_preserved=true"
echo "failed_v0_1_latent_weights_reused=false"
echo "prior_v0_2_infrastructure_failure_job=575673"
echo "prior_v0_2_infrastructure_failure_gradient_performed=false"
echo "mixed_precision_semantic_boundaries_fp32=true"
echo "competitive_cross_attention=true"
echo "duplicate_target_slot_reward_forbidden=true"
echo "available_source_view_semantic_recoverability=true"
echo "consensus_views_not_forced_into_artificial_distinctions=true"
echo "training_rows=384 dev_rows=128"
echo "public_latent_pool_v0_2_gradient_authorized_after_prep=true"
echo "private_identity_gradient=false"
echo "gradient_performed=false"
echo "gpu_required=false"
