#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
FAILED_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.1"
PARENT_CACHE="$FAILED_ROOT/parent_cache.pt"
FUSION_CACHE="$FAILED_ROOT/ratified_fusion_cache.pt"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_training_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
PREP_RECEIPT="$WORKDIR/adaptive-multi-view-latent-pool-prep-v0.2/preparation_receipt.json"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_v0_2_2_full_scale.py"
OUTPUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$PARENT_CACHE" \
  "$FUSION_CACHE" \
  "$LATENT_CONFIG" \
  "$TRAINING_CONFIG" \
  "$FUSION_RATIFICATION" \
  "$PREP_RECEIPT" \
  "$TRAINER"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing competitive latent-pool v0.2 training artifact: $required" >&2
    exit 2
  fi
done

python - "$PREP_RECEIPT" "$PARENT_CACHE" "$FUSION_CACHE" "$LATENT_CONFIG" "$TRAINING_CONFIG" "$FUSION_RATIFICATION" "$TRAINER" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path, parent_path, fusion_cache_path, latent_path, training_path, ratification_path, trainer_path = map(Path, sys.argv[1:8])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if receipt.get("status") != "PASS":
    raise SystemExit("competitive latent-pool v0.2 prep did not pass")
if receipt.get("git_revision") != head:
    raise SystemExit(f"competitive latent-pool v0.2 prep is stale: receipt={receipt.get('git_revision')} current={head}")
for path, key in [
    (parent_path, "parent_cache_sha256"),
    (fusion_cache_path, "fusion_cache_sha256"),
    (latent_path, "latent_config_sha256"),
    (training_path, "training_config_sha256"),
    (ratification_path, "fusion_ratification_sha256"),
    (trainer_path, "trainer_sha256"),
]:
    if receipt.get(key) != sha(path):
        raise SystemExit(f"competitive latent-pool v0.2 artifact drift: {key}")
if receipt.get("public_latent_pool_v0_2_gradient_authorized_after_prep") is not True:
    raise SystemExit("competitive latent-pool v0.2 gradient authorization missing")
if receipt.get("failed_v0_1_latent_weights_reused") is not False:
    raise SystemExit("failed latent-pool v0.1 weights must not be reused")
if receipt.get("private_identity_gradient") is not False:
    raise SystemExit("private identity gradient boundary crossed")
print("adaptive_latent_pool_v0_2_same_revision_gate=true")
print(f"git_revision={head}")
print(f"parent_cache_sha256={receipt['parent_cache_sha256']}")
print(f"fusion_cache_sha256={receipt['fusion_cache_sha256']}")
print(f"ratified_fusion_sha256={receipt['ratified_fusion_sha256']}")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Competitive latent-pool v0.2 training requires one visible CUDA device." >&2
  exit 3
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite competitive latent-pool v0.2 output: $OUTPUT_ROOT" >&2
  exit 4
fi

echo "===== N0 V0.2 COMPETITIVE ADAPTIVE MULTI-VIEW LATENT POOL TRAINING ====="
date -Is
echo "full_scale_model=true"
echo "fresh_latent_initialization=true"
echo "failed_v0_1_latent_weights_reused=false"
echo "competitive_cross_attention=true"
echo "duplicate_target_slot_reward_forbidden=true"
echo "source_view_semantic_recoverability=true"
echo "disagreement_weighted_view_specialization=true"
echo "ratified_fusion_frozen_initially=true"
echo "exact_routing_percentage_supervision=false"
echo "fixed_slot_trait_labels=false"
echo "public_identity_neutral=true"
echo "private_identity_gradient=false"

python "$TRAINER" \
  --parent-cache "$PARENT_CACHE" \
  --fusion-cache "$FUSION_CACHE" \
  --latent-config "$LATENT_CONFIG" \
  --training-config "$TRAINING_CONFIG" \
  --prep-receipt "$PREP_RECEIPT" \
  --output-dir "$OUTPUT_ROOT"

echo "===== N0 V0.2 COMPETITIVE ADAPTIVE MULTI-VIEW LATENT POOL TRAINING COMPLETE ====="
date -Is
echo "comparison=$OUTPUT_ROOT/adaptive_multi_view_latent_pool_v0_2_comparison.json"
