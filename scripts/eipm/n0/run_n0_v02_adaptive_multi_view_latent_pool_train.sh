#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST_PARENT="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
EVIDENCE_ADAPTER="$SPECIALIST_PARENT/evidence_view_adapter.safetensors"
EVIDENCE_GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
FUSION_CHECKPOINT="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.1.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_training_v0.1.json"
CURRICULUM_ROOT="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3"
CURRICULUM="$CURRICULUM_ROOT/cross_context_fusion_repair_curriculum.jsonl"
CURRICULUM_MANIFEST="$CURRICULUM_ROOT/cross_context_fusion_repair_manifest.json"
PREP_RECEIPT="$WORKDIR/adaptive-multi-view-latent-pool-prep-v0.1/preparation_receipt.json"
OUTPUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.1"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_full_scale.py"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CONFIG" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$STRUCTURED_CONFIG" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$STRUCTURED_CHECKPOINT/receipt.json" \
  "$EVIDENCE_ADAPTER" \
  "$EVIDENCE_GRAPH" \
  "$FUSION_CONFIG" \
  "$FUSION_RATIFICATION" \
  "$FUSION_CHECKPOINT/cross_context_fusion.safetensors" \
  "$LATENT_CONFIG" \
  "$TRAINING_CONFIG" \
  "$CURRICULUM" \
  "$CURRICULUM_MANIFEST" \
  "$PREP_RECEIPT" \
  "$TRAINER"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing latent-pool training artifact: $required" >&2
    exit 2
  fi
done

python - "$PREP_RECEIPT" "$LATENT_CONFIG" "$TRAINING_CONFIG" "$FUSION_RATIFICATION" "$FUSION_CHECKPOINT/cross_context_fusion.safetensors" "$CURRICULUM" "$CURRICULUM_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

receipt_path, latent_path, training_path, ratification_path, fusion_path, curriculum_path, manifest_path = map(Path, sys.argv[1:8])

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if receipt.get("status") != "PASS":
    raise SystemExit("latent-pool preparation did not pass")
if receipt.get("git_revision") != head:
    raise SystemExit(f"latent-pool prep revision is stale: receipt={receipt.get('git_revision')} current={head}")
for path, key in [
    (latent_path, "latent_config_sha256"),
    (training_path, "training_config_sha256"),
    (ratification_path, "fusion_ratification_sha256"),
    (fusion_path, "fusion_sha256"),
    (curriculum_path, "training_curriculum_sha256"),
    (manifest_path, "training_curriculum_manifest_sha256"),
]:
    if receipt.get(key) != sha(path):
        raise SystemExit(f"latent-pool same-revision artifact drift: {key}")
if receipt.get("public_latent_pool_gradient_authorized_after_prep") is not True:
    raise SystemExit("public latent-pool gradient authorization missing")
if receipt.get("private_identity_gradient") is not False:
    raise SystemExit("private gradient boundary crossed")
print("adaptive_latent_pool_same_revision_gate=true")
print(f"git_revision={head}")
print(f"fusion_sha256={receipt['fusion_sha256']}")
print(f"training_curriculum_sha256={receipt['training_curriculum_sha256']}")
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Adaptive latent-pool training requires one visible CUDA device." >&2
  exit 3
fi

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite latent-pool training output: $OUTPUT_ROOT" >&2
  exit 4
fi

echo "===== N0 V0.2 ADAPTIVE MULTI-VIEW LATENT POOL TRAINING ====="
date -Is
echo "full_scale_model=true"
echo "ratified_fusion_frozen_initially=true"
echo "exact_routing_percentage_supervision=false"
echo "fixed_slot_trait_labels=false"
echo "public_identity_neutral=true"
echo "private_identity_gradient=false"

python "$TRAINER" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$EVIDENCE_ADAPTER" \
  --evidence-graph "$EVIDENCE_GRAPH" \
  --fusion-config "$FUSION_CONFIG" \
  --fusion-ratification "$FUSION_RATIFICATION" \
  --fusion-checkpoint "$FUSION_CHECKPOINT" \
  --latent-config "$LATENT_CONFIG" \
  --training-config "$TRAINING_CONFIG" \
  --curriculum "$CURRICULUM" \
  --curriculum-manifest "$CURRICULUM_MANIFEST" \
  --prep-receipt "$PREP_RECEIPT" \
  --output-dir "$OUTPUT_ROOT" \
  --raw-max-length 96 \
  --field-max-length 96 \
  --encode-batch-size 32 \
  --fusion-cache-batch-size 8

echo "===== N0 V0.2 ADAPTIVE MULTI-VIEW LATENT POOL TRAINING COMPLETE ====="
date -Is
echo "comparison=$OUTPUT_ROOT/adaptive_multi_view_latent_pool_comparison.json"
