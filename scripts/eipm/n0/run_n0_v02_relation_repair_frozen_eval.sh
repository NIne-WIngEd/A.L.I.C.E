#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
REPAIR_ROOT="$WORKDIR/relation-repair-v0.1"
CHALLENGE_ROOT="$WORKDIR/relation-essential-challenge-v0.1"
CHALLENGE="$CHALLENGE_ROOT/relation_essential_challenge.jsonl"
CHALLENGE_MANIFEST="$CHALLENGE_ROOT/relation_essential_challenge_manifest.json"
OUT_ROOT="$WORKDIR/relation-repair-frozen-eval-v0.1"
OUT="$OUT_ROOT/relation_repair_frozen_ratification.json"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" \
  "$STRUCTURED_CHECKPOINT/structured_state.safetensors" \
  "$PARENT_ADAPTER" \
  "$CHALLENGE" \
  "$CHALLENGE_MANIFEST" \
  "$CONFIG" \
  "$STRUCTURED_CONFIG"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing frozen repair-eval artifact: $required" >&2
    exit 2
  fi
done

for step in 40 80 120 160; do
  ckpt="$REPAIR_ROOT/step-$(printf '%08d' "$step")"
  for required in "$ckpt/evidence_graph_dual_endpoint.safetensors" "$ckpt/receipt.json"; do
    if [[ ! -f "$required" ]]; then
      echo "Missing repaired checkpoint artifact: $required" >&2
      exit 3
    fi
  done
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing frozen repair evaluation: $OUT_ROOT" >&2
  exit 4
fi
mkdir -p "$OUT_ROOT"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/eval_n0_v02_relation_essential_challenge.py" \
  "$ROOT/scripts/eipm/n0/eval_n0_v02_relation_repair_frozen.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_dual_endpoint.py" \
  "$ROOT/src/alice_personality/n0/evidence_view_adapter.py"

python - "$CHALLENGE" "$CHALLENGE_MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
challenge = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
observed = hashlib.sha256(challenge.read_bytes()).hexdigest()
if observed != manifest.get("compiled_sha256"):
    raise SystemExit("frozen challenge hash mismatch")
if manifest.get("status") != "FROZEN_EVAL_ONLY":
    raise SystemExit("challenge is not frozen eval-only")
if manifest.get("training_authorized") is not False:
    raise SystemExit("challenge unexpectedly authorizes training")
if manifest.get("examples") != 80 or manifest.get("pairs") != 40:
    raise SystemExit("frozen challenge size drift")
print("frozen_relation_challenge_reuse_gate=true")
print("frozen_relation_challenge_sha256=" + observed)
PY

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Frozen relation repair evaluation requires one visible CUDA device." >&2
  exit 5
fi

echo "===== N0 V0.2 RELATION REPAIR FROZEN RATIFICATION ====="
date -Is
echo "training=false"
echo "challenge_regenerated=false"
echo "checkpoints=40,80,120,160"
echo "selection=earliest_checkpoint_clearing_all_capability_and_retention_gates"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

python "$ROOT/scripts/eipm/n0/eval_n0_v02_relation_repair_frozen.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --parent-adapter "$PARENT_ADAPTER" \
  --repair-root "$REPAIR_ROOT" \
  --challenge "$CHALLENGE" \
  --challenge-manifest "$CHALLENGE_MANIFEST" \
  --output "$OUT"

echo "===== N0 V0.2 RELATION REPAIR FROZEN RATIFICATION COMPLETE ====="
date -Is
echo "comparison=$OUT"
