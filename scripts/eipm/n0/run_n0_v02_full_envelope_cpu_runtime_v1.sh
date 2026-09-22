#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: full-envelope CPU qualification source drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no
  exit 91
fi

python - <<'PY'
import torch
if torch.cuda.is_available():
    raise SystemExit("STOP: CPU qualification unexpectedly sees CUDA")
print("PASS_N0_FULL_ENVELOPE_CPU_ONLY_RUNTIME")
PY

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

QUAL="$ROOT/configs/eipm/n0/n0_v02_full_envelope_cpu_runtime_qualification_v1.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
TOKENIZER="$WORKDIR/tokenizer-v0.2.1"
CORPUS="$WORKDIR/tokenizer-corpus-v0.2.1-offline"
SEMANTIC="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
RUN_ROOT="$WORKDIR/full-envelope-cpu-runtime-v1"
RESULT="$RUN_ROOT/result.json"

for required in   "$QUAL"   "$SEMANTIC_CONFIG"   "$SOURCE_CONFIG"   "$TOKENIZER/tokenizer.json"   "$TOKENIZER/tokenizer_receipt.json"   "$CORPUS/corpus_receipt.json"   "$SEMANTIC"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing full-envelope CPU prerequisite: $required" >&2
    exit 92
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  echo "STOP: preserve existing full-envelope CPU evidence root: $RUN_ROOT" >&2
  exit 93
fi

EXPECTED_SEMANTIC_SHA="6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"
OBSERVED_SEMANTIC_SHA="$(sha256sum "$SEMANTIC" | awk '{print $1}')"
if [[ "$OBSERVED_SEMANTIC_SHA" != "$EXPECTED_SEMANTIC_SHA" ]]; then
  echo "STOP: semantic checkpoint SHA drift: $OBSERVED_SEMANTIC_SHA" >&2
  exit 94
fi

python -m py_compile   "$ROOT/src/alice_personality/n0/semantic_backbone_interface_v1.py"   "$ROOT/src/alice_personality/n0/n0_full_envelope_stack_v1.py"   "$ROOT/scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py"

mkdir -p "$RUN_ROOT"

echo "===== N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION V1 ====="
date -Is
echo "source_revision=$HEAD"
echo "gpu=false"
echo "gradient=false"
echo "optimizer=false"
echo "model_training=false"
echo "final_validation_opened=false"
echo "semantic_checkpoint_sha256=$OBSERVED_SEMANTIC_SHA"

python "$ROOT/scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py"   --qualification-config "$QUAL"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --corpus-dir "$CORPUS"   --source-config "$SOURCE_CONFIG"   --output "$RESULT"

python - "$RESULT" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1"
assert r["cpu_only"] is True
assert r["inference_mode"] is True
assert r["gradient"] is False
assert r["optimizer"] is False
assert r["model_training"] is False
assert r["private_identity_data"] is False
assert r["final_validation_opened"] is False
assert r["semantic_parameters"]==136594435
assert r["successor_parameters"] > 0
assert r["segment_context_bridge_parameters"] > 0
assert r["combined_parameters"] > r["semantic_parameters"] + r["successor_parameters"]
assert r["long_context_bridge"]["segments"] >= 3
assert r["long_context_bridge"]["stitched_tokens"] == r["long_context_bridge"]["original_tokens"]
assert r["long_context_bridge"]["standalone_virtualizer_semantics_complete"] is False
assert r["long_context_bridge"]["bridge_report"]["cross_window_semantic_interaction"] is True
assert r["long_context_bridge"]["bridge_report"]["segment_count_ceiling"] is None
assert r["runtime_case"]["relations"] >= 8
assert r["runtime_case"]["factor_banks"] >= 9
assert r["runtime_case"]["fields"] >= 8
assert r["runtime_case"]["edges"] >= 10
assert r["runtime_case"]["views_after_additional"] >= 8
assert r["runtime_case"]["candidate_count"] >= 5
assert r["gpu_training_authorized"] is False
assert r["n0_complete"] is False
print("PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1")
print("combined_parameters="+str(r["combined_parameters"]))
print("successor_parameters="+str(r["successor_parameters"]))
print("segment_context_bridge_parameters="+str(r["segment_context_bridge_parameters"]))
print("peak_rss_mb="+str(r["memory_mb"]["peak_rss_mb"]))
print("long_context_bridge="+json.dumps(r["long_context_bridge"],sort_keys=True))
print("runtime_case="+json.dumps(r["runtime_case"],sort_keys=True))
PY

echo "===== N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION V1 COMPLETE ====="
date -Is
echo "result=$RESULT"
