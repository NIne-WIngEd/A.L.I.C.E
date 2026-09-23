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
EVIDENCE_ROOT="$RUN_ROOT/operator-evidence-alignment"
EVIDENCE_ROWS="$EVIDENCE_ROOT/rows.jsonl"
EVIDENCE_MANIFEST="$EVIDENCE_ROOT/manifest.json"
EVIDENCE_STATIC_AUDIT="$EVIDENCE_ROOT/static_audit.json"
EVIDENCE_TOKEN_AUDIT="$EVIDENCE_ROOT/token_alignment.json"
LONG_CONTEXT_ROOT="$RUN_ROOT/long-context"
LONG_CONTEXT_ROWS="$LONG_CONTEXT_ROOT/rows.jsonl"
LONG_CONTEXT_MANIFEST="$LONG_CONTEXT_ROOT/manifest.json"
LONG_CONTEXT_STATIC_AUDIT="$LONG_CONTEXT_ROOT/static_audit.json"
LONG_CONTEXT_TOKEN_BOUNDARY_AUDIT="$LONG_CONTEXT_ROOT/token_boundary_alignment.json"

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

python -m py_compile \
  "$ROOT/src/alice_personality/n0/semantic_backbone_interface_v1.py" \
  "$ROOT/src/alice_personality/n0/semantic_operator_evidence_targets_v1.py" \
  "$ROOT/src/alice_personality/n0/semantic_context_virtualizer_v1.py" \
  "$ROOT/src/alice_personality/n0/semantic_segment_context_bridge_v1.py" \
  "$ROOT/src/alice_personality/n0/full_envelope_semantic_input_v1.py" \
  "$ROOT/src/alice_personality/n0/n0_full_envelope_stack_v1.py" \
  "$ROOT/src/alice_personality/n0/n0_full_envelope_trainable_system_v1.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_semantic_operator_intervention_curriculum_v1.py" \
  "$ROOT/scripts/eipm/n0/audit_n0_v02_semantic_operator_curriculum_v1.py" \
  "$ROOT/scripts/eipm/n0/audit_n0_v02_operator_evidence_token_alignment_v1.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_full_envelope_long_context_curriculum_v1.py" \
  "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_curriculum_v1.py" \
  "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_token_boundaries_v1.py" \
  "$ROOT/scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py"

mkdir -p "$RUN_ROOT" "$EVIDENCE_ROOT" "$LONG_CONTEXT_ROOT"

python "$ROOT/scripts/eipm/n0/build_n0_v02_semantic_operator_intervention_curriculum_v1.py" \
  --output "$EVIDENCE_ROWS" \
  --manifest "$EVIDENCE_MANIFEST" \
  --examples-per-relation 20 \
  --candidate-counts 1,2,4,8,16

python "$ROOT/scripts/eipm/n0/audit_n0_v02_semantic_operator_curriculum_v1.py" \
  --rows "$EVIDENCE_ROWS" \
  --manifest "$EVIDENCE_MANIFEST" \
  --contract "$ROOT/configs/eipm/n0/n0_v02_semantic_operator_curriculum_contract_v1.json" \
  --output "$EVIDENCE_STATIC_AUDIT"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_operator_evidence_token_alignment_v1.py" \
  --rows "$EVIDENCE_ROWS" \
  --tokenizer-dir "$TOKENIZER" \
  --max-length 512 \
  --output "$EVIDENCE_TOKEN_AUDIT"

python - "$EVIDENCE_STATIC_AUDIT" "$EVIDENCE_TOKEN_AUDIT" <<'PY'
import json, sys
static=json.load(open(sys.argv[1]))
token=json.load(open(sys.argv[2]))
assert static["status"]=="PASS_SEMANTIC_OPERATOR_CURRICULUM_AUDIT"
assert static["query_relation_evidence_spans_required"] is True
assert static["relation_schema_evidence_spans_required"] is True
assert static["factor_schema_evidence_spans_required"] is True
assert static["step_factor_schema_evidence_spans_required"] is True
assert token["status"]=="PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1"
assert token["relation_steps"] > 0
assert token["positive_query_tokens"] > 0
assert token["positive_relation_schema_tokens"] > 0
assert token["positive_factor_schema_tokens"] > 0
assert token["positive_step_factor_schema_tokens"] > 0
assert token["training_authorized_by_audit"] is False
assert token["max_length_is_product_ceiling"] is False
print("PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1")
PY

python "$ROOT/scripts/eipm/n0/build_n0_v02_full_envelope_long_context_curriculum_v1.py" \
  --output "$LONG_CONTEXT_ROWS" \
  --manifest "$LONG_CONTEXT_MANIFEST" \
  --long-word-target 4608

python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_curriculum_v1.py" \
  --rows "$LONG_CONTEXT_ROWS" \
  --manifest "$LONG_CONTEXT_MANIFEST" \
  --contract "$ROOT/configs/eipm/n0/n0_v02_full_envelope_long_context_curriculum_contract_v1.json" \
  --output "$LONG_CONTEXT_STATIC_AUDIT"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_long_context_token_boundaries_v1.py" \
  --rows "$LONG_CONTEXT_ROWS" \
  --manifest "$LONG_CONTEXT_MANIFEST" \
  --contract "$ROOT/configs/eipm/n0/n0_v02_full_envelope_long_context_curriculum_contract_v1.json" \
  --tokenizer-dir "$TOKENIZER" \
  --output "$LONG_CONTEXT_TOKEN_BOUNDARY_AUDIT"

python - "$LONG_CONTEXT_STATIC_AUDIT" "$LONG_CONTEXT_TOKEN_BOUNDARY_AUDIT" <<'PY'
import json, sys
static=json.load(open(sys.argv[1]))
token=json.load(open(sys.argv[2]))
assert static["status"]=="PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_CURRICULUM_AUDIT_V1"
assert token["status"]=="PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1"
assert token["verified_pairs"]==token["expected_pairs"]
assert token["expected_pairs"]>0
assert token["exact_untrained_output_equality_claimed"] is False
assert token["training_authorized_by_audit"] is False
assert token["gradient"] is False
assert token["optimizer"] is False
assert token["gpu_training_authorized"] is False
assert token["final_opening_authorized"] is False
assert token["n0_complete"] is False
print("PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1")
PY

echo "===== N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION V1 ====="
date -Is
echo "source_revision=$HEAD"
echo "gpu=false"
echo "gradient=false"
echo "optimizer=false"
echo "model_training=false"
echo "final_validation_opened=false"
echo "semantic_checkpoint_sha256=$OBSERVED_SEMANTIC_SHA"
echo "operator_evidence_token_alignment=$EVIDENCE_TOKEN_AUDIT"
echo "long_context_token_boundary_alignment=$LONG_CONTEXT_TOKEN_BOUNDARY_AUDIT"

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
assert r["semantic_input_parameters"] > 0
assert r["successor_parameters"] > 0
assert r["segment_context_bridge_parameters"] > 0
assert r["combined_parameters"] == (
    r["semantic_parameters"]
    + r["semantic_input_parameters"]
    + r["successor_parameters"]
)
assert r["registered_trainable_system"]=="N0FullEnvelopeTrainableSystemV1"
assert r["single_shared_backbone"] is True
assert r["long_context_bridge"]["segments"] >= 3
assert r["long_context_bridge"]["dedicated_fixture_used"] is True
assert r["long_context_bridge"]["standalone_virtualizer_semantics_complete"] is False
assert r["long_context_bridge"]["bridge_report"]["cross_window_semantic_interaction"] is True
assert r["long_context_bridge"]["bridge_report"]["segment_count_ceiling"] is None
surface=r["text_surface_virtualization"]["surface_receipt"]
assert set(surface)=={
    "query",
    "relation_schema",
    "factor_schema",
    "field_text",
    "candidate_text",
    "descriptor_text",
    "internal_view_descriptor",
    "additional_view_descriptor",
    "additional_view_source",
}
assert all(surface.values())
assert r["system_report"]["semantic_replay_and_full_envelope_share_backbone"] is True
assert r["system_report"]["full_envelope_gradient_path_registered"] is True
assert r["system_report"]["all_text_surfaces_share_semantic_input"] is True
assert r["system_report"]["product_context_token_ceiling"] is None
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
print("semantic_input_parameters="+str(r["semantic_input_parameters"]))
print("peak_rss_mb="+str(r["memory_mb"]["peak_rss_mb"]))
print("text_surface_virtualization="+json.dumps(r["text_surface_virtualization"]["surface_receipt"],sort_keys=True))
print("long_context_bridge="+json.dumps(r["long_context_bridge"],sort_keys=True))
print("runtime_case="+json.dumps(r["runtime_case"],sort_keys=True))
PY

echo "===== N0 FULL-ENVELOPE CPU RUNTIME QUALIFICATION V1 COMPLETE ====="
date -Is
echo "result=$RESULT"
echo "operator_evidence_token_alignment=$EVIDENCE_TOKEN_AUDIT"
