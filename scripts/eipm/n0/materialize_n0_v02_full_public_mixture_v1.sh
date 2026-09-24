#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
FEWREL_ROOT="${ALICE_N0_FEWREL_ROOT:?ALICE_N0_FEWREL_ROOT is required}"
TEACHER_REGISTRY="${ALICE_N0_TEACHER_REGISTRY:?ALICE_N0_TEACHER_REGISTRY is required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?ALICE_N0_TEACHER_AUDIT is required}"
CPU_ROOT="${ALICE_N0_CPU_RUNTIME_ROOT:-$WORKDIR/full-envelope-cpu-runtime-v1}"
MIXTURE_ROOT="${ALICE_N0_FULL_MIXTURE_ROOT:?ALICE_N0_FULL_MIXTURE_ROOT is required}"
TOKENIZER="${ALICE_N0_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
CORPUS="${ALICE_N0_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: public-mixture materialization source drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: repository is not fully clean" >&2
  git status --short
  exit 91
fi
if [[ -e "$MIXTURE_ROOT" ]]; then
  echo "STOP: preserve existing public-mixture evidence root: $MIXTURE_ROOT" >&2
  exit 92
fi

SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
CORPUS_RECEIPT="$CORPUS/corpus_receipt.json"
NATURAL_SOURCES="$ROOT/configs/eipm/n0/n0_v02_natural_relation_sources_v1.json"
MIXTURE_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_public_mixture_contract_v1.json"
BEHAVIORAL_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_envelope_behavioral_curriculum_contract_v1.json"
RUNTIME_VIEW_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_envelope_runtime_view_curriculum_contract_v1.json"
SEMANTIC_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_semantic_operator_curriculum_contract_v1.json"
LONG_CONTEXT_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_envelope_long_context_curriculum_contract_v1.json"
FINAL_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_envelope_final_validation_contract_v2.json"
FINAL_PACKAGE_CONFIG="$ROOT/configs/eipm/n0/n0_v02_full_envelope_final_package_v1.json"
FINAL_EVALUATOR_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_full_envelope_final_v2_evaluator_contract_v1.json"
FINAL_GATE_REGISTRY="$ROOT/configs/eipm/n0/n0_v02_full_envelope_final_v2_gate_registry_v1.json"

for required in   "$SOURCE_CONFIG" "$CORPUS_RECEIPT" "$NATURAL_SOURCES" "$MIXTURE_CONTRACT"   "$TEACHER_REGISTRY" "$TEACHER_AUDIT" "$TOKENIZER/tokenizer.json"   "$CPU_ROOT/static_proof.json" "$CPU_ROOT/result.json" "$CPU_ROOT/tokenizer_stress.json"   "$CPU_ROOT/operator-evidence-alignment/rows.jsonl"   "$CPU_ROOT/operator-evidence-alignment/manifest.json"   "$CPU_ROOT/operator-evidence-alignment/static_audit.json"   "$CPU_ROOT/operator-evidence-alignment/token_alignment.json"   "$CPU_ROOT/semantic-operator-long-context/rows.jsonl"   "$CPU_ROOT/semantic-operator-long-context/manifest.json"   "$CPU_ROOT/semantic-operator-long-context/static_audit.json"   "$CPU_ROOT/semantic-operator-long-context/token_alignment.json"   "$CPU_ROOT/long-context/rows.jsonl"   "$CPU_ROOT/long-context/manifest.json"   "$CPU_ROOT/long-context/static_audit.json"   "$CPU_ROOT/long-context/token_boundary_alignment.json"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing P39PN prerequisite: $required" >&2
    exit 93
  fi
done

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

EXPECTED_FEWREL="$(python - "$NATURAL_SOURCES" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
sources=cfg.get("sources") or []
if len(sources)!=1 or sources[0].get("source_id")!="thunlp_fewrel_1_0":
    raise SystemExit("natural relation source registry drift")
print(sources[0]["revision"])
PY
)"
OBSERVED_FEWREL="$(git -C "$FEWREL_ROOT" rev-parse HEAD)"
if [[ "$OBSERVED_FEWREL" != "$EXPECTED_FEWREL" ]]; then
  echo "STOP: FewRel source drift HEAD=$OBSERVED_FEWREL expected=$EXPECTED_FEWREL" >&2
  exit 94
fi

python - "$HEAD" "$CPU_ROOT" <<'PY'
import json,sys
head,cpu_root=sys.argv[1:]
checks={
  "static_proof.json":"PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1",
  "result.json":"PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1",
  "tokenizer_stress.json":"PASS_N0_TOKENIZER_STRESS_V1",
  "operator-evidence-alignment/token_alignment.json":"PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1",
  "semantic-operator-long-context/token_alignment.json":"PASS_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1",
  "long-context/token_boundary_alignment.json":"PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1",
}
from pathlib import Path
root=Path(cpu_root)
for relative,status in checks.items():
    value=json.load(open(root/relative))
    assert value["status"]==status,(relative,value.get("status"))
    assert value["source_revision"]==head,(relative,value.get("source_revision"),head)
    assert value.get("final_results_observed") in (None,False)
    assert value.get("private_identity_data") in (None,False)
print("PASS_N0_P39PN_CPU_RECEIPT_INPUTS")
PY

STAGING="${MIXTURE_ROOT}.staging-${SLURM_JOB_ID:-$$}"
if [[ -e "$STAGING" ]]; then
  echo "STOP: preserve existing public-mixture staging evidence root: $STAGING" >&2
  exit 95
fi
mkdir -p   "$STAGING/runtime"   "$STAGING/semantic"   "$STAGING/semantic-long"   "$STAGING/behavioral"   "$STAGING/runtime-view"   "$STAGING/long-context"   "$STAGING/fewrel"   "$STAGING/final-v2"

# Carry forward the exact CPU-qualified artifacts. Byte-for-byte copies preserve
# the hashes that P40/P40A/P40B/P40C and the later trainer re-check.
cp "$CPU_ROOT/static_proof.json" "$STAGING/runtime/static_proof.json"
cp "$CPU_ROOT/result.json" "$STAGING/runtime/cpu_runtime.json"
cp "$CPU_ROOT/tokenizer_stress.json" "$STAGING/runtime/tokenizer_stress.json"
cp "$CPU_ROOT/operator-evidence-alignment/rows.jsonl" "$STAGING/semantic/rows.jsonl"
cp "$CPU_ROOT/operator-evidence-alignment/manifest.json" "$STAGING/semantic/manifest.json"
cp "$CPU_ROOT/operator-evidence-alignment/static_audit.json" "$STAGING/semantic/audit.json"
cp "$CPU_ROOT/operator-evidence-alignment/token_alignment.json" "$STAGING/semantic/token_alignment.json"
cp "$CPU_ROOT/semantic-operator-long-context/rows.jsonl" "$STAGING/semantic-long/rows.jsonl"
cp "$CPU_ROOT/semantic-operator-long-context/manifest.json" "$STAGING/semantic-long/manifest.json"
cp "$CPU_ROOT/semantic-operator-long-context/static_audit.json" "$STAGING/semantic-long/audit.json"
cp "$CPU_ROOT/semantic-operator-long-context/token_alignment.json" "$STAGING/semantic-long/token_alignment.json"
cp "$CPU_ROOT/long-context/rows.jsonl" "$STAGING/long-context/rows.jsonl"
cp "$CPU_ROOT/long-context/manifest.json" "$STAGING/long-context/manifest.json"
cp "$CPU_ROOT/long-context/static_audit.json" "$STAGING/long-context/audit.json"
cp "$CPU_ROOT/long-context/token_boundary_alignment.json" "$STAGING/long-context/token_boundary_alignment.json"

python "$ROOT/scripts/eipm/n0/build_n0_v02_full_envelope_behavioral_curriculum_v1.py"   --output "$STAGING/behavioral/rows.jsonl"   --manifest "$STAGING/behavioral/manifest.json"   --train-rows 280   --dev-rows 112
python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_behavioral_curriculum_v1.py"   --rows "$STAGING/behavioral/rows.jsonl"   --manifest "$STAGING/behavioral/manifest.json"   --contract "$BEHAVIORAL_CONTRACT"   --output "$STAGING/behavioral/audit.json"

python "$ROOT/scripts/eipm/n0/build_n0_v02_full_envelope_runtime_view_curriculum_v1.py"   --output "$STAGING/runtime-view/rows.jsonl"   --manifest "$STAGING/runtime-view/manifest.json"
python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_runtime_view_curriculum_v1.py"   --rows "$STAGING/runtime-view/rows.jsonl"   --manifest "$STAGING/runtime-view/manifest.json"   --contract "$RUNTIME_VIEW_CONTRACT"   --output "$STAGING/runtime-view/audit.json"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_shortcuts_v1.py"   --semantic-rows "$STAGING/semantic/rows.jsonl"   --semantic-manifest "$STAGING/semantic/manifest.json"   --behavioral-rows "$STAGING/behavioral/rows.jsonl"   --behavioral-manifest "$STAGING/behavioral/manifest.json"   --semantic-contract "$SEMANTIC_CONTRACT"   --behavioral-contract "$BEHAVIORAL_CONTRACT"   --final-contract "$FINAL_CONTRACT"   --output "$STAGING/shortcut_preflight.json"

python "$ROOT/scripts/eipm/n0/build_n0_v02_fewrel_natural_relation_curriculum_v1.py"   --fewrel-root "$FEWREL_ROOT"   --train-dev-rows-output "$STAGING/fewrel/train_dev_rows.jsonl"   --train-dev-bank-output "$STAGING/fewrel/train_dev_bank.json"   --final-rows-output "$STAGING/fewrel/final_rows.jsonl"   --final-bank-output "$STAGING/fewrel/final_bank.json"   --manifest-output "$STAGING/fewrel/manifest.json"
python "$ROOT/scripts/eipm/n0/audit_n0_v02_fewrel_natural_relation_curriculum_v1.py"   --train-dev-rows "$STAGING/fewrel/train_dev_rows.jsonl"   --train-dev-bank "$STAGING/fewrel/train_dev_bank.json"   --final-rows "$STAGING/fewrel/final_rows.jsonl"   --final-bank "$STAGING/fewrel/final_bank.json"   --manifest "$STAGING/fewrel/manifest.json"   --output "$STAGING/fewrel/audit.json"

python "$ROOT/scripts/eipm/n0/build_n0_v02_full_envelope_final_v2_package.py"   --behavioral-train-dev-rows "$STAGING/behavioral/rows.jsonl"   --fewrel-final-rows "$STAGING/fewrel/final_rows.jsonl"   --fewrel-final-bank "$STAGING/fewrel/final_bank.json"   --fewrel-manifest "$STAGING/fewrel/manifest.json"   --final-contract "$FINAL_CONTRACT"   --synthetic-output "$STAGING/final-v2/synthetic_final_rows.jsonl"   --semantic-final-output "$STAGING/final-v2/semantic_final_rows.jsonl"   --runtime-view-final-output "$STAGING/final-v2/runtime_view_final_rows.jsonl"   --long-context-final-output "$STAGING/final-v2/long_context_final_rows.jsonl"   --manifest-output "$STAGING/final-v2/package_manifest.json"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_envelope_final_v2_package.py"   --package-config "$FINAL_PACKAGE_CONFIG"   --evaluator-contract "$FINAL_EVALUATOR_CONTRACT"   --final-contract "$FINAL_CONTRACT"   --behavioral-train-dev-rows "$STAGING/behavioral/rows.jsonl"   --semantic-train-dev-rows "$STAGING/semantic/rows.jsonl"   --runtime-view-train-dev-rows "$STAGING/runtime-view/rows.jsonl"   --long-context-train-dev-rows "$STAGING/long-context/rows.jsonl"   --synthetic-final-rows "$STAGING/final-v2/synthetic_final_rows.jsonl"   --semantic-final-rows "$STAGING/final-v2/semantic_final_rows.jsonl"   --runtime-view-final-rows "$STAGING/final-v2/runtime_view_final_rows.jsonl"   --long-context-final-rows "$STAGING/final-v2/long_context_final_rows.jsonl"   --package-manifest "$STAGING/final-v2/package_manifest.json"   --fewrel-final-rows "$STAGING/fewrel/final_rows.jsonl"   --fewrel-final-bank "$STAGING/fewrel/final_bank.json"   --fewrel-manifest "$STAGING/fewrel/manifest.json"   --fewrel-audit "$STAGING/fewrel/audit.json"   --output "$STAGING/final-v2/audit.json"

python "$ROOT/scripts/eipm/n0/freeze_n0_v02_full_envelope_final_v2_package.py"   --source-revision "$HEAD"   --package-config "$FINAL_PACKAGE_CONFIG"   --evaluator-contract "$FINAL_EVALUATOR_CONTRACT"   --evaluator-implementation "$ROOT/scripts/eipm/n0/evaluate_n0_v02_full_envelope_final_v2.py"   --gate-registry "$FINAL_GATE_REGISTRY"   --opening-authorizer "$ROOT/scripts/eipm/n0/authorize_n0_v02_full_envelope_final_v2_opening.py"   --final-contract "$FINAL_CONTRACT"   --synthetic-final-rows "$STAGING/final-v2/synthetic_final_rows.jsonl"   --semantic-final-rows "$STAGING/final-v2/semantic_final_rows.jsonl"   --runtime-view-final-rows "$STAGING/final-v2/runtime_view_final_rows.jsonl"   --long-context-final-rows "$STAGING/final-v2/long_context_final_rows.jsonl"   --package-manifest "$STAGING/final-v2/package_manifest.json"   --fewrel-final-rows "$STAGING/fewrel/final_rows.jsonl"   --fewrel-final-bank "$STAGING/fewrel/final_bank.json"   --fewrel-manifest "$STAGING/fewrel/manifest.json"   --audit "$STAGING/final-v2/audit.json"   --output "$STAGING/final-v2/freeze_receipt.json"

python "$ROOT/scripts/eipm/n0/build_n0_v02_full_public_mixture_manifest_v1.py"   --contract "$MIXTURE_CONTRACT"   --source-revision "$HEAD"   --source-config "$SOURCE_CONFIG"   --corpus-receipt "$CORPUS_RECEIPT"   --teacher-registry "$TEACHER_REGISTRY"   --teacher-audit "$TEACHER_AUDIT"   --semantic-rows "$STAGING/semantic/rows.jsonl"   --semantic-manifest "$STAGING/semantic/manifest.json"   --semantic-audit "$STAGING/semantic/audit.json"   --semantic-long-rows "$STAGING/semantic-long/rows.jsonl"   --semantic-long-manifest "$STAGING/semantic-long/manifest.json"   --semantic-long-audit "$STAGING/semantic-long/audit.json"   --behavioral-rows "$STAGING/behavioral/rows.jsonl"   --behavioral-manifest "$STAGING/behavioral/manifest.json"   --behavioral-audit "$STAGING/behavioral/audit.json"   --runtime-view-rows "$STAGING/runtime-view/rows.jsonl"   --runtime-view-manifest "$STAGING/runtime-view/manifest.json"   --runtime-view-audit "$STAGING/runtime-view/audit.json"   --long-context-rows "$STAGING/long-context/rows.jsonl"   --long-context-manifest "$STAGING/long-context/manifest.json"   --long-context-audit "$STAGING/long-context/audit.json"   --fewrel-train-dev-rows "$STAGING/fewrel/train_dev_rows.jsonl"   --fewrel-train-dev-bank "$STAGING/fewrel/train_dev_bank.json"   --fewrel-manifest "$STAGING/fewrel/manifest.json"   --fewrel-audit "$STAGING/fewrel/audit.json"   --final-freeze-receipt "$STAGING/final-v2/freeze_receipt.json"   --output "$STAGING/full_public_mixture_manifest.json"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_full_public_mixture_manifest_v1.py"   --manifest "$STAGING/full_public_mixture_manifest.json"   --contract "$MIXTURE_CONTRACT"   --source-revision "$HEAD"   --source-config "$SOURCE_CONFIG"   --corpus-receipt "$CORPUS_RECEIPT"   --teacher-registry "$TEACHER_REGISTRY"   --teacher-audit "$TEACHER_AUDIT"   --semantic-rows "$STAGING/semantic/rows.jsonl"   --semantic-manifest "$STAGING/semantic/manifest.json"   --semantic-audit "$STAGING/semantic/audit.json"   --semantic-long-rows "$STAGING/semantic-long/rows.jsonl"   --semantic-long-manifest "$STAGING/semantic-long/manifest.json"   --semantic-long-audit "$STAGING/semantic-long/audit.json"   --behavioral-rows "$STAGING/behavioral/rows.jsonl"   --behavioral-manifest "$STAGING/behavioral/manifest.json"   --behavioral-audit "$STAGING/behavioral/audit.json"   --runtime-view-rows "$STAGING/runtime-view/rows.jsonl"   --runtime-view-manifest "$STAGING/runtime-view/manifest.json"   --runtime-view-audit "$STAGING/runtime-view/audit.json"   --long-context-rows "$STAGING/long-context/rows.jsonl"   --long-context-manifest "$STAGING/long-context/manifest.json"   --long-context-audit "$STAGING/long-context/audit.json"   --fewrel-train-dev-rows "$STAGING/fewrel/train_dev_rows.jsonl"   --fewrel-train-dev-bank "$STAGING/fewrel/train_dev_bank.json"   --fewrel-manifest "$STAGING/fewrel/manifest.json"   --fewrel-audit "$STAGING/fewrel/audit.json"   --final-freeze-receipt "$STAGING/final-v2/freeze_receipt.json"   --output "$STAGING/full_public_mixture_audit.json"

python - "$HEAD" "$STAGING" <<'PY'
import json,sys
from pathlib import Path
head,root=sys.argv[1:]
root=Path(root)
checks={
  "behavioral/audit.json":"PASS_N0_FULL_ENVELOPE_BEHAVIORAL_CURRICULUM_AUDIT_V1",
  "runtime-view/audit.json":"PASS_N0_FULL_ENVELOPE_RUNTIME_VIEW_CURRICULUM_AUDIT_V1",
  "fewrel/audit.json":"PASS_FEWREL_NATURAL_RELATION_AUDIT_V3",
  "shortcut_preflight.json":"PASS_N0_FULL_ENVELOPE_SHORTCUT_PREFLIGHT_V1",
  "final-v2/audit.json":"PASS_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_AUDIT_V1",
  "final-v2/freeze_receipt.json":"FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT",
  "full_public_mixture_audit.json":"PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1",
}
for relative,status in checks.items():
    value=json.load(open(root/relative))
    assert value["status"]==status,(relative,value.get("status"))
    if "source_revision" in value:
        assert value["source_revision"]==head,(relative,value["source_revision"],head)
    assert value.get("final_results_observed") in (None,False),(relative,"FINAL observed")
    assert value.get("private_identity_data") in (None,False),(relative,"private identity")
manifest=json.load(open(root/"full_public_mixture_manifest.json"))
assert manifest["source_revision"]==head
assert manifest["final_results_observed"] is False
assert manifest["final_rows_in_training"]==0
assert manifest["final_rows_in_model_selection"]==0
assert manifest["private_identity_data"] is False
assert manifest["gradient"] is False
assert manifest["optimizer"] is False
assert manifest["gpu_training_authorized"] is False
assert manifest["n0_complete"] is False
freeze=json.load(open(root/"final-v2/freeze_receipt.json"))
assert freeze["results_observed"] is False
assert freeze["training_authorized"] is False
assert freeze["model_selection_authorized"] is False
assert freeze["final_opening_authorized"] is False
print("PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1")
PY

# Publish only after the entire staged materialization/audit succeeds. Failed
# staging roots are intentionally preserved for diagnosis instead of deleted.
mv "$STAGING" "$MIXTURE_ROOT"

echo "===== N0 FULL PUBLIC MIXTURE MATERIALIZATION COMPLETE ====="
date -Is
echo "source_revision=$HEAD"
echo "mixture_root=$MIXTURE_ROOT"
echo "manifest=$MIXTURE_ROOT/full_public_mixture_manifest.json"
echo "audit=$MIXTURE_ROOT/full_public_mixture_audit.json"
echo "final_freeze=$MIXTURE_ROOT/final-v2/freeze_receipt.json"
echo "static_proof=$MIXTURE_ROOT/runtime/static_proof.json"
echo "cpu_runtime=$MIXTURE_ROOT/runtime/cpu_runtime.json"
echo "optimizer=false"
echo "gradient=false"
echo "training=false"
echo "final_opening=false"
