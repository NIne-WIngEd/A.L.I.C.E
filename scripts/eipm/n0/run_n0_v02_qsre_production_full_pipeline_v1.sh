#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
cd "$ROOT"

HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: production source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no
  exit 91
fi
if ! python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PY
then
  echo "STOP: one visible CUDA device is required" >&2
  exit 92
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

PIPE="$WORKDIR/qsre-production-core-v1"
if [[ -e "$PIPE" ]]; then
  echo "STOP: governed production output already exists; preserve it: $PIPE" >&2
  exit 93
fi
mkdir -p "$PIPE/p0" "$PIPE/final-frozen"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_production_training_plan_v1.json"
PROD_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_production_relation_schema_v1.json"
FINAL_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_final_self_validation_relation_schema_v1.json"
FINAL_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_final_self_validation_contract_v1.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"

TOKENIZER="$WORKDIR/tokenizer-v0.2.1"
SEMANTIC_DIR="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
SEMANTIC="$SEMANTIC_DIR/alice_n0_v02.safetensors"
STRUCTURED="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
SPECIALIST="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080"
ADAPTER="$SPECIALIST/evidence_view_adapter.safetensors"
GRAPH="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
FUSION="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"

for required in   "$PLAN" "$PROD_SCHEMA" "$FINAL_SCHEMA" "$FINAL_CONTRACT"   "$SEMANTIC_CONFIG" "$STRUCTURED_CONFIG" "$FUSION_CONFIG" "$FUSION_RATIFICATION" "$LATENT_CONFIG"   "$TOKENIZER/tokenizer.json" "$SEMANTIC" "$STRUCTURED/structured_state.safetensors"   "$ADAPTER" "$GRAPH" "$FUSION/cross_context_fusion.safetensors" "$LATENT"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing governed prerequisite: $required" >&2
    exit 94
  fi
done

SEMANTIC_SHA="$(sha256sum "$SEMANTIC" | awk '{print $1}')"
if [[ "$SEMANTIC_SHA" != "6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43" ]]; then
  echo "STOP: ratified semantic checkpoint hash drift: $SEMANTIC_SHA" >&2
  exit 95
fi
if [[ "$(sha256sum "$FUSION/cross_context_fusion.safetensors" | awk '{print $1}')" != "4d51494beb788f74ddc03590da05ea00f0e36574294649c2cd5d438f9472e577" ]]; then
  echo "STOP: ratified fusion checkpoint hash drift" >&2
  exit 96
fi
if [[ "$(sha256sum "$LATENT" | awk '{print $1}')" != "503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4" ]]; then
  echo "STOP: preselected latent candidate hash drift" >&2
  exit 97
fi

echo "===== N0 PRODUCTION CORE V1 SINGLE GOVERNED PIPELINE ====="
date -Is
echo "source_revision=$HEAD"
echo "gpu_runs_authorized=one_pipeline_job"
echo "automatic_rerun=false"
echo "automatic_hotfix=false"
echo "learning_rate_search=false"
echo "step_search=false"
echo "width_search=false"
echo "private_identity_gradient=false"
echo "final_validator=alice_native_self_validation_harness"

P0="$PIPE/p0"
PROD_CURRICULUM="$P0/qsre_production_curriculum_v1.jsonl"
PROD_SCHEMA_CACHE="$P0/qsre_production_schema_v1.pt"
PROD_SCHEMA_RECEIPT="$P0/qsre_production_schema_v1.receipt.json"
PROD_CACHE="$P0/qsre_production_cache_v1.pt"
PROD_CACHE_RECEIPT="$P0/qsre_production_cache_v1.receipt.json"
P0_RECEIPT="$P0/p0_runtime_receipt.json"

FINAL="$PIPE/final-frozen"
FINAL_GENERAL="$FINAL/general.jsonl"
FINAL_RELATIONAL="$FINAL/relational.jsonl"
FINAL_MANIFEST="$FINAL/manifest.json"
FINAL_SCHEMA_CACHE="$FINAL/final_schema_v1.pt"
FINAL_SCHEMA_RECEIPT="$FINAL/final_schema_v1.receipt.json"
FINAL_CACHE="$FINAL/final_relational_cache_v1.pt"
FINAL_CACHE_RECEIPT="$FINAL/final_relational_cache_v1.receipt.json"
FINAL_FREEZE="$FINAL/freeze_receipt.json"

echo "===== P0A: FREEZE PRODUCTION + FINAL CORPORA ====="
python "$ROOT/scripts/eipm/n0/build_n0_v02_qsre_production_curriculum_v1.py"   --output "$PROD_CURRICULUM"   --train-groups-per-family 20   --dev-groups-per-family 8   --open-schema-groups 32

python "$ROOT/scripts/eipm/n0/build_n0_v02_final_self_validation_v1.py"   --general-output "$FINAL_GENERAL"   --relational-output "$FINAL_RELATIONAL"   --manifest "$FINAL_MANIFEST"

echo "===== P0B: MATERIALIZE DYNAMIC SCHEMAS ====="
python "$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_production_schema_v1.py"   --schema "$PROD_SCHEMA"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$PROD_SCHEMA_CACHE"   --receipt "$PROD_SCHEMA_RECEIPT"   --device cuda

python "$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_production_schema_v1.py"   --schema "$FINAL_SCHEMA"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$FINAL_SCHEMA_CACHE"   --receipt "$FINAL_SCHEMA_RECEIPT"   --device cuda

echo "===== P0C: MATERIALIZE FROZEN SEMANTIC CACHES ====="
python "$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_production_cache_v1.py"   --curriculum "$PROD_CURRICULUM"   --schema-cache "$PROD_SCHEMA_CACHE"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$PROD_CACHE"   --receipt "$PROD_CACHE_RECEIPT"   --device cuda   --batch-size 32

python "$ROOT/scripts/eipm/n0/prepare_n0_v02_final_self_validation_cache_v1.py"   --relational-corpus "$FINAL_RELATIONAL"   --manifest "$FINAL_MANIFEST"   --schema-cache "$FINAL_SCHEMA_CACHE"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --output "$FINAL_CACHE"   --receipt "$FINAL_CACHE_RECEIPT"   --device cuda   --batch-size 32

echo "===== P0D: FREEZE FINAL VALIDATION BEFORE ANY PRODUCTION GRADIENT ====="
python "$ROOT/scripts/eipm/n0/freeze_n0_v02_final_self_validation_v1.py"   --contract "$FINAL_CONTRACT"   --manifest "$FINAL_MANIFEST"   --general-corpus "$FINAL_GENERAL"   --relational-corpus "$FINAL_RELATIONAL"   --final-schema-cache "$FINAL_SCHEMA_CACHE"   --relational-cache "$FINAL_CACHE"   --output "$FINAL_FREEZE"

echo "===== P0E: REAL-ARTIFACT PRODUCTION RUNTIME ====="
python "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_production_p0_v1.py"   --plan "$PLAN"   --curriculum "$PROD_CURRICULUM"   --schema-cache "$PROD_SCHEMA_CACHE"   --prepared-cache "$PROD_CACHE"   --output "$P0_RECEIPT"

echo "===== P1: DYNAMIC SCHEMA ENCODER + RELATIONAL EXECUTOR ====="
P1="$PIPE/p1"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p1_v1.py"   --plan "$PLAN"   --p0-receipt "$P0_RECEIPT"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --output-dir "$P1"

echo "===== P2: NATURAL-LANGUAGE DYNAMIC OPERATOR ====="
P2="$PIPE/p2"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p2_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --output-dir "$P2"

echo "===== P3: ADAPTIVE SUPPORT + LEARNED PATH FOCUS ====="
P3="$PIPE/p3"
python "$ROOT/scripts/eipm/n0/train_n0_v02_qsre_production_p3_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --output-dir "$P3"

echo "===== P4: FULLY PREDICTED QSRE; ZERO GRADIENT ====="
P4="$PIPE/p4"
mkdir -p "$P4"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v1.py"   --plan "$PLAN"   --prepared-cache "$PROD_CACHE"   --schema-cache "$PROD_SCHEMA_CACHE"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --output "$P4/result.json"

echo "===== P5 + FINAL: NATIVE FULL N0 OBJECTIVE SELF-VALIDATION ====="
FINAL_RESULT="$PIPE/final-self-validation-result.json"
python "$ROOT/scripts/eipm/n0/eval_n0_v02_final_self_validation_v1.py"   --contract "$FINAL_CONTRACT"   --manifest "$FINAL_MANIFEST"   --freeze-receipt "$FINAL_FREEZE"   --general-corpus "$FINAL_GENERAL"   --relational-corpus "$FINAL_RELATIONAL"   --relational-cache "$FINAL_CACHE"   --final-schema-cache "$FINAL_SCHEMA_CACHE"   --plan "$PLAN"   --p1-result "$P1/result.json"   --p1-root "$P1"   --p2-result "$P2/result.json"   --p2-root "$P2"   --p3-result "$P3/result.json"   --p3-root "$P3"   --p4-result "$P4/result.json"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED"   --evidence-adapter "$ADAPTER"   --evidence-graph "$GRAPH"   --fusion-checkpoint "$FUSION"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --latent-candidate "$LATENT"   --latent-config "$LATENT_CONFIG"   --output "$FINAL_RESULT"

python - "$FINAL_RESULT" <<'PY'
import json,sys
result=json.load(open(sys.argv[1]))
if result.get("status")!="PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE":
    raise SystemExit("N0 final native objective gate did not pass")
if result.get("n0_complete") is not True or result.get("n1_authorized") is not True:
    raise SystemExit("N0 did not authorize N1")
print("===== N0 FULL-SCALE OBJECTIVE REACHED =====")
print("n0_complete=true")
print("n1_authorized=true")
print("external_validator=false")
print("private_identity_gradient=false")
PY

echo "===== N0 PRODUCTION CORE V1 PIPELINE COMPLETE ====="
date -Is
echo "final_result=$FINAL_RESULT"
