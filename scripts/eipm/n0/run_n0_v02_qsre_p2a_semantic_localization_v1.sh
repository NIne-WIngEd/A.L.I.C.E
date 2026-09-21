#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: semantic-localization source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no
  exit 91
fi

python - <<'PY'
import torch
raise SystemExit(0 if not torch.cuda.is_available() else 1)
PY

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

SOURCE="$WORKDIR/qsre-production-core-v1"
P0="$SOURCE/p0"
SOURCE_FAIL="$WORKDIR/qsre-n0-frozen-semantic-authority-v3/p2a/result.json"
RUN_ROOT="$WORKDIR/qsre-n0-p2a-semantic-localization-v1"
RESULT="$RUN_ROOT/result.json"

META="$ROOT/configs/eipm/n0/n0_v02_qsre_closure_schema_meta_v2.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
PROD_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_production_relation_schema_v1.json"
PROD_CURRICULUM="$P0/qsre_production_curriculum_v1.jsonl"
PROD_CACHE="$P0/qsre_production_cache_v1.pt"
TOKENIZER="$WORKDIR/tokenizer-v0.2.1"
SEMANTIC="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"

for required in   "$SOURCE_FAIL" "$META" "$SEMANTIC_CONFIG" "$PROD_SCHEMA"   "$PROD_CURRICULUM" "$PROD_CACHE" "$TOKENIZER/tokenizer.json" "$SEMANTIC"
do
  if [[ ! -e "$required" ]]; then
    echo "STOP: missing semantic-localization prerequisite: $required" >&2
    exit 92
  fi
done

if [[ -e "$RUN_ROOT" ]]; then
  echo "STOP: semantic-localization evidence root already exists; preserve it: $RUN_ROOT" >&2
  exit 93
fi

python -m py_compile "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_p2a_semantic_localization_v1.py"

python - "$SOURCE_FAIL" <<'PY'
import json, math, sys
p=sys.argv[1]
r=json.load(open(p))
assert r["status"]=="FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY"
assert r["p2_authorized"] is False
expected={
  "production_core_single_relation_top1":0.4068181812763214,
  "auxiliary_seen_relation_top1":0.2708333432674408,
  "auxiliary_holdout_relation_top1":0.2604166567325592,
  "heldout_factor_macro_accuracy":0.36666667064030967,
}
for k,v in expected.items():
    assert math.isclose(float(r["metrics"][k]),v,rel_tol=0.0,abs_tol=1e-12),(k,r["metrics"][k])
print("PASS_JOB575986_SOURCE_FAILURE_LINEAGE")
PY

SEMANTIC_SHA="$(sha256sum "$SEMANTIC" | awk '{print $1}')"
if [[ "$SEMANTIC_SHA" != "6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43" ]]; then
  echo "STOP: semantic checkpoint SHA drift: $SEMANTIC_SHA" >&2
  exit 94
fi

# Only after every source/read-only preflight has passed do we create the
# diagnostic evidence root.
mkdir -p "$RUN_ROOT"

echo "===== N0 JOB575986 P2A SEMANTIC LOCALIZATION V1 ====="
date -Is
echo "source_revision=$HEAD"
echo "source_failure_job=575986"
echo "gpu=false"
echo "gradient=false"
echo "optimizer=false"
echo "model_training=false"
echo "threshold_change=false"
echo "automatic_rerun=false"
echo "production_p2_authorized=false"
echo "semantic_backbone_retraining_authorized=false"

python "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_p2a_semantic_localization_v1.py"   --source-failure-result "$SOURCE_FAIL"   --meta-config "$META"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC"   --tokenizer-dir "$TOKENIZER"   --production-cache "$PROD_CACHE"   --production-curriculum "$PROD_CURRICULUM"   --production-relation-schema "$PROD_SCHEMA"   --output "$RESULT"   --batch-size 8

python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["schema"]=="alice.eipm.n0.qsre-p2a-semantic-localization.v1"
assert r["status"]=="COMPLETE_ZERO_GRADIENT_SEMANTIC_LOCALIZATION"
assert r["gradient"] is False
assert r["optimizer"] is False
assert r["model_training"] is False
assert r["gpu"] is False
assert r["production_p2_authorized"] is False
assert r["semantic_backbone_retraining_authorized"] is False
assert r["reconstruction"]["matches_job575986_within_1e-6"] is True
print("PASS_N0_P2A_SEMANTIC_LOCALIZATION_V1")
print("current="+json.dumps(r["reconstruction"]["metrics"],sort_keys=True))
print("teacher_native="+json.dumps(r["teacher_native_prompt_geometry"],sort_keys=True))
print("native_plus_best_layer_oracle="+json.dumps(r["teacher_native_plus_best_layer_oracle"]["metrics"],sort_keys=True))
PY

echo "===== N0 P2A SEMANTIC LOCALIZATION V1 COMPLETE ====="
date -Is
echo "result=$RESULT"
