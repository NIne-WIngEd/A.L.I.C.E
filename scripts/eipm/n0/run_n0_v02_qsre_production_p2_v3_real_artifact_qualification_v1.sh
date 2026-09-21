#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: P2 v3 qualification source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repo changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 91
fi

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=0

SOURCE="$WORKDIR/qsre-production-core-v1"
FAILED_RECOVERY="$WORKDIR/qsre-production-core-v1-stop-tail-recovery-v1"
LOCALIZATION="$WORKDIR/qsre-production-p2-failure-localization-v1/result.json"
OUTROOT="$WORKDIR/qsre-production-p2-v3-real-artifact-qualification-v1"
RESULT="$OUTROOT/result.json"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_production_training_plan_v3.json"
PREPARED="$SOURCE/p0/qsre_production_cache_v1.pt"
SCHEMA_CACHE="$SOURCE/p0/qsre_production_schema_v1.pt"
P1_RESULT="$FAILED_RECOVERY/p1/result.json"
P1_ROOT="$FAILED_RECOVERY/p1"
FAILED_P2_RESULT="$FAILED_RECOVERY/p2/result.json"
QUALIFIER="$ROOT/scripts/eipm/n0/qualify_n0_v02_qsre_production_p2_v3_real_artifact_v1.py"

for required in   "$PLAN" "$PREPARED" "$SCHEMA_CACHE" "$P1_RESULT"   "$FAILED_P2_RESULT" "$LOCALIZATION" "$QUALIFIER"
do
  [[ -e "$required" ]] || {
    echo "STOP: missing governed v3 qualification prerequisite: $required" >&2
    exit 92
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "STOP: v3 real-artifact qualification already exists; preserve it: $OUTROOT" >&2
  exit 93
fi
mkdir "$OUTROOT"

echo "===== PRODUCTION P2 V3 ZERO-GRADIENT REAL-ARTIFACT QUALIFICATION ====="
date -Is
echo "source_revision=$HEAD"
echo "device=cpu"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "test_open=false"
echo "frozen_final_open=false"
echo "private_identity_gradient=false"

python "$QUALIFIER"   --plan "$PLAN"   --prepared-cache "$PREPARED"   --schema-cache "$SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --failed-p2-result "$FAILED_P2_RESULT"   --localization-result "$LOCALIZATION"   --output "$RESULT"

echo
echo "===== P2 V3 QUALIFICATION HASH ====="
sha256sum "$RESULT"

echo
echo "===== P2 V3 QUALIFICATION SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_QSRE_PRODUCTION_P2_V3_ZERO_GRADIENT_REAL_ARTIFACT_RUNTIME"
assert r["optimizer_created"] is False
assert r["gradient_performed"] is False
assert r["gpu_required"] is False
assert r["test_split_opened"] is False
assert r["frozen_final_opened"] is False
assert r["private_identity_gradient"] is False
assert r["p2_v3_gpu_training_authorized_by_this_receipt"] is False
print("status="+r["status"])
print("runtime_relation_count="+str(r["runtime_relation_count"]))
print("runtime_steps="+str(r["runtime_steps"]))
print("view_0="+json.dumps(r["views"]["view_0"],sort_keys=True))
print("view_1="+json.dumps(r["views"]["view_1"],sort_keys=True))
print("optimizer_created=false")
print("gradient_performed=false")
PY

echo "===== PRODUCTION P2 V3 ZERO-GRADIENT REAL-ARTIFACT QUALIFICATION COMPLETE ====="
date -Is
