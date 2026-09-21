#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"

cd "$ROOT"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: P2 localization source revision drift HEAD=$HEAD expected=$EXPECTED" >&2
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
RECOVERY="$WORKDIR/qsre-production-core-v1-stop-tail-recovery-v1"
OUTROOT="$WORKDIR/qsre-production-p2-failure-localization-v1"
RESULT="$OUTROOT/result.json"

PLAN="$ROOT/configs/eipm/n0/n0_v02_qsre_production_training_plan_v1.json"
PREPARED="$SOURCE/p0/qsre_production_cache_v1.pt"
SCHEMA_CACHE="$SOURCE/p0/qsre_production_schema_v1.pt"
P1_RESULT="$RECOVERY/p1/result.json"
P1_ROOT="$RECOVERY/p1"
P2_RESULT="$RECOVERY/p2/result.json"
P2_ROOT="$RECOVERY/p2"
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_qsre_production_p2_failure_v1.py"

for required in   "$PLAN" "$PREPARED" "$SCHEMA_CACHE"   "$P1_RESULT" "$P2_RESULT" "$DIAGNOSTIC"
do
  [[ -f "$required" ]] || {
    echo "STOP: missing required P2 diagnostic input: $required" >&2
    exit 92
  }
done

[[ -d "$P1_ROOT" ]] || {
  echo "STOP: P1 root missing: $P1_ROOT" >&2
  exit 93
}
[[ -d "$P2_ROOT" ]] || {
  echo "STOP: P2 root missing: $P2_ROOT" >&2
  exit 94
}
if [[ -e "$OUTROOT" ]]; then
  echo "STOP: P2 localization evidence already exists; preserve it: $OUTROOT" >&2
  exit 95
fi

python - "$P1_RESULT" "$P2_RESULT" <<'PY'
import json,sys
p1=json.load(open(sys.argv[1]))
p2=json.load(open(sys.argv[2]))
assert p1["status"]=="PASS_QSRE_PRODUCTION_P1_EXECUTOR", p1["status"]
assert p1["selected"]["step"]==50, p1["selected"]
assert p1["selected"]["checkpoint_sha256"]=="91f2c78dcc35967064189af4a7110cdee83e5652fb037d3d88f58500ac3aaab9"
assert p2["status"]=="FAIL_QSRE_PRODUCTION_P2_OPERATOR", p2["status"]
assert p2["selected"] is None
assert p2["best_observed"]["step"]==1600, p2["best_observed"]
assert p2["best_observed"]["checkpoint_sha256"]=="16dd2681a18e9b5e371232c9206ca4a3b4d0ace8e5ec47e7ee492e9ab25364a0"
assert p2["operator_gradient"] is True
assert p2["p3_authorized"] is False
print("PASS_PRESERVED_P2_V1_FAILURE_LINEAGE")
PY

mkdir "$OUTROOT"

echo "===== ZERO-GRADIENT PRODUCTION P2 FAILURE LOCALIZATION ====="
date -Is
echo "source_revision=$HEAD"
echo "failed_model_revision=dba3d4b100f081b0c09fe63ae002205244af5b44"
echo "failed_magnolia_job=575956"
echo "device=cpu"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "test_open=false"
echo "frozen_final_open=false"
echo "private_identity_gradient=false"

python "$DIAGNOSTIC"   --plan "$PLAN"   --prepared-cache "$PREPARED"   --schema-cache "$SCHEMA_CACHE"   --p1-result "$P1_RESULT"   --p1-root "$P1_ROOT"   --p2-result "$P2_RESULT"   --p2-root "$P2_ROOT"   --output "$RESULT"   --batch-size 8

echo
echo "===== P2 FAILURE LOCALIZATION HASH ====="
sha256sum "$RESULT"

echo
echo "===== P2 FAILURE LOCALIZATION SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_ZERO_GRADIENT_P2_FAILURE_LOCALIZATION"
assert r["optimizer_created"] is False
assert r["gradient_performed"] is False
assert r["parameters_mutated"] is False
assert r["test_split_opened"] is False
assert r["frozen_final_opened"] is False
assert r["private_identity_gradient"] is False

print("status="+r["status"])
for view in ("0","1"):
    v=r["views"][view]
    print(f"view_{view}_identity_exact={v['relation_identity_exact_ignoring_activity']}")
    print(f"view_{view}_active_mask_exact={v['active_mask_exact']}")
    print(f"view_{view}_full_program_exact={v['full_program_exact']}")
    print(f"view_{view}_identity_correct_activity_wrong={v['identity_correct_but_activity_wrong_rate']}")
    print(f"view_{view}_open_schema_identity={v['open_schema_relation_identity_exact']}")
    print(f"view_{view}_reverse_match={v['multi_step_reversed_target_match_rate']}")
    print(f"view_{view}_repeated_relation_collapse={v['multi_step_repeated_relation_collapse_rate']}")
    print(f"view_{view}_unknown_terminal={json.dumps(v['unknown_terminal'],sort_keys=True)}")
print("alignment="+json.dumps(r["query_schema_alignment"],sort_keys=True))
print("optimizer_created=false")
print("gradient_performed=false")
PY

echo "===== ZERO-GRADIENT PRODUCTION P2 FAILURE LOCALIZATION COMPLETE ====="
date -Is
