#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"

SOURCE_CACHE="$WORKDIR/query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt"
OUTROOT="$WORKDIR/qsre-t1-executor-v0.1/pretraining-v0.2"
CURRICULUM="$OUTROOT/qsre_t1_curriculum_v0.2.jsonl"
CURRICULUM_RECEIPT="$OUTROOT/qsre_t1_curriculum_v0.2.receipt.json"
PREPARED="$OUTROOT/qsre_t1_real_cache_v0.1.pt"
PREP_RECEIPT="$OUTROOT/qsre_t1_real_cache_v0.1.receipt.json"

CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t1_curriculum_contract_v0_2.json"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.47.json"
BUILD="$ROOT/scripts/eipm/n0/build_n0_v02_qsre_t1_curriculum_v0_2.py"
EVAL="$ROOT/scripts/eipm/n0/evaluate_n0_v02_qsre_t1_curriculum_v0_2.py"
PREP="$ROOT/scripts/eipm/n0/prepare_n0_v02_qsre_t1_real_cache_v0_1.py"

EXPECTED_SOURCE_SHA="5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
EXPECTED_CURRICULUM_SHA="155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

for required in   "$SOURCE_CACHE" "$CONTRACT" "$STATE" "$BUILD" "$EVAL" "$PREP"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: T1 v0.2 pretraining directory already exists: $OUTROOT" >&2
  echo "Preserve it as evidence; do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.47":
    raise SystemExit("v0.47 state drift")
p=s["execution_policy"]
if p["real_cache_preparation_authorized"] is not True:
    raise SystemExit("real-cache preparation not authorized")
for key in (
    "t1_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
):
    if p[key] is not False:
        raise SystemExit(f"training boundary drift: {key}")
print("v0_47_t1_real_cache_preparation_gate=PASS")
PY

SOURCE_SHA="$(sha256sum "$SOURCE_CACHE" | awk '{print $1}')"
echo "source_cache_sha256=$SOURCE_SHA"
test "$SOURCE_SHA" = "$EXPECTED_SOURCE_SHA" || {
  echo "STOP: source cache hash drift" >&2
  exit 5
}

mkdir -p "$OUTROOT"

python "$BUILD"   --contract "$CONTRACT"   --output "$CURRICULUM"

python "$EVAL"   --contract "$CONTRACT"   --curriculum "$CURRICULUM"   --output "$CURRICULUM_RECEIPT"

CURRICULUM_SHA="$(sha256sum "$CURRICULUM" | awk '{print $1}')"
echo "curriculum_sha256=$CURRICULUM_SHA"
test "$CURRICULUM_SHA" = "$EXPECTED_CURRICULUM_SHA" || {
  echo "STOP: curriculum hash drift" >&2
  exit 6
}

python "$PREP"   --curriculum "$CURRICULUM"   --source-cache "$SOURCE_CACHE"   --output "$PREPARED"   --receipt "$PREP_RECEIPT"   --expected-source-cache-sha256 "$EXPECTED_SOURCE_SHA"

echo
echo "===== QSRE T1 PREPARATION HASHES ====="
sha256sum   "$CURRICULUM"   "$CURRICULUM_RECEIPT"   "$PREPARED"   "$PREP_RECEIPT"

echo
echo "===== QSRE T1 PREPARATION RECEIPT ====="
cat "$PREP_RECEIPT"
