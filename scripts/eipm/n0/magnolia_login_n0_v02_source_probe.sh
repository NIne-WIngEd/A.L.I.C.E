#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
PLAN="$ROOT/configs/eipm/n0/public_corpus_v0.2.plan.json"
OUT="${N0_V02_SOURCE_PROBE_OUT:-$WORKDIR/source-probe/source-probe.json}"

mkdir -p "$(dirname "$OUT")" "$WORKDIR/hf-cache-probe"

export RAYAN_UDOCKER_NVIDIA=0
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache-probe}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2 SOURCE SCHEMA PROBE ====="
date -Is
echo "repo_root=$ROOT"
echo "plan=$PLAN"
echo "output=$OUT"
echo "sample_rows=${N0_V02_SOURCE_PROBE_ROWS:-64}"
echo "gpu_requested=false"
echo "training_authorized=false"
echo "private_identity_gradient=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python scripts/eipm/n0/probe_n0_v02_sources.py \
    --plan configs/eipm/n0/public_corpus_v0.2.plan.json \
    --output '$OUT' \
    --sample-rows '${N0_V02_SOURCE_PROBE_ROWS:-64}'"

echo "===== N0 V0.2 SOURCE SCHEMA PROBE COMPLETE ====="
date -Is
