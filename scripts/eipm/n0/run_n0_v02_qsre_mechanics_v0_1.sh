#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$PWD}}"
FIXTURES="$ROOT/evaluation/eipm/n0/n0_v02_qsre_mechanics_v0_1.jsonl"
WORKLOAD="$ROOT/configs/eipm/n0/n0_v02_qsre_workload_envelope_v0_1.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_static_mechanics_contract_v0_1.json"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.42.json"
EVALUATOR="$ROOT/scripts/eipm/n0/evaluate_n0_v02_qsre_mechanics_v0_1.py"
OUTPUT="${ALICE_N0_QSRE_MECHANICS_RECEIPT:-$ROOT/.tmp/n0_v02_qsre_mechanics_v0_1.result.json}"

cd "$ROOT"

for required in "$FIXTURES" "$WORKLOAD" "$CONTRACT" "$STATE" "$EVALUATOR"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 2; }
done

python -m py_compile "$EVALUATOR"
mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"

echo "===== QSRE CPU/NO-GRADIENT REFERENCE MECHANICS ====="
echo "learned_parameters=false"
echo "optimizer=false"
echo "gradient=false"
echo "gpu=false"
echo "test_open=false"
echo "challenge_open=false"

python "$EVALUATOR"   --fixtures "$FIXTURES"   --workload-contract "$WORKLOAD"   --mechanics-contract "$CONTRACT"   --state "$STATE"   --output "$OUTPUT"

echo
echo "===== QSRE MECHANICS RECEIPT ====="
cat "$OUTPUT"
