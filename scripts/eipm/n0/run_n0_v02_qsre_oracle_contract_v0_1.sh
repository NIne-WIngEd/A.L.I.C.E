#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-${ALICE_REPO_ROOT:-$PWD}}"
FIXTURES="$ROOT/evaluation/eipm/n0/n0_v02_qsre_oracle_contract_v0_1.jsonl"
EVALUATOR="$ROOT/scripts/eipm/n0/evaluate_n0_v02_qsre_oracle_contract_v0_1.py"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.40.json"
PROOF="$ROOT/configs/eipm/n0/n0_v02_relational_execution_proof_obligations_v0_1.json"
DIAG="$ROOT/configs/eipm/n0/n0_v02_qsre_preimplementation_diagnostics_v0_1.json"
OUTPUT="${ALICE_N0_QSRE_ORACLE_RECEIPT:-$ROOT/.tmp/n0_v02_qsre_oracle_contract_v0_1.result.json}"

cd "$ROOT"

for required in "$FIXTURES" "$EVALUATOR" "$STATE" "$PROOF" "$DIAG"; do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

python - "$STATE" "$PROOF" "$DIAG" <<'PY'
import json,sys

state=json.load(open(sys.argv[1]))
proof=json.load(open(sys.argv[2]))
diag=json.load(open(sys.argv[3]))

if state.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.40":
    raise SystemExit("v0.40 state drift")
if state["selected_family"]["name"]!="QSRE_QUERY_CONDITIONED_SPARSE_RELATIONAL_EXECUTOR":
    raise SystemExit("selected family drift")
if state["selected_family"]["implementation_authorized"] is not False:
    raise SystemExit("trainable implementation unexpectedly authorized")
if proof.get("schema")!="alice.eipm.n0.relational-execution-proof-obligations.v0.1":
    raise SystemExit("proof contract drift")
if [x["id"] for x in proof["obligations"]] != [f"P{i}" for i in range(17)]:
    raise SystemExit("proof obligations P0-P16 incomplete or reordered")
if diag.get("schema")!="alice.eipm.n0.qsre-preimplementation-diagnostic-contract.v0.1":
    raise SystemExit("diagnostic contract drift")
if [x["id"] for x in diag["diagnostics"]] != [f"D{i}" for i in range(1,6)]:
    raise SystemExit("diagnostics D1-D5 incomplete or reordered")
for obj,name in ((state["execution_policy"],"state"),(diag,"diagnostic")):
    for key in (
        "optimizer_authorized",
        "gradient_authorized",
        "gpu_training_authorized",
    ):
        if obj[key] is not False:
            raise SystemExit(f"{name} training boundary drift: {key}")
print("v0_40_qsre_deterministic_contract_gate=PASS")
PY

python -m py_compile "$EVALUATOR"

mkdir -p "$(dirname "$OUTPUT")"
rm -f "$OUTPUT"

python "$EVALUATOR"   --fixtures "$FIXTURES"   --output "$OUTPUT"

echo
echo "===== QSRE ORACLE CONTRACT RECEIPT ====="
cat "$OUTPUT"
