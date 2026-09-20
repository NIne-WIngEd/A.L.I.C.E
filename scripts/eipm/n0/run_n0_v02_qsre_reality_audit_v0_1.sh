#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"
STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.56.json"
AUDIT_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_reality_audit_contract_v0_1.json"
T2_TRAINING_CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_1.json"

T2_ROOT="$WORKDIR/qsre-t2-operator-v0.1"
T2_CURRICULUM="$T2_ROOT/qsre_t2_operator_curriculum_v0.1.jsonl"
T2_PREPARED="$T2_ROOT/qsre_t2_tokenized_preparation_v0.1.pt"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
T1_CHECKPOINT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1/step-00000050/qsre_t1_executor.pt"

OUTROOT="$WORKDIR/qsre-reality-audit-v0.1"
REALITY="$OUTROOT/qsre_operator_reality_corpus_v0.1.jsonl"
RUNTIME="$OUTROOT/runtime_parity_v0.1.json"
SHORTCUT="$OUTROOT/t2_shortcut_audit_v0.1.json"
SEMANTIC="$OUTROOT/frozen_semantic_reality_probe_v0.1.json"
FUZZ="$OUTROOT/t2_t1_boundary_fuzz_v0.1.json"
DECISION="$OUTROOT/combined_architecture_decision_v0.1.json"

EXPECTED_T2_CURRICULUM_SHA="40e2d608eaa59ad13e2f9b37a2f3da5d6c20101a28ab76434c11be3d36cb5590"
EXPECTED_T2_PREPARED_SHA="774722d57ccaacc7dba8e53ff3aec66bc4b4b5c6efd20db6907ac5a70654e781"
EXPECTED_SEMANTIC_SHA="6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"
EXPECTED_T1_SHA="483bd59499e9bea890072af9c01c12961f41504d48f988ae3b3ee9a3ae8154fb"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

for required in \
  "$STATE" "$AUDIT_CONTRACT" "$T2_TRAINING_CONTRACT" \
  "$T2_CURRICULUM" "$T2_PREPARED" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" \
  "$TOKENIZER_DIR/tokenizer.json" "$T1_CHECKPOINT"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: audit output already exists: $OUTROOT" >&2
  echo "Preserve it. Do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python3 - "$STATE" <<'PY'
import json, subprocess, sys
state=json.load(open(sys.argv[1]))
if state.get("schema") != "alice.eipm.n0.latent-pool-stage-state.v0.56":
    raise SystemExit("v0.56 state drift")
policy=state["execution_policy"]
if policy.get("cpu_reality_audit_authorized") is not True:
    raise SystemExit("CPU reality audit not authorized")
for key in (
    "optimizer_authorized",
    "gradient_authorized",
    "gpu_training_authorized",
    "automatic_rerun_authorized",
    "learned_support_training_authorized",
    "causal_test_opening_authorized",
    "private_identity_gradient_authorized",
):
    if policy.get(key) is not False:
        raise SystemExit(f"premature authorization: {key}")
core=state["audit_core_revision"]
subprocess.run(["git","merge-base","--is-ancestor",core,"HEAD"],check=True)
print("v0_56_qsre_reality_audit_cpu_gate=PASS")
print("audit_core_revision="+core)
PY

check_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  echo "$label=$actual"
  [[ "$actual" == "$expected" ]] || {
    echo "STOP: $label hash drift" >&2
    exit 5
  }
}

check_sha "$T2_CURRICULUM" "$EXPECTED_T2_CURRICULUM_SHA" "t2_curriculum_sha256"
check_sha "$T2_PREPARED" "$EXPECTED_T2_PREPARED_SHA" "t2_prepared_cache_sha256"
check_sha "$SEMANTIC_CHECKPOINT" "$EXPECTED_SEMANTIC_SHA" "semantic_checkpoint_sha256"
check_sha "$T1_CHECKPOINT" "$EXPECTED_T1_SHA" "t1_checkpoint_sha256"

mkdir -p "$OUTROOT"

echo "===== TRACK 1/5 REALISTIC PUBLIC OPERATOR CORPUS ====="
python3 "$ROOT/scripts/eipm/n0/build_n0_v02_qsre_operator_reality_corpus_v0_1.py" \
  --output "$REALITY"

echo "===== TRACK 2/5 RUNTIME BEHAVIORAL PARITY ====="
python3 "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_runtime_parity_v0_1.py" \
  --repo-root "$ROOT" \
  --workdir "$WORKDIR" \
  --output "$RUNTIME"

echo "===== TRACK 3/5 SHALLOW SHORTCUT ATTACKS ====="
python3 "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_t2_shortcuts_v0_1.py" \
  --t2-curriculum "$T2_CURRICULUM" \
  --reality-corpus "$REALITY" \
  --output "$SHORTCUT"

echo "===== TRACK 4/5 FROZEN SEMANTIC REALITY PROBES ====="
python3 "$ROOT/scripts/eipm/n0/probe_n0_v02_qsre_semantic_reality_v0_1.py" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --t2-curriculum "$T2_CURRICULUM" \
  --reality-corpus "$REALITY" \
  --output "$SEMANTIC" \
  --batch-size 24 \
  --max-length 96

echo "===== TRACK 5/5 T2->T1 TOTALITY FUZZ ====="
python3 "$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_t2_t1_boundary_fuzz_v0_1.py" \
  --contract "$T2_TRAINING_CONTRACT" \
  --prepared-cache "$T2_PREPARED" \
  --t1-checkpoint "$T1_CHECKPOINT" \
  --output "$FUZZ"

echo "===== SINGLE COMBINED ARCHITECTURE DECISION ====="
python3 "$ROOT/scripts/eipm/n0/decide_n0_v02_qsre_reality_audit_v0_1.py" \
  --contract "$AUDIT_CONTRACT" \
  --runtime "$RUNTIME" \
  --shortcut "$SHORTCUT" \
  --semantic "$SEMANTIC" \
  --fuzz "$FUZZ" \
  --output "$DECISION"

echo
echo "===== AUDIT OUTPUT HASHES ====="
sha256sum "$REALITY" "$RUNTIME" "$SHORTCUT" "$SEMANTIC" "$FUZZ" "$DECISION"

echo
echo "===== FINAL DECISION ====="
python3 - "$DECISION" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("status="+str(r["status"]))
print("decision="+str(r["decision"]))
print("reason="+str(r["reason"]))
print("evidence="+json.dumps(r["evidence"],sort_keys=True))
print("optimizer="+str(r["governance"]["optimizer"]))
print("gradient="+str(r["governance"]["gradient"]))
print("gpu="+str(r["governance"]["gpu"]))
print("automatic_training_authorization="+str(r["governance"]["automatic_training_authorization"]))
PY
