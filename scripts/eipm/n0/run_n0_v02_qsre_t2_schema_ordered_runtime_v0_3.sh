#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:?ALICE_N0_REPO_ROOT is required}"
WORKDIR="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR is required}"

STATE="$ROOT/configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.58.json"
CONTRACT="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_3.json"
AUDIT="$ROOT/scripts/eipm/n0/audit_n0_v02_qsre_t2_schema_ordered_runtime_v0_3.py"
RELATION_SCHEMA="$ROOT/configs/eipm/n0/n0_v02_qsre_t2_relation_schema_v0_1.json"

PREPARED="$WORKDIR/qsre-t2-operator-v0.2/qsre_t2_tokenized_preparation_v0.2.pt"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
T1_CHECKPOINT="$WORKDIR/qsre-t1-executor-v0.1/training-v0.1/step-00000050/qsre_t1_executor.pt"

OUTROOT="$WORKDIR/qsre-t2-operator-v0.3/runtime-qualification-v0.1"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=0

for required in   "$STATE" "$CONTRACT" "$AUDIT" "$RELATION_SCHEMA" "$PREPARED"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" "$TOKENIZER_DIR/tokenizer.json"   "$T1_CHECKPOINT"
do
  [[ -f "$required" ]] || {
    echo "MISSING: $required" >&2
    exit 2
  }
done

if [[ -e "$OUTROOT" ]]; then
  echo "REFUSING: T2 v0.3 runtime evidence already exists: $OUTROOT" >&2
  echo "Preserve it. Do not delete it to rerun." >&2
  exit 3
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python3 - "$STATE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s.get("schema")!="alice.eipm.n0.latent-pool-stage-state.v0.58":
    raise SystemExit("v0.58 state drift")
p=s["execution_policy"]
if p.get("t2_v03_cpu_real_artifact_runtime_qualification_authorized") is not True:
    raise SystemExit("T2 v0.3 CPU runtime qualification not authorized")
if p.get("t2_v03_gpu_training_authorized_after_static_and_cpu_runtime_pass") is not True:
    raise SystemExit("dependent T2 v0.3 GPU boundary not pre-authorized")
if p.get("max_t2_v03_gpu_runs")!=1:
    raise SystemExit("T2 v0.3 one-shot count drift")
for key in (
    "t2_v02_rerun_authorized",
    "automatic_rerun_authorized",
    "automatic_hotfix_chain_authorized",
    "learning_rate_search_authorized",
    "step_search_authorized",
    "model_size_search_authorized",
    "semantic_backbone_gradient_authorized",
    "t1_executor_gradient_authorized",
    "learned_support_training_authorized",
    "causal_test_opening_authorized",
    "private_identity_gradient_authorized",
):
    if p.get(key) is not False:
        raise SystemExit(f"boundary drift: {key}")
print("v0_58_t2_v03_real_artifact_cpu_gate=PASS")
PY

python3 "$AUDIT"   --contract "$CONTRACT"   --prepared-cache "$PREPARED"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --t1-checkpoint "$T1_CHECKPOINT"   --relation-schema "$RELATION_SCHEMA"   --output-dir "$OUTROOT"

echo
echo "===== QSRE T2 v0.3 RUNTIME QUALIFICATION HASHES ====="
sha256sum "$OUTROOT/real-artifact-hidden-smoke.pt" "$OUTROOT/receipt.json"

echo
echo "===== QSRE T2 v0.3 RUNTIME RECEIPT ====="
cat "$OUTROOT/receipt.json"
