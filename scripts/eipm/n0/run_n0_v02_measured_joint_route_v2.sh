#!/usr/bin/env bash
# One complete protocol inside a verified two-GPU Linux runtime.
# The public diagnostic optimizer states are quarantined in a fresh root.
set -euo pipefail

ROOT="${ALICE_N0_REPO_ROOT:-$PWD}"
cd "$ROOT"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?expected exact source revision required}"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED" ]] || [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: source revision or clean-worktree proof failed" >&2
  exit 90
fi

WORK="${ALICE_N0_WORKDIR:?work root required}"
MIXTURE="${ALICE_N0_FULL_MIXTURE_ROOT:?fresh exact-head public mixture required}"
TEACHER="${ALICE_N0_TEACHER_REGISTRY:?registered teacher bank required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?teacher audit required}"
EVIDENCE="${ALICE_N0_MEASURED_JOINT_ROOT:?fresh evidence root required}"
TOKENIZER="${ALICE_N0_TOKENIZER_DIR:-$WORK/tokenizer-v0.2.1}"
CORPUS="${ALICE_N0_CORPUS_DIR:-$WORK/tokenizer-corpus-v0.2.1-offline}"
SEMANTIC="${ALICE_N0_SEMANTIC_CHECKPOINT:-$WORK/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors}"

if [[ -e "$EVIDENCE" ]]; then
  echo "STOP: preserve previous measured joint evidence: $EVIDENCE" >&2
  exit 92
fi
for path in "$TEACHER" "$TEACHER_AUDIT" "$SEMANTIC" \
  "$MIXTURE/full_public_mixture_manifest.json" \
  "$MIXTURE/full_public_mixture_audit.json" \
  "$CORPUS/corpus_receipt.json" "$TOKENIZER/tokenizer.json"; do
  test -f "$path" || { echo "STOP: missing input: $path" >&2; exit 93; }
done
python - "$EXPECTED" "$MIXTURE/full_public_mixture_manifest.json" <<'PY'
import json,sys
with open(sys.argv[2],encoding="utf-8") as file:
    manifest=json.load(file)
if manifest.get("source_revision")!=sys.argv[1]:
    raise SystemExit("STOP: mixture was built against a different source head")
PY

mkdir -p "$EVIDENCE"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
PORT="${N0_MAIN_PROCESS_PORT:-29543}"

ARGS=(
  --topology-config configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json
  --semantic-config configs/eipm/n0/alice_n0_semantic_v0.2.json
  --semantic-checkpoint "$SEMANTIC"
  --tokenizer-dir "$TOKENIZER"
  --corpus-dir "$CORPUS"
  --source-config configs/eipm/n0/public_corpus_v0.2.1.activated.json
  --teacher-registry "$TEACHER"
  --teacher-audit "$TEACHER_AUDIT"
  --semantic-rows "$MIXTURE/semantic/rows.jsonl"
  --semantic-long-rows "$MIXTURE/semantic-long/rows.jsonl"
  --behavioral-rows "$MIXTURE/behavioral/rows.jsonl"
  --runtime-view-rows "$MIXTURE/runtime-view/rows.jsonl"
  --long-context-rows "$MIXTURE/long-context/rows.jsonl"
  --fewrel-rows "$MIXTURE/fewrel/train_dev_rows.jsonl"
  --fewrel-bank "$MIXTURE/fewrel/train_dev_bank.json"
  --mixture-manifest "$MIXTURE/full_public_mixture_manifest.json"
  --mixture-audit "$MIXTURE/full_public_mixture_audit.json"
)

echo "N0_MEASURED_ROUTE_SOURCE=$EXPECTED"
echo "N0_INFERENCE_PROJECTION_DIAGNOSTIC_START"
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision fp16 --dynamo_backend no --main_process_port "$PORT" \
  scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py \
  --qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json \
  --preserve-projection-failure "${ARGS[@]}" \
  --output "$EVIDENCE/no_gradient_projection.json"

python - "$EVIDENCE/no_gradient_projection.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding="utf-8") as file:
    receipt=json.load(file)
if receipt.get("status") not in {
    "PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
    "FAIL_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1",
}:
    raise SystemExit("STOP: no-gradient receipt incomplete")
if receipt.get("world_size")!=2 or receipt.get("stress_pair_count")!=18:
    raise SystemExit("STOP: no-gradient full stress coverage incomplete")
print("OLD_PROJECTION_STATUS="+receipt["status"])
PY

echo "N0_COMPLETE_JOINT_MEASURED_STEP_START"
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
  --mixed_precision fp16 --dynamo_backend no --main_process_port "$PORT" \
  scripts/eipm/n0/qualify_n0_v02_complete_joint_route_v2.py \
  --qualification-config configs/eipm/n0/n0_v02_full_envelope_measured_joint_memory_v2.json \
  --old-qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json \
  --no-gradient-receipt "$EVIDENCE/no_gradient_projection.json" \
  --training-plan configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json \
  "${ARGS[@]}" --output-dir "$EVIDENCE/measured"

python - "$EVIDENCE/measured/measured_gpu_memory_result.json" \
  "$EVIDENCE/measured/complete_joint_route_result.json" <<'PY'
import hashlib,json,sys
with open(sys.argv[1],encoding="utf-8") as file:
    gpu=json.load(file)
with open(sys.argv[2],encoding="utf-8") as file:
    route=json.load(file)
if gpu.get("status")!="PASS_N0_FULL_ENVELOPE_MEASURED_JOINT_MEMORY_V2":
    raise SystemExit("STOP: actual memory receipt not PASS")
if route.get("status")!="PASS_N0_COMPLETE_JOINT_ROUTE_GRADIENT_OPTIMIZER_RESUME_V2":
    raise SystemExit("STOP: complete route receipt not PASS")
if hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()!=route.get("gpu_memory_receipt_sha256"):
    raise SystemExit("STOP: measured memory and route hashes differ")
print("COMPLETE_MEASURED_JOINT_PROOF_NOT_PRODUCTION_TRAINING")
PY
