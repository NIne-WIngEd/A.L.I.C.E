#!/usr/bin/env bash
#SBATCH --job-name=n0-j3-grad-memory
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
EXPECTED=a19f8e8893422702c138182f239064385addf91c
SCRIPT_SHA=4cb68fd590225e7c240e2e065f310bd07a36ab27de1e9356697e1dcb83ae39e2
if [[ "$(git rev-parse HEAD)" != "$EXPECTED" || -n "$(git status --porcelain)" ]]; then
  echo 'STOP: requires the pinned, clean scientific N0 source' >&2
  exit 90
fi
WORK="${ALICE_N0_WORKDIR:?workdir required}"
SCRIPT="${ALICE_N0_GRADIENT_DIAGNOSTIC_SCRIPT:?diagnostic script required}"
MIXTURE="${ALICE_N0_FULL_MIXTURE_ROOT:?audited mixture required}"
TEACHER_REGISTRY="${ALICE_N0_TEACHER_REGISTRY:?teacher registry required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?teacher audit required}"
OUT="${ALICE_N0_GRADIENT_DIAGNOSTIC_ROOT:?fresh output root required}"
if [[ -e "$OUT" ]]; then
  echo "STOP: preserve existing gradient diagnostic evidence root: $OUT" >&2
  exit 92
fi
if [[ "$(sha256sum "$SCRIPT" | cut -d ' ' -f 1)" != "$SCRIPT_SHA" ]]; then
  echo 'STOP: gradient diagnostic script hash mismatch' >&2
  exit 93
fi
test -f "$MIXTURE/full_public_mixture_manifest.json"
test -f "$MIXTURE/full_public_mixture_audit.json"
test -f "$TEACHER_REGISTRY"
test -f "$TEACHER_AUDIT"
mkdir -p "$OUT"

export PYTHONPATH="$REPO/src:$REPO/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=1
PORT="$((20000 + (${SLURM_JOB_ID:-1337} % 20000)))"

echo "SOURCE=$EXPECTED"
echo 'FULL_J3=true WORLD_SIZE=2 BACKWARD=true OPTIMIZER=false WEIGHT_UPDATE=false'
echo 'P43_AUTHORITY=false TRAINING_AUTHORITY=false FINAL=false'
echo "EVIDENCE_ROOT=$OUT"

bash "$REPO/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc '
  set -euo pipefail
  cd "$1"
  accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
    --mixed_precision fp16 --dynamo_backend no --main_process_port "$8" \
    "$2" \
    --qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json \
    --training-plan configs/eipm/n0/n0_v02_semantic_operator_joint_training_plan_v1.json \
    --topology-config configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json \
    --semantic-config configs/eipm/n0/alice_n0_semantic_v0.2.json \
    --semantic-checkpoint "$3/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors" \
    --tokenizer-dir "$3/tokenizer-v0.2.1" \
    --corpus-dir "$3/tokenizer-corpus-v0.2.1-offline" \
    --source-config configs/eipm/n0/public_corpus_v0.2.1.activated.json \
    --teacher-registry "$4" \
    --teacher-audit "$5" \
    --semantic-rows "$6/semantic/rows.jsonl" \
    --semantic-long-rows "$6/semantic-long/rows.jsonl" \
    --behavioral-rows "$6/behavioral/rows.jsonl" \
    --runtime-view-rows "$6/runtime-view/rows.jsonl" \
    --long-context-rows "$6/long-context/rows.jsonl" \
    --fewrel-rows "$6/fewrel/train_dev_rows.jsonl" \
    --fewrel-bank "$6/fewrel/train_dev_bank.json" \
    --mixture-manifest "$6/full_public_mixture_manifest.json" \
    --mixture-audit "$6/full_public_mixture_audit.json" \
    --output-root "$7"
' _ "$REPO" "$SCRIPT" "$WORK" "$TEACHER_REGISTRY" "$TEACHER_AUDIT" "$MIXTURE" "$OUT" "$PORT"

python - "$OUT/diagnostic.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["status"] == "COMPLETE_TWO_RANK_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY"
assert len(result["ranks"]) == 2
assert all(row["stress_pair_count"] == 18 for row in result["ranks"])
assert all(row["weight_update"] is False for row in result["ranks"])
assert result["optimizer_step_measured"] is False
assert result["training_authorized"] is False
print("COMPLETE_TWO_RANK_GRADIENT_DIAGNOSTIC_NOT_P43_AUTHORITY")
PY
