#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-fp16-j3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?expected source revision required}"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: fp16 diagnostic source drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: fp16 diagnostic requires a fully clean checkout" >&2
  exit 91
fi

WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
TOKENIZER="${ALICE_N0_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
CORPUS="${ALICE_N0_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
SEMANTIC="${ALICE_N0_SEMANTIC_CHECKPOINT:-$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors}"
TEACHER_REGISTRY="${ALICE_N0_TEACHER_REGISTRY:?teacher registry required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?teacher audit required}"
MIXTURE_ROOT="${ALICE_N0_FULL_MIXTURE_ROOT:?audited public mixture required}"
OUT_ROOT="${ALICE_N0_GPU_PRECISION_PREFLIGHT_ROOT:?fresh diagnostic evidence root required}"
RESULT="$OUT_ROOT/diagnostic.json"

if [[ -e "$OUT_ROOT" ]]; then
  echo "STOP: preserve existing fp16 diagnostic evidence root: $OUT_ROOT" >&2
  exit 92
fi
mkdir -p "$OUT_ROOT"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=1

echo "===== EXACT J3 ONE-GPU FP16 FORWARD DIAGNOSTIC ====="
echo "source_revision=$HEAD"
echo "no_gradient=true no_optimizer=true memory_authority=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc '
  set -euo pipefail
  python scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py \
    --diagnostic-only \
    --qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json \
    --topology-config configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json \
    --semantic-config configs/eipm/n0/alice_n0_semantic_v0.2.json \
    --semantic-checkpoint "$1" \
    --tokenizer-dir "$2" \
    --corpus-dir "$3" \
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
    --output "$7"
' _ "$SEMANTIC" "$TOKENIZER" "$CORPUS" \
  "$TEACHER_REGISTRY" "$TEACHER_AUDIT" "$MIXTURE_ROOT" "$RESULT"

python - "$RESULT" "$HEAD" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="COMPLETE_N0_P43_FP16_J3_DIAGNOSTIC_NOT_AUTHORITY"
assert r["source_revision"]==sys.argv[2]
assert r["world_size"]==1 and r["ddp_replica_wrapped"] is False
assert r["stress_pair_count"]==18 and len(r["stress_pair_receipts"])==18
assert r["full_j3_counterfactual_path"] is True
assert r["all_public_training_lanes_executed"] is True
assert r["gradient"] is False and r["optimizer_object_created"] is False
assert r["gpu_memory_projection_measured"] is False
assert r["gpu_training_authorized"] is False and r["final_results_observed"] is False
print("COMPLETE_N0_P43_FP16_J3_DIAGNOSTIC_NOT_AUTHORITY")
PY
