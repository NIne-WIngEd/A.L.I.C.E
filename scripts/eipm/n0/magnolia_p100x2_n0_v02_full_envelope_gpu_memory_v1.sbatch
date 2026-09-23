#!/usr/bin/env bash
#SBATCH --job-name=rayan-n0-gpu-mem
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=rayan-n0-gpu-mem-%j.out
#SBATCH --error=rayan-n0-gpu-mem-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
EXPECTED="${ALICE_N0_EXPECTED_REVISION:?ALICE_N0_EXPECTED_REVISION is required}"
HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED" ]]; then
  echo "STOP: GPU memory dry-run source drift HEAD=$HEAD expected=$EXPECTED" >&2
  exit 90
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "STOP: tracked repository changes exist" >&2
  exit 91
fi

WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
TOKENIZER="${ALICE_N0_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
CORPUS="${ALICE_N0_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
SEMANTIC="${ALICE_N0_SEMANTIC_CHECKPOINT:-$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors}"
TEACHER_REGISTRY="${ALICE_N0_TEACHER_REGISTRY:?ALICE_N0_TEACHER_REGISTRY is required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?ALICE_N0_TEACHER_AUDIT is required}"
MIXTURE_ROOT="${ALICE_N0_FULL_MIXTURE_ROOT:?ALICE_N0_FULL_MIXTURE_ROOT is required}"
OUT_ROOT="${ALICE_N0_GPU_MEMORY_ROOT:-$WORKDIR/full-envelope-gpu-memory-v1}"
RESULT="$OUT_ROOT/result.json"

if [[ -e "$OUT_ROOT" ]]; then
  echo "STOP: preserve existing GPU memory evidence root: $OUT_ROOT" >&2
  exit 92
fi
mkdir -p "$OUT_ROOT"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=1

echo "===== N0 FULL-ENVELOPE GPU NO-GRAD MEMORY DRY RUN ====="
date -Is
echo "source_revision=$HEAD"
echo "gradient=false"
echo "optimizer=false"
echo "training=false"
echo "final_opening=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc "
  torchrun --standalone --nproc_per_node=2     scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py     --qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json     --topology-config configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json     --semantic-config configs/eipm/n0/alice_n0_semantic_v0.2.json     --semantic-checkpoint '$SEMANTIC'     --tokenizer-dir '$TOKENIZER'     --corpus-dir '$CORPUS'     --source-config configs/eipm/n0/public_corpus_v0.2.1.activated.json     --teacher-registry '$TEACHER_REGISTRY'     --teacher-audit '$TEACHER_AUDIT'     --semantic-rows '$MIXTURE_ROOT/semantic/rows.jsonl'     --behavioral-rows '$MIXTURE_ROOT/behavioral/rows.jsonl'     --runtime-view-rows '$MIXTURE_ROOT/runtime-view/rows.jsonl'     --long-context-rows '$MIXTURE_ROOT/long-context/rows.jsonl'     --fewrel-rows '$MIXTURE_ROOT/fewrel/train_dev_rows.jsonl'     --fewrel-bank '$MIXTURE_ROOT/fewrel/train_dev_bank.json'     --mixture-manifest '$MIXTURE_ROOT/full_public_mixture_manifest.json'     --mixture-audit '$MIXTURE_ROOT/full_public_mixture_audit.json'     --output '$RESULT'
"

python - "$RESULT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1"
assert r["gradient"] is False
assert r["backward"] is False
assert r["optimizer_object_created"] is False
assert r["weight_update"] is False
assert r["model_training_performed"] is False
assert r["projection_is_training_authorization"] is False
assert r["gpu_training_authorized"] is False
assert r["private_identity_data"] is False
assert r["final_results_observed"] is False
assert r["final_opening_authorized"] is False
assert r["n0_complete"] is False
assert r["real_optimizer_facing_public_lanes"] is True
assert r["full_j3_counterfactual_path"] is True
assert all(x["projection_pass"] for x in r["ranks"])
print("PASS_N0_FULL_ENVELOPE_GPU_MEMORY_DRY_RUN_V1")
PY

echo "===== N0 FULL-ENVELOPE GPU MEMORY DRY RUN COMPLETE ====="
date -Is
echo "result=$RESULT"
