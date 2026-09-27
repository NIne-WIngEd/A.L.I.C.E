#!/usr/bin/env bash
#SBATCH --job-name=n0-j3-forward-attr
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
SCRIPT_SHA=bd2d39fe507c02cf7e3e2fb479c651f4d2c98abb446eb506cc8c89b2e197f406
if [[ "$(git rev-parse HEAD)" != "$EXPECTED" || -n "$(git status --porcelain)" ]]; then
  echo 'STOP: exact pinned clean N0 source required' >&2
  exit 90
fi

WORK="${ALICE_N0_WORKDIR:?ALICE_N0_WORKDIR required}"
PROFILE="${ALICE_N0_FORWARD_TRACE_SCRIPT:?trace script path required}"
MIXTURE="${ALICE_N0_FULL_MIXTURE_ROOT:?audited mixture required}"
TEACHER_REGISTRY="${ALICE_N0_TEACHER_REGISTRY:?teacher registry required}"
TEACHER_AUDIT="${ALICE_N0_TEACHER_AUDIT:?teacher audit required}"
OUTPUT="${ALICE_N0_FORWARD_TRACE_ROOT:?fresh forward trace root required}"
if [[ -e "$OUTPUT" ]]; then
  echo "STOP: preserve existing trace evidence root: $OUTPUT" >&2
  exit 92
fi
if [[ "$(sha256sum "$PROFILE" | cut -d ' ' -f 1)" != "$SCRIPT_SHA" ]]; then
  echo 'STOP: trace script hash mismatch' >&2
  exit 93
fi
test -f "$MIXTURE/full_public_mixture_manifest.json"
test -f "$TEACHER_REGISTRY"
test -f "$TEACHER_AUDIT"
mkdir -p "$OUTPUT"

export PYTHONPATH="$REPO/src:$REPO/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export RAYAN_UDOCKER_NVIDIA=1

echo "SOURCE=$EXPECTED"
echo 'FULL_J3_FORWARD=true SINGLE_RANK_DIAGNOSTIC=true GRADIENT=false TRAINING_AUTHORITY=false'

bash "$REPO/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc '
  set -euo pipefail
  cd "$1"
  python "$2" \
    --trace-output "$7/forward_trace.json" \
    --diagnostic-only \
    --qualification-config configs/eipm/n0/n0_v02_full_envelope_gpu_memory_dry_run_v1.json \
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
    --output "$7/diagnostic.json"
' _ "$REPO" "$PROFILE" "$WORK" "$TEACHER_REGISTRY" "$TEACHER_AUDIT" "$MIXTURE" "$OUTPUT"

python - "$OUTPUT/forward_trace.json" "$OUTPUT/diagnostic.json" <<'PY'
import json, sys
trace = json.load(open(sys.argv[1]))
diag = json.load(open(sys.argv[2]))
assert trace['status'] == 'COMPLETE_FORWARD_ATTRIBUTION_NOT_P43_AUTHORITY'
assert diag['status'] == 'COMPLETE_N0_P43_FP16_J3_DIAGNOSTIC_NOT_AUTHORITY'
assert trace['stress_pair_count'] == diag['stress_pair_count'] == 18
assert trace['training_authorized'] is False
print('COMPLETE_FORWARD_ATTRIBUTION_NOT_P43_AUTHORITY')
PY
