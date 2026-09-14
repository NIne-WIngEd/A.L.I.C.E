#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKDIR="${ALICE_N0_WORKDIR:-$ROOT/.alice-private/n0}"
CONFIG="${ALICE_N0_CONFIG:-$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json}"
SOURCE_CONFIG="${ALICE_N0_SOURCE_CONFIG:-$ROOT/configs/eipm/n0/public_corpus_bootstrap_v0.1.json}"
CORPUS_DIR="$WORKDIR/corpus"
TOKENIZER_DIR="$WORKDIR/tokenizer"
CHECKPOINT_DIR="$WORKDIR/checkpoints"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CUDA_COUNT="$($PYTHON - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 2 ]]; then
  echo "2xP100 trainer requires at least two visible CUDA devices; found $CUDA_COUNT." >&2
  exit 2
fi

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
for name in "${GPU_NAMES[@]:0:2}"; do
  if [[ "$name" != *"P100"* ]]; then
    echo "2xP100 trainer expected P100 devices but observed: $name" >&2
    exit 3
  fi
done

"$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG"

if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" || ! -f "$TOKENIZER_DIR/tokenizer_receipt.json" ]]; then
  echo "Verified tokenizer artifacts are required before P100 training." >&2
  exit 4
fi

mapfile -t SHARDS < <(find "$CORPUS_DIR/shards" -type f -name '*.jsonl' | sort)
if [[ "${#SHARDS[@]}" -eq 0 ]]; then
  echo "No verified training shards found under $CORPUS_DIR/shards." >&2
  exit 5
fi

TRAIN_ARGS=()
for shard in "${SHARDS[@]}"; do
  TRAIN_ARGS+=(--train "$shard")
done

RESUME_ARGS=()
if [[ -n "${N0_RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume-from "$N0_RESUME_FROM")
fi

PORT="${N0_MAIN_PROCESS_PORT:-$((20000 + (${SLURM_JOB_ID:-1337} % 20000)))}"

mkdir -p "$CHECKPOINT_DIR"

echo "===== A.L.I.C.E. N0 2xP100 MLM ====="
echo "workdir=$WORKDIR"
echo "cuda_count=$CUDA_COUNT"
echo "gpu0=${GPU_NAMES[0]}"
echo "gpu1=${GPU_NAMES[1]}"
echo "main_process_port=$PORT"
echo "sequence_length=${N0_SEQUENCE_LENGTH:-512}"
echo "micro_batch_size=${N0_MICRO_BATCH_SIZE:-1}"
echo "grad_accum=${N0_GRAD_ACCUM:-16}"
echo "max_steps=${N0_MAX_STEPS:-200}"
echo "save_every=${N0_SAVE_EVERY:-100}"
echo "mixed_precision=${N0_MIXED_PRECISION:-fp16}"

accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision "${N0_MIXED_PRECISION:-fp16}" \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  "$ROOT/scripts/eipm/n0/train_mlm.py" \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
  --tokenizer-receipt "$TOKENIZER_DIR/tokenizer_receipt.json" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG" \
  "${TRAIN_ARGS[@]}" \
  --output-dir "$CHECKPOINT_DIR" \
  --sequence-length "${N0_SEQUENCE_LENGTH:-512}" \
  --micro-batch-size "${N0_MICRO_BATCH_SIZE:-1}" \
  --grad-accum "${N0_GRAD_ACCUM:-16}" \
  --max-steps "${N0_MAX_STEPS:-200}" \
  --warmup-steps "${N0_WARMUP_STEPS:-20}" \
  --save-every "${N0_SAVE_EVERY:-100}" \
  --learning-rate "${N0_LEARNING_RATE:-3e-4}" \
  --mixed-precision "${N0_MIXED_PRECISION:-fp16}" \
  "${RESUME_ARGS[@]}"
