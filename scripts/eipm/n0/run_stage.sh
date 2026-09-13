#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKDIR="${ALICE_N0_WORKDIR:-$ROOT/.alice-private/n0}"
CONFIG="${ALICE_N0_CONFIG:-$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json}"
SOURCE_CONFIG="${ALICE_N0_SOURCE_CONFIG:-$ROOT/configs/eipm/n0/public_corpus_bootstrap_v0.1.json}"
CORPUS_DIR="$WORKDIR/corpus"
TOKENIZER_DIR="$WORKDIR/tokenizer"
CHECKPOINT_DIR="$WORKDIR/checkpoints"

mkdir -p "$WORKDIR" "$CHECKPOINT_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

case "$STAGE" in
  corpus-smoke)
    "$PYTHON" "$ROOT/scripts/eipm/n0/materialize_public_corpus.py" \
      --source-config "$SOURCE_CONFIG" \
      --output-dir "$CORPUS_DIR" \
      --max-docs-per-source "${N0_SMOKE_DOCS_PER_SOURCE:-100}" \
      --shard-mb 32
    ;;

  corpus)
    "$PYTHON" "$ROOT/scripts/eipm/n0/materialize_public_corpus.py" \
      --source-config "$SOURCE_CONFIG" \
      --output-dir "$CORPUS_DIR" \
      --shard-mb "${N0_SHARD_MB:-256}"
    ;;

  tokenizer)
    mapfile -t SHARDS < <(find "$CORPUS_DIR/shards" -type f -name '*.jsonl' | sort)
    if [[ "${#SHARDS[@]}" -eq 0 ]]; then
      echo "No N0 corpus shards found under $CORPUS_DIR/shards" >&2
      exit 2
    fi
    ARGS=()
    for shard in "${SHARDS[@]}"; do ARGS+=(--input "$shard"); done
    "$PYTHON" "$ROOT/scripts/eipm/n0/train_tokenizer.py" \
      "${ARGS[@]}" \
      --output-dir "$TOKENIZER_DIR" \
      --vocab-size 48000
    ;;

  preflight)
    TOKENIZER_ARG=()
    BACKWARD_ARG=()
    if [[ -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
      TOKENIZER_ARG=(--tokenizer "$TOKENIZER_DIR/tokenizer.json")
    fi
    if [[ -n "${N0_PREFLIGHT_BACKWARD:-}" ]]; then
      BACKWARD_ARG=(--backward)
    fi
    "$PYTHON" "$ROOT/scripts/eipm/n0/preflight.py" \
      --config "$CONFIG" \
      "${TOKENIZER_ARG[@]}" \
      "${BACKWARD_ARG[@]}"
    ;;

  train-mlm)
    mapfile -t SHARDS < <(find "$CORPUS_DIR/shards" -type f -name '*.jsonl' | sort)
    if [[ "${#SHARDS[@]}" -eq 0 || ! -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
      echo "Corpus and tokenizer must exist before train-mlm" >&2
      exit 2
    fi
    ARGS=()
    for shard in "${SHARDS[@]}"; do ARGS+=(--train "$shard"); done
    accelerate launch "$ROOT/scripts/eipm/n0/train_mlm.py" \
      --config "$CONFIG" \
      --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
      "${ARGS[@]}" \
      --output-dir "$CHECKPOINT_DIR" \
      --sequence-length "${N0_SEQUENCE_LENGTH:-512}" \
      --micro-batch-size "${N0_MICRO_BATCH_SIZE:-2}" \
      --grad-accum "${N0_GRAD_ACCUM:-16}" \
      --max-steps "${N0_MAX_STEPS:-10000}" \
      --warmup-steps "${N0_WARMUP_STEPS:-500}" \
      --save-every "${N0_SAVE_EVERY:-1000}" \
      --learning-rate "${N0_LEARNING_RATE:-3e-4}" \
      --mixed-precision "${N0_MIXED_PRECISION:-auto}"
    ;;

  *)
    cat >&2 <<EOF
usage: $0 {corpus-smoke|corpus|tokenizer|preflight|train-mlm}

Private work directory defaults to:
  $WORKDIR

Override with ALICE_N0_WORKDIR. The script never writes private identity data.
EOF
    exit 2
    ;;
esac
