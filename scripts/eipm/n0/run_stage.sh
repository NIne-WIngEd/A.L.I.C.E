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
RANKER_DIR="$WORKDIR/ranker"
EVAL_DIR="$WORKDIR/evaluation"
CURRICULUM="${ALICE_N0_CURRICULUM:-$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.jsonl}"
CURRICULUM_MANIFEST="${ALICE_N0_CURRICULUM_MANIFEST:-$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.origin.json}"

mkdir -p "$WORKDIR" "$CHECKPOINT_DIR" "$RANKER_DIR" "$EVAL_DIR"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

latest_mlm_checkpoint() {
  if [[ -n "${N0_MLM_CHECKPOINT:-}" ]]; then
    printf '%s\n' "$N0_MLM_CHECKPOINT"
    return
  fi
  local latest
  latest="$(find "$CHECKPOINT_DIR" -maxdepth 2 -type d -path '*/model' | sort | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "No MLM checkpoint found. Set N0_MLM_CHECKPOINT or run train-mlm first." >&2
    exit 2
  fi
  printf '%s\n' "$latest"
}

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

  train-curriculum)
    if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" ]]; then
      echo "Tokenizer must exist before train-curriculum" >&2
      exit 2
    fi
    MLM_CHECKPOINT="$(latest_mlm_checkpoint)"
    accelerate launch "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
      --config "$CONFIG" \
      --tokenizer-dir "$TOKENIZER_DIR" \
      --mlm-checkpoint "$MLM_CHECKPOINT" \
      --curriculum "$CURRICULUM" \
      --curriculum-manifest "$CURRICULUM_MANIFEST" \
      --output-dir "$RANKER_DIR" \
      --max-length "${N0_CURRICULUM_MAX_LENGTH:-512}" \
      --batch-size "${N0_CURRICULUM_BATCH_SIZE:-4}" \
      --epochs "${N0_CURRICULUM_EPOCHS:-3}" \
      --learning-rate "${N0_CURRICULUM_LR:-2e-5}"
    ;;

  evaluate-curriculum)
    if [[ ! -f "$RANKER_DIR/ranker.safetensors" ]]; then
      echo "Ranker checkpoint must exist before evaluate-curriculum" >&2
      exit 2
    fi
    MLM_CHECKPOINT="$(latest_mlm_checkpoint)"
    "$PYTHON" "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
      --config "$CONFIG" \
      --tokenizer-dir "$TOKENIZER_DIR" \
      --mlm-checkpoint "$MLM_CHECKPOINT" \
      --ranker "$RANKER_DIR/ranker.safetensors" \
      --curriculum "$CURRICULUM" \
      --curriculum-manifest "$CURRICULUM_MANIFEST" \
      --output-dir "$EVAL_DIR" \
      --split "${N0_EVAL_SPLIT:-dev}" \
      --max-length "${N0_CURRICULUM_MAX_LENGTH:-512}" \
      --batch-size "${N0_EVAL_BATCH_SIZE:-8}" \
      --device "${N0_EVAL_DEVICE:-auto}"
    ;;

  *)
    cat >&2 <<EOF
usage: $0 {corpus-smoke|corpus|tokenizer|preflight|train-mlm|train-curriculum|evaluate-curriculum}

Private work directory defaults to:
  $WORKDIR

Override with ALICE_N0_WORKDIR. N0 stages do not write private identity data.
EOF
    exit 2
    ;;
esac
