#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKDIR="${ALICE_N0_WORKDIR:-$ROOT/.alice-private/n0}"
CONFIG="${ALICE_N0_CONFIG:-$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json}"
SOURCE_CONFIG="${ALICE_N0_SOURCE_CONFIG:-$ROOT/configs/eipm/n0/public_corpus_bootstrap_v0.1.json}"
SMOKE_CORPUS_DIR="$WORKDIR/corpus-smoke"
SMOKE_TOKENIZER_DIR="$WORKDIR/tokenizer-smoke"
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

latest_mlm_step_dir() {
  if [[ -n "${N0_MLM_STEP_DIR:-}" ]]; then
    printf '%s\n' "$N0_MLM_STEP_DIR"
    return
  fi
  local latest
  latest="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name 'step-*' | sort | tail -n 1 || true)"
  if [[ -z "$latest" || ! -f "$latest/receipt.json" || ! -d "$latest/model" ]]; then
    echo "No complete MLM step checkpoint found. Set N0_MLM_STEP_DIR or run train-mlm first." >&2
    exit 2
  fi
  printf '%s\n' "$latest"
}

collect_shards() {
  local root="$1"
  find "$root/shards" -type f -name '*.jsonl' 2>/dev/null | sort
}

require_empty_corpus_target() {
  local root="$1"
  if [[ -e "$root/corpus_receipt.json" || -d "$root/shards" ]]; then
    cat >&2 <<EOF
Corpus target already contains materialized state:
  $root

Refusing to append into an existing receipt lineage. Set ALICE_N0_WORKDIR to a fresh
runtime directory, or explicitly remove/archive that runtime corpus before rebuilding.
EOF
    exit 2
  fi
}

train_tokenizer_from_dir() {
  local corpus_root="$1"
  local tokenizer_root="$2"
  mapfile -t SHARDS < <(collect_shards "$corpus_root")
  if [[ "${#SHARDS[@]}" -eq 0 ]]; then
    echo "No N0 corpus shards found under $corpus_root/shards" >&2
    exit 2
  fi
  ARGS=()
  for shard in "${SHARDS[@]}"; do ARGS+=(--input "$shard"); done
  "$PYTHON" "$ROOT/scripts/eipm/n0/train_tokenizer.py" \
    "${ARGS[@]}" \
    --output-dir "$tokenizer_root" \
    --vocab-size 48000
}

case "$STAGE" in
  corpus-smoke)
    require_empty_corpus_target "$SMOKE_CORPUS_DIR"
    "$PYTHON" "$ROOT/scripts/eipm/n0/materialize_public_corpus.py" \
      --source-config "$SOURCE_CONFIG" \
      --output-dir "$SMOKE_CORPUS_DIR" \
      --max-docs-per-source "${N0_SMOKE_DOCS_PER_SOURCE:-100}" \
      --shard-mb 32
    ;;

  tokenizer-smoke)
    "$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
      --corpus-dir "$SMOKE_CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG"
    train_tokenizer_from_dir "$SMOKE_CORPUS_DIR" "$SMOKE_TOKENIZER_DIR"
    ;;

  preflight-smoke)
    if [[ ! -f "$SMOKE_TOKENIZER_DIR/tokenizer.json" ]]; then
      echo "Smoke tokenizer must exist before preflight-smoke" >&2
      exit 2
    fi
    "$PYTHON" "$ROOT/scripts/eipm/n0/preflight.py" \
      --config "$CONFIG" \
      --tokenizer "$SMOKE_TOKENIZER_DIR/tokenizer.json" \
      --backward
    ;;

  corpus-bootstrap)
    require_empty_corpus_target "$CORPUS_DIR"
    "$PYTHON" "$ROOT/scripts/eipm/n0/materialize_public_corpus.py" \
      --source-config "$SOURCE_CONFIG" \
      --output-dir "$CORPUS_DIR" \
      --max-chars-per-source "${N0_BOOTSTRAP_CHARS_PER_SOURCE:-100000000}" \
      --shard-mb "${N0_SHARD_MB:-128}"
    ;;

  corpus)
    if [[ "${N0_ALLOW_UNBOUNDED_CORPUS:-}" != "1" ]]; then
      cat >&2 <<EOF
Unbounded corpus materialization is disabled by default.
Use 'corpus-bootstrap' for the first real bounded N0 checkpoint, or set:
  N0_ALLOW_UNBOUNDED_CORPUS=1
when an intentionally unbounded corpus run is desired and storage has been checked.
EOF
      exit 2
    fi
    require_empty_corpus_target "$CORPUS_DIR"
    "$PYTHON" "$ROOT/scripts/eipm/n0/materialize_public_corpus.py" \
      --source-config "$SOURCE_CONFIG" \
      --output-dir "$CORPUS_DIR" \
      --shard-mb "${N0_SHARD_MB:-256}"
    ;;

  verify-corpus)
    "$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
      --corpus-dir "$CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG"
    ;;

  tokenizer)
    "$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
      --corpus-dir "$CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG"
    train_tokenizer_from_dir "$CORPUS_DIR" "$TOKENIZER_DIR"
    ;;

  preflight)
    if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" || ! -f "$TOKENIZER_DIR/tokenizer_receipt.json" ]]; then
      echo "Verified N0 tokenizer and tokenizer receipt must exist before preflight" >&2
      exit 2
    fi
    BACKWARD_ARG=()
    if [[ -n "${N0_PREFLIGHT_BACKWARD:-}" ]]; then
      BACKWARD_ARG=(--backward)
    fi
    "$PYTHON" "$ROOT/scripts/eipm/n0/preflight.py" \
      --config "$CONFIG" \
      --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
      "${BACKWARD_ARG[@]}"
    ;;

  train-mlm)
    "$PYTHON" "$ROOT/scripts/eipm/n0/verify_public_corpus.py" \
      --corpus-dir "$CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG"
    mapfile -t SHARDS < <(collect_shards "$CORPUS_DIR")
    if [[ "${#SHARDS[@]}" -eq 0 || ! -f "$TOKENIZER_DIR/tokenizer.json" || ! -f "$TOKENIZER_DIR/tokenizer_receipt.json" ]]; then
      echo "Verified corpus and tokenizer artifacts must exist before train-mlm" >&2
      exit 2
    fi
    ARGS=()
    for shard in "${SHARDS[@]}"; do ARGS+=(--train "$shard"); done
    RESUME_ARG=()
    if [[ -n "${N0_RESUME_FROM:-}" ]]; then
      RESUME_ARG=(--resume-from "$N0_RESUME_FROM")
    fi
    accelerate launch "$ROOT/scripts/eipm/n0/train_mlm.py" \
      --config "$CONFIG" \
      --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
      --tokenizer-receipt "$TOKENIZER_DIR/tokenizer_receipt.json" \
      --corpus-dir "$CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG" \
      "${ARGS[@]}" \
      --output-dir "$CHECKPOINT_DIR" \
      --sequence-length "${N0_SEQUENCE_LENGTH:-512}" \
      --micro-batch-size "${N0_MICRO_BATCH_SIZE:-2}" \
      --grad-accum "${N0_GRAD_ACCUM:-16}" \
      --max-steps "${N0_MAX_STEPS:-10000}" \
      --warmup-steps "${N0_WARMUP_STEPS:-500}" \
      --scheduler-total-steps "${N0_SCHEDULER_TOTAL_STEPS:-10000}" \
      --save-every "${N0_SAVE_EVERY:-1000}" \
      --learning-rate "${N0_LEARNING_RATE:-3e-4}" \
      --mixed-precision "${N0_MIXED_PRECISION:-auto}" \
      "${RESUME_ARG[@]}"
    ;;

  evaluate-mlm)
    if [[ ! -f "$TOKENIZER_DIR/tokenizer.json" || ! -f "$TOKENIZER_DIR/tokenizer_receipt.json" ]]; then
      echo "Tokenizer must exist before evaluate-mlm" >&2
      exit 2
    fi
    STEP_DIR="$(latest_mlm_step_dir)"
    "$PYTHON" "$ROOT/scripts/eipm/n0/evaluate_mlm.py" \
      --config "$CONFIG" \
      --checkpoint-dir "$STEP_DIR" \
      --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
      --tokenizer-receipt "$TOKENIZER_DIR/tokenizer_receipt.json" \
      --corpus-dir "$CORPUS_DIR" \
      --source-config "$SOURCE_CONFIG" \
      --sequence-length "${N0_EVAL_SEQUENCE_LENGTH:-512}" \
      --batch-size "${N0_MLM_EVAL_BATCH_SIZE:-4}" \
      --max-batches "${N0_MLM_EVAL_MAX_BATCHES:-256}" \
      --seed "${N0_MLM_EVAL_SEED:-424242}" \
      --device "${N0_MLM_EVAL_DEVICE:-auto}" \
      --output "$EVAL_DIR/mlm-dev-$(basename "$STEP_DIR").json"
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
usage: $0 {corpus-smoke|tokenizer-smoke|preflight-smoke|corpus-bootstrap|corpus|verify-corpus|tokenizer|preflight|train-mlm|evaluate-mlm|train-curriculum|evaluate-curriculum}

Private work directory defaults to:
  $WORKDIR

The first real corpus path is 'corpus-bootstrap', bounded by default to 100M accepted
characters per configured source (override N0_BOOTSTRAP_CHARS_PER_SOURCE).

Override ALICE_N0_WORKDIR for a fresh runtime lineage. N0 stages do not write private identity data.
EOF
    exit 2
    ;;
esac
