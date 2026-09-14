#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
CORPUS_DIR="${N0_V02_TOKENIZER_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
AUDIT_OUT="$TOKENIZER_DIR/tokenizer_audit.json"

if [[ ! -f "$CORPUS_DIR/corpus_receipt.json" ]]; then
  echo "Missing verified offline tokenizer corpus receipt: $CORPUS_DIR/corpus_receipt.json" >&2
  exit 2
fi
if [[ -e "$TOKENIZER_DIR" && -n "$(find "$TOKENIZER_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing tokenizer directory: $TOKENIZER_DIR" >&2
  exit 2
fi

mkdir -p "$TOKENIZER_DIR"
export RAYAN_UDOCKER_NVIDIA=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2 GOVERNED TOKENIZER BUILD ====="
date -Is
echo "repo_root=$ROOT"
echo "corpus_dir=$CORPUS_DIR"
echo "source_config=$SOURCE_CONFIG"
echo "tokenizer_dir=$TOKENIZER_DIR"
echo "vocab_size=48000"
echo "fit_split=train_only"
echo "network_access_required=false"
echo "gpu_requested=false"
echo "private_identity_gradient=false"

df -h "$WORKDIR" || true

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python -m py_compile \
    scripts/eipm/n0/train_tokenizer_v02.py \
    scripts/eipm/n0/audit_tokenizer_v02.py && \
   python scripts/eipm/n0/train_tokenizer_v02.py \
    --corpus-dir '$CORPUS_DIR' \
    --source-config configs/eipm/n0/public_corpus_v0.2.1.activated.json \
    --output-dir '$TOKENIZER_DIR' \
    --vocab-size 48000 \
    --min-frequency 2 && \
   python scripts/eipm/n0/audit_tokenizer_v02.py \
    --corpus-dir '$CORPUS_DIR' \
    --tokenizer-dir '$TOKENIZER_DIR' \
    --output '$AUDIT_OUT' \
    --train-rows-per-source 64"

echo "===== N0 V0.2 GOVERNED TOKENIZER BUILD COMPLETE ====="
date -Is
