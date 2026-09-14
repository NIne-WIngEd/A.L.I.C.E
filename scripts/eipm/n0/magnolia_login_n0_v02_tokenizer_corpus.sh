#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.activated.json"
OUT="${N0_V02_TOKENIZER_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.1}"
TARGET_CHARS="${N0_V02_TOKENIZER_CORPUS_CHARS:-120000000}"

if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing tokenizer corpus: $OUT" >&2
  exit 2
fi

mkdir -p "$WORKDIR/hf-cache-tokenizer-corpus"
export RAYAN_UDOCKER_NVIDIA=0
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache-tokenizer-corpus}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2 BALANCED TOKENIZER CORPUS ====="
date -Is
echo "repo_root=$ROOT"
echo "source_config=$SOURCE_CONFIG"
echo "output=$OUT"
echo "target_total_chars=$TARGET_CHARS"
echo "gpu_requested=false"
echo "private_identity_gradient=false"

df -h "$WORKDIR" || true

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python scripts/eipm/n0/materialize_public_corpus_v02.py \
    --source-config configs/eipm/n0/public_corpus_v0.2.activated.json \
    --output-dir '$OUT' \
    --target-total-chars '$TARGET_CHARS' \
    --partition-index 0 \
    --partition-count 1 \
    --minimum-fill-ratio 0.95 \
    --shard-mb 64"

echo "===== N0 V0.2 BALANCED TOKENIZER CORPUS COMPLETE ====="
date -Is
