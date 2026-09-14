#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
PARENT="${N0_V02_PARENT_TOKENIZER_CORPUS:-$WORKDIR/tokenizer-corpus-v0.1}"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_v0.2.1.activated.json"
OUT="${N0_V02_OFFLINE_TOKENIZER_CORPUS_DIR:-$WORKDIR/tokenizer-corpus-v0.2.1-offline}"
TARGET_CHARS="${N0_V02_OFFLINE_TOKENIZER_CHARS:-100000000}"

if [[ ! -f "$PARENT/corpus_receipt.json" ]]; then
  echo "Missing parent corpus receipt: $PARENT/corpus_receipt.json" >&2
  exit 2
fi
if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing offline tokenizer corpus: $OUT" >&2
  exit 2
fi

export RAYAN_UDOCKER_NVIDIA=0
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== N0 V0.2.1 OFFLINE TOKENIZER CORPUS DERIVATION ====="
date -Is
echo "repo_root=$ROOT"
echo "parent_corpus=$PARENT"
echo "source_config=$SOURCE_CONFIG"
echo "output=$OUT"
echo "target_total_chars=$TARGET_CHARS"
echo "network_access_required=false"
echo "gpu_requested=false"
echo "private_identity_gradient=false"

bash "$ROOT/scripts/eipm/n0/magnolia_udocker_exec.sh" -lc \
  "python -m py_compile scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py && \
   python scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py \
     --parent-corpus-dir '$PARENT' \
     --source-config configs/eipm/n0/public_corpus_v0.2.1.activated.json \
     --output-dir '$OUT' \
     --target-total-chars '$TARGET_CHARS' \
     --max-document-chunk-chars 16000 \
     --min-chars 80 \
     --shard-mb 64"

echo "===== N0 V0.2.1 OFFLINE TOKENIZER CORPUS DERIVATION COMPLETE ====="
date -Is
