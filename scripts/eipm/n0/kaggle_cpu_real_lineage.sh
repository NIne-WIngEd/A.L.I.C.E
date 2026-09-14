#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKDIR="${ALICE_N0_WORKDIR:-/kaggle/working/rayan-n0/n0-v01}"
EXPORT_DIR="${N0_KAGGLE_EXPORT_DIR:-/kaggle/working/rayan-n0-export}"
HF_HOME="${HF_HOME:-/kaggle/working/hf-cache}"
CHARS_PER_SOURCE="${N0_BOOTSTRAP_CHARS_PER_SOURCE:-100000000}"
SHARD_MB="${N0_SHARD_MB:-128}"
RESET="${N0_KAGGLE_RESET:-0}"

mkdir -p "$EXPORT_DIR" "$HF_HOME"

if ! "$PYTHON" - <<'PY'
import datasets, huggingface_hub, tokenizers
print("kaggle_lineage_dependencies=present")
PY
then
  "$PYTHON" -m pip install -q \
    'datasets>=3.6,<5' \
    'huggingface-hub>=0.30,<2' \
    'tokenizers>=0.21,<1'
fi

if [[ "$RESET" == "1" ]]; then
  rm -rf "$WORKDIR/corpus" "$WORKDIR/tokenizer" "$WORKDIR/checkpoints" "$WORKDIR/ranker" "$WORKDIR/evaluation"
fi

if [[ -e "$WORKDIR/corpus" || -e "$WORKDIR/tokenizer" ]]; then
  echo "Refusing to overwrite existing real lineage under $WORKDIR." >&2
  echo "Use a fresh Kaggle session/workdir or set N0_KAGGLE_RESET=1 intentionally." >&2
  exit 2
fi

export ALICE_N0_WORKDIR="$WORKDIR"
export HF_HOME
export N0_BOOTSTRAP_CHARS_PER_SOURCE="$CHARS_PER_SOURCE"
export N0_SHARD_MB="$SHARD_MB"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

mkdir -p "$WORKDIR"

cat <<EOF
===== KAGGLE CPU REAL N0 LINEAGE =====
repo_root=$ROOT
workdir=$WORKDIR
export_dir=$EXPORT_DIR
hf_home=$HF_HOME
chars_per_source=$N0_BOOTSTRAP_CHARS_PER_SOURCE
shard_mb=$N0_SHARD_MB
private_identity_data=false
model_training=false
EOF

bash "$ROOT/scripts/eipm/n0/run_stage.sh" corpus-bootstrap
bash "$ROOT/scripts/eipm/n0/run_stage.sh" verify-corpus
bash "$ROOT/scripts/eipm/n0/run_stage.sh" tokenizer

"$PYTHON" - "$ROOT" "$WORKDIR" "$EXPORT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
workdir = Path(sys.argv[2]).resolve()
export_dir = Path(sys.argv[3]).resolve()
corpus = workdir / "corpus"
tokenizer = workdir / "tokenizer"
corpus_receipt = corpus / "corpus_receipt.json"
tokenizer_receipt = tokenizer / "tokenizer_receipt.json"

for path in (corpus_receipt, tokenizer_receipt, tokenizer / "tokenizer.json"):
    if not path.is_file():
        raise SystemExit(f"required lineage artifact missing: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

files = {}
for base_name, base in (("corpus", corpus), ("tokenizer", tokenizer)):
    for path in sorted(base.rglob("*")):
        if path.is_file():
            rel = Path(base_name) / path.relative_to(base)
            files[str(rel)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

manifest = {
    "schema": "alice.eipm.n0.external-public-lineage-manifest.v0.1",
    "git_revision": git_revision(),
    "corpus_receipt_sha256": sha256_file(corpus_receipt),
    "tokenizer_receipt_sha256": sha256_file(tokenizer_receipt),
    "files": files,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "model_training_performed": False,
    "weights_created": False,
}
export_dir.mkdir(parents=True, exist_ok=True)
manifest_path = export_dir / "transfer_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

archive_path = export_dir / "rayan-n0-v01-public-lineage.tar.gz"
with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(corpus, arcname="corpus", recursive=True)
    archive.add(tokenizer, arcname="tokenizer", recursive=True)
    archive.add(manifest_path, arcname="transfer_manifest.json")

receipt = {
    "schema": "alice.eipm.n0.external-public-lineage-transfer.v0.1",
    "archive_name": archive_path.name,
    "archive_sha256": sha256_file(archive_path),
    "archive_bytes": archive_path.stat().st_size,
    "manifest_sha256": sha256_file(manifest_path),
    "git_revision": manifest["git_revision"],
    "corpus_receipt_sha256": manifest["corpus_receipt_sha256"],
    "tokenizer_receipt_sha256": manifest["tokenizer_receipt_sha256"],
    "private_identity_data": False,
    "private_identity_gradient": False,
    "model_training_performed": False,
    "weights_created": False,
}
receipt_path = export_dir / "transfer_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(json.dumps({
    "status": "PASS",
    "archive": str(archive_path),
    "archive_sha256": receipt["archive_sha256"],
    "archive_bytes": receipt["archive_bytes"],
    "manifest": str(manifest_path),
    "receipt": str(receipt_path),
    "git_revision": receipt["git_revision"],
}, sort_keys=True))
PY

echo "===== KAGGLE CPU REAL N0 LINEAGE COMPLETE ====="
echo "archive=$EXPORT_DIR/rayan-n0-v01-public-lineage.tar.gz"
echo "receipt=$EXPORT_DIR/transfer_receipt.json"
echo "manifest=$EXPORT_DIR/transfer_manifest.json"
