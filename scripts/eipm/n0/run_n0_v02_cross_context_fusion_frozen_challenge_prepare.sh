#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1"
CHALLENGE="$OUT_ROOT/challenge.jsonl"
MANIFEST="$OUT_ROOT/manifest.json"
FREEZE_RECEIPT="$OUT_ROOT/freeze_receipt.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.1.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge.py"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge.py"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing frozen fusion challenge: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$BUILDER" "$EVALUATOR"

mkdir -p "$OUT_ROOT"
python "$BUILDER" \
  --output "$CHALLENGE" \
  --manifest "$MANIFEST" \
  --spec "$SPEC"

python - "$SPEC" "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

spec_path = Path(sys.argv[1])
challenge_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

spec = json.loads(spec_path.read_text())
manifest = json.loads(manifest_path.read_text())
if spec.get("status") != "FROZEN_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
    raise SystemExit("challenge spec status drift")
if spec.get("training_use") != "FORBIDDEN":
    raise SystemExit("challenge training use must be forbidden")
if spec.get("ratification_gate", {}).get("thresholds_are_frozen_before_results") is not True:
    raise SystemExit("ratification thresholds are not frozen")
if manifest.get("status") != "FROZEN_UNTOUCHED_NOT_EVALUATED":
    raise SystemExit("challenge manifest status drift")
if manifest.get("challenge_sha256") != sha(challenge_path):
    raise SystemExit("challenge hash mismatch")
if manifest.get("spec_sha256") != sha(spec_path):
    raise SystemExit("challenge spec hash mismatch")
if manifest.get("training_authorized") is not False:
    raise SystemExit("challenge may not authorize training")
if manifest.get("results_observed_at_compile_time") is not False:
    raise SystemExit("challenge was not frozen before result observation")
if manifest.get("rows") != 96 or manifest.get("family_count") != 12:
    raise SystemExit("frozen challenge coverage drift")

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
receipt = {
    "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-freeze.v0.1",
    "status": "FROZEN_BEFORE_EVALUATION",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": head,
    "challenge_sha256": sha(challenge_path),
    "manifest_sha256": sha(manifest_path),
    "spec_sha256": sha(spec_path),
    "rows": 96,
    "families": 12,
    "thresholds_frozen_before_results": True,
    "challenge_rows_used_for_training": False,
    "training_authorized": False,
    "gradient_performed": False,
    "results_observed": False,
    "private_identity_data": False,
    "private_identity_gradient": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print("fusion_frozen_challenge_prepare_pass=true")
print(f"git_revision={head}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print("challenge_rows=96 families=12")
print("thresholds_frozen_before_results=true")
print("challenge_rows_used_for_training=false")
print("gradient_performed=false")
print(f"freeze_receipt={receipt_path}")
PY
