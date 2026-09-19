#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
MAP="$OUT_ROOT/design/relation_conditioned_layer_map.json"
QUAL_DIR="$OUT_ROOT/qualification"
RESULT="$QUAL_DIR/runtime_contract_qualification.json"
QUALIFIER="$ROOT/scripts/eipm/n0/qualify_n0_v02_relation_conditioned_multilayer_interface_v0_1.py"

EXPECTED_LAYER_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_SOURCE_AUDIT_SHA256="ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=true

if [[ -e "$QUAL_DIR" ]]; then
  echo "REFUSING: runtime qualification output already exists: $QUAL_DIR" >&2
  exit 2
fi

for required in "$MAP" "$QUALIFIER"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$QUALIFIER"

ACTUAL_MAP_SHA256="$(sha256sum "$MAP" | awk '{print $1}')"
if [[ "$ACTUAL_MAP_SHA256" != "$EXPECTED_LAYER_MAP_SHA256" ]]; then
  echo "STOP: compiled layer-map hash drift: $ACTUAL_MAP_SHA256" >&2
  exit 5
fi

mkdir -p "$QUAL_DIR"

echo "===== N0 RELATION-CONDITIONED MULTILAYER INTERFACE RUNTIME QUALIFICATION ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "layer_map_sha256=$ACTUAL_MAP_SHA256"
echo "source_audit_sha256=$EXPECTED_SOURCE_AUDIT_SHA256"
echo "gradient_performed=false"
echo "optimizer_created=false"
echo "gpu_required=false"
echo "heldout_rows_used=false"
echo "frozen_challenge_rows_used=false"
echo "training_authorized=false"
echo "scale_authorized=false"

python "$QUALIFIER" \
  --repo-root "$ROOT" \
  --layer-map "$MAP" \
  --expected-layer-map-sha256 "$EXPECTED_LAYER_MAP_SHA256" \
  --expected-source-audit-sha256 "$EXPECTED_SOURCE_AUDIT_SHA256" \
  --output "$RESULT"

echo
echo "===== QUALIFICATION RECEIPT HASH ====="
sha256sum "$RESULT"

echo
echo "===== QUALIFICATION SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"git_revision={r.get('git_revision')}")
print(f"layer_map_sha256={r.get('layer_map_sha256')}")
print(f"candidate_layer_bank={r.get('candidate_layer_bank')}")
print(f"parameter_state_exactly_unchanged={r.get('parameter_state_exactly_unchanged')}")
print(f"gradient_performed={r.get('gradient_performed')}")
print(f"optimizer_created={r.get('optimizer_created')}")
print(f"gpu_required={r.get('gpu_required')}")
print(f"training_authorized={r.get('training_authorized')}")
print(f"scale_authorized={r.get('scale_authorized')}")
print(f"heldout_opening_authorized={r.get('heldout_opening_authorized')}")
print(f"frozen_challenge_rerun_authorized={r.get('frozen_challenge_rerun_authorized')}")
print(f"n0_complete={r.get('n0_complete')}")
print(f"next_action={r.get('next_action')}")
PY

echo "===== N0 RELATION-CONDITIONED MULTILAYER INTERFACE QUALIFICATION COMPLETE ====="
date -Is
