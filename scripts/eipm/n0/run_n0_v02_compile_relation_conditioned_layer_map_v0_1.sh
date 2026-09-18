#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
AUDIT="$WORKDIR/query-semantics-layerwise-audit-v0.1/query_semantics_layerwise_audit.json"
COMPILER="$ROOT/scripts/eipm/n0/compile_n0_v02_relation_conditioned_layer_map_v0_1.py"
MAP="$OUT_ROOT/design/relation_conditioned_layer_map.json"

EXPECTED_AUDIT_SHA256="ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab"

cd "$ROOT"

if [[ -e "$OUT_ROOT" ]]; then
  echo "REFUSING: multilayer-interface design output already exists: $OUT_ROOT" >&2
  exit 2
fi

for required in "$AUDIT" "$COMPILER"; do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$COMPILER"

ACTUAL_AUDIT_SHA256="$(sha256sum "$AUDIT" | awk '{print $1}')"
if [[ "$ACTUAL_AUDIT_SHA256" != "$EXPECTED_AUDIT_SHA256" ]]; then
  echo "STOP: 575804 audit hash drift: $ACTUAL_AUDIT_SHA256" >&2
  exit 5
fi

python - "$AUDIT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if r.get("status")!="COMPLETE_NO_GRADIENT_LAYERWISE_RELATION_SEMANTICS_AUDIT":
    raise SystemExit("575804 audit status drift")
if r.get("gradient_performed") is not False:
    raise SystemExit("575804 gradient contract drift")
if r.get("model_parameters_mutated") is not False:
    raise SystemExit("575804 mutation contract drift")
if r.get("scale_authorized") is not False:
    raise SystemExit("575804 scale contract drift")
print("layerwise_audit_575804_binding=PASS")
PY

mkdir -p "$OUT_ROOT/design"

python "$COMPILER"   --audit "$AUDIT"   --output "$MAP"

echo
echo "===== LAYER MAP HASH ====="
sha256sum "$MAP"

echo
echo "===== LAYER MAP SUMMARY ====="
python - "$MAP" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"schema={r.get('schema')}")
print(f"source_audit_sha256={r.get('source_audit_sha256')}")
print(f"global_candidate_layer_bank={r.get('global_candidate_layer_bank')}")
for relation,info in sorted(r.get("relation_map",{}).items()):
    print(
        f"{relation}: best={info.get('best_token_role_accuracy')} "
        f"exact={info.get('exact_best_layers')} "
        f"candidates={info.get('candidate_layers')}"
    )
print(f"training_authorized={r.get('training_authorized')}")
print(f"scale_authorized={r.get('scale_authorized')}")
print(f"n0_complete={r.get('n0_complete')}")
PY
