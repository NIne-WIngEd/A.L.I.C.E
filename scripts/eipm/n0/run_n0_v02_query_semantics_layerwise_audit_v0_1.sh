#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_QUERY_SEMANTICS_LAYER_AUDIT_DIR:-$WORKDIR/query-semantics-layerwise-audit-v0.1}"
RESULT="$OUT_ROOT/query_semantics_layerwise_audit.json"

CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
PREVIOUS_AUDIT="$WORKDIR/query-semantics-architecture-audit-v0.1/query_semantics_architecture_audit.json"
AUDITOR="$ROOT/scripts/eipm/n0/audit_n0_v02_query_semantic_layers_v0_1.py"
BASE_AUDITOR="$ROOT/scripts/eipm/n0/audit_n0_v02_query_semantic_representations_v0_1.py"
RESEARCH_NOTE="$ROOT/docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md"

EXPECTED_PREVIOUS_AUDIT_SHA256="8bb4aa0f249b763342d78f5bf0f2f960a0bd73fafa67d18f8e0e7342f62f24b6"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

if [[ -e "$OUT_ROOT" ]]; then
  echo "REFUSING: layerwise audit output already exists: $OUT_ROOT" >&2
  exit 2
fi

for required in   "$CONFIG" "$CHECKPOINT" "$TOKENIZER_DIR/tokenizer.json"   "$PREVIOUS_AUDIT" "$AUDITOR" "$BASE_AUDITOR" "$RESEARCH_NOTE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$AUDITOR" "$BASE_AUDITOR"

python - "$PREVIOUS_AUDIT" "$EXPECTED_PREVIOUS_AUDIT_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
expected=sys.argv[2]
h=hashlib.sha256(p.read_bytes()).hexdigest()
if h!=expected:
    raise SystemExit(f"575801 audit result hash drift: {h}")
r=json.loads(p.read_text(encoding="utf-8"))
if r.get("status")!="COMPLETE_NO_GRADIENT_ARCHITECTURE_AUDIT":
    raise SystemExit("575801 audit status drift")
if r.get("gradient_performed") is not False or r.get("model_parameters_mutated") is not False:
    raise SystemExit("575801 no-gradient contract drift")
acc=r.get("representation_role_accuracy",{})
if float(acc.get("token_late_interaction",-1))!=0.6875:
    raise SystemExit("575801 token late-interaction evidence drift")
if float(acc.get("raw_mean_pool",-1))!=0.5625:
    raise SystemExit("575801 raw pool evidence drift")
if float(acc.get("trained_semantic_projection",-1))!=0.5625:
    raise SystemExit("575801 projection evidence drift")
per=r.get("per_relation_role_accuracy",{})
for family in ("causes","supports"):
    if float(per.get(family,{}).get("token_late_interaction",-1))!=0.5:
        raise SystemExit(f"575801 {family} token evidence drift")
print("query_semantics_575801_binding=PASS")
PY

mkdir -p "$OUT_ROOT"

echo "===== N0 QUERY SEMANTICS LAYERWISE AUDIT ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "gradient_performed=false"
echo "model_mutation=false"
echo "heldout_rows_used=false"
echo "scale_authorized=false"

python "$AUDITOR"   --repo-root "$ROOT"   --config "$CONFIG"   --checkpoint "$CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --previous-audit "$PREVIOUS_AUDIT"   --output "$RESULT"

echo
echo "===== LAYERWISE AUDIT HASH ====="
sha256sum "$RESULT"

echo
echo "===== LAYERWISE AUDIT SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"best_token_late_interaction={r.get('best_token_late_interaction')}")
print(f"best_mean_pool={r.get('best_mean_pool')}")
print(f"final_layer_token_late_interaction_accuracy={r.get('final_layer_token_late_interaction_accuracy')}")
print(f"gradient_performed={r.get('gradient_performed')}")
print(f"model_parameters_mutated={r.get('model_parameters_mutated')}")
print(f"scale_authorized={r.get('scale_authorized')}")
print(f"n0_complete={r.get('n0_complete')}")
PY

echo "===== N0 QUERY SEMANTICS LAYERWISE AUDIT COMPLETE ====="
date -Is
