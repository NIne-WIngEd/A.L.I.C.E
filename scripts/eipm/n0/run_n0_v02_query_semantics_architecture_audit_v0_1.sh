#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_QUERY_SEMANTICS_AUDIT_DIR:-$WORKDIR/query-semantics-architecture-audit-v0.1}"
RESULT="$OUT_ROOT/query_semantics_architecture_audit.json"

CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
AUDITOR="$ROOT/scripts/eipm/n0/audit_n0_v02_query_semantic_representations_v0_1.py"
RESEARCH_NOTE="$ROOT/docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md"
QRR_RESULT="$WORKDIR/query-relation-role-router-v0.3-retry-575799/training/result.json"

EXPECTED_QRR_SHA256="0f7917446d9f23c5d58e1eb1e1e5641a8df38da1a4730926a90d73ec044305a0"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

if [[ -e "$OUT_ROOT" ]]; then
  echo "REFUSING: audit output already exists: $OUT_ROOT" >&2
  exit 2
fi

for required in   "$CONFIG" "$CHECKPOINT" "$TOKENIZER_DIR/tokenizer.json"   "$AUDITOR" "$RESEARCH_NOTE" "$QRR_RESULT"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$AUDITOR"

python - "$QRR_RESULT" "$EXPECTED_QRR_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
expected=sys.argv[2]
h=hashlib.sha256(p.read_bytes()).hexdigest()
if h!=expected:
    raise SystemExit(f"575800 QRR result hash drift: {h}")
r=json.loads(p.read_text(encoding="utf-8"))
if r.get("status")!="FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE":
    raise SystemExit("architecture audit requires preserved 575800 dev failure")
if r.get("next_action")!="architecture_level_audit_required_before_any_further_gradient_run":
    raise SystemExit("575800 anti-loop next action drift")
if r.get("semantic_heldout_test_evaluated") is not False:
    raise SystemExit("575800 heldout exposure drift")
if r.get("semantic_heldout_rows_opened_after_training") is not False:
    raise SystemExit("575800 heldout exposure drift")
if r.get("eligible_candidate_keys")!=[]:
    raise SystemExit("575800 eligibility drift")
if r.get("parent_graph_parameters_exactly_unchanged") is not True:
    raise SystemExit("575800 parent immutability drift")
if r.get("scale_authorized") is not False:
    raise SystemExit("575800 scale governance drift")
print("qrr_575800_binding=PASS")
PY

mkdir -p "$OUT_ROOT"

echo "===== N0 QUERY SEMANTICS ARCHITECTURE AUDIT ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "gradient_performed=false"
echo "model_mutation=false"
echo "heldout_rows_used=false"
echo "scale_authorized=false"

python "$AUDITOR"   --repo-root "$ROOT"   --config "$CONFIG"   --checkpoint "$CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --output "$RESULT"

echo
echo "===== AUDIT HASH ====="
sha256sum "$RESULT"

echo
echo "===== AUDIT SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"representation_role_accuracy={r.get('representation_role_accuracy')}")
print(f"mean_opposite_role_pair_distance={r.get('mean_opposite_role_pair_distance')}")
print(f"gradient_performed={r.get('gradient_performed')}")
print(f"model_parameters_mutated={r.get('model_parameters_mutated')}")
print(f"scale_authorized={r.get('scale_authorized')}")
print(f"n0_complete={r.get('n0_complete')}")
PY

echo "===== N0 QUERY SEMANTICS ARCHITECTURE AUDIT COMPLETE ====="
date -Is
