#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TEACHER_DIR="${N0_V02_TEACHER_DIR:-$WORKDIR/teacher-bank-v0.5}"
TEACHER_REGISTRY="$TEACHER_DIR/n0_v02_teacher_bank_v0.5.runtime.json"
TEACHER_AUDIT="$TEACHER_DIR/teacher-bank-v0.5-audit.json"
OUT_ROOT="${N0_V02_STRUCTURED_PREP_ROOT:-$WORKDIR/structured-state-curriculum-v0.2}"
OUTPUT="$OUT_ROOT/structured_state_curriculum.jsonl"
MANIFEST="$OUT_ROOT/structured_state_curriculum_manifest.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in "$TEACHER_REGISTRY" "$TEACHER_AUDIT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required teacher artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing structured-state preparation: $OUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"

python -m py_compile \
  "$ROOT/src/alice_personality/n0/structured_state.py" \
  "$ROOT/src/alice_personality/n0/structured_state_objectives.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_structured_state_curriculum.py"

python -m pytest -q \
  "$ROOT/tests/eipm/test_n0_structured_state.py" \
  "$ROOT/tests/eipm/test_n0_structured_state_objectives.py" \
  "$ROOT/tests/eipm/test_n0_v02_design.py"

python "$ROOT/scripts/eipm/n0/build_n0_v02_structured_state_curriculum.py" \
  --teacher-registry "$TEACHER_REGISTRY" \
  --teacher-audit "$TEACHER_AUDIT" \
  --output "$OUTPUT" \
  --manifest "$MANIFEST"

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != "alice.eipm.n0.v02-structured-state-curriculum.v0.2":
    raise SystemExit("structured-state manifest schema mismatch")
if manifest.get("status") != "COMPILED_NOT_ACTIVATED":
    raise SystemExit("structured-state curriculum unexpectedly activated")
if int(manifest.get("source_registered_rows", -1)) != 1020:
    raise SystemExit("structured-state compiler did not bind the governed 1020-row teacher bank")
if int(manifest.get("competency_count", -1)) != 51:
    raise SystemExit("structured-state compiler lost teacher competency coverage")
if manifest.get("semantic_target_mode") != "context_candidate_pair":
    raise SystemExit("structured-state semantic target must be the context/candidate pair")
if manifest.get("rationale_target_separate") is not True:
    raise SystemExit("structured-state rationale target must remain separate")
labels = manifest.get("compatibility_label_counts", {})
if set(labels) != {"0", "1"} or min(int(value) for value in labels.values()) < 1:
    raise SystemExit("structured-state curriculum lacks positive/negative compatibility coverage")
if manifest.get("text_generated_by_compiler") is not False:
    raise SystemExit("structured-state compiler must not generate new text")
if manifest.get("activation_authorized") is not False:
    raise SystemExit("structured-state preparation must not authorize gradient activation")
print("structured_state_prepare_pass=true")
print(f"compiled_rows={manifest['compiled_rows']}")
print(f"compiled_split_counts={manifest['compiled_split_counts']}")
print(f"compatibility_label_counts={manifest['compatibility_label_counts']}")
print("semantic_and_rationale_targets_separate=true")
PY

echo "===== N0 V0.2 STRUCTURED-STATE PREPARATION COMPLETE ====="
echo "manifest=$MANIFEST"
echo "gradient_performed=false"
echo "gpu_required=false"
