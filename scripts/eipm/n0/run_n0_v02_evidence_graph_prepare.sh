#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="${N0_V02_GRAPH_PREP_ROOT:-$WORKDIR/evidence-graph-curriculum-v0.1}"
CURRICULUM="$OUT_ROOT/evidence_graph_curriculum.jsonl"
MANIFEST="$OUT_ROOT/evidence_graph_curriculum_manifest.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m py_compile \
  "$ROOT/src/alice_personality/n0/evidence_graph.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_data.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph_objectives.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_evidence_graph_curriculum.py"

python -m pytest -q \
  "$ROOT/tests/eipm/test_n0_evidence_graph.py" \
  "$ROOT/tests/eipm/test_n0_evidence_graph_data.py" \
  "$ROOT/tests/eipm/test_n0_evidence_graph_objectives.py" \
  "$ROOT/tests/eipm/test_n0_evidence_graph_curriculum.py"

mkdir -p "$OUT_ROOT"
python "$ROOT/scripts/eipm/n0/build_n0_v02_evidence_graph_curriculum.py" \
  --output "$CURRICULUM" \
  --manifest "$MANIFEST"

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("compiled_rows") != 240:
    raise SystemExit("evidence graph curriculum row count mismatch")
if manifest.get("split_counts") != {"dev": 60, "train": 180}:
    raise SystemExit("evidence graph curriculum split mismatch")
if len(manifest.get("families", [])) != 10:
    raise SystemExit("evidence graph curriculum family coverage mismatch")
if manifest.get("private_identity_content") is not False:
    raise SystemExit("private identity content is forbidden in N0 graph preparation")
if manifest.get("hard_parameter_ceiling", "missing") is not None:
    raise SystemExit("graph preparation may not impose a hard parameter ceiling")
print("evidence_graph_prepare_pass=true")
print("compiled_rows=240 train_rows=180 dev_rows=60 families=10")
print("soft_evidence_targets=true")
print("relation_counterfactual_training=true")
print("historical_query_coverage=true")
print("hard_parameter_ceiling=none")
print("gradient_performed=false")
print("gpu_required=false")
PY
