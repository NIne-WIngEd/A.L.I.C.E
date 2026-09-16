#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/cross-context-fusion-curriculum-v0.2"
CURRICULUM="$OUT_ROOT/cross_context_fusion_curriculum.jsonl"
MANIFEST="$OUT_ROOT/cross_context_fusion_manifest.json"
CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

python -m py_compile \
  "$ROOT/src/alice_personality/n0/cross_context_fusion.py" \
  "$ROOT/src/alice_personality/n0/cross_context_fusion_objectives.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum_v0_2.py"

pytest -q \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_objectives.py"

mkdir -p "$OUT_ROOT"
python "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum_v0_2.py" \
  --output "$CURRICULUM" \
  --manifest "$MANIFEST"

python - "$CONFIG" "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
if config.get("status") != "FULL_SCALE_FRONTIER_MECHANICS_READY_NO_GRADIENT_AUTHORIZED":
    raise SystemExit("fusion config status drift")
policy = config.get("architecture_policy", {})
if policy.get("full_scale_model") is not True or policy.get("reduced_capability_pilot") is not False:
    raise SystemExit("fusion must be governed as the full-scale model, not a reduced pilot")
if policy.get("generic_fallback_architecture") is not False:
    raise SystemExit("generic fallback fusion architecture is forbidden")
arch = config.get("architecture", {})
if arch.get("family") != "tri_stream_self_refinement_plus_gated_bidirectional_cross_attention":
    raise SystemExit("frontier fusion architecture family drift")
if arch.get("bidirectional_cross_attention") is not True or arch.get("query_conditioned_cross_view_gating") is not True:
    raise SystemExit("frontier cross-view exchange mechanics missing")
if arch.get("hard_parameter_ceiling") is not None:
    raise SystemExit("unexpected fusion parameter ceiling")
if config.get("gradient_authorization", {}).get("authorized") is not False:
    raise SystemExit("fusion gradient unexpectedly authorized")
if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-curriculum.v0.2":
    raise SystemExit("fusion curriculum schema mismatch")
if manifest.get("rows") != 320 or manifest.get("train_rows") != 240 or manifest.get("dev_rows") != 80:
    raise SystemExit("fusion curriculum split drift")
if manifest.get("family_count") != 10:
    raise SystemExit("fusion family coverage drift")
if manifest.get("data_origin") != "deterministic_public_synthetic_template":
    raise SystemExit("fusion data origin drift")
if manifest.get("source_authority") != "public_synthetic_training_only":
    raise SystemExit("fusion source authority drift")
if manifest.get("generated_text") is not True:
    raise SystemExit("fusion generated-text provenance drift")
if manifest.get("identity_authority") is not False:
    raise SystemExit("synthetic fusion data cannot be identity authority")
if manifest.get("private_identity_content") is not False:
    raise SystemExit("fusion curriculum crossed private boundary")
if manifest.get("training_authorized") is not False:
    raise SystemExit("fusion curriculum unexpectedly authorizes gradient")
print("cross_context_fusion_prepare_pass=true")
print("fusion_architecture=full_scale_frontier_tri_stream_gated_bidirectional_cross_attention")
print("reduced_capability_pilot=false")
print("generic_fallback_architecture=false")
print("fusion_curriculum_schema=v0.2")
print("fusion_rows=320 train_rows=240 dev_rows=80 families=10")
print("data_origin=deterministic_public_synthetic_template")
print("source_authority=public_synthetic_training_only")
print("hard_parameter_ceiling=none")
print("gradient_performed=false")
print("gpu_required=false")
PY
