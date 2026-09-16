#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/cross-context-fusion-curriculum-v0.2"
CURRICULUM="$OUT_ROOT/cross_context_fusion_curriculum.jsonl"
MANIFEST="$OUT_ROOT/cross_context_fusion_manifest.json"
RECEIPT="$OUT_ROOT/preparation_receipt.json"
CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
TRAINING_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_training_v0.1.json"
AUDIT_DOC="$ROOT/docs/eipm/EIPM_CAPABILITY_LIMIT_AUDIT_2026-09-16.md"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

python -m py_compile \
  "$ROOT/src/alice_personality/n0/v02_model.py" \
  "$ROOT/src/alice_personality/n0/structured_state.py" \
  "$ROOT/src/alice_personality/n0/evidence_view_adapter.py" \
  "$ROOT/src/alice_personality/n0/evidence_graph.py" \
  "$ROOT/src/alice_personality/n0/cross_context_fusion.py" \
  "$ROOT/src/alice_personality/n0/cross_context_fusion_objectives.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum.py" \
  "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum_v0_2.py" \
  "$ROOT/scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py"

pytest -q \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion.py" \
  "$ROOT/tests/eipm/test_n0_cross_context_fusion_objectives.py" \
  "$ROOT/tests/eipm/test_n0_no_accidental_capability_ceilings.py"

mkdir -p "$OUT_ROOT"
python "$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum_v0_2.py" \
  --output "$CURRICULUM" \
  --manifest "$MANIFEST"

python - "$CONFIG" "$TRAINING_CONFIG" "$MANIFEST" "$CURRICULUM" "$AUDIT_DOC" "$RECEIPT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

config_path = Path(sys.argv[1])
training_config_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
curriculum_path = Path(sys.argv[4])
audit_path = Path(sys.argv[5])
receipt_path = Path(sys.argv[6])
config = json.loads(config_path.read_text())
training_config = json.loads(training_config_path.read_text())
manifest = json.loads(manifest_path.read_text())

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if config.get("status") != "FULL_SCALE_FRONTIER_MECHANICS_READY_NO_GRADIENT_AUTHORIZED":
    raise SystemExit("fusion config status drift")
policy = config.get("architecture_policy", {})
if policy.get("full_scale_model") is not True or policy.get("reduced_capability_pilot") is not False:
    raise SystemExit("fusion must be governed as the full-scale model, not a reduced pilot")
if policy.get("generic_fallback_architecture") is not False:
    raise SystemExit("generic fallback fusion architecture is forbidden")
if policy.get("finite_checkpoint_tensor_shapes_are_migratable_not_permanent_product_limits") is not True:
    raise SystemExit("finite checkpoint shapes must remain migratable")
arch = config.get("architecture", {})
if arch.get("family") != "multi_stream_self_refinement_plus_gated_bidirectional_cross_attention":
    raise SystemExit("frontier fusion architecture family drift")
if arch.get("bidirectional_cross_attention") is not True or arch.get("query_conditioned_cross_view_gating") is not True:
    raise SystemExit("frontier cross-view exchange mechanics missing")
if arch.get("view_count_ceiling") is not None:
    raise SystemExit("unexpected fusion view-count ceiling")
if arch.get("checkpoint_view_vocabulary_migratable") is not True:
    raise SystemExit("fusion checkpoint view vocabulary must remain migratable")
if arch.get("hard_parameter_ceiling") is not None:
    raise SystemExit("unexpected fusion parameter ceiling")
if config.get("gradient_authorization", {}).get("authorized") is not False:
    raise SystemExit("base fusion mechanics config unexpectedly authorizes gradient")
if training_config.get("status") != "CONDITIONALLY_AUTHORIZED_AFTER_SAME_REVISION_PREP_PASS":
    raise SystemExit("fusion training config status drift")
if training_config.get("authorization", {}).get("public_fusion_gradient_authorized") is not True:
    raise SystemExit("conditional public fusion training authorization missing")
if training_config.get("authorization", {}).get("private_identity_gradient") is not False:
    raise SystemExit("private identity gradient boundary drift")
if training_config.get("capacity_policy", {}).get("hard_parameter_ceiling") is not None:
    raise SystemExit("unexpected training parameter ceiling")
if manifest.get("schema") != "alice.eipm.n0.v02-cross-context-fusion-curriculum.v0.2":
    raise SystemExit("fusion curriculum schema mismatch")
if manifest.get("compiled_sha256") != sha(curriculum_path):
    raise SystemExit("fusion curriculum hash mismatch")
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
    raise SystemExit("compiled curriculum itself must remain non-authoritative; activation comes from training config + this receipt")
if not audit_path.is_file():
    raise SystemExit("capability-limit audit document missing")

git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
receipt = {
    "schema": "alice.eipm.n0.v02-cross-context-fusion-preparation-receipt.v0.1",
    "status": "PASS",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": git_revision,
    "fusion_config_sha256": sha(config_path),
    "training_config_sha256": sha(training_config_path),
    "curriculum_sha256": sha(curriculum_path),
    "curriculum_manifest_sha256": sha(manifest_path),
    "capability_limit_audit_sha256": sha(audit_path),
    "full_scale_model": True,
    "reduced_capability_pilot": False,
    "generic_fallback_architecture": False,
    "current_instantiated_public_views": 3,
    "view_count_ceiling": None,
    "field_node_edge_runtime_ceilings": None,
    "semantic_token_level_representation": True,
    "finite_checkpoint_vocabularies_migratable": True,
    "no_accidental_capability_ceilings_gate": True,
    "gradient_performed": False,
    "gpu_required": False,
    "private_identity_data": False,
    "private_identity_gradient": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print("cross_context_fusion_prepare_pass=true")
print("fusion_architecture=full_scale_frontier_multi_stream_gated_bidirectional_cross_attention")
print("current_instantiated_public_views=3")
print("view_count_ceiling=none")
print("field_node_edge_runtime_ceilings=none")
print("semantic_token_level_representation=true")
print("finite_checkpoint_vocabularies_migratable=true")
print("no_accidental_capability_ceilings_gate=true")
print("reduced_capability_pilot=false")
print("generic_fallback_architecture=false")
print("fusion_curriculum_schema=v0.2")
print("fusion_rows=320 train_rows=240 dev_rows=80 families=10")
print("data_origin=deterministic_public_synthetic_template")
print("source_authority=public_synthetic_training_only")
print("hard_parameter_ceiling=none")
print("gradient_performed=false")
print("gpu_required=false")
print(f"preparation_receipt={receipt_path}")
PY
