#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
INTERFACE_ROOT="${ALICE_N0_MULTILAYER_INTERFACE_DIR:-$WORKDIR/relation-conditioned-multilayer-interface-v0.1}"
STUDY_ROOT="${ALICE_N0_MULTILAYER_CAUSAL_STUDY_DIR:-$INTERFACE_ROOT/causal-interface-study-v0.1}"
PREP_ROOT="$STUDY_ROOT/preparation"

MAP="$INTERFACE_ROOT/design/relation_conditioned_layer_map.json"
QUAL_RECEIPT="$INTERFACE_ROOT/qualification/runtime_contract_qualification.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_relation_conditioned_multilayer_interface_curriculum_v0_1.py"
PRESERVATION="$ROOT/configs/eipm/n0/n0_v02_relation_conditioned_multilayer_preservation_contract_v0_1.json"
CURRICULUM="$PREP_ROOT/relation_conditioned_multilayer_interface_curriculum.jsonl"
CURRICULUM_MANIFEST="$PREP_ROOT/relation_conditioned_multilayer_interface_curriculum_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080/structured_state.safetensors"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
ORDINARY_REPLAY_CACHE="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
ENDPOINT_REPLAY_CACHE="$WORKDIR/relation-endpoint-repair-v0.2/training/endpoint_repair_semantic_cache.pt"

EXPECTED_MAP_SHA256="ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304"
EXPECTED_QUALIFICATION_SHA256="779b19459fa8ab67bb1c3ed56b3e9cdb1b89b8382f145e1b3d7c4d98312b2b57"
EXPECTED_AUDIT_SHA256="ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab"
EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -e "$STUDY_ROOT" ]]; then
  echo "REFUSING: causal interface study output already exists: $STUDY_ROOT" >&2
  exit 2
fi

for required in \
  "$MAP" "$QUAL_RECEIPT" "$BUILDER" "$PRESERVATION" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" \
  "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE"
do
  [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist" >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$BUILDER"

check_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "STOP: $label hash drift: $actual" >&2
    exit 5
  fi
}

check_sha "$MAP" "$EXPECTED_MAP_SHA256" "compiled layer map"
check_sha "$QUAL_RECEIPT" "$EXPECTED_QUALIFICATION_SHA256" "runtime qualification receipt"
check_sha "$PARENT_GRAPH" "$EXPECTED_PARENT_GRAPH_SHA256" "selected parent graph"

mkdir -p "$PREP_ROOT"
python "$BUILDER" --output "$CURRICULUM" --manifest "$CURRICULUM_MANIFEST"

python - \
  "$ROOT" "$MAP" "$QUAL_RECEIPT" "$PRESERVATION" \
  "$CURRICULUM" "$CURRICULUM_MANIFEST" "$BUILDER" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT" \
  "$PARENT_ADAPTER" "$PARENT_GRAPH" \
  "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE" \
  "$PREP_RECEIPT" "$EXPECTED_MAP_SHA256" "$EXPECTED_QUALIFICATION_SHA256" \
  "$EXPECTED_AUDIT_SHA256" "$EXPECTED_PARENT_GRAPH_SHA256" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

(
    root, layer_map, qualification, preservation,
    curriculum, curriculum_manifest, builder,
    semantic_config, semantic_checkpoint,
    structured_config, structured_checkpoint,
    parent_adapter, parent_graph,
    ordinary_replay_cache, endpoint_replay_cache,
    prep_receipt, expected_map, expected_qualification,
    expected_audit, expected_parent_graph,
) = sys.argv[1:]

def sha(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=root, text=True
).strip()

layer_map_obj = json.loads(Path(layer_map).read_text(encoding="utf-8"))
if sha(layer_map) != expected_map:
    raise SystemExit("layer-map binding drift")
if layer_map_obj.get("source_audit_sha256") != expected_audit:
    raise SystemExit("source layerwise audit binding drift")
if layer_map_obj.get("training_authorized") is not False:
    raise SystemExit("layer map prematurely authorizes training")

qual = json.loads(Path(qualification).read_text(encoding="utf-8"))
if sha(qualification) != expected_qualification:
    raise SystemExit("qualification receipt binding drift")
if qual.get("status") != "PASS_EXACT_MAP_NO_GRADIENT_RUNTIME_CONTRACT":
    raise SystemExit("runtime qualification did not pass")
for key in (
    "gradient_performed",
    "optimizer_created",
    "model_parameters_mutated",
    "gpu_required",
    "heldout_rows_used",
    "frozen_challenge_rows_used",
    "private_identity_data_used",
    "training_authorized",
    "scale_authorized",
    "heldout_opening_authorized",
    "frozen_challenge_rerun_authorized",
    "n0_complete",
):
    if qual.get(key) is not False:
        raise SystemExit(f"qualification governance drift: {key}")
if qual.get("parameter_state_exactly_unchanged") is not True:
    raise SystemExit("runtime qualification mutated interface state")

contract = json.loads(Path(preservation).read_text(encoding="utf-8"))
if contract.get("schema") != (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-preservation-contract.v0.1"
):
    raise SystemExit("preservation contract schema drift")
for key in (
    "training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "new_gpu_run_authorized",
):
    if contract.get(key) is not False:
        raise SystemExit(f"preservation contract authorization drift: {key}")
if contract["lineage"]["selected_parent_graph_sha256"] != expected_parent_graph:
    raise SystemExit("preservation parent graph binding drift")
if contract["numerical_tolerance_policy"]["candidate_result_may_not_define_tolerances"] is not True:
    raise SystemExit("candidate-dependent tolerance selection leaked")
if contract["numerical_tolerance_policy"]["tolerances_must_be_frozen_before_candidate_training"] is not True:
    raise SystemExit("preservation tolerance freeze missing")

manifest = json.loads(Path(curriculum_manifest).read_text(encoding="utf-8"))
if manifest.get("schema") != (
    "alice.eipm.n0.v02-relation-conditioned-multilayer-interface-curriculum.v0.1"
):
    raise SystemExit("causal curriculum manifest schema drift")
if manifest.get("rows") != 720 or manifest.get("quad_count") != 180:
    raise SystemExit("causal curriculum size drift")
if manifest.get("compiled_sha256") != sha(curriculum):
    raise SystemExit("causal curriculum hash drift")
for key in (
    "interface_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "dev_split_training_authorized",
    "test_split_training_authorized",
    "test_split_opening_authorized",
):
    if manifest.get(key) is not False:
        raise SystemExit(f"curriculum authorization drift: {key}")
if manifest["factorial_design"]["target_must_flip_with_query_role"] is not True:
    raise SystemExit("query-role causal control missing")
if manifest["factorial_design"]["target_must_flip_with_edge_direction"] is not True:
    raise SystemExit("edge-direction causal control missing")

artifacts = {
    "layer_map": layer_map,
    "qualification_receipt": qualification,
    "preservation_contract": preservation,
    "curriculum": curriculum,
    "curriculum_manifest": curriculum_manifest,
    "curriculum_builder": builder,
    "semantic_config": semantic_config,
    "semantic_checkpoint": semantic_checkpoint,
    "structured_config": structured_config,
    "structured_checkpoint": structured_checkpoint,
    "parent_adapter": parent_adapter,
    "parent_graph": parent_graph,
    "ordinary_replay_cache": ordinary_replay_cache,
    "endpoint_replay_cache": endpoint_replay_cache,
}

receipt = {
    "schema": "alice.eipm.n0.v02-relation-conditioned-multilayer-causal-study-preparation.v0.1",
    "status": "PASS_CAUSAL_STUDY_INPUTS_FROZEN_TRAINING_DECISION_STILL_REQUIRED",
    "git_revision": head,
    "artifact_sha256": {name: sha(path) for name, path in artifacts.items()},
    "runtime_qualification_receipt_sha256": expected_qualification,
    "compiled_layer_map_sha256": expected_map,
    "source_layerwise_audit_sha256": expected_audit,
    "parent_graph_sha256": expected_parent_graph,
    "curriculum_rows": manifest["rows"],
    "curriculum_quad_count": manifest["quad_count"],
    "curriculum_relations": manifest["relations"],
    "causal_design": manifest["factorial_design"],
    "preservation_lanes": sorted(contract["preservation_lanes"]),
    "candidate_result_may_define_tolerances": False,
    "tolerance_calibration_required_before_training": True,
    "semantic_backbone_trainable": False,
    "structured_state_trainable": False,
    "parent_adapter_trainable": False,
    "parent_graph_trainable": False,
    "interface_training_authorized": False,
    "optimizer_authorized": False,
    "gradient_authorized": False,
    "new_gpu_run_authorized": False,
    "heldout_opening_authorized": False,
    "frozen_challenge_rerun_authorized": False,
    "scale_authorized": False,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "production_promotion_authorized": False,
    "n0_complete": False,
    "next_action": (
        "review frozen causal curriculum and preservation bindings, then make one "
        "explicit architecture-training decision before implementing any optimizer"
    ),
}
Path(prep_receipt).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo
echo "===== N0 RELATION-CONDITIONED MULTILAYER CAUSAL STUDY PREPARATION ====="
echo "source_revision=$(git rev-parse HEAD)"
sha256sum "$CURRICULUM" "$CURRICULUM_MANIFEST" "$PRESERVATION" "$PREP_RECEIPT"
python - "$PREP_RECEIPT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for key in (
    "status",
    "git_revision",
    "curriculum_rows",
    "curriculum_quad_count",
    "preservation_lanes",
    "interface_training_authorized",
    "optimizer_authorized",
    "gradient_authorized",
    "new_gpu_run_authorized",
    "heldout_opening_authorized",
    "scale_authorized",
    "n0_complete",
    "next_action",
):
    print(f"{key}={r.get(key)}")
PY
