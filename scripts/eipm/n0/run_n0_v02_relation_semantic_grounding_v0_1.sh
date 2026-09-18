#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_RELATION_SEMANTIC_DIR:-$WORKDIR/relation-semantic-grounding-v0.1}"
PREP_ROOT="$OUT_ROOT/preparation"
TRAIN_ROOT="$OUT_ROOT/training"

TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
PARENT_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
PARENT_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
ORDINARY_REPLAY_CACHE="$WORKDIR/evidence-graph-pilot-v0.1/graph_semantic_cache.pt"
ENDPOINT_REPLAY_CACHE="$WORKDIR/relation-endpoint-repair-v0.2/training/endpoint_repair_semantic_cache.pt"
LOCALIZATION_RESULT="$WORKDIR/missing-evidence-path-localization-v0.1/path_localization_result.json"
ARBITRATION_RESULT="$WORKDIR/downstream-causal-arbitration-v0.1-run-575760/arbitration_result.json"

BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_relation_semantic_grounding_curriculum_v0_1.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_relation_semantic_grounding_v0_1.py"
CURRICULUM="$PREP_ROOT/relation_semantic_grounding_curriculum.jsonl"
MANIFEST="$PREP_ROOT/relation_semantic_grounding_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_LOCALIZATION_SHA256="1733bf1a5fdc4f979beab6d5198bfb6a77a0e17b4fc96adc134dafd03dd5ebbe"
EXPECTED_ARBITRATION_SHA256="430e1b5c51f6157c0d40d95728f278b5400df13fb6d30e822bb197ab9fe110a5"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUT_ROOT" ]]; then
  echo "REFUSING: relation-semantic output already exists: $OUT_ROOT" >&2
  exit 2
fi

for required in   "$BUILDER" "$TRAINER"   "$TOKENIZER_DIR/tokenizer.json"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"   "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors"   "$PARENT_ADAPTER" "$PARENT_GRAPH"   "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE"   "$LOCALIZATION_RESULT" "$ARBITRATION_RESULT"
do
  if [[ ! -f "$required" ]]; then
    echo "MISSING: $required" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository changes exist." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$BUILDER" "$TRAINER"

mkdir -p "$PREP_ROOT"
python "$BUILDER" --output "$CURRICULUM" --manifest "$MANIFEST"

python -   "$ROOT" "$CURRICULUM" "$MANIFEST"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"   "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG"   "$STRUCTURED_CHECKPOINT/structured_state.safetensors"   "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE"   "$TRAINER" "$LOCALIZATION_RESULT" "$ARBITRATION_RESULT" "$PREP_RECEIPT"   "$EXPECTED_PARENT_GRAPH_SHA256" "$EXPECTED_LOCALIZATION_SHA256" "$EXPECTED_ARBITRATION_SHA256" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(
 root,curriculum,manifest,semantic_config,semantic_checkpoint,tokenizer,
 structured_config,structured_checkpoint,parent_adapter,parent_graph,
 ordinary_replay_cache,endpoint_replay_cache,trainer,localization_result,
 arbitration_result,prep_path,expected_parent,expected_localization,expected_arbitration
)=sys.argv[1:]
paths={name:Path(value).resolve() for name,value in {
 "curriculum":curriculum,
 "manifest":manifest,
 "semantic_config":semantic_config,
 "semantic_checkpoint":semantic_checkpoint,
 "tokenizer":tokenizer,
 "structured_config":structured_config,
 "structured_checkpoint":structured_checkpoint,
 "parent_adapter":parent_adapter,
 "parent_graph":parent_graph,
 "ordinary_replay_cache":ordinary_replay_cache,
 "endpoint_replay_cache":endpoint_replay_cache,
 "trainer":trainer,
}.items()}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
manifest_obj=json.loads(Path(manifest).read_text(encoding="utf-8"))
if manifest_obj.get("schema")!="alice.eipm.n0.v02-relation-semantic-grounding-curriculum.v0.1":
    raise SystemExit("relation-semantic manifest schema drift")
if manifest_obj.get("compiled_sha256")!=sha(Path(curriculum)):
    raise SystemExit("relation-semantic curriculum hash drift")
for key in (
    "frozen_latent_challenge_rows_used_for_training",
    "missing_evidence_localization_rows_used_for_training",
    "endpoint_repair_v0_2_heldout_rows_reused",
):
    if manifest_obj.get(key) is not False:
        raise SystemExit(f"forbidden data reuse: {key}")
if manifest_obj.get("query_names_endpoint_role_explicitly") is not False:
    raise SystemExit("semantic grounding curriculum leaked endpoint-role wording")

if sha(Path(parent_graph))!=expected_parent:
    raise SystemExit("selected step-200 parent graph hash drift")
if sha(Path(localization_result))!=expected_localization:
    raise SystemExit("575795 localization result hash drift")
if sha(Path(arbitration_result))!=expected_arbitration:
    raise SystemExit("575792 arbitration result hash drift")

loc=json.loads(Path(localization_result).read_text(encoding="utf-8"))
if loc.get("status")!="COMPLETE_DIAGNOSTIC_ONLY":
    raise SystemExit("575795 localization status drift")
if loc.get("localization")!="EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE":
    raise SystemExit("semantic grounding repair is not bound to the localized graph-selection defect")
if loc.get("dataset",{}).get("frozen_challenge_rows_reused") is not False:
    raise SystemExit("575795 lineage drift")
if loc.get("interpretation_contract",{}).get("gradient_performed") is not False:
    raise SystemExit("575795 unexpectedly performed gradient")

arb=json.loads(Path(arbitration_result).read_text(encoding="utf-8"))
if arb.get("classification")!="IMPROVEMENT":
    raise SystemExit("selected parent lacks downstream causal improvement evidence")
if arb.get("candidate_graph",{}).get("sha256") not in (None, expected_parent):
    raise SystemExit("arbitration candidate graph drift")
if arb.get("candidate_graph_sha256") not in (None, expected_parent):
    raise SystemExit("arbitration candidate graph drift")

receipt={
 "schema":"alice.eipm.n0.v02-relation-semantic-grounding-preparation.v0.1",
 "status":"PASS_RELATION_SEMANTIC_GROUNDING_READY_FOR_GPU",
 "git_revision":head,
 "artifact_sha256":{name:sha(path) for name,path in paths.items()},
 "trigger_localization_result_sha256":expected_localization,
 "trigger_localization":"EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE",
 "downstream_arbitration_result_sha256":expected_arbitration,
 "parent_graph_sha256":expected_parent,
 "parent_graph_downstream_classification":"IMPROVEMENT",
 "training_scope":"query_conditioned_source_target_relation_read_only",
 "message_passing_trainable":False,
 "conflict_pool_trainable":False,
 "adapter_trainable":False,
 "semantic_parent_trainable":False,
 "structured_parent_trainable":False,
 "endpoint_repair_v0_2_heldout_test_reused":False,
 "frozen_latent_challenge_rows_used_for_training":False,
 "missing_evidence_localization_rows_used_for_training":False,
 "ordinary_replay_required":True,
 "endpoint_role_replay_required":True,
 "hard_parameter_ceiling":None,
 "hard_parameter_floor":None,
 "scale_authorized":False,
 "private_identity_data":False,
 "private_identity_gradient":False,
 "production_promotion_authorized":False,
 "n0_complete":False,
}
Path(prep_path).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

echo "===== N0 RELATION-SEMANTIC GROUNDING REPAIR ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "parent_graph=$PARENT_GRAPH"
echo "localization_result=$LOCALIZATION_RESULT"
echo "curriculum=$CURRICULUM"
echo "training_enabled=true"
echo "trainable_scope=query_conditioned_source_target_relation_read_only"
echo "frozen_challenge_rows_used_for_training=false"
echo "localization_rows_used_for_training=false"
echo "endpoint_heldout_test_reused=false"
echo "scale_authorized=false"
echo "automatic_rerun=false"

python "$TRAINER"   --repo-root "$ROOT"   --prep-receipt "$PREP_RECEIPT"   --curriculum "$CURRICULUM"   --manifest "$MANIFEST"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED_CHECKPOINT"   --parent-adapter "$PARENT_ADAPTER"   --parent-graph "$PARENT_GRAPH"   --ordinary-replay-cache "$ORDINARY_REPLAY_CACHE"   --endpoint-replay-cache "$ENDPOINT_REPLAY_CACHE"   --output-dir "$TRAIN_ROOT"

echo
echo "===== RELATION-SEMANTIC OUTPUT HASHES ====="
sha256sum "$CURRICULUM" "$MANIFEST" "$PREP_RECEIPT" "$TRAIN_ROOT/result.json"

echo
echo "===== RELATION-SEMANTIC DECISION ====="
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"selected_checkpoint={r.get('selected_checkpoint')}")
print(f"selected_graph_sha256={r.get('selected_graph_sha256')}")
test=r.get("semantic_heldout_test_metrics",{})
print(f"heldout_pair_accuracy={test.get('pair_accuracy')}")
print(f"heldout_family_min_pair_accuracy={test.get('family_min_pair_accuracy')}")
print(f"heldout_row_accuracy={test.get('row_accuracy')}")
print(f"heldout_mean_graph_target_margin={test.get('mean_graph_target_margin')}")
print(f"heldout_role_summary={r.get('semantic_heldout_role_summary')}")
print(f"endpoint_role_dev_preservation={r.get('endpoint_role_dev_preservation_metrics')}")
print(f"ordinary_replay_preservation={r.get('ordinary_replay_preservation_metrics')}")
print(f"next_action={r.get('next_action')}")
print("scale_authorized=false")
print("n0_complete=false")
PY

echo "===== N0 RELATION-SEMANTIC GROUNDING REPAIR COMPLETE ====="
date -Is
