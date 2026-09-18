#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_RELATION_SEMANTIC_ROLE_RESIDUAL_DIR:-$WORKDIR/relation-semantic-role-residual-v0.2}"
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
FAILED_SEMANTIC_RESULT="$WORKDIR/relation-semantic-grounding-v0.1/training/result.json"

BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_relation_semantic_role_residual_curriculum_v0_2.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_relation_semantic_role_residual_v0_2.py"
SEMANTIC_ROLE_SOURCE="$ROOT/src/alice_personality/n0/evidence_graph_semantic_role_residual.py"
CURRICULUM="$PREP_ROOT/relation_semantic_role_residual_curriculum.jsonl"
MANIFEST="$PREP_ROOT/relation_semantic_role_residual_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

EXPECTED_PARENT_GRAPH_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_LOCALIZATION_SHA256="1733bf1a5fdc4f979beab6d5198bfb6a77a0e17b4fc96adc134dafd03dd5ebbe"
EXPECTED_FAILED_SEMANTIC_SHA256="cfd2a6aee3ba2e441363700cf0d57917cf7cf16c3c81b2a789b5c1ecf3614980"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUT_ROOT" ]]; then
  echo "REFUSING: role-residual output already exists: $OUT_ROOT" >&2
  exit 2
fi

for required in   "$BUILDER" "$TRAINER" "$SEMANTIC_ROLE_SOURCE"   "$TOKENIZER_DIR/tokenizer.json"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"   "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors"   "$PARENT_ADAPTER" "$PARENT_GRAPH"   "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE"   "$LOCALIZATION_RESULT" "$FAILED_SEMANTIC_RESULT"
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

python -m py_compile "$BUILDER" "$TRAINER" "$SEMANTIC_ROLE_SOURCE"

mkdir -p "$PREP_ROOT"
python "$BUILDER" --output "$CURRICULUM" --manifest "$MANIFEST"

python -   "$ROOT" "$CURRICULUM" "$MANIFEST"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"   "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG"   "$STRUCTURED_CHECKPOINT/structured_state.safetensors"   "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE"   "$BUILDER" "$TRAINER" "$SEMANTIC_ROLE_SOURCE"   "$LOCALIZATION_RESULT" "$FAILED_SEMANTIC_RESULT" "$PREP_RECEIPT"   "$EXPECTED_PARENT_GRAPH_SHA256" "$EXPECTED_LOCALIZATION_SHA256" "$EXPECTED_FAILED_SEMANTIC_SHA256" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path

(
 root,curriculum,manifest,semantic_config,semantic_checkpoint,tokenizer,
 structured_config,structured_checkpoint,parent_adapter,parent_graph,
 ordinary_replay_cache,endpoint_replay_cache,builder,trainer,semantic_role_source,
 localization_result,failed_semantic_result,prep_path,
 expected_parent,expected_localization,expected_failed_semantic
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
 "builder":builder,
 "trainer":trainer,
 "semantic_role_source":semantic_role_source,
}.items()}

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()

manifest_obj=json.loads(Path(manifest).read_text(encoding="utf-8"))
if manifest_obj.get("schema")!="alice.eipm.n0.v02-relation-semantic-role-residual-curriculum.v0.2":
    raise SystemExit("role-residual manifest schema drift")
if manifest_obj.get("compiled_sha256")!=sha(Path(curriculum)):
    raise SystemExit("role-residual curriculum hash drift")
for key in (
    "frozen_latent_challenge_rows_used_for_training",
    "missing_evidence_localization_rows_used_for_training",
    "relation_semantic_v0_1_rows_reused",
    "endpoint_repair_v0_2_heldout_rows_reused",
):
    if manifest_obj.get(key) is not False:
        raise SystemExit(f"forbidden data reuse: {key}")
if manifest_obj.get("query_names_endpoint_role_explicitly") is not False:
    raise SystemExit("role-residual curriculum leaked endpoint-role wording")
if manifest_obj.get("hard_parameter_ceiling") is not None:
    raise SystemExit("hard parameter ceiling drift")

if sha(Path(parent_graph))!=expected_parent:
    raise SystemExit("step-200 endpoint parent graph hash drift")
if sha(Path(localization_result))!=expected_localization:
    raise SystemExit("575795 localization result hash drift")
if sha(Path(failed_semantic_result))!=expected_failed_semantic:
    raise SystemExit("575797 failed semantic result hash drift")

loc=json.loads(Path(localization_result).read_text(encoding="utf-8"))
if loc.get("status")!="COMPLETE_DIAGNOSTIC_ONLY":
    raise SystemExit("575795 localization status drift")
if loc.get("localization")!="EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE":
    raise SystemExit("role-residual repair not bound to graph-selection localization")
if loc.get("interpretation_contract",{}).get("gradient_performed") is not False:
    raise SystemExit("575795 unexpectedly performed gradient")

failed=json.loads(Path(failed_semantic_result).read_text(encoding="utf-8"))
if failed.get("schema")!="alice.eipm.n0.v02-relation-semantic-grounding-result.v0.1":
    raise SystemExit("575797 result schema drift")
if failed.get("status")!="FAIL_RELATION_SEMANTIC_GROUNDING_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX":
    raise SystemExit("role-residual repair requires preserved 575797 fail")
if failed.get("automatic_rerun_or_hotfix_authorized") is not False:
    raise SystemExit("575797 anti-loop contract drift")
if failed.get("parent_graph_sha256")!=expected_parent:
    raise SystemExit("575797 parent graph drift")
if failed.get("relation_endpoint_repair_step200_used_as_parent") is not True:
    raise SystemExit("575797 lineage drift")
if failed.get("semantic_heldout_test_evaluated_once") is not True:
    raise SystemExit("575797 heldout evidence drift")
if failed.get("selected_checkpoint")!="step-00000080":
    raise SystemExit("575797 selected checkpoint drift")
if failed.get("scale_authorized") is not False or failed.get("n0_complete") is not False:
    raise SystemExit("575797 governance drift")

baseline_ep=float(failed["baseline_endpoint_role_dev"]["pair_accuracy"])
selected_ep=float(failed["endpoint_role_dev_preservation_metrics"]["pair_accuracy"])
if not selected_ep < baseline_ep:
    raise SystemExit("575797 does not demonstrate endpoint preservation regression")
if float(failed["semantic_heldout_test_metrics"]["family_min_pair_accuracy"]) != 0.0:
    raise SystemExit("575797 heldout failure family-min drift")

receipt={
 "schema":"alice.eipm.n0.v02-relation-semantic-role-residual-preparation.v0.2",
 "status":"PASS_RELATION_SEMANTIC_ROLE_RESIDUAL_READY_FOR_GPU",
 "git_revision":head,
 "artifact_sha256":{name:sha(path) for name,path in paths.items()},
 "trigger_localization_result_sha256":expected_localization,
 "trigger_failed_semantic_result_sha256":expected_failed_semantic,
 "trigger_failure_status":"FAIL_RELATION_SEMANTIC_GROUNDING_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX",
 "trigger_failure_interpretation":"shared_parent_relation_read_training_improves_some_semantic_behavior_but_degrades_previously_learned_endpoint_role_behavior_and_never_reaches_nonzero_worst_family_semantic_dev_accuracy",
 "parent_graph_sha256":expected_parent,
 "parent_graph_role":"immutable_endpoint_role_parent",
 "new_architecture":"zero_initialized_signed_source_target_semantic_role_residual",
 "trainable_scope":"new_semantic_role_query_adapter_and_delta_mlp_only",
 "parent_graph_parameters_trainable":False,
 "message_passing_trainable":False,
 "conflict_pool_trainable":False,
 "adapter_trainable":False,
 "semantic_parent_trainable":False,
 "structured_parent_trainable":False,
 "frozen_latent_challenge_rows_used_for_training":False,
 "missing_evidence_localization_rows_used_for_training":False,
 "relation_semantic_v0_1_rows_reused":False,
 "endpoint_repair_v0_2_heldout_test_reused":False,
 "v0_2_heldout_may_open_only_after_dev_readiness_and_preservation":True,
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

echo "===== N0 RELATION-SEMANTIC ROLE-RESIDUAL REPAIR ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "parent_graph=$PARENT_GRAPH"
echo "failed_semantic_result=$FAILED_SEMANTIC_RESULT"
echo "localization_result=$LOCALIZATION_RESULT"
echo "training_enabled=true"
echo "trainable_scope=new_zero_initialized_relation_semantic_role_residual_only"
echo "parent_endpoint_graph_trainable=false"
echo "frozen_challenge_rows_used_for_training=false"
echo "localization_rows_used_for_training=false"
echo "relation_semantic_v0_1_rows_reused=false"
echo "endpoint_heldout_test_reused=false"
echo "v0_2_heldout_requires_dev_readiness=true"
echo "scale_authorized=false"
echo "automatic_rerun=false"

python "$TRAINER"   --repo-root "$ROOT"   --prep-receipt "$PREP_RECEIPT"   --curriculum "$CURRICULUM"   --manifest "$MANIFEST"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED_CHECKPOINT"   --parent-adapter "$PARENT_ADAPTER"   --parent-graph "$PARENT_GRAPH"   --ordinary-replay-cache "$ORDINARY_REPLAY_CACHE"   --endpoint-replay-cache "$ENDPOINT_REPLAY_CACHE"   --builder "$BUILDER"   --semantic-role-source "$SEMANTIC_ROLE_SOURCE"   --output-dir "$TRAIN_ROOT"

echo
echo "===== ROLE-RESIDUAL OUTPUT HASHES ====="
sha256sum "$CURRICULUM" "$MANIFEST" "$PREP_RECEIPT" "$TRAIN_ROOT/result.json"

echo
echo "===== ROLE-RESIDUAL DECISION ====="
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"preservation_candidate_keys={r.get('preservation_candidate_keys')}")
print(f"eligible_candidate_keys={r.get('eligible_candidate_keys')}")
print(f"selected_checkpoint={r.get('selected_checkpoint')}")
print(f"best_dev_checkpoint={r.get('best_dev_checkpoint')}")
print(f"selected_graph_sha256={r.get('selected_graph_sha256')}")
print(f"semantic_heldout_test_evaluated={r.get('semantic_heldout_test_evaluated')}")
print(f"semantic_heldout_rows_opened_after_training={r.get('semantic_heldout_rows_opened_after_training')}")
if r.get("semantic_heldout_test_evaluated"):
    test=r.get("semantic_heldout_test_metrics",{})
    print(f"heldout_pair_accuracy={test.get('pair_accuracy')}")
    print(f"heldout_family_min_pair_accuracy={test.get('family_min_pair_accuracy')}")
    print(f"heldout_row_accuracy={test.get('row_accuracy')}")
    print(f"heldout_mean_graph_target_margin={test.get('mean_graph_target_margin')}")
    print(f"heldout_role_summary={r.get('semantic_heldout_role_summary')}")
print(f"parent_endpoint_graph_parameters_exactly_unchanged={r.get('parent_endpoint_graph_parameters_exactly_unchanged')}")
print(f"next_action={r.get('next_action')}")
print("scale_authorized=false")
print("n0_complete=false")
PY

echo "===== N0 RELATION-SEMANTIC ROLE-RESIDUAL REPAIR COMPLETE ====="
date -Is
