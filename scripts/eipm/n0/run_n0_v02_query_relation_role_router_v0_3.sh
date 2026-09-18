#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
OUT_ROOT="${ALICE_N0_QUERY_RELATION_ROLE_ROUTER_DIR:-$WORKDIR/query-relation-role-router-v0.3}"
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
FAILED_V01="$WORKDIR/relation-semantic-grounding-v0.1/training/result.json"
FAILED_V02="$WORKDIR/relation-semantic-role-residual-v0.2/training/result.json"

BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_query_relation_role_router_curriculum_v0_3.py"
TRAINER="$ROOT/scripts/eipm/n0/train_n0_v02_query_relation_role_router_v0_3.py"
ROUTER_SOURCE="$ROOT/src/alice_personality/n0/evidence_graph_query_relation_role_router.py"
RESEARCH_NOTE="$ROOT/docs/research/eipm-n0-query-relation-role-router-external-research-v0.1.md"
CURRICULUM="$PREP_ROOT/query_relation_role_router_curriculum.jsonl"
MANIFEST="$PREP_ROOT/query_relation_role_router_manifest.json"
PREP_RECEIPT="$PREP_ROOT/preparation_receipt.json"

EXPECTED_PARENT="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
EXPECTED_LOCALIZATION="1733bf1a5fdc4f979beab6d5198bfb6a77a0e17b4fc96adc134dafd03dd5ebbe"
EXPECTED_V01="cfd2a6aee3ba2e441363700cf0d57917cf7cf16c3c81b2a789b5c1ecf3614980"
EXPECTED_V02="d9abbd6d3da3030361201a8a32f84aa8bf458fb5bebcc07db241a217b3557d8e"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUT_ROOT" ]]; then echo "REFUSING existing QRR output: $OUT_ROOT" >&2; exit 2; fi
for f in "$BUILDER" "$TRAINER" "$ROUTER_SOURCE" "$RESEARCH_NOTE" "$TOKENIZER_DIR/tokenizer.json"  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors" "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors"  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE" "$LOCALIZATION_RESULT" "$FAILED_V01" "$FAILED_V02"; do
 [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 3; }
done
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "REFUSING tracked repo changes" >&2; git status --short --untracked-files=no; exit 4; }

python -m py_compile "$BUILDER" "$TRAINER" "$ROUTER_SOURCE"
mkdir -p "$PREP_ROOT"
python "$BUILDER" --output "$CURRICULUM" --manifest "$MANIFEST"

python - "$ROOT" "$CURRICULUM" "$MANIFEST" "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT/alice_n0_v02.safetensors"  "$TOKENIZER_DIR/tokenizer.json" "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT/structured_state.safetensors"  "$PARENT_ADAPTER" "$PARENT_GRAPH" "$ORDINARY_REPLAY_CACHE" "$ENDPOINT_REPLAY_CACHE" "$BUILDER" "$TRAINER" "$ROUTER_SOURCE" "$RESEARCH_NOTE"  "$LOCALIZATION_RESULT" "$FAILED_V01" "$FAILED_V02" "$PREP_RECEIPT" "$EXPECTED_PARENT" "$EXPECTED_LOCALIZATION" "$EXPECTED_V01" "$EXPECTED_V02" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
(root,curriculum,manifest,semantic_config,semantic_checkpoint,tokenizer,structured_config,structured_checkpoint,parent_adapter,parent_graph,
 ordinary_replay_cache,endpoint_replay_cache,builder,trainer,router_source,research_note,localization,v01,v02,prep_path,
 expected_parent,expected_localization,expected_v01,expected_v02)=sys.argv[1:]
def sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
if sha(parent_graph)!=expected_parent:raise SystemExit("parent graph hash drift")
if sha(localization)!=expected_localization:raise SystemExit("575795 result hash drift")
if sha(v01)!=expected_v01:raise SystemExit("575797 result hash drift")
if sha(v02)!=expected_v02:raise SystemExit("575798 result hash drift")
v02j=json.loads(Path(v02).read_text())
if v02j.get("status")!="FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_DEV_STOP_NO_HELDOUT_EXPOSURE":raise SystemExit("575798 status drift")
if v02j.get("semantic_heldout_test_evaluated") is not False or v02j.get("semantic_heldout_rows_opened_after_training") is not False:raise SystemExit("575798 heldout contract drift")
if v02j.get("parent_endpoint_graph_parameters_exactly_unchanged") is not True:raise SystemExit("575798 parent immutability drift")
if v02j.get("eligible_candidate_keys")!=[]:raise SystemExit("575798 eligibility drift")
manifestj=json.loads(Path(manifest).read_text())
if manifestj.get("schema")!="alice.eipm.n0.v02-query-relation-role-router-curriculum.v0.3":raise SystemExit("QRR manifest drift")
if manifestj.get("compiled_sha256")!=sha(curriculum):raise SystemExit("QRR curriculum hash drift")
for k in ("frozen_latent_challenge_rows_used_for_training","missing_evidence_localization_rows_used_for_training","relation_semantic_v0_1_rows_reused","role_residual_v0_2_rows_reused","endpoint_repair_v0_2_heldout_rows_reused"):
 if manifestj.get(k) is not False:raise SystemExit("forbidden reuse "+k)
if manifestj.get("train_dev_test_templates_lexically_disjoint") is not True:raise SystemExit("lexical split drift")
paths={"curriculum":curriculum,"manifest":manifest,"semantic_config":semantic_config,"semantic_checkpoint":semantic_checkpoint,"tokenizer":tokenizer,
 "structured_config":structured_config,"structured_checkpoint":structured_checkpoint,"parent_adapter":parent_adapter,"parent_graph":parent_graph,
 "ordinary_replay_cache":ordinary_replay_cache,"endpoint_replay_cache":endpoint_replay_cache,"builder":builder,"trainer":trainer,
 "router_source":router_source,"research_note":research_note}
receipt={"schema":"alice.eipm.n0.v02-query-relation-role-router-preparation.v0.3","status":"PASS_QUERY_RELATION_ROLE_ROUTER_READY_FOR_GPU",
 "git_revision":head,"artifact_sha256":{k:sha(v) for k,v in paths.items()},"trigger_575798_result_sha256":expected_v02,
 "trigger_575798_status":v02j["status"],"external_architecture_research_completed":True,
 "architecture":"frozen_parent_plus_raw_query_relation_conditioned_source_target_defer_router",
 "composition":"probability_level_gated_expert_mixture","direct_role_supervision":True,"parent_graph_parameters_trainable":False,
 "parent_query_projection_is_router_input":False,"raw_query_semantic_is_router_input":True,
 "frozen_latent_challenge_rows_used_for_training":False,"missing_evidence_localization_rows_used_for_training":False,
 "relation_semantic_v0_1_rows_reused":False,"role_residual_v0_2_rows_reused":False,"endpoint_repair_v0_2_heldout_test_reused":False,
 "heldout_requires_preservation_and_dev_readiness":True,"failure_requires_architecture_level_audit_before_further_gradient_run":True,
 "hard_parameter_ceiling":None,"scale_authorized":False,"private_identity_gradient":False,"production_promotion_authorized":False,"n0_complete":False}
Path(prep_path).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,indent=2,sort_keys=True))
PY

echo "===== N0 QUERY-RELATION ROLE ROUTER ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "training_enabled=true"
echo "architecture=frozen_parent_raw_query_relation_role_router"
echo "composition=probability_level_gated_expert_mixture"
echo "direct_role_supervision=true"
echo "parent_graph_trainable=false"
echo "v0_1_rows_reused=false"
echo "v0_2_rows_reused=false"
echo "frozen_challenge_rows_used_for_training=false"
echo "localization_rows_used_for_training=false"
echo "endpoint_heldout_test_reused=false"
echo "scale_authorized=false"
echo "automatic_hotfix=false"

python "$TRAINER" --repo-root "$ROOT" --prep-receipt "$PREP_RECEIPT" --curriculum "$CURRICULUM" --manifest "$MANIFEST"  --semantic-config "$SEMANTIC_CONFIG" --semantic-checkpoint "$SEMANTIC_CHECKPOINT" --tokenizer-dir "$TOKENIZER_DIR"  --structured-config "$STRUCTURED_CONFIG" --structured-checkpoint "$STRUCTURED_CHECKPOINT" --parent-adapter "$PARENT_ADAPTER"  --parent-graph "$PARENT_GRAPH" --ordinary-replay-cache "$ORDINARY_REPLAY_CACHE" --endpoint-replay-cache "$ENDPOINT_REPLAY_CACHE"  --builder "$BUILDER" --router-source "$ROUTER_SOURCE" --research-note "$RESEARCH_NOTE" --output-dir "$TRAIN_ROOT"

sha256sum "$CURRICULUM" "$MANIFEST" "$PREP_RECEIPT" "$TRAIN_ROOT/result.json"
python - "$TRAIN_ROOT/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
for k in ("status","preservation_candidate_keys","eligible_candidate_keys","selected_checkpoint","best_dev_checkpoint",
          "semantic_heldout_test_evaluated","semantic_heldout_rows_opened_after_training","next_action"):
 print(f"{k}={r.get(k)}")
print("scale_authorized=false")
print("n0_complete=false")
PY
