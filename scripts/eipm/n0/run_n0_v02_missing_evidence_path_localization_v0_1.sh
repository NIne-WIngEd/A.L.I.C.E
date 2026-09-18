#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
FINAL_CHALLENGE_DIR="${ALICE_N0_FINAL_CHALLENGE_DIR:-$WORKDIR/adaptive-multi-view-latent-pool-final-frozen-challenge-v0.1-after-575792}"
OUTDIR="${ALICE_N0_MISSING_EVIDENCE_DIAG_DIR:-$WORKDIR/missing-evidence-path-localization-v0.1}"

BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_missing_evidence_fresh_localization_v0_1.py"
DIAGNOSTIC="$ROOT/scripts/eipm/n0/diagnose_n0_v02_missing_evidence_path_v0_1.py"
DATASET="$OUTDIR/fresh_relation_flip_rows.jsonl"
MANIFEST="$OUTDIR/fresh_relation_flip_manifest.json"
RESULT="$OUTDIR/path_localization_result.json"
TRIGGER="$FINAL_CHALLENGE_DIR/execution_receipt.json"

LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
EVIDENCE_ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
REPAIRED_GRAPH="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
FUSION_CHECKPOINT="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -e "$OUTDIR" ]]; then
  echo "REFUSING: localization output directory already exists: $OUTDIR" >&2
  exit 2
fi

for required in   "$BUILDER" "$DIAGNOSTIC" "$TRIGGER"   "$LATENT" "$LATENT_CONFIG"   "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" "$TOKENIZER_DIR"   "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT" "$EVIDENCE_ADAPTER"   "$REPAIRED_GRAPH" "$FUSION_CHECKPOINT" "$FUSION_CONFIG" "$FUSION_RATIFICATION"
do
  if [[ ! -e "$required" ]]; then
    echo "MISSING: $required" >&2
    exit 3
  fi
done

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository worktree is not clean." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

python -m py_compile "$BUILDER" "$DIAGNOSTIC"

mkdir -p "$OUTDIR"
python "$BUILDER" --output "$DATASET" --manifest "$MANIFEST"

python - "$MANIFEST" "$TRIGGER" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
trigger=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("frozen_challenge_rows_reused") is not False:
    raise SystemExit("fresh diagnostic reused frozen challenge rows")
if manifest.get("prior_diagnostic_rows_reused") is not False:
    raise SystemExit("fresh diagnostic reused prior diagnostic rows")
if manifest.get("training_authorized") is not False:
    raise SystemExit("fresh diagnostic training contract drift")
if trigger.get("model_conclusion")!="FAIL_STOP_AND_LOCALIZE_WITHOUT_AUTOMATIC_HOTFIX":
    raise SystemExit("unexpected final challenge trigger")
if trigger.get("gate_pass") is not False:
    raise SystemExit("final challenge trigger is not a valid fail")
print("fresh_localization_preflight=true")
print(f"rows={manifest.get('rows')}")
print(f"pairs={manifest.get('pairs')}")
print("frozen_challenge_rows_reused=false")
print("training_authorized=false")
PY

echo "===== N0 FRESH MISSING-EVIDENCE PATH LOCALIZATION ====="
date -Is
echo "source_revision=$(git rev-parse HEAD)"
echo "trigger=$TRIGGER"
echo "dataset=$DATASET"
echo "repaired_graph=$REPAIRED_GRAPH"
echo "latent_checkpoint=$LATENT"
echo "training_enabled=false"
echo "promotion_authorized=false"
echo "scale_authorized=false"
echo "automatic_rerun=false"

python "$DIAGNOSTIC"   --dataset "$DATASET"   --manifest "$MANIFEST"   --trigger-receipt "$TRIGGER"   --latent-checkpoint "$LATENT"   --latent-config "$LATENT_CONFIG"   --semantic-config "$SEMANTIC_CONFIG"   --semantic-checkpoint "$SEMANTIC_CHECKPOINT"   --tokenizer-dir "$TOKENIZER_DIR"   --structured-config "$STRUCTURED_CONFIG"   --structured-checkpoint "$STRUCTURED_CHECKPOINT"   --evidence-adapter "$EVIDENCE_ADAPTER"   --repaired-graph "$REPAIRED_GRAPH"   --fusion-checkpoint "$FUSION_CHECKPOINT"   --fusion-config "$FUSION_CONFIG"   --fusion-ratification "$FUSION_RATIFICATION"   --output "$RESULT"

echo
echo "===== LOCALIZATION OUTPUT HASHES ====="
sha256sum "$DATASET" "$MANIFEST" "$RESULT"

echo
echo "===== LOCALIZATION SUMMARY ====="
python - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"status={r.get('status')}")
print(f"localization={r.get('localization')}")
for name,stage in r.get("stage_summaries",{}).items():
    print(
        f"{name}: pair_accuracy={stage.get('relation_flip_pair_accuracy')} "
        f"target_preference_rate={stage.get('target_preference_rate')} "
        f"mean_margin={stage.get('mean_margin')}"
    )
agg=r.get("aggregate_causal_metrics",{})
for key in (
    "graph_argmax_target_rate",
    "mean_relation_bias_target_minus_foil",
    "mean_fusion_evidence_view_weight",
    "mean_fusion_semantic_transfer_delta",
    "mean_latent_probe_margin_drop",
    "mean_latent_summary_margin_drop",
    "mean_absolute_target_cosine_drop",
    "mean_latent_evidence_view_attention",
    "mean_counterfactual_latent_evidence_view_attention",
):
    print(f"{key}={agg.get(key)}")
print("training_authorized=false")
print("scale_authorized=false")
print("n0_complete=false")
PY

echo
echo "===== N0 FRESH MISSING-EVIDENCE PATH LOCALIZATION COMPLETE ====="
date -Is
