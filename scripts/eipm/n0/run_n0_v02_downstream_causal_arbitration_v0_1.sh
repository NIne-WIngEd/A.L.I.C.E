#!/usr/bin/env bash
set -euo pipefail

ROOT="${ALICE_REPO_ROOT:-$HOME/rayan-compute/rayan-eipm-main}"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v02}"
CALDIR="${ALICE_N0_ARB_CAL_DIR:-$WORKDIR/downstream-causal-arbitration-metric-calibration-v0.1}"
OUTDIR="${ALICE_N0_ARB_DIR:-$WORKDIR/downstream-causal-arbitration-v0.1}"

cd "$ROOT"

# Preparation and frozen execution spawn evaluator/helper processes. Keep the
# repository package root and N0 script root explicit inside the container.
export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"

CANON="$WORKDIR/relation-repair-v0.1/step-00000080/evidence_graph_dual_endpoint.safetensors"
CANDIDATE="$WORKDIR/relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors"
EXPECTED_CANDIDATE_SHA256="3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"

LATENT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
EVAL="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2/challenge.jsonl"
ADAPTER="$WORKDIR/evidence-graph-specialist-v0.1/specialized_expanded_640x3_graph512x2/step-00000080/evidence_view_adapter.safetensors"
FUSION="$WORKDIR/cross-context-fusion-repair-training-v0.2/repair-step-00000240"

SEMANTIC_CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
SEMANTIC_CHECKPOINT="$WORKDIR/targeted-repair-v0.1/checkpoints/step-00000080"
TOKENIZER_DIR="$WORKDIR/tokenizer-v0.2.1"
STRUCTURED_CONFIG="$ROOT/configs/eipm/n0/n0_v02_structured_state_v0.1.json"
STRUCTURED_CHECKPOINT="$WORKDIR/structured-state-pilot-v0.1/step-00000080"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
FUSION_CONFIG="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"

POLICY="$CALDIR/n0_v02_downstream_causal_arbitration_metric_policy_v0.1.json"
CAL_RECEIPT="$CALDIR/calibration_receipt.json"

STACK_MANIFEST="$OUTDIR/full_stack_manifest.json"
ARBITRATION_MANIFEST="$OUTDIR/arbitration_manifest.json"
PREFLIGHT="$OUTDIR/preflight_receipt.json"
ARBITRATION_RESULT="$OUTDIR/arbitration_result.json"
GATE_CONFIG="$OUTDIR/post_arbitration_gate_config.json"
GATE_RECEIPT="$OUTDIR/post_arbitration_gate_receipt.json"
FINALIZATION_RECEIPT="$OUTDIR/finalization_receipt.json"

PREP="$ROOT/scripts/eipm/n0/prepare_downstream_causal_arbitration_v0_1.py"
EXECUTE="$ROOT/scripts/eipm/n0/execute_frozen_downstream_causal_arbitration_v0_1.py"
FINALIZE="$ROOT/scripts/eipm/n0/finalize_downstream_causal_arbitration_v0_1.py"
ARBITRATION="$ROOT/scripts/eipm/n0/downstream_causal_arbitration_v0_2.py"
EVALUATOR="$ROOT/scripts/eipm/n0/downstream_full_stack_graph_evaluator_v0_1.py"
GATE="$ROOT/scripts/eipm/n0/post_arbitration_decision_gate_v0_1.py"

if [[ -e "$OUTDIR" ]]; then
  echo "REFUSING: arbitration output directory already exists: $OUTDIR" >&2
  exit 2
fi

for path in \
  "$CANON" "$CANDIDATE" "$LATENT" "$EVAL" "$ADAPTER" "$FUSION" \
  "$SEMANTIC_CONFIG" "$SEMANTIC_CHECKPOINT" "$TOKENIZER_DIR" \
  "$STRUCTURED_CONFIG" "$STRUCTURED_CHECKPOINT" "$LATENT_CONFIG" \
  "$FUSION_CONFIG" "$FUSION_RATIFICATION" "$POLICY" "$CAL_RECEIPT" \
  "$PREP" "$EXECUTE" "$FINALIZE" "$ARBITRATION" "$EVALUATOR" "$GATE"
do
  if [[ ! -e "$path" ]]; then
    echo "MISSING: $path" >&2
    exit 3
  fi
done

SOURCE_REVISION="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "REFUSING: tracked repository worktree is not clean." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

CANDIDATE_SHA256="$(sha256sum "$CANDIDATE" | awk '{print $1}')"
if [[ "$CANDIDATE_SHA256" != "$EXPECTED_CANDIDATE_SHA256" ]]; then
  echo "REFUSING: candidate graph hash drift." >&2
  echo "expected=$EXPECTED_CANDIDATE_SHA256" >&2
  echo "observed=$CANDIDATE_SHA256" >&2
  exit 5
fi

CANON_SHA256="$(sha256sum "$CANON" | awk '{print $1}')"

python - "$CAL_RECEIPT" "$CANON_SHA256" "$POLICY" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
canonical_sha = sys.argv[2]
policy_path = Path(sys.argv[3])

receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("protocol_version") != "n0_downstream_causal_arbitration_metric_calibration_v0.1":
    raise SystemExit("calibration receipt protocol drift")
if receipt.get("status") != "CALIBRATION_COMPLETE":
    raise SystemExit("calibration is not complete")
if receipt.get("candidate_graph_supplied") is not False:
    raise SystemExit("calibration unexpectedly received a candidate graph")
if receipt.get("candidate_arm_result_observed") is not False:
    raise SystemExit("calibration unexpectedly observed candidate-arm evidence")
frozen = receipt.get("frozen_identity") or {}
if frozen.get("graph_sha256") != canonical_sha:
    raise SystemExit(
        "calibration canonical graph differs from arbitration canonical graph: "
        f"calibration={frozen.get('graph_sha256')} arbitration={canonical_sha}"
    )
if not policy_path.is_file():
    raise SystemExit(f"metric policy missing: {policy_path}")
policy = json.loads(policy_path.read_text(encoding="utf-8"))
if policy.get("protocol_version") != "n0_downstream_causal_arbitration_metric_policy_v0.1":
    raise SystemExit("metric policy protocol drift")
if policy.get("selected_before_arm_results") is not True:
    raise SystemExit("metric policy was not frozen before arm results")
if policy.get("results_observed") is not False:
    raise SystemExit("metric policy indicates arm results were already observed")
print("calibration_binding_valid=true")
print(f"canonical_graph_sha256={canonical_sha}")
print(f"metric_policy_sha256={hashlib.sha256(policy_path.read_bytes()).hexdigest()}")
PY

python -m py_compile \
  "$PREP" "$EXECUTE" "$FINALIZE" "$ARBITRATION" "$EVALUATOR" "$GATE"

EXPERIMENT_ID="n0-v02-downstream-causal-arbitration-v0.1-${SOURCE_REVISION:0:12}-${CANDIDATE_SHA256:0:12}"

mkdir -p "$OUTDIR"

echo "===== N0 DOWNSTREAM CAUSAL ARBITRATION ====="
date -Is
echo "source_revision=$SOURCE_REVISION"
echo "experiment_id=$EXPERIMENT_ID"
echo "canonical_graph=$CANON"
echo "canonical_sha256=$CANON_SHA256"
echo "candidate_graph=$CANDIDATE"
echo "candidate_sha256=$CANDIDATE_SHA256"
echo "latent_checkpoint=$LATENT"
echo "metric_policy=$POLICY"
echo "training_enabled=false"
echo "promotion_authorized=false"
echo "scale_authorized=false"
echo "automatic_final_challenge=false"

python "$PREP" \
  --experiment-id "$EXPERIMENT_ID" \
  --metric-policy "$POLICY" \
  --evaluation-set "$EVAL" \
  --latent-checkpoint "$LATENT" \
  --latent-config "$LATENT_CONFIG" \
  --semantic-config "$SEMANTIC_CONFIG" \
  --semantic-checkpoint "$SEMANTIC_CHECKPOINT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --structured-config "$STRUCTURED_CONFIG" \
  --structured-checkpoint "$STRUCTURED_CHECKPOINT" \
  --evidence-adapter "$ADAPTER" \
  --fusion-checkpoint "$FUSION" \
  --fusion-config "$FUSION_CONFIG" \
  --fusion-ratification "$FUSION_RATIFICATION" \
  --canonical-graph "$CANON" \
  --candidate-graph "$CANDIDATE" \
  --stack-manifest-output "$STACK_MANIFEST" \
  --arbitration-manifest-output "$ARBITRATION_MANIFEST" \
  --preflight-output "$PREFLIGHT" \
  --evaluator "$EVALUATOR"

python "$EXECUTE" \
  --manifest "$ARBITRATION_MANIFEST" \
  --preflight-receipt "$PREFLIGHT" \
  --output "$ARBITRATION_RESULT"

python "$FINALIZE" \
  --preflight-receipt "$PREFLIGHT" \
  --arbitration-result "$ARBITRATION_RESULT" \
  --gate-config-output "$GATE_CONFIG" \
  --gate-receipt-output "$GATE_RECEIPT" \
  --finalization-receipt-output "$FINALIZATION_RECEIPT"

echo
echo "===== ARBITRATION OUTPUT HASHES ====="
sha256sum \
  "$POLICY" \
  "$CAL_RECEIPT" \
  "$STACK_MANIFEST" \
  "$ARBITRATION_MANIFEST" \
  "$PREFLIGHT" \
  "$ARBITRATION_RESULT" \
  "$GATE_CONFIG" \
  "$GATE_RECEIPT" \
  "$FINALIZATION_RECEIPT"

echo
echo "===== POST-ARBITRATION DECISION ====="
python - "$ARBITRATION_RESULT" "$GATE_RECEIPT" "$FINALIZATION_RECEIPT" <<'PY'
import json
import sys
from pathlib import Path

arb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
final = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

print(f"arbitration_classification={arb.get('classification')}")
print(f"gate_decision={gate.get('decision')}")
print(
    "final_frozen_challenge_authorized="
    + str(gate.get("final_frozen_challenge_authorized")).lower()
)
print(f"candidate_graph_promoted={str(gate.get('candidate_graph_promoted')).lower()}")
print(f"training_authorized={str(gate.get('training_authorized')).lower()}")
print(f"scale_authorized={str(gate.get('scale_authorized')).lower()}")
print(f"n0_complete={str(gate.get('n0_complete')).lower()}")
if final.get("status") not in {"COMPLETE", "FINALIZATION_COMPLETE"}:
    raise SystemExit(f"unexpected finalization status: {final.get('status')}")
PY

echo
echo "===== N0 DOWNSTREAM CAUSAL ARBITRATION COMPLETE ====="
date -Is
echo "result=$ARBITRATION_RESULT"
echo "gate_receipt=$GATE_RECEIPT"
echo "finalization_receipt=$FINALIZATION_RECEIPT"
