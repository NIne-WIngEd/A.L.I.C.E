#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
TEACHER_DIR="${N0_V02_TEACHER_DIR:-$WORKDIR/teacher-bank-v0.5}"
TEACHER_REGISTRY="$TEACHER_DIR/n0_v02_teacher_bank_v0.5.runtime.json"
TEACHER_AUDIT="$TEACHER_DIR/teacher-bank-v0.5-audit.json"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
TRANCHE_ROOT="${N0_V02_FIRST_TRANCHE_ROOT:-$WORKDIR/first-tranche-v0.1}"
OUT_ROOT="${N0_V02_TEACHER_DEV_EVAL_ROOT:-$WORKDIR/teacher-dev-challenge-v0.1}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$TEACHER_REGISTRY" \
  "$TEACHER_AUDIT" \
  "$CONFIG" \
  "$TRANCHE_ROOT/checkpoints/step-00000250/receipt.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000250/mlm/config.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000250/ranker.safetensors" \
  "$TRANCHE_ROOT/checkpoints/step-00000500/receipt.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000500/mlm/config.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000500/ranker.safetensors"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required teacher-dev challenge artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing challenge evaluation: $OUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/preflight_n0_v02_teacher_dev_challenge.py" \
  "$ROOT/scripts/eipm/n0/evaluate_n0_v02_teacher_dev.py"

# Bind this evaluation to the exact first-tranche artifacts before consuming
# P100 time. This verifies both checkpoint receipts/hashes and proves that the
# 255-row dev split is disjoint from the 765-row teacher train split by row ID.
python "$ROOT/scripts/eipm/n0/preflight_n0_v02_teacher_dev_challenge.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --teacher-registry "$TEACHER_REGISTRY" \
  --teacher-audit "$TEACHER_AUDIT" \
  --tranche-root "$TRANCHE_ROOT" \
  | tee "$OUT_ROOT/challenge_preflight.json"

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Teacher-dev challenge requires one visible CUDA device." >&2
  exit 4
fi

echo "===== N0 V0.2 HELD-OUT TEACHER DEV CHALLENGE ====="
date -Is
echo "workdir=$WORKDIR"
echo "tranche_root=$TRANCHE_ROOT"
echo "teacher_dev_rows_expected=255"
echo "candidate_order_variants=3"
echo "checkpoints=250,500"
echo "challenge_scope=held_out_teacher_dev_only"
echo "full_expanded_challenge_satisfied=false"
echo "additional_gradient_authorized=false"
echo "private_identity_gradient=false"

for STEP in 250 500; do
  KEY="step-$(printf '%08d' "$STEP")"
  STEP_DIR="$TRANCHE_ROOT/checkpoints/$KEY"
  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_teacher_dev.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --teacher-registry "$TEACHER_REGISTRY" \
    --teacher-audit "$TEACHER_AUDIT" \
    --output-dir "$OUT_ROOT/$KEY" \
    --max-length 256 \
    --batch-size 8 \
    --order-variants 3 \
    --device cuda
done

python - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "schema": "alice.eipm.n0.v02-teacher-dev-challenge-comparison.v0.2",
    "status": "PASS",
    "status_meaning": "evaluation_completed_and_artifacts_valid_not_model_promotion",
    "scope": "held_out_teacher_dev_only",
    "eval_only": True,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "full_expanded_challenge_satisfied": False,
    "additional_gradient_authorized": False,
    "novel_cross_competency_challenge_required": True,
    "checkpoints": {},
}
for step in (250, 500):
    key = f"step-{step:08d}"
    metrics = json.loads((root / key / "teacher_dev_metrics.json").read_text(encoding="utf-8"))
    summary["checkpoints"][key] = {
        "base_rows": metrics["base_rows"],
        "compiled_rows": metrics["compiled_rows"],
        "top1_accuracy": metrics["top1_accuracy"],
        "separation_rate": metrics["separation_rate"],
        "full_order_invariance_pass_rate": metrics["full_order_invariance_pass_rate"],
        "mean_margin": metrics["mean_margin"],
        "median_margin": metrics["median_margin"],
        "margin_p10": metrics["margin_p10"],
        "margin_p90": metrics["margin_p90"],
        "mean_abs_score": metrics["mean_abs_score"],
        "max_abs_score": metrics["max_abs_score"],
        "failure_count": len(metrics["full_order_invariance_failures"]),
        "failures": metrics["full_order_invariance_failures"],
    }

s250 = summary["checkpoints"]["step-00000250"]
s500 = summary["checkpoints"]["step-00000500"]
if s500["full_order_invariance_pass_rate"] > s250["full_order_invariance_pass_rate"]:
    provisional = "step-00000500"
elif s500["full_order_invariance_pass_rate"] < s250["full_order_invariance_pass_rate"]:
    provisional = "step-00000250"
elif s500["top1_accuracy"] > s250["top1_accuracy"]:
    provisional = "step-00000500"
elif s500["top1_accuracy"] < s250["top1_accuracy"]:
    provisional = "step-00000250"
else:
    provisional = "tie_requires_novel_challenge_and_mlm_consideration"
summary["provisional_generalization_winner"] = provisional
summary["step500_minus_step250"] = {
    field: s500[field] - s250[field]
    for field in (
        "top1_accuracy",
        "separation_rate",
        "full_order_invariance_pass_rate",
        "mean_margin",
        "median_margin",
        "margin_p10",
        "margin_p90",
        "mean_abs_score",
        "max_abs_score",
    )
}
summary["next_rule"] = (
    "inspect_dev_failures_and_score_growth_then_author_novel_cross_competency_"
    "hard_negative_ambiguity_multiturn_and_voice_challenge_before_any_new_gradient"
)

(root / "teacher_dev_challenge_comparison.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "===== N0 V0.2 TEACHER DEV CHALLENGE COMPLETE ====="
date -Is
