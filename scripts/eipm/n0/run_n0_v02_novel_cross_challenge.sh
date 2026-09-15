#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
TOKENIZER_DIR="${N0_V02_TOKENIZER_DIR:-$WORKDIR/tokenizer-v0.2.1}"
TRANCHE_ROOT="${N0_V02_FIRST_TRANCHE_ROOT:-$WORKDIR/first-tranche-v0.1}"
DEV_ROOT="${N0_V02_TEACHER_DEV_EVAL_ROOT:-$WORKDIR/teacher-dev-challenge-v0.2}"
OUT_ROOT="$WORKDIR/novel-cross-challenge-v0.1"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.2.json"
BASE="$ROOT/evaluation/eipm/n0/n0_v02_novel_cross_competency_base_v0.1.jsonl"
COMPILED="$OUT_ROOT/novel_cross_competency_compiled.jsonl"
MANIFEST="$OUT_ROOT/novel_cross_competency_manifest.json"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in \
  "$TOKENIZER_DIR/tokenizer.json" \
  "$CONFIG" \
  "$BASE" \
  "$DEV_ROOT/challenge_preflight.json" \
  "$DEV_ROOT/teacher_dev_challenge_comparison.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000250/mlm/config.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000250/ranker.safetensors" \
  "$TRANCHE_ROOT/checkpoints/step-00000500/mlm/config.json" \
  "$TRANCHE_ROOT/checkpoints/step-00000500/ranker.safetensors"
do
  if [[ ! -f "$required" ]]; then
    echo "Missing required novel-challenge artifact: $required" >&2
    exit 2
  fi
done

python - "$DEV_ROOT/challenge_preflight.json" "$DEV_ROOT/teacher_dev_challenge_comparison.json" <<'PY'
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
comparison = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if preflight.get("schema") != "alice.eipm.n0.v02-teacher-dev-challenge-preflight.v0.2":
    raise SystemExit("teacher-dev preflight schema mismatch")
if preflight.get("status") != "PASS":
    raise SystemExit("teacher-dev preflight did not pass")
if comparison.get("schema") != "alice.eipm.n0.v02-teacher-dev-challenge-comparison.v0.2":
    raise SystemExit("teacher-dev comparison schema mismatch")
if comparison.get("status") != "PASS":
    raise SystemExit("teacher-dev comparison did not complete cleanly")
if comparison.get("additional_gradient_authorized") is not False:
    raise SystemExit("unexpected gradient authorization in teacher-dev comparison")
if comparison.get("novel_cross_competency_challenge_required") is not True:
    raise SystemExit("teacher-dev comparison did not require novel challenge")
print("teacher_dev_gate_verified=true")
PY

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing novel challenge evaluation: $OUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/compile_n0_v02_fixed_eval.py" \
  "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py"

python "$ROOT/scripts/eipm/n0/compile_n0_v02_fixed_eval.py" \
  --base "$BASE" \
  --output "$COMPILED" \
  --manifest "$MANIFEST" \
  --order-variants 3 \
  --minimum-cases-per-competency 3 \
  --expected-competencies 8

CUDA_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [[ "$CUDA_COUNT" -lt 1 ]]; then
  echo "Novel cross-competency challenge requires one visible CUDA device." >&2
  exit 4
fi

echo "===== N0 V0.2 NOVEL CROSS-COMPETENCY CHALLENGE ====="
date -Is
echo "workdir=$WORKDIR"
echo "challenge_base=$BASE"
echo "base_cases=24"
echo "challenge_families=8"
echo "paraphrase_variants=2"
echo "candidate_order_variants=3"
echo "compiled_examples_expected=144"
echo "checkpoints=250,500"
echo "eval_only=true"
echo "full_expanded_challenge_satisfied=false"
echo "additional_gradient_authorized=false"
echo "private_identity_data=false"
echo "private_identity_gradient=false"

for STEP in 250 500; do
  KEY="step-$(printf '%08d' "$STEP")"
  STEP_DIR="$TRANCHE_ROOT/checkpoints/$KEY"
  python "$ROOT/scripts/eipm/n0/evaluate_n0_v02_fixed.py" \
    --config "$CONFIG" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --mlm-checkpoint "$STEP_DIR/mlm" \
    --ranker "$STEP_DIR/ranker.safetensors" \
    --benchmark "$COMPILED" \
    --benchmark-receipt "$MANIFEST" \
    --output-dir "$OUT_ROOT/$KEY" \
    --max-length 512 \
    --batch-size 8 \
    --device cuda
done

python - "$OUT_ROOT" "$MANIFEST" "$DEV_ROOT/teacher_dev_challenge_comparison.json" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
dev_comparison_path = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
dev_comparison = json.loads(dev_comparison_path.read_text(encoding="utf-8"))


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def enrich(key):
    step_root = root / key
    metrics = json.loads((step_root / "fixed_metrics.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (step_root / "fixed_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    margins = [float(row["separation_margin"]) for row in records]
    abs_scores = [abs(float(score)) for row in records for score in row["scores"]]
    failures = [
        row for row in records
        if not bool(row["top_supported"]) or not bool(row["supported_set_separated"])
    ]
    failed_abs = [abs(float(score)) for row in failures for score in row["scores"]]
    return {
        "base_cases": metrics["base_cases"],
        "compiled_examples": metrics["examples"],
        "top1_accuracy": metrics["top1_accuracy"],
        "separation_rate": metrics["separation_rate"],
        "full_invariance_pass_rate": metrics["full_invariance_pass_rate"],
        "full_invariance_failures": metrics["full_invariance_failures"],
        "mean_margin": statistics.fmean(margins),
        "median_margin": statistics.median(margins),
        "margin_p10": percentile(margins, 0.10),
        "margin_p90": percentile(margins, 0.90),
        "mean_abs_score": statistics.fmean(abs_scores),
        "max_abs_score": max(abs_scores),
        "failed_compiled_examples": len(failures),
        "mean_abs_score_on_failed_examples": statistics.fmean(failed_abs) if failed_abs else 0.0,
        "max_abs_score_on_failed_examples": max(failed_abs) if failed_abs else 0.0,
        "per_competency": metrics["per_competency"],
    }

summary = {
    "schema": "alice.eipm.n0.v02-novel-cross-challenge-comparison.v0.1",
    "status": "PASS",
    "status_meaning": "novel_evaluation_completed_not_model_promotion",
    "scope": "frozen_novel_cross_competency_hard_negative_ambiguity_multiturn_voice",
    "eval_only": True,
    "private_identity_data": False,
    "private_identity_gradient": False,
    "additional_gradient_authorized": False,
    "benchmark_base_rows": manifest["base_rows"],
    "benchmark_compiled_rows": manifest["compiled_rows"],
    "benchmark_base_sha256": manifest["base_sha256"],
    "benchmark_compiled_sha256": manifest["compiled_sha256"],
    "teacher_dev_gate_schema": dev_comparison["schema"],
    "teacher_dev_gate_status": dev_comparison["status"],
    "checkpoints": {},
}
for step in (250, 500):
    key = f"step-{step:08d}"
    summary["checkpoints"][key] = enrich(key)

s250 = summary["checkpoints"]["step-00000250"]
s500 = summary["checkpoints"]["step-00000500"]
fields = (
    "top1_accuracy",
    "separation_rate",
    "full_invariance_pass_rate",
    "mean_margin",
    "median_margin",
    "margin_p10",
    "margin_p90",
    "mean_abs_score",
    "max_abs_score",
    "mean_abs_score_on_failed_examples",
    "max_abs_score_on_failed_examples",
)
summary["step500_minus_step250"] = {field: s500[field] - s250[field] for field in fields}

if s500["full_invariance_pass_rate"] > s250["full_invariance_pass_rate"]:
    winner = "step-00000500"
elif s500["full_invariance_pass_rate"] < s250["full_invariance_pass_rate"]:
    winner = "step-00000250"
elif s500["top1_accuracy"] > s250["top1_accuracy"]:
    winner = "step-00000500"
elif s500["top1_accuracy"] < s250["top1_accuracy"]:
    winner = "step-00000250"
elif s500["mean_abs_score_on_failed_examples"] < s250["mean_abs_score_on_failed_examples"]:
    winner = "step-00000500"
elif s500["mean_abs_score_on_failed_examples"] > s250["mean_abs_score_on_failed_examples"]:
    winner = "step-00000250"
else:
    winner = "tie_requires_case_level_review"
summary["provisional_generalization_winner"] = winner
summary["winner_policy"] = (
    "full_invariance_then_top1_then_lower_absolute_score_on_failed_examples; "
    "this is checkpoint triage only and does not authorize promotion or training"
)
summary["next_rule"] = (
    "inspect_case_level_failures_and_checkpoint_disagreements_then_author_targeted_repair_rows_"
    "only_for_observed_failures_before_any_new_gradient"
)

(root / "novel_cross_challenge_comparison.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "===== N0 V0.2 NOVEL CROSS-COMPETENCY CHALLENGE COMPLETE ====="
date -Is
