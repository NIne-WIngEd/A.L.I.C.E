#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
STEP_DIR="${N0_MLM_STEP_DIR:-$WORKDIR/checkpoints/step-00001000}"
TOKENIZER_DIR="$WORKDIR/tokenizer"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json"

SEED_CURRICULUM="$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.jsonl"
SEED_MANIFEST="$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.origin.json"
COVERAGE_CURRICULUM="$ROOT/training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl"
COVERAGE_MANIFEST="$ROOT/training/eipm/n0/sol_curriculum_coverage_v0.2.origin.json"
REPAIR_CURRICULUM="$ROOT/training/eipm/n0/sol_curriculum_repair_v0.3.jsonl"
REPAIR_MANIFEST="$ROOT/training/eipm/n0/sol_curriculum_repair_v0.3.origin.json"

BASE_PROBE_ROOT="${N0_BASE_TEACHER_PROBE_DIR:-$WORKDIR/teacher-probes/step-00001000-sol-v0.2}"
BASE_RANKER="$BASE_PROBE_ROOT/frozen-ranker/ranker.safetensors"
PROBE_ROOT="${N0_REPAIR_PROBE_DIR:-$WORKDIR/teacher-probes/step-00001000-sol-repair-v0.3}"
BASELINE_EVAL="$PROBE_ROOT/baseline-expanded-dev"
RANKER_DIR="$PROBE_ROOT/frozen-ranker"
TRAIN_EVAL="$PROBE_ROOT/repaired-train"
DEV_EVAL="$PROBE_ROOT/repaired-dev"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$STEP_DIR/receipt.json" || ! -d "$STEP_DIR/model" ]]; then
  echo "Complete step-1000 checkpoint required: $STEP_DIR" >&2
  exit 2
fi
if [[ ! -f "$BASE_RANKER" ]]; then
  echo "Prior frozen teacher probe ranker required for fair baseline: $BASE_RANKER" >&2
  exit 2
fi
for path in \
  "$SEED_CURRICULUM" "$SEED_MANIFEST" \
  "$COVERAGE_CURRICULUM" "$COVERAGE_MANIFEST" \
  "$REPAIR_CURRICULUM" "$REPAIR_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Required governed teaching artifact missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$PROBE_ROOT" ]]; then
  echo "Repair probe output already exists; refusing duplicate diagnostic: $PROBE_ROOT" >&2
  exit 3
fi
mkdir -p "$BASELINE_EVAL" "$RANKER_DIR" "$TRAIN_EVAL" "$DEV_EVAL"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
  "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  "$ROOT/src/alice_personality/n0/curriculum.py" \
  "$ROOT/src/alice_personality/n0/curriculum_data.py" \
  "$ROOT/src/alice_personality/n0/ranker.py"

echo "===== FAILURE-DRIVEN CURRICULUM PREFLIGHT ====="
python - \
  "$SEED_CURRICULUM" "$SEED_MANIFEST" \
  "$COVERAGE_CURRICULUM" "$COVERAGE_MANIFEST" \
  "$REPAIR_CURRICULUM" "$REPAIR_MANIFEST" <<'PY'
import json
import sys
from alice_personality.n0.curriculum import validate_curriculum_manifest, validate_curriculum_rows
from alice_personality.n0.curriculum_data import CurriculumDataset

pairs = [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4]), (sys.argv[5], sys.argv[6])]
entries = []
for curriculum, manifest in pairs:
    loaded = validate_curriculum_manifest(curriculum, manifest)
    summary = validate_curriculum_rows(curriculum)
    entries.append({
        "curriculum": curriculum,
        "actor": loaded.get("actor"),
        "origin_type": loaded.get("origin_type"),
        "summary": summary,
    })
paths = [pair[0] for pair in pairs]
train = CurriculumDataset(paths, "train")
dev = CurriculumDataset(paths, "dev")
print(json.dumps({
    "status": "PASS",
    "curricula": entries,
    "combined_train_rows": len(train),
    "combined_dev_rows": len(dev),
    "train_competencies": len({str(row["competency"]) for row in train.rows}),
    "dev_competencies": len({str(row["competency"]) for row in dev.rows}),
    "private_identity_gradient": False,
}, indent=2, sort_keys=True))
PY

COMMON_CURRICULUM_ARGS=(
  --curriculum "$SEED_CURRICULUM"
  --curriculum-manifest "$SEED_MANIFEST"
  --curriculum "$COVERAGE_CURRICULUM"
  --curriculum-manifest "$COVERAGE_MANIFEST"
  --curriculum "$REPAIR_CURRICULUM"
  --curriculum-manifest "$REPAIR_MANIFEST"
)

echo "===== PRE-REPAIR RANKER ON EXPANDED DEV ====="
python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --ranker "$BASE_RANKER" \
  "${COMMON_CURRICULUM_ARGS[@]}" \
  --output-dir "$BASELINE_EVAL" \
  --split dev \
  --max-length 512 \
  --batch-size 8 \
  --device cuda

echo "===== FROZEN-BACKBONE FAILURE-DRIVEN REPAIR ====="
python "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  "${COMMON_CURRICULUM_ARGS[@]}" \
  --output-dir "$RANKER_DIR" \
  --max-length 512 \
  --batch-size "${N0_REPAIR_BATCH_SIZE:-4}" \
  --epochs "${N0_REPAIR_EPOCHS:-12}" \
  --learning-rate "${N0_REPAIR_LR:-2e-4}" \
  --freeze-backbone

echo "===== REPAIRED TRAIN-SPLIT DIAGNOSTIC ====="
python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --ranker "$RANKER_DIR/ranker.safetensors" \
  "${COMMON_CURRICULUM_ARGS[@]}" \
  --output-dir "$TRAIN_EVAL" \
  --split train \
  --max-length 512 \
  --batch-size 8 \
  --device cuda

echo "===== REPAIRED DEV-SPLIT DIAGNOSTIC ====="
python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --ranker "$RANKER_DIR/ranker.safetensors" \
  "${COMMON_CURRICULUM_ARGS[@]}" \
  --output-dir "$DEV_EVAL" \
  --split dev \
  --max-length 512 \
  --batch-size 8 \
  --device cuda

echo "===== N0 FAILURE-DRIVEN REPAIR PROBE COMPLETE ====="
python - \
  "$STEP_DIR/receipt.json" \
  "$BASELINE_EVAL/dev_metrics.json" \
  "$RANKER_DIR/receipt.json" \
  "$TRAIN_EVAL/train_metrics.json" \
  "$DEV_EVAL/dev_metrics.json" \
  "$DEV_EVAL/dev_failures.jsonl" <<'PY'
import json
import sys
from pathlib import Path

step = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baseline = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
probe = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
train = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
dev = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
failures = [json.loads(line) for line in Path(sys.argv[6]).read_text(encoding="utf-8").splitlines() if line.strip()]

print(json.dumps({
    "checkpoint_step": step.get("step"),
    "tokens_seen_total": step.get("tokens_seen_total"),
    "backbone_frozen": probe.get("freeze_backbone"),
    "trainable_parameters": probe.get("trainable_parameters"),
    "baseline_expanded_dev_top1": baseline.get("top1_accuracy"),
    "baseline_expanded_dev_separation": baseline.get("supported_set_separation_rate"),
    "baseline_expanded_dev_failures": baseline.get("failures"),
    "repaired_train_top1": train.get("top1_accuracy"),
    "repaired_dev_top1": dev.get("top1_accuracy"),
    "repaired_dev_separation": dev.get("supported_set_separation_rate"),
    "repaired_dev_margin": dev.get("mean_separation_margin"),
    "repaired_dev_failures": dev.get("failures"),
    "remaining_failure_competencies": sorted({str(row["competency"]) for row in failures}),
    "private_identity_gradient": False,
}, indent=2, sort_keys=True))
PY
