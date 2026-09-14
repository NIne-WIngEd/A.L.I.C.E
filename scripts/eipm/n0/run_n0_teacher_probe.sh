#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_WORKDIR:-$HOME/rayan-compute/rayan-n0/n0-v01}"
STEP_DIR="${N0_MLM_STEP_DIR:-$WORKDIR/checkpoints/step-00001000}"
TOKENIZER_DIR="$WORKDIR/tokenizer"
CORPUS_DIR="$WORKDIR/corpus"
CONFIG="$ROOT/configs/eipm/n0/alice_n0_semantic_v0.1.json"
SOURCE_CONFIG="$ROOT/configs/eipm/n0/public_corpus_bootstrap_v0.1.json"
SEED_CURRICULUM="$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.jsonl"
SEED_MANIFEST="$ROOT/training/eipm/n0/sol_curriculum_seed_v0.1.origin.json"
COVERAGE_CURRICULUM="$ROOT/training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl"
COVERAGE_MANIFEST="$ROOT/training/eipm/n0/sol_curriculum_coverage_v0.2.origin.json"
PROBE_ROOT="${N0_TEACHER_PROBE_DIR:-$WORKDIR/teacher-probes/step-00001000-sol-v0.2}"
MLM_EVAL="$PROBE_ROOT/mlm-dev-multimask.json"
RANKER_DIR="$PROBE_ROOT/frozen-ranker"
RANKER_EVAL="$PROBE_ROOT/ranker-eval"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$STEP_DIR/receipt.json" || ! -d "$STEP_DIR/model" ]]; then
  echo "Complete step checkpoint required: $STEP_DIR" >&2
  exit 2
fi
if [[ -e "$PROBE_ROOT" ]]; then
  echo "Teacher probe output already exists; refusing duplicate diagnostic: $PROBE_ROOT" >&2
  exit 3
fi
mkdir -p "$PROBE_ROOT" "$RANKER_DIR" "$RANKER_EVAL"

python -m py_compile \
  "$ROOT/scripts/eipm/n0/evaluate_mlm.py" \
  "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
  "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  "$ROOT/src/alice_personality/n0/curriculum_data.py" \
  "$ROOT/src/alice_personality/n0/ranker.py"

echo "===== GOVERNED CURRICULUM PREFLIGHT ====="
python - "$SEED_CURRICULUM" "$SEED_MANIFEST" "$COVERAGE_CURRICULUM" "$COVERAGE_MANIFEST" <<'PY'
import json
import sys
from alice_personality.n0.curriculum import validate_curriculum_manifest, validate_curriculum_rows
from alice_personality.n0.curriculum_data import CurriculumDataset

curricula = [sys.argv[1], sys.argv[3]]
manifests = [sys.argv[2], sys.argv[4]]
entries = []
for curriculum, manifest in zip(curricula, manifests):
    loaded = validate_curriculum_manifest(curriculum, manifest)
    summary = validate_curriculum_rows(curriculum)
    entries.append({
        "curriculum": curriculum,
        "actor": loaded.get("actor"),
        "origin_type": loaded.get("origin_type"),
        "summary": summary,
    })
train = CurriculumDataset(curricula, "train")
dev = CurriculumDataset(curricula, "dev")
print(json.dumps({
    "status": "PASS",
    "curricula": entries,
    "combined_train_rows": len(train),
    "combined_dev_rows": len(dev),
    "train_competencies": len({str(row["competency"]) for row in train.rows}),
    "dev_competencies": len({str(row["competency"]) for row in dev.rows}),
}, indent=2, sort_keys=True))
PY

echo "===== ROBUST TRUE-HELD-OUT MLM EVALUATION ====="
python "$ROOT/scripts/eipm/n0/evaluate_mlm.py" \
  --config "$CONFIG" \
  --checkpoint-dir "$STEP_DIR" \
  --tokenizer "$TOKENIZER_DIR/tokenizer.json" \
  --tokenizer-receipt "$TOKENIZER_DIR/tokenizer_receipt.json" \
  --corpus-dir "$CORPUS_DIR" \
  --source-config "$SOURCE_CONFIG" \
  --sequence-length 512 \
  --batch-size 4 \
  --max-batches 256 \
  --seed 424242 \
  --mask-repeats "${N0_MLM_EVAL_MASK_REPEATS:-8}" \
  --device cuda \
  --output "$MLM_EVAL"

echo "===== FROZEN-BACKBONE SOL SEMANTIC PROBE ====="
python "$ROOT/scripts/eipm/n0/train_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --curriculum "$SEED_CURRICULUM" \
  --curriculum-manifest "$SEED_MANIFEST" \
  --curriculum "$COVERAGE_CURRICULUM" \
  --curriculum-manifest "$COVERAGE_MANIFEST" \
  --output-dir "$RANKER_DIR" \
  --max-length 512 \
  --batch-size "${N0_PROBE_BATCH_SIZE:-4}" \
  --epochs "${N0_PROBE_EPOCHS:-30}" \
  --learning-rate "${N0_PROBE_LR:-5e-4}" \
  --freeze-backbone

echo "===== PROBE TRAIN-SPLIT DIAGNOSTIC ====="
python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --ranker "$RANKER_DIR/ranker.safetensors" \
  --curriculum "$SEED_CURRICULUM" \
  --curriculum-manifest "$SEED_MANIFEST" \
  --curriculum "$COVERAGE_CURRICULUM" \
  --curriculum-manifest "$COVERAGE_MANIFEST" \
  --output-dir "$RANKER_EVAL/train" \
  --split train \
  --max-length 512 \
  --batch-size 8 \
  --device cuda

echo "===== PROBE DEV-SPLIT DIAGNOSTIC ====="
python "$ROOT/scripts/eipm/n0/evaluate_curriculum_ranker.py" \
  --config "$CONFIG" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --mlm-checkpoint "$STEP_DIR/model" \
  --ranker "$RANKER_DIR/ranker.safetensors" \
  --curriculum "$SEED_CURRICULUM" \
  --curriculum-manifest "$SEED_MANIFEST" \
  --curriculum "$COVERAGE_CURRICULUM" \
  --curriculum-manifest "$COVERAGE_MANIFEST" \
  --output-dir "$RANKER_EVAL/dev" \
  --split dev \
  --max-length 512 \
  --batch-size 8 \
  --device cuda

echo "===== N0 TEACHER PROBE COMPLETE ====="
python - "$STEP_DIR/receipt.json" "$MLM_EVAL" "$RANKER_DIR/receipt.json" "$RANKER_EVAL/dev/dev_metrics.json" <<'PY'
import json
import sys
from pathlib import Path
step = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mlm = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
probe = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
dev = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
print(json.dumps({
    "checkpoint_step": step.get("step"),
    "tokens_seen_total": step.get("tokens_seen_total"),
    "mlm_mean_nll": mlm.get("mean_masked_token_nll"),
    "mlm_perplexity": mlm.get("masked_token_perplexity"),
    "mlm_repeat_nll_stddev": mlm.get("repeat_nll_stddev"),
    "probe_freeze_backbone": probe.get("freeze_backbone"),
    "probe_trainable_parameters": probe.get("trainable_parameters"),
    "probe_dev_top1": dev.get("top1_accuracy"),
    "probe_dev_separation": dev.get("supported_set_separation_rate"),
    "probe_dev_margin": dev.get("mean_separation_margin"),
    "probe_dev_failures": dev.get("failures"),
    "private_identity_gradient": False,
}, indent=2, sort_keys=True))
PY
