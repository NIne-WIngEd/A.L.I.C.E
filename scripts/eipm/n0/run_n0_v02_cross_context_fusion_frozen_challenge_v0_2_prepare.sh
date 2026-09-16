#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.2"
CHALLENGE="$OUT_ROOT/challenge.jsonl"
MANIFEST="$OUT_ROOT/manifest.json"
FREEZE_RECEIPT="$OUT_ROOT/freeze_receipt.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.2.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py"
ORIGINAL_CURRICULUM="$WORKDIR/cross-context-fusion-curriculum-v0.2/cross_context_fusion_curriculum.jsonl"
REPAIR_CURRICULUM="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3/cross_context_fusion_repair_curriculum.jsonl"
RETIRED_CHALLENGE="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1/challenge.jsonl"
REPAIR_ROOT="$WORKDIR/cross-context-fusion-repair-training-v0.2"
REPAIR_COMPARISON="$REPAIR_ROOT/cross_context_fusion_repair_comparison.json"
CANDIDATE="$REPAIR_ROOT/repair-step-00000240"
CANDIDATE_MODEL="$CANDIDATE/cross_context_fusion.safetensors"
CANDIDATE_RECEIPT="$CANDIDATE/receipt.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

for required in "$SPEC" "$BUILDER" "$EVALUATOR" "$REPAIR_COMPARISON" "$CANDIDATE_MODEL" "$CANDIDATE_RECEIPT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing confirmatory challenge artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing confirmatory challenge v0.2: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$BUILDER" "$EVALUATOR"
mkdir -p "$OUT_ROOT"
python "$BUILDER" --output "$CHALLENGE" --manifest "$MANIFEST" --spec "$SPEC"

python - "$SPEC" "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" "$REPAIR_COMPARISON" "$CANDIDATE_MODEL" "$CANDIDATE_RECEIPT" "$ORIGINAL_CURRICULUM" "$REPAIR_CURRICULUM" "$RETIRED_CHALLENGE" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

spec_path, challenge_path, manifest_path, receipt_path = map(Path, sys.argv[1:5])
repair_comparison_path, candidate_model_path, candidate_receipt_path = map(Path, sys.argv[5:8])
comparison_sources = [Path(p) for p in sys.argv[8:11]]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(str(json.loads(line)["id"]))
    return result

spec = json.loads(spec_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
repair = json.loads(repair_comparison_path.read_text(encoding="utf-8"))
candidate_receipt = json.loads(candidate_receipt_path.read_text(encoding="utf-8"))

if spec.get("status") != "FROZEN_CONFIRMATORY_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
    raise SystemExit("confirmatory spec status drift")
if spec.get("training_use") != "FORBIDDEN":
    raise SystemExit("confirmatory training use must be forbidden")
if spec.get("candidate_selection", {}).get("preselected_before_challenge") is not True:
    raise SystemExit("confirmatory candidate was not preselected")
if spec.get("candidate_selection", {}).get("challenge_may_not_select_a_different_checkpoint") is not True:
    raise SystemExit("confirmatory challenge may not select a different checkpoint")
if spec.get("ratification_gate", {}).get("thresholds_are_frozen_before_results") is not True:
    raise SystemExit("confirmatory thresholds are not frozen")
if spec.get("challenge", {}).get("rows") != 136 or spec.get("challenge", {}).get("families") != 17:
    raise SystemExit("confirmatory spec coverage drift")

if manifest.get("status") != "FROZEN_UNTOUCHED_CONFIRMATORY_NOT_EVALUATED":
    raise SystemExit("confirmatory manifest status drift")
if manifest.get("challenge_sha256") != sha(challenge_path):
    raise SystemExit("confirmatory challenge hash mismatch")
if manifest.get("spec_sha256") != sha(spec_path):
    raise SystemExit("confirmatory spec hash mismatch")
if manifest.get("rows") != 136 or manifest.get("family_count") != 17:
    raise SystemExit("confirmatory manifest coverage drift")
for key in (
    "original_training_rows_reused",
    "repair_training_rows_reused",
    "retired_challenge_rows_reused",
    "training_entities_reused",
    "training_text_templates_reused",
    "retired_challenge_entities_reused",
    "retired_challenge_text_templates_reused",
):
    if manifest.get(key) is not False:
        raise SystemExit(f"confirmatory independence flag drift: {key}")
if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False:
    raise SystemExit("confirmatory challenge was not frozen cleanly")

challenge_ids = ids(challenge_path)
for source in comparison_sources:
    overlap = challenge_ids & ids(source)
    if overlap:
        raise SystemExit(f"confirmatory challenge id overlap with {source}: {sorted(overlap)[:3]}")

candidate = spec["candidate_selection"]
expected_step = int(candidate["repair_step"])
expected_sha = str(candidate["fusion_sha256"])
if repair.get("status") != "PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE":
    raise SystemExit("repair comparison is not ready for confirmatory challenge")
if int(repair.get("provisional_winner_repair_step", -1)) != expected_step:
    raise SystemExit("repair winner step differs from frozen candidate")
if repair.get("winner", {}).get("fusion_sha256") != expected_sha:
    raise SystemExit("repair winner hash differs from frozen candidate")
if repair.get("winner", {}).get("repair_gate_pass") is not True:
    raise SystemExit("repair winner did not pass repair gate")
if candidate_receipt.get("status") != "TRAINED_FULL_SCALE_REPAIR_NOT_RATIFIED":
    raise SystemExit("candidate receipt status drift")
if int(candidate_receipt.get("repair_step", -1)) != expected_step:
    raise SystemExit("candidate receipt step drift")
if candidate_receipt.get("fusion_sha256") != expected_sha or sha(candidate_model_path) != expected_sha:
    raise SystemExit("candidate hash differs from frozen spec")
if candidate_receipt.get("private_identity_gradient") is not False:
    raise SystemExit("candidate crossed private gradient boundary")

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
receipt = {
    "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-freeze.v0.2",
    "status": "FROZEN_CONFIRMATORY_BEFORE_EVALUATION",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": head,
    "challenge_sha256": sha(challenge_path),
    "manifest_sha256": sha(manifest_path),
    "spec_sha256": sha(spec_path),
    "repair_comparison_sha256": sha(repair_comparison_path),
    "candidate_fusion_sha256": expected_sha,
    "candidate_repair_step": expected_step,
    "candidate_preselected_before_challenge": True,
    "checkpoint_selection_on_challenge_forbidden": True,
    "rows": 136,
    "families": 17,
    "thresholds_frozen_before_results": True,
    "challenge_rows_used_for_training": False,
    "training_authorized": False,
    "gradient_performed": False,
    "results_observed": False,
    "private_identity_data": False,
    "private_identity_gradient": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("fusion_confirmatory_v02_prepare_pass=true")
print(f"git_revision={head}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print("challenge_rows=136 families=17")
print("candidate_preselected_before_challenge=true")
print("candidate_repair_step=240")
print(f"candidate_fusion_sha256={expected_sha}")
print("checkpoint_selection_on_challenge_forbidden=true")
print("thresholds_frozen_before_results=true")
print("challenge_rows_used_for_training=false")
print("gradient_performed=false")
print(f"freeze_receipt={receipt_path}")
PY
