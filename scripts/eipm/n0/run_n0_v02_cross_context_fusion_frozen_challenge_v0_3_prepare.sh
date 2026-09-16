#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.3"
CHALLENGE="$OUT_ROOT/challenge.jsonl"
MANIFEST="$OUT_ROOT/manifest.json"
FREEZE_RECEIPT="$OUT_ROOT/freeze_receipt.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_frozen_challenge_v0.3.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge_v0_3.py"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_cross_context_fusion_frozen_challenge_v0_3.py"
DOCTRINE="$ROOT/docs/eipm/EIPM_FUSION_ROUTING_VALIDATION_DOCTRINE_2026-09-16.md"
REPAIR_ROOT="$WORKDIR/cross-context-fusion-repair-training-v0.2"
REPAIR_COMPARISON="$REPAIR_ROOT/cross_context_fusion_repair_comparison.json"
CANDIDATE="$REPAIR_ROOT/repair-step-00000240/cross_context_fusion.safetensors"
V02_RESULT="$WORKDIR/cross-context-fusion-frozen-challenge-v0.2/result.json"
BASE_CURRICULUM="$WORKDIR/cross-context-fusion-curriculum-v0.2/cross_context_fusion_curriculum.jsonl"
REPAIR_CURRICULUM="$WORKDIR/cross-context-fusion-repair-curriculum-v0.3/cross_context_fusion_repair_curriculum.jsonl"
V01_CHALLENGE="$WORKDIR/cross-context-fusion-frozen-challenge-v0.1/challenge.jsonl"
V02_CHALLENGE="$WORKDIR/cross-context-fusion-frozen-challenge-v0.2/challenge.jsonl"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in "$SPEC" "$BUILDER" "$EVALUATOR" "$DOCTRINE" "$REPAIR_COMPARISON" "$CANDIDATE" "$V02_RESULT" "$BASE_CURRICULUM" "$REPAIR_CURRICULUM" "$V01_CHALLENGE" "$V02_CHALLENGE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing constraint-confirmatory artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing v0.3 frozen challenge: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile \
  "$ROOT/src/alice_personality/n0/cross_context_fusion_routing_constraints.py" \
  "$BUILDER" "$EVALUATOR"
pytest -q "$ROOT/tests/eipm/test_n0_cross_context_fusion_routing_constraints.py"

mkdir -p "$OUT_ROOT"
python "$BUILDER" --output "$CHALLENGE" --manifest "$MANIFEST" --spec "$SPEC"

python - "$SPEC" "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" "$DOCTRINE" "$REPAIR_COMPARISON" "$CANDIDATE" "$V02_RESULT" "$BASE_CURRICULUM" "$REPAIR_CURRICULUM" "$V01_CHALLENGE" "$V02_CHALLENGE" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

(spec_path, challenge_path, manifest_path, receipt_path, doctrine_path, repair_path, candidate_path, v02_result_path, *prior_paths) = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ids(path: Path) -> set[str]:
    return {str(json.loads(line)["id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

spec=json.loads(spec_path.read_text()); manifest=json.loads(manifest_path.read_text()); repair=json.loads(repair_path.read_text()); v02=json.loads(v02_result_path.read_text())
if spec.get("status") != "FROZEN_CONSTRAINT_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED": raise SystemExit("v0.3 spec status drift")
if spec.get("training_use") != "FORBIDDEN": raise SystemExit("v0.3 training use must be forbidden")
if spec.get("v0_2_result_reclassified") is not False: raise SystemExit("v0.2 may not be retroactively reclassified")
if spec.get("candidate_selection",{}).get("model_weights_changed_after_v0_2") is not False: raise SystemExit("v0.3 requires unchanged candidate weights")
if spec.get("ratification_gate",{}).get("thresholds_are_frozen_before_results") is not True: raise SystemExit("v0.3 thresholds not frozen")
if manifest.get("status") != "FROZEN_UNTOUCHED_CONSTRAINT_CONFIRMATORY_NOT_EVALUATED": raise SystemExit("v0.3 manifest status drift")
if manifest.get("challenge_sha256") != sha(challenge_path) or manifest.get("spec_sha256") != sha(spec_path): raise SystemExit("v0.3 challenge/spec hash drift")
if manifest.get("exact_target_routing_distribution_used_for_ratification") is not False: raise SystemExit("exact routing distribution remains a ratification target")
if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False: raise SystemExit("v0.3 governance drift")
if v02.get("status") != "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_RATIFICATION" or v02.get("gradient_performed") is not False: raise SystemExit("expected immutable failed v0.2 result")
if repair.get("status") != "PASS_FULL_SCALE_REPAIR_READY_FOR_NEW_UNTOUCHED_CHALLENGE": raise SystemExit("repair comparison status drift")
expected_sha=str(spec["candidate_selection"]["fusion_sha256"])
if sha(candidate_path) != expected_sha or repair.get("winner",{}).get("fusion_sha256") != expected_sha: raise SystemExit("unchanged candidate hash drift")
new_ids=ids(challenge_path)
for prior in prior_paths:
    overlap=new_ids.intersection(ids(prior))
    if overlap: raise SystemExit(f"v0.3 challenge reuses prior row ids from {prior}: {sorted(overlap)[:5]}")
text=challenge_path.read_text(encoding="utf-8")
for forbidden in ("Cipher Ash","Relay Aster","Beacon Alder"):
    if forbidden in text: raise SystemExit(f"v0.3 challenge reused a prior canonical entity: {forbidden}")
head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
receipt={
  "schema":"alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-freeze.v0.3",
  "status":"FROZEN_CONSTRAINT_CONFIRMATORY_BEFORE_EVALUATION",
  "created_at":datetime.now(timezone.utc).isoformat(),
  "git_revision":head,
  "challenge_sha256":sha(challenge_path),"manifest_sha256":sha(manifest_path),"spec_sha256":sha(spec_path),"doctrine_sha256":sha(doctrine_path),
  "repair_comparison_sha256":sha(repair_path),"v0_2_failed_result_sha256":sha(v02_result_path),"candidate_fusion_sha256":sha(candidate_path),
  "candidate_repair_step":240,"candidate_preselected_before_challenge":True,"candidate_weights_changed_after_v0_2":False,"checkpoint_selection_on_challenge_forbidden":True,
  "routing_validation":"behavioral_constraints_not_exact_soft_distribution_similarity","exact_target_routing_distribution_used_for_ratification":False,
  "rows":160,"families":20,"thresholds_frozen_before_results":True,"challenge_rows_used_for_training":False,"training_authorized":False,"gradient_performed":False,"results_observed":False,
  "private_identity_data":False,"private_identity_gradient":False
}
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print("fusion_constraint_confirmatory_v03_prepare_pass=true")
print(f"git_revision={head}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print("challenge_rows=160 families=20")
print("candidate_preselected_before_challenge=true")
print("candidate_weights_changed_after_v0_2=false")
print("routing_validation=behavioral_constraints_not_exact_soft_distribution_similarity")
print("exact_target_routing_distribution_used_for_ratification=false")
print("thresholds_frozen_before_results=true")
print("challenge_rows_used_for_training=false")
print("gradient_performed=false")
print(f"freeze_receipt={receipt_path}")
PY
