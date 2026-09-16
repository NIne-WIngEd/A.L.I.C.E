#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.2"
CHALLENGE="$OUT_ROOT/challenge.jsonl"
MANIFEST="$OUT_ROOT/manifest.json"
FREEZE_RECEIPT="$OUT_ROOT/freeze_receipt.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0.2.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2.py"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2.py"
AUDIT="$ROOT/docs/eipm/N0_LATENT_POOL_V01_CHALLENGE_FAILURE_AND_SCALE_ADEQUACY_AUDIT_2026-09-16.md"
TRAIN_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1"
COMPARISON="$TRAIN_ROOT/adaptive_multi_view_latent_pool_v0_2_comparison.json"
CANDIDATE="$TRAIN_ROOT/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
V01_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.1"
V01_CHALLENGE="$V01_ROOT/challenge.jsonl"
V01_RESULT="$V01_ROOT/result.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in "$SPEC" "$BUILDER" "$EVALUATOR" "$AUDIT" "$COMPARISON" "$CANDIDATE" "$V01_CHALLENGE" "$V01_RESULT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing latent v0.2 confirmatory artifact: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing latent v0.2 frozen challenge: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$BUILDER" "$EVALUATOR"
mkdir -p "$OUT_ROOT"
python "$BUILDER" --output "$CHALLENGE" --manifest "$MANIFEST" --spec "$SPEC"

python - "$SPEC" "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" "$BUILDER" "$EVALUATOR" "$AUDIT" "$COMPARISON" "$CANDIDATE" "$V01_CHALLENGE" "$V01_RESULT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

(spec_path, challenge_path, manifest_path, receipt_path, builder_path, evaluator_path, audit_path, comparison_path, candidate_path, v01_challenge_path, v01_result_path) = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def ids(path: Path) -> set[str]:
    return {str(json.loads(line)["id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

spec = json.loads(spec_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
v01 = json.loads(v01_result_path.read_text(encoding="utf-8"))

if spec.get("status") != "FROZEN_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED":
    raise SystemExit("latent v0.2 spec status drift")
if spec.get("challenge", {}).get("training_use") != "FORBIDDEN":
    raise SystemExit("latent v0.2 training use must be forbidden")
if spec.get("historical_v0_1", {}).get("reclassified_as_pass") is not False:
    raise SystemExit("latent v0.1 may not be retroactively reclassified")
if spec.get("candidate_selection", {}).get("candidate_weights_changed_after_v0_1") is not False:
    raise SystemExit("latent v0.2 requires unchanged candidate weights")
if spec.get("ratification_gate", {}).get("thresholds_are_frozen_before_results") is not True:
    raise SystemExit("latent v0.2 thresholds are not frozen")
if spec.get("challenge", {}).get("counterfactual_intervention_point") != "before_parent_cache_and_before_cross_context_fusion":
    raise SystemExit("latent v0.2 intervention point drift")

if manifest.get("status") != "FROZEN_UNTOUCHED_NOT_EVALUATED":
    raise SystemExit("latent v0.2 manifest status drift")
if manifest.get("challenge_sha256") != sha(challenge_path) or manifest.get("spec_sha256") != sha(spec_path):
    raise SystemExit("latent v0.2 challenge/spec hash drift")
if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False:
    raise SystemExit("latent v0.2 governance drift")
if manifest.get("counterfactual_intervention_point") != "before_parent_cache_and_before_cross_context_fusion":
    raise SystemExit("latent v0.2 manifest intervention drift")

if v01.get("status") != "FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION":
    raise SystemExit("expected immutable latent v0.1 failed result")
if v01.get("gradient_performed") is not False:
    raise SystemExit("latent v0.1 challenge must remain no-gradient")

expected_sha = str(spec["candidate_selection"]["latent_pool_sha256"])
if sha(candidate_path) != expected_sha:
    raise SystemExit("latent v0.2 candidate hash drift")
if comparison.get("status") != "PASS_COMPETITIVE_LATENT_POOL_READY_FOR_UNTOUCHED_CHALLENGE":
    raise SystemExit("latent training comparison status drift")
if comparison.get("winner", {}).get("latent_pool_sha256") != expected_sha or int(comparison.get("winner", {}).get("step", -1)) != 360:
    raise SystemExit("latent preselected winner drift")

new_ids = ids(challenge_path)
old_ids = ids(v01_challenge_path)
if new_ids.intersection(old_ids):
    raise SystemExit("latent v0.2 challenge reuses v0.1 row ids")
text = challenge_path.read_text(encoding="utf-8")
if "Constellation " in text or "N0V02-LATCHAL1-" in text:
    raise SystemExit("latent v0.2 challenge reused v0.1 entity/id namespace")

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
receipt = {
    "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-freeze.v0.2",
    "status": "FROZEN_CONFIRMATORY_BEFORE_EVALUATION",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": head,
    "challenge_sha256": sha(challenge_path),
    "manifest_sha256": sha(manifest_path),
    "spec_sha256": sha(spec_path),
    "builder_sha256": sha(builder_path),
    "evaluator_sha256": sha(evaluator_path),
    "audit_sha256": sha(audit_path),
    "training_comparison_sha256": sha(comparison_path),
    "v0_1_failed_result_sha256": sha(v01_result_path),
    "candidate_latent_pool_sha256": sha(candidate_path),
    "candidate_step": 360,
    "candidate_preselected_before_challenge": True,
    "candidate_weights_changed_after_v0_1": False,
    "checkpoint_selection_on_challenge_forbidden": True,
    "rows": 160,
    "families": 20,
    "counterfactual_required_rows": 32,
    "counterfactual_intervention_point": "before_parent_cache_and_before_cross_context_fusion",
    "counterfactual_rebuilds_parent_and_fusion_representations": True,
    "thresholds_frozen_before_results": True,
    "challenge_rows_used_for_training": False,
    "training_authorized": False,
    "gradient_performed": False,
    "results_observed": False,
    "historical_v0_1_reclassified_as_pass": False,
    "private_identity_data": False,
    "private_identity_gradient": False,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("adaptive_latent_pool_confirmatory_v02_prepare_pass=true")
print(f"git_revision={head}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print("challenge_rows=160 families=20")
print("counterfactual_required_rows=32")
print("counterfactual_intervention_point=before_parent_cache_and_before_cross_context_fusion")
print("candidate_step=360")
print("candidate_preselected_before_challenge=true")
print("candidate_weights_changed_after_v0_1=false")
print("checkpoint_selection_on_challenge_forbidden=true")
print("thresholds_frozen_before_results=true")
print("challenge_rows_used_for_training=false")
print("gradient_performed=false")
print(f"freeze_receipt={receipt_path}")
PY
