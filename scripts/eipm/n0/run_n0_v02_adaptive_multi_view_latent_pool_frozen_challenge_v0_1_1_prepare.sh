#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="${ALICE_N0_V02_WORKDIR:-$(dirname "$ROOT")/rayan-n0/n0-v02}"
OUT_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-frozen-challenge-v0.1"
CHALLENGE="$OUT_ROOT/challenge.jsonl"
MANIFEST="$OUT_ROOT/manifest.json"
FREEZE_RECEIPT="$OUT_ROOT/freeze_receipt.json"
SPEC="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0.1.json"
BUILDER="$ROOT/scripts/eipm/n0/build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1_1.py"
EVALUATOR="$ROOT/scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py"
SOURCE_ROOT="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.2.1"
SOURCE_COMPARISON="$SOURCE_ROOT/adaptive_multi_view_latent_pool_v0_2_comparison.json"
CANDIDATE="$SOURCE_ROOT/step-00000360/adaptive_multi_view_latent_pool_v0_2.safetensors"
TRAIN_PARENT_CACHE="$WORKDIR/adaptive-multi-view-latent-pool-training-v0.1/parent_cache.pt"
LATENT_CONFIG="$ROOT/configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.2.json"
FUSION_RATIFICATION="$ROOT/configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json"

export PYTHONPATH="$ROOT/src:$ROOT/scripts/eipm/n0${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required in "$SPEC" "$BUILDER" "$EVALUATOR" "$SOURCE_COMPARISON" "$CANDIDATE" "$TRAIN_PARENT_CACHE" "$LATENT_CONFIG" "$FUSION_RATIFICATION"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing latent confirmatory prerequisite: $required" >&2
    exit 2
  fi
done

if [[ -e "$OUT_ROOT" && -n "$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite existing latent frozen challenge: $OUT_ROOT" >&2
  exit 3
fi

python -m py_compile "$BUILDER" "$EVALUATOR"
mkdir -p "$OUT_ROOT"
python "$BUILDER" --output "$CHALLENGE" --manifest "$MANIFEST" --spec "$SPEC"

python - "$SPEC" "$CHALLENGE" "$MANIFEST" "$FREEZE_RECEIPT" "$SOURCE_COMPARISON" "$CANDIDATE" "$TRAIN_PARENT_CACHE" "$LATENT_CONFIG" "$FUSION_RATIFICATION" "$BUILDER" "$EVALUATOR" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import torch

(spec_path, challenge_path, manifest_path, receipt_path, comparison_path, candidate_path, train_cache_path, latent_config_path, fusion_ratification_path, builder_path, evaluator_path) = map(Path, sys.argv[1:])

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

spec=json.loads(spec_path.read_text(encoding="utf-8"))
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
comparison=json.loads(comparison_path.read_text(encoding="utf-8"))
ratification=json.loads(fusion_ratification_path.read_text(encoding="utf-8"))
if spec.get("status") != "FROZEN_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED": raise SystemExit("latent frozen spec status drift")
if spec.get("training_use") != "FORBIDDEN": raise SystemExit("latent challenge training use must be forbidden")
if spec.get("ratification_gate",{}).get("thresholds_are_frozen_before_results") is not True: raise SystemExit("latent challenge thresholds not frozen")
if comparison.get("status") != "PASS_COMPETITIVE_LATENT_POOL_READY_FOR_UNTOUCHED_CHALLENGE": raise SystemExit("latent source comparison is not challenge-ready")
winner=comparison.get("winner",{})
if int(winner.get("step",-1)) != 360: raise SystemExit("latent candidate was not preselected step360")
expected=str(spec["candidate_selection"]["latent_pool_sha256"])
if winner.get("latent_pool_sha256") != expected or sha(candidate_path) != expected: raise SystemExit("latent preselected candidate hash drift")
if manifest.get("status") != "FROZEN_UNTOUCHED_NOT_EVALUATED": raise SystemExit("latent challenge manifest status drift")
if manifest.get("challenge_sha256") != sha(challenge_path) or manifest.get("spec_sha256") != sha(spec_path): raise SystemExit("latent challenge hash drift")
if manifest.get("training_authorized") is not False or manifest.get("results_observed_at_compile_time") is not False: raise SystemExit("latent challenge governance drift")
if manifest.get("counterfactual_policy") != "only_uniquely_necessary_authority_view_is_removed": raise SystemExit("latent challenge counterfactual contract drift")
if set(manifest.get("counterfactual_families",[])) != {"novel_semantic_authority","novel_structured_authority","missing_structured_evidence","semantic_reliability_reversal"}: raise SystemExit("latent challenge counterfactual family drift")
if ratification.get("status") != "RATIFIED_PUBLIC_N0_FUSION_BASE_NOT_FULL_EIPM_PROMOTION": raise SystemExit("fusion ratification missing")

challenge_rows=rows(challenge_path)
new_ids={str(row["id"]) for row in challenge_rows}
train_cache=torch.load(train_cache_path,map_location="cpu",weights_only=False)
train_ids={str(value) for value in train_cache.get("ids",[])}
if new_ids.intersection(train_ids): raise SystemExit("latent challenge reuses training row ids")
if len(challenge_rows) != 160 or len({row["family"] for row in challenge_rows}) != 20: raise SystemExit("latent challenge coverage drift")
if any(row.get("training_authorized") is not False for row in challenge_rows): raise SystemExit("latent challenge row authorizes training")
text=challenge_path.read_text(encoding="utf-8")
for forbidden in ("Array Juniper", "Cipher Ash", "Relay Aster", "Beacon Alder"):
    if forbidden in text: raise SystemExit(f"latent challenge reused prior canonical entity: {forbidden}")

head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
receipt={
  "schema":"alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-freeze.v0.1.1",
  "status":"FROZEN_CONFIRMATORY_BEFORE_EVALUATION",
  "created_at":datetime.now(timezone.utc).isoformat(),
  "git_revision":head,
  "challenge_sha256":sha(challenge_path),"manifest_sha256":sha(manifest_path),"spec_sha256":sha(spec_path),
  "source_comparison_sha256":sha(comparison_path),"candidate_latent_pool_sha256":sha(candidate_path),
  "latent_config_sha256":sha(latent_config_path),"fusion_ratification_sha256":sha(fusion_ratification_path),
  "builder_sha256":sha(builder_path),"evaluator_sha256":sha(evaluator_path),
  "candidate_step":360,"candidate_preselected_before_challenge":True,"checkpoint_selection_on_challenge_forbidden":True,
  "rows":160,"families":20,"counterfactual_required_rows":int(manifest["counterfactual_required_rows"]),
  "counterfactual_policy":manifest["counterfactual_policy"],
  "thresholds_frozen_before_results":True,"challenge_rows_used_for_training":False,"training_authorized":False,
  "gradient_performed":False,"results_observed":False,"private_identity_data":False,"private_identity_gradient":False,
  "mean_max_offdiag_slot_cosine_is_gate":False
}
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("adaptive_latent_pool_confirmatory_v011_prepare_pass=true")
print(f"git_revision={head}")
print(f"challenge_sha256={receipt['challenge_sha256']}")
print("challenge_rows=160 families=20")
print(f"counterfactual_required_rows={receipt['counterfactual_required_rows']}")
print("counterfactual_policy=only_uniquely_necessary_authority_view_is_removed")
print("candidate_step=360")
print("candidate_preselected_before_challenge=true")
print("checkpoint_selection_on_challenge_forbidden=true")
print("mean_max_offdiag_slot_cosine_is_gate=false")
print("thresholds_frozen_before_results=true")
print("challenge_rows_used_for_training=false")
print("gradient_performed=false")
print(f"freeze_receipt={receipt_path}")
PY
