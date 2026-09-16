#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1_1 as prior

# New entities/values for a genuinely new untouched v0.2 corpus. The family
# semantics are preserved, but no v0.1 row ids, subjects, or concrete values are
# reused.
NEW_SUBJECTS = [
    "Matrix Alder", "Matrix Briar", "Matrix Copper", "Matrix Drift",
    "Matrix Elm", "Matrix Flint", "Matrix Garnet", "Matrix Heath",
    "Matrix Iris", "Matrix Juniper", "Matrix Kyanite", "Matrix Laurel",
    "Matrix Moss", "Matrix Nacre", "Matrix Osprey", "Matrix Pine",
]
NEW_ATTRIBUTES = [
    ("handoff mode", "manual", "coordinated"),
    ("control epoch", "17", "41"),
    ("service profile", "quiet", "burst"),
    ("refresh period", "13 seconds", "29 seconds"),
    ("replica policy", "solo", "paired"),
    ("telemetry class", "basic", "extended"),
    ("routing lane", "secondary", "primary"),
    ("snapshot interval", "12 minutes", "34 minutes"),
    ("worker quota", "4", "9"),
    ("sampling mode", "sparse", "dense"),
]


def build_row(family: str, index: int) -> dict:
    prior.base.SUBJECTS = NEW_SUBJECTS
    prior.base.ATTRIBUTES = NEW_ATTRIBUTES
    item = prior.build_row(family, index)
    item["id"] = item["id"].replace("N0V02-LATCHAL1-", "N0V02-LATCHAL2-")
    item["challenge_version"] = "v0.2"
    item["counterfactual_intervention_point"] = (
        "before_parent_cache_and_before_cross_context_fusion"
        if item.get("counterfactual_required")
        else None
    )
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("latent v0.2 challenge spec status drift")
    if spec.get("challenge", {}).get("counterfactual_intervention_point") != "before_parent_cache_and_before_cross_context_fusion":
        raise SystemExit("latent v0.2 challenge intervention-point drift")

    rows = [build_row(family, index) for family in prior.base.FAMILIES for index in range(prior.base.ROWS_PER_FAMILY)]
    if len(rows) != 160 or len({row["id"] for row in rows}) != 160:
        raise SystemExit("latent v0.2 row count/id uniqueness drift")
    counts = Counter(str(row["family"]) for row in rows)
    if set(counts) != set(prior.base.FAMILIES) or any(value != prior.base.ROWS_PER_FAMILY for value in counts.values()):
        raise SystemExit("latent v0.2 family coverage drift")

    counterfactual_rows = [row for row in rows if row.get("counterfactual_required")]
    if len(counterfactual_rows) != 32:
        raise SystemExit(f"latent v0.2 expected 32 counterfactual rows, got {len(counterfactual_rows)}")
    if any(row.get("counterfactual_intervention_point") != "before_parent_cache_and_before_cross_context_fusion" for row in counterfactual_rows):
        raise SystemExit("latent v0.2 counterfactual row intervention drift")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-manifest.v0.2",
        "status": "FROZEN_UNTOUCHED_NOT_EVALUATED",
        "rows": len(rows),
        "family_count": len(counts),
        "family_counts": dict(sorted(counts.items())),
        "counterfactual_required_rows": len(counterfactual_rows),
        "counterfactual_families": sorted({str(row["family"]) for row in counterfactual_rows}),
        "challenge_sha256": prior.base.sha(output),
        "spec_sha256": prior.base.sha(spec_path),
        "data_origin": "deterministic_public_synthetic_challenge_template_new_entities_values",
        "generated_text": True,
        "identity_authority": False,
        "private_identity_content": False,
        "training_authorized": False,
        "results_observed_at_compile_time": False,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "slot_permutation_invariant_evaluation": True,
        "counterfactual_policy": "only_uniquely_necessary_authority_view_is_removed",
        "counterfactual_intervention_point": "before_parent_cache_and_before_cross_context_fusion",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
