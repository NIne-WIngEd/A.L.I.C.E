#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1 as base

NO_COUNTERFACTUAL = {
    "supersession_current",
    "supersession_historical",
    "correction_chain_current",
    "correction_chain_historical",
    "temporal_successor_current",
    "temporal_successor_previous",
    "derived_support_chain",
    "causal_chain",
    "missing_semantic_relational",
    "evidence_reliability_reversal",
    "lexical_high_overlap_distractor",
    "paired_current_state",
    "paired_historical_state",
}


def build_row(family: str, index: int) -> dict:
    item = base.build_row(family, index)
    if family in NO_COUNTERFACTUAL:
        item["counterfactual_required"] = False
        item["counterfactual_view"] = None

    if family == "novel_structured_authority":
        # Make the structured channel uniquely authoritative for the causal-use
        # test. The evidence view is deliberately unavailable rather than a
        # redundant copy of the same field text.
        item["view_available"] = [True, True, False]
        item["view_reliability"] = [0.35, 0.995, 0.0]
        item["target_view_distribution"] = base.compat_distribution(item["view_available"])
        item["counterfactual_required"] = True
        item["counterfactual_view"] = 1

    if family == "missing_structured_evidence":
        # Here the semantic channel is intentionally non-answering and the
        # structured channel is absent, so relation/evidence is uniquely
        # necessary for the requested current state.
        subject, attribute, previous, current = base.parts(index, base.FAMILIES.index(family))
        item["raw_text"] = (
            f"Two records disagree about {subject}'s {attribute}; the prose does not identify which record is current."
        )
        item["query_text"] = f"According to the correction relation, what is {subject}'s current {attribute}?"
        item["target_summary_text"] = f"{subject}'s current {attribute} is {current}."
        item["view_reliability"] = [0.40, 0.0, 0.99]
        item["counterfactual_required"] = True
        item["counterfactual_view"] = 2

    # These two semantic-authority families are genuinely unique because the
    # other channels contain only an unknown/contradictory lower-authority state.
    if family in {"novel_semantic_authority", "semantic_reliability_reversal"}:
        item["counterfactual_required"] = True
        item["counterfactual_view"] = 0

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
        raise SystemExit("latent challenge spec status drift")

    rows = [build_row(family, index) for family in base.FAMILIES for index in range(base.ROWS_PER_FAMILY)]
    if len(rows) != 160 or len({row["id"] for row in rows}) != 160:
        raise SystemExit("latent challenge row count/id uniqueness drift")
    counts = Counter(str(row["family"]) for row in rows)
    if set(counts) != set(base.FAMILIES) or any(value != base.ROWS_PER_FAMILY for value in counts.values()):
        raise SystemExit("latent challenge family coverage drift")
    counterfactual_families = {
        str(row["family"]) for row in rows if row.get("counterfactual_required")
    }
    expected_counterfactual_families = {
        "novel_semantic_authority",
        "novel_structured_authority",
        "missing_structured_evidence",
        "semantic_reliability_reversal",
    }
    if counterfactual_families != expected_counterfactual_families:
        raise SystemExit(
            f"counterfactual family drift: {sorted(counterfactual_families)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-manifest.v0.1.1",
        "status": "FROZEN_UNTOUCHED_NOT_EVALUATED",
        "rows": len(rows),
        "family_count": len(counts),
        "family_counts": dict(sorted(counts.items())),
        "counterfactual_required_rows": sum(1 for row in rows if row.get("counterfactual_required")),
        "counterfactual_families": sorted(counterfactual_families),
        "challenge_sha256": base.sha(output),
        "spec_sha256": base.sha(spec_path),
        "data_origin": "deterministic_public_synthetic_challenge_template",
        "generated_text": True,
        "identity_authority": False,
        "private_identity_content": False,
        "training_authorized": False,
        "results_observed_at_compile_time": False,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "slot_permutation_invariant_evaluation": True,
        "counterfactual_policy": "only_uniquely_necessary_authority_view_is_removed",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
