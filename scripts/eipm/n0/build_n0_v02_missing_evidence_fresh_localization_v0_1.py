#!/usr/bin/env python3
"""Build fresh public relation-flip rows for missing-evidence path localization.

These rows are diagnostic-only. They are not frozen challenge rows, are never
training-authorized, and are deliberately constructed after the observed
challenge failure to localize where a uniquely necessary evidence relation is
lost in the N0 full stack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from alice_personality.n0.evidence_graph import relation_type_id

SUBJECTS = [
    "Beacon Aster", "Beacon Birch", "Beacon Cobalt", "Beacon Dune",
    "Beacon Ember", "Beacon Fjord", "Beacon Grove", "Beacon Harbor",
    "Beacon Ion", "Beacon Jade", "Beacon Kestrel", "Beacon Linden",
    "Beacon Marrow", "Beacon Nimbus", "Beacon Onyx", "Beacon Prism",
]

ATTRIBUTES = [
    ("dispatch phase", "amber", "violet"),
    ("audit window", "23 seconds", "47 seconds"),
    ("replica stance", "passive", "active"),
    ("handover tier", "delta", "sigma"),
    ("buffer target", "6", "14"),
    ("archive lane", "west", "east"),
    ("inspection span", "15 minutes", "38 minutes"),
    ("control band", "narrow", "broad"),
    ("recovery mode", "cold", "warm"),
    ("sampling tier", "light", "heavy"),
    ("queue policy", "serial", "parallel"),
    ("telemetry lane", "inner", "outer"),
    ("refresh mode", "deferred", "immediate"),
    ("mirror count", "5", "11"),
    ("handoff class", "local", "federated"),
    ("checkpoint cadence", "18 minutes", "43 minutes"),
]

ROWS_PER_PAIR = 2
PAIRS = 16


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compat_distribution(available: list[bool]) -> list[float]:
    count = sum(1 for value in available if value)
    if count < 1:
        raise ValueError("at least one view must remain available")
    return [1.0 / count if value else 0.0 for value in available]


def field(name: str, text: str) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": 1,
        "provenance_id": 2,
        "relation_role_id": 1,
        "temporal_scope_id": 1,
        "confidence": 0.99,
        "missing": False,
    }


def edge(source: int, target: int) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": "corrects",
        "relation_type_id": relation_type_id("corrects"),
        "confidence": 1.0,
    }


def make_row(pair_index: int, orientation: int) -> dict[str, Any]:
    subject = SUBJECTS[pair_index]
    attribute, value_a, value_b = ATTRIBUTES[pair_index]
    if orientation == 0:
        source, target = 1, 0
        answer, foil = value_b, value_a
    else:
        source, target = 0, 1
        answer, foil = value_a, value_b

    fields = [
        field("record_a", f"Evidence record A lists {value_a} for {subject}'s {attribute}."),
        field("record_b", f"Evidence record B lists {value_b} for {subject}'s {attribute}."),
    ]
    available = [True, False, True]
    return {
        "id": f"N0V02-MSEFRESH-P{pair_index + 1:02d}-O{orientation}",
        "pair_id": f"N0V02-MSEFRESH-P{pair_index + 1:02d}",
        "orientation": orientation,
        "family": "missing_structured_evidence_fresh_relation_flip",
        "split": "diagnostic",
        "raw_text": (
            f"Two records disagree about {subject}'s {attribute}. "
            "The prose copy does not identify which record corrects the other."
        ),
        "query_text": (
            f"Using only the correction relation, what is {subject}'s current {attribute}?"
        ),
        "fields": fields,
        "relations": [edge(source, target)],
        "target_summary_text": (
            f"According to the correction relation, {subject}'s current {attribute} is {answer}."
        ),
        "diagnostic_foil_summary_text": (
            f"According to the correction relation, {subject}'s current {attribute} is {foil}."
        ),
        "diagnostic_target_probe_text": f"{subject} {attribute} {answer}",
        "diagnostic_foil_probe_text": f"{subject} {attribute} {foil}",
        "diagnostic_target_value": answer,
        "diagnostic_foil_value": foil,
        "diagnostic_target_field_index": source,
        "diagnostic_foil_field_index": target,
        "target_view_distribution": compat_distribution(available),
        "target_view_distribution_role": (
            "parent_cache_compatibility_only_not_used_as_diagnostic_ground_truth"
        ),
        "view_reliability": [0.40, 0.0, 0.99],
        "view_available": available,
        "cross_view_conflict": True,
        "counterfactual_required": True,
        "counterfactual_view": 2,
        "counterfactual_intervention_point": (
            "before_parent_cache_and_before_cross_context_fusion"
        ),
        "diagnostic_ground_truth_role": (
            "relation_source_endpoint_of_corrects_edge_is_current_value"
        ),
        "training_authorized": False,
        "identity_authority": False,
        "private_identity_content": False,
    }


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != PAIRS * ROWS_PER_PAIR:
        raise SystemExit(f"expected {PAIRS * ROWS_PER_PAIR} rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("diagnostic row ids are not unique")

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)
        if row["view_available"] != [True, False, True]:
            raise SystemExit("fresh diagnostic must keep structured unavailable")
        if row["view_reliability"] != [0.40, 0.0, 0.99]:
            raise SystemExit("fresh diagnostic reliability contract drift")
        if row["counterfactual_view"] != 2:
            raise SystemExit("fresh diagnostic must ablate evidence view")
        if row["training_authorized"] is not False:
            raise SystemExit("fresh diagnostic may not authorize training")

    if len(by_pair) != PAIRS:
        raise SystemExit(f"expected {PAIRS} pairs, got {len(by_pair)}")

    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2:
            raise SystemExit(f"pair {pair_id} must contain two orientations")
        pair_rows = sorted(pair_rows, key=lambda item: int(item["orientation"]))
        a, b = pair_rows
        for key in ("raw_text", "query_text", "fields", "view_reliability", "view_available"):
            if a[key] != b[key]:
                raise SystemExit(f"pair {pair_id} changed {key}; only relation direction may flip")
        if a["relations"][0]["source"] != b["relations"][0]["target"]:
            raise SystemExit(f"pair {pair_id} source/target flip mismatch")
        if a["relations"][0]["target"] != b["relations"][0]["source"]:
            raise SystemExit(f"pair {pair_id} source/target flip mismatch")
        if a["diagnostic_target_value"] != b["diagnostic_foil_value"]:
            raise SystemExit(f"pair {pair_id} target/foil values did not swap")
        if a["diagnostic_foil_value"] != b["diagnostic_target_value"]:
            raise SystemExit(f"pair {pair_id} target/foil values did not swap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        make_row(pair_index, orientation)
        for pair_index in range(PAIRS)
        for orientation in range(ROWS_PER_PAIR)
    ]
    validate(rows)

    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if output.exists() or manifest.exists():
        raise SystemExit("refusing to overwrite fresh diagnostic artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    payload = {
        "schema": "alice.eipm.n0.missing-evidence-fresh-relation-flip-diagnostic-manifest.v0.1",
        "status": "FRESH_DIAGNOSTIC_ROWS_MATERIALIZED_NOT_EVALUATED",
        "rows": len(rows),
        "pairs": PAIRS,
        "rows_per_pair": ROWS_PER_PAIR,
        "family": "missing_structured_evidence_fresh_relation_flip",
        "dataset_sha256": sha256_file(output),
        "data_origin": "new_public_synthetic_relation_flip_diagnostic_after_frozen_failure",
        "only_relation_direction_changes_within_pair": True,
        "semantic_channel_deliberately_non_answering": True,
        "structured_view_available": False,
        "evidence_view_uniquely_answer_authoritative": True,
        "counterfactual_view": 2,
        "counterfactual_intervention_point": (
            "before_parent_cache_and_before_cross_context_fusion"
        ),
        "frozen_challenge_rows_reused": False,
        "prior_diagnostic_rows_reused": False,
        "challenge_rows_used_for_training": False,
        "training_authorized": False,
        "gradient_performed": False,
        "ratification_effect": False,
        "promotion_authorized": False,
        "scale_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "n0_complete": False,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
