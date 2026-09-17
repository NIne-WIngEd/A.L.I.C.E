#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import build_n0_v02_evidence_selector_repair_curriculum_v0_1 as prior

# Fresh entities/values and fresh query wording. The v0.1 held-out rows were
# observed before this repair was designed, so none of them may be reused as
# the confirmatory test for the endpoint-read repair.
NEW_SUBJECTS = [f"Endpoint {name}" for name in [
    "Aquila","Beryl","Cairn","Dahlia","Eider","Fennel","Glacier","Hearth",
    "Ivory","Jade","Kite","Lagoon","Maple","Nectar","Onyx","Pollen",
    "Quarry","Raven","Sorrel","Topaz","Ulna","Vesper","Wheat","Xylem",
    "Yucca","Zinnia","Alder","Basin","Cedar","Dew","Echo","Flora",
]]
NEW_ATTRIBUTES = [
    ("handover band", "copper", "indigo"),
    ("control nonce", "418", "863"),
    ("service track", "idle", "surge"),
    ("refresh window", "23 seconds", "51 seconds"),
    ("replica class", "single", "duplex"),
    ("telemetry grade", "core", "expanded"),
    ("routing sector", "north", "south"),
    ("snapshot period", "27 minutes", "53 minutes"),
    ("worker allotment", "8", "14"),
    ("sampling density", "thin", "rich"),
    ("archive state", "locked", "rotating"),
    ("sync width", "compact", "broad"),
]

QUERY_REWRITES = {
    "corrects_source": "From the correction link alone, which record is the correcting source for {attribute} on {subject}?",
    "corrects_target": "From the correction link alone, which record is the object being corrected for {attribute} on {subject}?",
    "supersedes_source": "From the supersession link alone, which record is the replacement source for {subject}'s {attribute}?",
    "supersedes_target": "From the supersession link alone, which record is the one being replaced for {subject}'s {attribute}?",
    "temporal_successor_source": "From the temporal-successor link alone, which record is the successor endpoint for {subject}'s {attribute}?",
    "supports_target": "From the support link alone, which {attribute} claim for {subject} receives the support?",
    "conflict_support_target": "In the conflict, which {attribute} claim for {subject} receives the arbiter's support?",
    "mixed_correct_support_source": "Using the correction and validator-support links together, which record is the correcting supported source for {subject}'s {attribute}?",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freshen(rows: list[dict[str, Any]], family: str, index: int) -> list[dict[str, Any]]:
    subject = NEW_SUBJECTS[(index + 3 * [name for name, _ in prior.FAMILIES].index(family)) % len(NEW_SUBJECTS)]
    attribute = NEW_ATTRIBUTES[(index + [name for name, _ in prior.FAMILIES].index(family)) % len(NEW_ATTRIBUTES)][0]
    query = QUERY_REWRITES[family].format(subject=subject, attribute=attribute)
    for row in rows:
        row["id"] = str(row["id"]).replace("N0V02-SELREPAIR-", "N0V02-ENDPOINTREPAIR2-")
        row["pair_id"] = str(row["pair_id"]).replace("N0V02-SELREPAIR-", "N0V02-ENDPOINTREPAIR2-")
        row["query_text"] = query
        row["data_origin"] = "deterministic_public_synthetic_relation_endpoint_repair_v0_2"
        row["curriculum_version"] = "v0.2"
        row["prior_selector_repair_v0_1_test_row_reused"] = False
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    # The prior factories are deterministic and refer to their module globals at
    # call time. Rebinding these gives a genuinely fresh corpus without copying
    # the relation semantics into a second implementation.
    prior.SUBJECTS = NEW_SUBJECTS
    prior.ATTRIBUTES = NEW_ATTRIBUTES

    rows: list[dict[str, Any]] = []
    for family, factory in prior.FAMILIES:
        for index in range(prior.PAIRS_PER_FAMILY):
            rows.extend(freshen(factory(index), family, index))

    expected_rows = len(prior.FAMILIES) * prior.PAIRS_PER_FAMILY * 2
    if len(rows) != expected_rows or len({str(row["id"]) for row in rows}) != expected_rows:
        raise SystemExit("endpoint-repair v0.2 row count/id uniqueness drift")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    for pair_id, members in grouped.items():
        members = sorted(members, key=lambda item: str(item["variant"]))
        if len(members) != 2 or {str(m["variant"]) for m in members} != {"A", "B"}:
            raise SystemExit(f"malformed endpoint-repair pair: {pair_id}")
        a, b = members
        if a["fields"] != b["fields"] or a["query_text"] != b["query_text"]:
            raise SystemExit(f"non-relation input drift inside pair: {pair_id}")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise SystemExit(f"target did not flip inside pair: {pair_id}")
        if a["split"] != b["split"]:
            raise SystemExit(f"split drift inside pair: {pair_id}")
        if a.get("prior_selector_repair_v0_1_test_row_reused") is not False or b.get("prior_selector_repair_v0_1_test_row_reused") is not False:
            raise SystemExit("v0.1 held-out row reuse detected")

    expected_split = {
        "train": prior.TRAIN_PAIRS_PER_FAMILY,
        "dev": prior.DEV_PAIRS_PER_FAMILY,
        "test": prior.TEST_PAIRS_PER_FAMILY,
    }
    for family, _factory in prior.FAMILIES:
        counts: dict[str, int] = defaultdict(int)
        seen: set[str] = set()
        for row in rows:
            if row["family"] != family or row["pair_id"] in seen:
                continue
            seen.add(str(row["pair_id"]))
            counts[str(row["split"])] += 1
        if dict(counts) != expected_split:
            raise SystemExit(f"split count drift for {family}: {dict(counts)}")

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    families = [name for name, _factory in prior.FAMILIES]
    manifest = {
        "schema": "alice.eipm.n0.v02-relation-endpoint-repair-curriculum.v0.2",
        "status": "TRAIN_DEV_TEST_COMPILED_TEST_UNTOUCHED",
        "rows": len(rows),
        "family_count": len(families),
        "families": families,
        "source_role_families": [
            "corrects_source", "supersedes_source", "temporal_successor_source", "mixed_correct_support_source"
        ],
        "target_role_families": [
            "corrects_target", "supersedes_target", "supports_target", "conflict_support_target"
        ],
        "pairs_per_family": prior.PAIRS_PER_FAMILY,
        "train_pairs_per_family": prior.TRAIN_PAIRS_PER_FAMILY,
        "dev_pairs_per_family": prior.DEV_PAIRS_PER_FAMILY,
        "test_pairs_per_family": prior.TEST_PAIRS_PER_FAMILY,
        "compiled_sha256": sha(output),
        "generated_text": True,
        "data_origin": "deterministic_public_synthetic_relation_endpoint_repair_v0_2",
        "identity_authority": False,
        "private_identity_content": False,
        "train_split_training_authorized": True,
        "dev_split_training_authorized": False,
        "test_split_training_authorized": False,
        "prior_selector_repair_v0_1_test_rows_reused": False,
        "frozen_latent_challenge_rows_used_for_training": False,
        "parent_value_path_diagnostic_rows_used_for_training": False,
        "paired_non_relation_inputs_identical": True,
        "target_flips_only_with_relation_graph": True,
        "graph_pooled_cosine_is_optimization_target": False,
        "curriculum_counts_are_operating_budget_not_architecture_ceiling": True,
        "hard_parameter_ceiling": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
