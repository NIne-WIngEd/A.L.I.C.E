#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from alice_personality.n0.evidence_graph import relation_type_id

SUBJECTS = [f"Residual {name}" for name in [
    "Alder","Breeze","Canyon","Dahlia","Estuary","Flint","Garnet","Harbor",
    "Islet","Juniper","Kite","Lagoon","Meadow","Nectar","Opal","Prairie",
    "Quill","Ridge","Sable","Terrace","Upland","Violet","Warden","Xylem",
    "Yucca","Zenith","Apricot","Birch","Cedar","Delta","Ember","Fjord",
    "Glade","Heather","Indigo","Jasper",
]]

ATTRIBUTES = [
    ("handover policy", "manual", "coordinated"),
    ("control epoch", "41", "73"),
    ("service posture", "standby", "active"),
    ("refresh window", "12 seconds", "29 seconds"),
    ("replica setting", "single", "paired"),
    ("telemetry mode", "local", "extended"),
    ("routing lane", "auxiliary", "primary"),
    ("snapshot period", "13 minutes", "37 minutes"),
    ("worker allocation", "4", "13"),
    ("sampling regime", "sparse", "dense"),
    ("archive tier", "cold", "warm"),
    ("synchronization mode", "deferred", "immediate"),
]

PAIRS_PER_FAMILY = 36
TRAIN_PAIRS_PER_FAMILY = 24
DEV_PAIRS_PER_FAMILY = 6
TEST_PAIRS_PER_FAMILY = 6

QUERY_TEMPLATES: dict[str, list[str]] = {
    "corrects_current": [
        "After resolving the correction, what {attribute} should now be used for {subject}?",
        "What {attribute} remains valid once the correction is taken into account for {subject}?",
        "Which {attribute} is the corrected value for {subject}?",
        "Once the mistaken record is fixed, what {attribute} applies to {subject}?",
        "What should {subject}'s {attribute} be after incorporating the correction?",
        "Which {attribute} represents the corrected state of {subject}?",
    ],
    "corrects_previous": [
        "What {attribute} is displaced by the correction for {subject}?",
        "Which earlier {attribute} is no longer valid after the correction for {subject}?",
        "What {attribute} did the corrected record replace for {subject}?",
        "Which {attribute} belongs to the mistaken state that was corrected for {subject}?",
        "What was {subject}'s superseded-by-correction {attribute}?",
        "Which {attribute} should be discarded as the pre-correction value for {subject}?",
    ],
    "supersedes_current": [
        "What {attribute} applies after the newer state takes over for {subject}?",
        "Which {attribute} belongs to the replacement state for {subject}?",
        "After supersession, which {attribute} should be retained for {subject}?",
        "What is {subject}'s {attribute} in the state that takes precedence?",
        "Which {attribute} survives the supersession for {subject}?",
        "What {attribute} belongs to the state that replaces the other one for {subject}?",
    ],
    "supersedes_previous": [
        "Which {attribute} belongs to the state that lost precedence for {subject}?",
        "What {attribute} was replaced when the newer state took over for {subject}?",
        "Which {attribute} is associated with the superseded state for {subject}?",
        "What was {subject}'s earlier {attribute} before the replacement state took precedence?",
        "Which {attribute} should be treated as historical after supersession for {subject}?",
        "What {attribute} belonged to the state that was replaced for {subject}?",
    ],
    "temporal_current": [
        "What {attribute} belongs to the next state in time for {subject}?",
        "Which {attribute} is associated with the later state for {subject}?",
        "Following the temporal transition, what {attribute} applies to {subject}?",
        "What {attribute} is present in the succeeding state for {subject}?",
        "Which {attribute} comes later in the recorded sequence for {subject}?",
        "What is {subject}'s {attribute} after the documented temporal transition?",
    ],
    "temporal_previous": [
        "What {attribute} belongs to the state immediately before the next one for {subject}?",
        "Which {attribute} is associated with the earlier state for {subject}?",
        "Before the temporal transition, what {attribute} applied to {subject}?",
        "What {attribute} is present in the preceding state for {subject}?",
        "Which {attribute} comes earlier in the recorded sequence for {subject}?",
        "What was {subject}'s {attribute} prior to the documented temporal transition?",
    ],
    "causes_cause": [
        "Which {attribute} initiated the linked outcome for {subject}?",
        "What {attribute} is responsible for the documented result for {subject}?",
        "Which {attribute} should be treated as the causal condition for {subject}?",
        "What {attribute} produced the other linked state for {subject}?",
        "Which {attribute} is upstream in the causal relation for {subject}?",
        "What {attribute} acted as the cause for {subject}?",
    ],
    "causes_effect": [
        "Which {attribute} resulted from the linked cause for {subject}?",
        "What {attribute} is the documented outcome for {subject}?",
        "Which {attribute} should be treated as the causal consequence for {subject}?",
        "What {attribute} was produced by the other linked state for {subject}?",
        "Which {attribute} is downstream in the causal relation for {subject}?",
        "What {attribute} occurred as the effect for {subject}?",
    ],
    "supports_supported": [
        "Which {attribute} is backed by the linked evidence for {subject}?",
        "What {attribute} is the claim receiving evidentiary support for {subject}?",
        "Which {attribute} should be retained as supported for {subject}?",
        "What {attribute} is corroborated by the other linked record for {subject}?",
        "Which {attribute} is the supported conclusion for {subject}?",
        "What {attribute} receives support in the evidence link for {subject}?",
    ],
    "supports_supporter": [
        "Which {attribute} belongs to the record doing the supporting for {subject}?",
        "What {attribute} is carried by the evidence-provider record for {subject}?",
        "Which {attribute} comes from the record that backs the other claim for {subject}?",
        "What {attribute} is on the supporting side of the evidence link for {subject}?",
        "Which {attribute} belongs to the corroborating record for {subject}?",
        "What {attribute} comes from the record providing evidentiary support for {subject}?",
    ],
    "derived_item": [
        "Which {attribute} belongs to the record produced from the other one for {subject}?",
        "What {attribute} is the derived result for {subject}?",
        "Which {attribute} was obtained from the linked basis for {subject}?",
        "What {attribute} belongs to the dependent derived record for {subject}?",
        "Which {attribute} is the product of the derivation for {subject}?",
        "What {attribute} is derived in the relation for {subject}?",
    ],
    "derived_basis": [
        "Which {attribute} belongs to the record used as the basis for the other one for {subject}?",
        "What {attribute} is the underlying basis of the derivation for {subject}?",
        "Which {attribute} did the derived record depend on for {subject}?",
        "What {attribute} belongs to the antecedent material for the derivation for {subject}?",
        "Which {attribute} served as the basis from which the other record was produced for {subject}?",
        "What {attribute} is underneath the derived result for {subject}?",
    ],
}

FAMILIES: list[tuple[str, str, str]] = [
    ("corrects_current", "corrects", "source"),
    ("corrects_previous", "corrects", "target"),
    ("supersedes_current", "supersedes", "source"),
    ("supersedes_previous", "supersedes", "target"),
    ("temporal_current", "temporal_successor", "source"),
    ("temporal_previous", "temporal_successor", "target"),
    ("causes_cause", "causes", "source"),
    ("causes_effect", "causes", "target"),
    ("supports_supported", "supports", "target"),
    ("supports_supporter", "supports", "source"),
    ("derived_item", "derived_from", "source"),
    ("derived_basis", "derived_from", "target"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(index: int) -> str:
    if index < TRAIN_PAIRS_PER_FAMILY:
        return "train"
    if index < TRAIN_PAIRS_PER_FAMILY + DEV_PAIRS_PER_FAMILY:
        return "dev"
    return "test"


def field(name: str, text: str) -> dict[str, Any]:
    # Equal non-relational metadata prevents confidence or temporal priors from
    # defining the answer. The relation direction is the only flipped signal.
    return {
        "name": name,
        "text": text,
        "field_type_id": 1,
        "provenance_id": 2,
        "relation_role_id": 1,
        "temporal_scope_id": 1,
        "confidence": 0.97,
        "missing": False,
    }


def edge(source: int, target: int, relation: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_type_id": relation_type_id(relation),
        "confidence": 1.0,
    }


def make_pair(
    *,
    family: str,
    relation: str,
    desired_role: str,
    index: int,
    family_offset: int,
) -> list[dict[str, Any]]:
    subject = SUBJECTS[(index + 7 * family_offset) % len(SUBJECTS)]
    attribute, value_a, value_b = ATTRIBUTES[(index + 3 * family_offset) % len(ATTRIBUTES)]
    fields = [
        field(
            "record_one",
            f"One record for {subject} reports {value_a} for {attribute}.",
        ),
        field(
            "record_two",
            f"Another record for {subject} reports {value_b} for {attribute}.",
        ),
    ]
    query = QUERY_TEMPLATES[family][index % len(QUERY_TEMPLATES[family])]
    query = query.format(subject=subject, attribute=attribute)
    split = split_for(index)
    pair_id = f"N0V02-RELROLE2-{family.upper()}-{index + 1:02d}"

    def row(variant: str, source: int, target: int) -> dict[str, Any]:
        answer_index = source if desired_role == "source" else target
        values = [value_a, value_b]
        distribution = [0.0, 0.0]
        distribution[answer_index] = 1.0
        return {
            "id": f"{pair_id}-{variant}",
            "pair_id": pair_id,
            "variant": variant,
            "family": family,
            "relation": relation,
            "semantic_role": desired_role,
            "split": split,
            "query_text": query,
            "fields": fields,
            "relations": [edge(source, target, relation)],
            "target_evidence_distribution": distribution,
            "target_summary_text": (
                f"The relation-semantic answer for {subject}'s {attribute} is "
                f"{values[answer_index]}."
            ),
            "relation_essential": True,
            "paired_non_relation_inputs_identical": True,
            "target_flips_only_with_relation_direction": True,
            "query_names_endpoint_role_explicitly": False,
            "generated_text": True,
            "data_origin": "deterministic_public_synthetic_relation_semantic_role_residual_v0_2",
            "identity_authority": False,
            "private_identity_content": False,
            "training_authorized": split == "train",
            "frozen_latent_challenge_row_reused": False,
            "missing_evidence_localization_row_reused": False,
            "relation_semantic_v0_1_row_reused": False,
            "endpoint_repair_v0_2_heldout_row_reused": False,
        }

    return [
        row("A", 1, 0),
        row("B", 0, 1),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for family_offset, (family, relation, desired_role) in enumerate(FAMILIES):
        for index in range(PAIRS_PER_FAMILY):
            rows.extend(
                make_pair(
                    family=family,
                    relation=relation,
                    desired_role=desired_role,
                    index=index,
                    family_offset=family_offset,
                )
            )

    expected_rows = len(FAMILIES) * PAIRS_PER_FAMILY * 2
    if len(rows) != expected_rows:
        raise SystemExit("relation-semantic role-residual row count drift")
    if len({str(row["id"]) for row in rows}) != expected_rows:
        raise SystemExit("relation-semantic role-residual id collision")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)

    for pair_id, members in grouped.items():
        members = sorted(members, key=lambda item: str(item["variant"]))
        if len(members) != 2 or {str(m["variant"]) for m in members} != {"A", "B"}:
            raise SystemExit(f"malformed role-residual pair: {pair_id}")
        a, b = members
        for key in ("fields", "query_text", "family", "relation", "semantic_role", "split"):
            if a[key] != b[key]:
                raise SystemExit(f"non-relation drift in {pair_id}: {key}")
        edge_a = a["relations"][0]
        edge_b = b["relations"][0]
        if edge_a["source"] != edge_b["target"] or edge_a["target"] != edge_b["source"]:
            raise SystemExit(f"relation direction failed to flip in {pair_id}")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise SystemExit(f"target failed to flip in {pair_id}")

    expected_split = {
        "train": TRAIN_PAIRS_PER_FAMILY,
        "dev": DEV_PAIRS_PER_FAMILY,
        "test": TEST_PAIRS_PER_FAMILY,
    }
    for family, _relation, _role in FAMILIES:
        seen: set[str] = set()
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if row["family"] != family or row["pair_id"] in seen:
                continue
            seen.add(str(row["pair_id"]))
            counts[str(row["split"])] += 1
        if dict(counts) != expected_split:
            raise SystemExit(f"split count drift for {family}: {dict(counts)}")

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite role-residual curriculum artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = {
        "schema": "alice.eipm.n0.v02-relation-semantic-role-residual-curriculum.v0.2",
        "status": "TRAIN_DEV_TEST_COMPILED_TEST_UNTOUCHED",
        "rows": len(rows),
        "family_count": len(FAMILIES),
        "families": [family for family, _relation, _role in FAMILIES],
        "relations_covered": sorted({relation for _family, relation, _role in FAMILIES}),
        "source_semantic_families": [
            family for family, _relation, role in FAMILIES if role == "source"
        ],
        "target_semantic_families": [
            family for family, _relation, role in FAMILIES if role == "target"
        ],
        "pairs_per_family": PAIRS_PER_FAMILY,
        "train_pairs_per_family": TRAIN_PAIRS_PER_FAMILY,
        "dev_pairs_per_family": DEV_PAIRS_PER_FAMILY,
        "test_pairs_per_family": TEST_PAIRS_PER_FAMILY,
        "compiled_sha256": sha(output),
        "data_origin": "deterministic_public_synthetic_relation_semantic_role_residual_v0_2",
        "generated_text": True,
        "identity_authority": False,
        "private_identity_content": False,
        "train_split_training_authorized": True,
        "dev_split_training_authorized": False,
        "test_split_training_authorized": False,
        "paired_non_relation_inputs_identical": True,
        "target_flips_only_with_relation_direction": True,
        "query_names_endpoint_role_explicitly": False,
        "equal_non_relation_metadata_within_and_across_pair": True,
        "frozen_latent_challenge_rows_used_for_training": False,
        "missing_evidence_localization_rows_used_for_training": False,
        "relation_semantic_v0_1_rows_reused": False,
        "endpoint_repair_v0_2_heldout_rows_reused": False,
        "graph_pooled_cosine_is_optimization_target": False,
        "curriculum_counts_are_operating_budget_not_architecture_ceiling": True,
        "hard_parameter_ceiling": None,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
