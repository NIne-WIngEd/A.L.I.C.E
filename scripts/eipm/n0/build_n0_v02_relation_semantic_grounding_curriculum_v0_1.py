#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from alice_personality.n0.evidence_graph import relation_type_id

SUBJECTS = [f"Grounding {name}" for name in [
    "Acorn","Brook","Citrine","Drift","Elm","Flare","Grove","Horizon",
    "Iris","Jet","Kelp","Lark","Moss","Nova","Orchid","Pine",
    "Quartz","Reef","Spruce","Thistle","Umber","Vale","Willow","Xenon",
    "Yew","Zephyr","Aster","Basin","Clover","Dawn","Ember","Frost",
]]

ATTRIBUTES = [
    ("handoff mode", "manual", "coordinated"),
    ("control epoch", "31", "67"),
    ("service profile", "quiet", "burst"),
    ("refresh period", "14 seconds", "33 seconds"),
    ("replica policy", "solo", "paired"),
    ("telemetry class", "basic", "extended"),
    ("routing lane", "secondary", "primary"),
    ("snapshot interval", "16 minutes", "39 minutes"),
    ("worker quota", "5", "12"),
    ("sampling mode", "sparse", "dense"),
    ("archive lane", "inner", "outer"),
    ("sync regime", "deferred", "immediate"),
]

PAIRS_PER_FAMILY = 32
TRAIN_PAIRS_PER_FAMILY = 20
DEV_PAIRS_PER_FAMILY = 6
TEST_PAIRS_PER_FAMILY = 6

QUERY_TEMPLATES: dict[str, list[str]] = {
    "corrects_current": [
        "After the correction is applied, what {attribute} should be treated as current for {subject}?",
        "Which {attribute} value survives the correction for {subject}?",
        "What is the corrected current {attribute} for {subject}?",
        "Which {attribute} should replace the erroneous value for {subject}?",
    ],
    "corrects_previous": [
        "Which {attribute} value is the correction replacing for {subject}?",
        "What {attribute} becomes outdated after the correction for {subject}?",
        "Which earlier {attribute} is being corrected for {subject}?",
        "What {attribute} should no longer be treated as current after the correction for {subject}?",
    ],
    "supersedes_current": [
        "After supersession, what {attribute} is current for {subject}?",
        "Which {attribute} replaces the superseded value for {subject}?",
        "What {attribute} should be retained after the newer record supersedes the older one for {subject}?",
        "Which {attribute} belongs to the replacement state for {subject}?",
    ],
    "supersedes_previous": [
        "Which {attribute} belonged to the state that was superseded for {subject}?",
        "What {attribute} was replaced by the newer state for {subject}?",
        "Which previous {attribute} is no longer current after supersession for {subject}?",
        "What {attribute} belongs to the superseded record for {subject}?",
    ],
    "temporal_current": [
        "Which {attribute} belongs to the temporal successor for {subject}?",
        "What is the later {attribute} state for {subject}?",
        "Which {attribute} comes next in the temporal chain for {subject}?",
        "What {attribute} is associated with the successor state for {subject}?",
    ],
    "temporal_previous": [
        "Which {attribute} immediately precedes the temporal successor for {subject}?",
        "What is the earlier {attribute} state for {subject}?",
        "Which {attribute} comes before the successor state for {subject}?",
        "What {attribute} belongs to the predecessor in the temporal chain for {subject}?",
    ],
    "causes_cause": [
        "Which {attribute} value caused the linked downstream state for {subject}?",
        "What {attribute} is the cause in this causal relation for {subject}?",
        "Which {attribute} initiated the documented effect for {subject}?",
        "What {attribute} should be identified as the causal input for {subject}?",
    ],
    "causes_effect": [
        "Which {attribute} value is the effect of the linked cause for {subject}?",
        "What {attribute} is the resulting state in this causal relation for {subject}?",
        "Which {attribute} was produced by the documented cause for {subject}?",
        "What {attribute} should be identified as the causal result for {subject}?",
    ],
    "supports_supported": [
        "Which {attribute} value is supported by the linked evidence for {subject}?",
        "What {attribute} is the supported claim for {subject}?",
        "Which {attribute} receives support in the evidence relation for {subject}?",
        "What {attribute} should be retained as the supported value for {subject}?",
    ],
    "supports_supporter": [
        "Which {attribute} value belongs to the record providing support for {subject}?",
        "What {attribute} is on the supporting side of the evidence relation for {subject}?",
        "Which {attribute} comes from the supporter record for {subject}?",
        "What {attribute} belongs to the evidence source that supports the other record for {subject}?",
    ],
    "derived_item": [
        "Which {attribute} value is derived from the linked basis for {subject}?",
        "What {attribute} belongs to the derived record for {subject}?",
        "Which {attribute} is the result of the derivation relation for {subject}?",
        "What {attribute} was derived from the other record for {subject}?",
    ],
    "derived_basis": [
        "Which {attribute} value is the basis from which the other record was derived for {subject}?",
        "What {attribute} belongs to the derivation source material for {subject}?",
        "Which {attribute} is the underlying basis in the derived-from relation for {subject}?",
        "What {attribute} is the record that the derived value depends on for {subject}?",
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


def field(name: str, text: str, *, temporal_scope_id: int, confidence: float) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": 1,
        "provenance_id": 2,
        "relation_role_id": 1,
        "temporal_scope_id": temporal_scope_id,
        "confidence": confidence,
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


def make_pair(family: str, relation: str, desired_role: str, index: int, family_offset: int) -> list[dict[str, Any]]:
    subject = SUBJECTS[(index + 5 * family_offset) % len(SUBJECTS)]
    attribute, value_a, value_b = ATTRIBUTES[(index + 2 * family_offset) % len(ATTRIBUTES)]
    # Metadata is deliberately weak and alternates so it cannot define the answer.
    metadata_flip = (index + family_offset) % 2
    if metadata_flip == 0:
        metadata = [(1, 0.99), (2, 0.94)]
    else:
        metadata = [(2, 0.94), (1, 0.99)]
    fields = [
        field("record_alpha", f"Record alpha for {subject} reports {value_a} for {attribute}.",
              temporal_scope_id=metadata[0][0], confidence=metadata[0][1]),
        field("record_beta", f"Record beta for {subject} reports {value_b} for {attribute}.",
              temporal_scope_id=metadata[1][0], confidence=metadata[1][1]),
    ]
    query = QUERY_TEMPLATES[family][index % len(QUERY_TEMPLATES[family])]
    query = query.format(subject=subject, attribute=attribute)
    split = split_for(index)
    pair_id = f"N0V02-RELGROUND-{family.upper()}-{index + 1:02d}"

    def row(variant: str, source: int, target: int) -> dict[str, Any]:
        answer_index = source if desired_role == "source" else target
        foil_index = target if desired_role == "source" else source
        distribution = [0.0, 0.0]
        distribution[answer_index] = 1.0
        values = [value_a, value_b]
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
                f"For {subject}, the answer to the relation-semantic query is "
                f"{values[answer_index]} for {attribute}."
            ),
            "diagnostic_foil_value": values[foil_index],
            "relation_essential": True,
            "paired_adapter_inputs_identical": True,
            "target_flips_only_with_relation_graph": True,
            "query_names_endpoint_role_explicitly": False,
            "generated_text": True,
            "data_origin": "deterministic_public_synthetic_relation_semantic_grounding_v0_1",
            "identity_authority": False,
            "private_identity_content": False,
            "training_authorized": split == "train",
            "frozen_latent_challenge_row_reused": False,
            "missing_evidence_localization_row_reused": False,
            "endpoint_repair_v0_2_heldout_row_reused": False,
        }

    return [row("A", 1, 0), row("B", 0, 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for family_offset, (family, relation, desired_role) in enumerate(FAMILIES):
        for index in range(PAIRS_PER_FAMILY):
            rows.extend(make_pair(family, relation, desired_role, index, family_offset))

    expected_rows = len(FAMILIES) * PAIRS_PER_FAMILY * 2
    if len(rows) != expected_rows or len({str(row["id"]) for row in rows}) != expected_rows:
        raise SystemExit("relation-semantic curriculum row count/id uniqueness drift")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)

    for pair_id, members in grouped.items():
        members = sorted(members, key=lambda item: str(item["variant"]))
        if len(members) != 2 or {str(m["variant"]) for m in members} != {"A", "B"}:
            raise SystemExit(f"malformed semantic-role pair: {pair_id}")
        a, b = members
        for key in ("fields", "query_text", "family", "relation", "semantic_role"):
            if a[key] != b[key]:
                raise SystemExit(f"non-relation input drift in {pair_id}: {key}")
        if a["relations"][0]["source"] != b["relations"][0]["target"]:
            raise SystemExit(f"relation source/target did not flip in {pair_id}")
        if a["relations"][0]["target"] != b["relations"][0]["source"]:
            raise SystemExit(f"relation source/target did not flip in {pair_id}")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise SystemExit(f"semantic-role target did not flip in {pair_id}")
        if a["split"] != b["split"]:
            raise SystemExit(f"split drift in {pair_id}")

    expected_split = {
        "train": TRAIN_PAIRS_PER_FAMILY,
        "dev": DEV_PAIRS_PER_FAMILY,
        "test": TEST_PAIRS_PER_FAMILY,
    }
    for family, _relation, _role in FAMILIES:
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
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite relation-semantic grounding artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    source_families = [family for family, _relation, role in FAMILIES if role == "source"]
    target_families = [family for family, _relation, role in FAMILIES if role == "target"]
    manifest = {
        "schema": "alice.eipm.n0.v02-relation-semantic-grounding-curriculum.v0.1",
        "status": "TRAIN_DEV_TEST_COMPILED_TEST_UNTOUCHED",
        "rows": len(rows),
        "family_count": len(FAMILIES),
        "families": [family for family, _relation, _role in FAMILIES],
        "source_semantic_families": source_families,
        "target_semantic_families": target_families,
        "relations_covered": sorted({relation for _family, relation, _role in FAMILIES}),
        "pairs_per_family": PAIRS_PER_FAMILY,
        "train_pairs_per_family": TRAIN_PAIRS_PER_FAMILY,
        "dev_pairs_per_family": DEV_PAIRS_PER_FAMILY,
        "test_pairs_per_family": TEST_PAIRS_PER_FAMILY,
        "compiled_sha256": sha(output),
        "generated_text": True,
        "data_origin": "deterministic_public_synthetic_relation_semantic_grounding_v0_1",
        "identity_authority": False,
        "private_identity_content": False,
        "train_split_training_authorized": True,
        "dev_split_training_authorized": False,
        "test_split_training_authorized": False,
        "query_names_endpoint_role_explicitly": False,
        "paired_non_relation_inputs_identical": True,
        "target_flips_only_with_relation_graph": True,
        "frozen_latent_challenge_rows_used_for_training": False,
        "missing_evidence_localization_rows_used_for_training": False,
        "endpoint_repair_v0_2_heldout_rows_reused": False,
        "graph_pooled_cosine_is_optimization_target": False,
        "curriculum_counts_are_operating_budget_not_architecture_ceiling": True,
        "hard_parameter_ceiling": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
