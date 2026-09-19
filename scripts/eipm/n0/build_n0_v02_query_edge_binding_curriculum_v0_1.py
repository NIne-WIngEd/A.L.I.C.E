#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "alice.eipm.n0.v02-query-edge-binding-curriculum.v0.1"

RELATIONS = (
    "corrects",
    "supersedes",
    "derived_from",
    "causes",
    "supports",
    "temporal_successor",
)
DISTRACTOR_RELATION = {
    "corrects": "supports",
    "supersedes": "derived_from",
    "derived_from": "temporal_successor",
    "causes": "supports",
    "supports": "corrects",
    "temporal_successor": "causes",
}

TRAIN_QUADS_PER_RELATION = 18
DEV_QUADS_PER_RELATION = 6
TEST_QUADS_PER_RELATION = 6
QUADS_PER_RELATION = 30

SUBJECTS = {
    "train": [
        "Axiom", "Beryl", "Cygnus", "Drift", "Eon", "Fjord",
        "Glyph", "Helix", "Ion", "Jade", "Kepler", "Lumen",
        "Mosaic", "Nacre", "Orbit", "Pylon", "Quill", "Rune",
        "Solace", "Talon", "Umber", "Vega", "Warden", "Xylem",
    ],
    "dev": [
        "Alto", "Bracken", "Cinder", "Dahlia", "Estuary", "Fable",
        "Garnet", "Hollow", "Indigo", "Jasper", "Kodiak", "Lotus",
    ],
    "test": [
        "Anvil", "Basin", "Comet", "Dorado", "Elm", "Foxtrot",
        "Ginkgo", "Hearth", "Islet", "Jet", "Krypton", "Lyric",
    ],
}

ATTRIBUTES = {
    "train": [
        ("routing phase", "alpha", "omega"),
        ("buffer mode", "direct", "staged"),
        ("clock profile", "slow", "rapid"),
        ("handover code", "17", "83"),
        ("cache policy", "local", "shared"),
        ("sensor band", "narrow", "wide"),
        ("queue state", "paused", "flowing"),
        ("replica count", "two", "seven"),
        ("audit tier", "bronze", "platinum"),
    ],
    "dev": [
        ("worker mode", "isolated", "pooled"),
        ("refresh code", "29", "61"),
        ("telemetry band", "quiet", "verbose"),
        ("archive profile", "frozen", "live"),
        ("dispatch tier", "single", "mesh"),
        ("sampling code", "13", "47"),
    ],
    "test": [
        ("failover plan", "manual", "autonomous"),
        ("retention code", "21", "77"),
        ("replication mode", "sparse", "dense"),
        ("control band", "inner", "outer"),
        ("snapshot tier", "cold", "warm"),
        ("handoff code", "31", "59"),
    ],
}

ROLE_LANGUAGE: dict[str, tuple[str, str]] = {
    "corrects": ("the record doing the correction", "the record receiving the correction"),
    "supersedes": ("the record taking over", "the record being displaced"),
    "derived_from": ("the derived record", "the basis record"),
    "causes": ("the causal antecedent", "the resulting consequence"),
    "supports": ("the supporting evidence", "the record being supported"),
    "temporal_successor": ("the later record", "the earlier record"),
}

SPLIT_FRAMING = {
    "train": (
        "For the {subject} case about {attribute}, identify {role}.",
        "Within {subject}'s {attribute} records, select {role}.",
        "Focus only on {subject}'s {attribute}. Which entry is {role}?",
    ),
    "dev": (
        "Resolve the {subject} scenario for {attribute}: which entry is {role}?",
        "In the {attribute} evidence for {subject}, locate {role}.",
    ),
    "test": (
        "Inspect {subject}'s {attribute} case and return {role}.",
        "For {subject} under {attribute}, which entry functions as {role}?",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(index: int) -> str:
    if index < TRAIN_QUADS_PER_RELATION:
        return "train"
    if index < TRAIN_QUADS_PER_RELATION + DEV_QUADS_PER_RELATION:
        return "dev"
    return "test"


def local_index(index: int, split: str) -> int:
    if split == "train":
        return index
    if split == "dev":
        return index - TRAIN_QUADS_PER_RELATION
    return index - TRAIN_QUADS_PER_RELATION - DEV_QUADS_PER_RELATION


def field(name: str, text: str, *, subject: str, attribute: str) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "subject": subject,
        "attribute": attribute,
        "provenance": "PUBLIC_SYNTHETIC",
        "confidence": 1.0,
    }


def edge(
    source: int,
    target: int,
    relation: str,
    *,
    binding_group: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": 1.0,
        "binding_group": binding_group,
    }


def target_distribution(count: int, index: int) -> list[float]:
    values = [0.0] * count
    values[index] = 1.0
    return values


def pair_payload(
    *,
    slot: int,
    subject: str,
    attribute: tuple[str, str, str],
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    attr, left, right = attribute
    start = 2 * slot
    fields = [
        field(
            f"slot_{slot}_left",
            f"Case {subject}: {attr} is reported as {left}.",
            subject=subject,
            attribute=attr,
        ),
        field(
            f"slot_{slot}_right",
            f"Case {subject}: {attr} is reported as {right}.",
            subject=subject,
            attribute=attr,
        ),
    ]
    return fields, (start, start + 1)


def make_quad(
    *,
    relation: str,
    relation_offset: int,
    index: int,
) -> list[dict[str, Any]]:
    split = split_for(index)
    local = local_index(index, split)
    subjects = SUBJECTS[split]
    attrs = ATTRIBUTES[split]

    # Three same-relation pairs plus one different-relation distractor pair.
    # The relevant pair moves among the three same-relation slots so field
    # position cannot identify it.
    relevant_pair = (local + relation_offset) % 3
    subject_start = (3 * local + relation_offset) % len(subjects)
    pair_subjects = [
        subjects[(subject_start + offset) % len(subjects)]
        for offset in range(4)
    ]
    attr_start = (2 * local + relation_offset) % len(attrs)
    pair_attrs = [
        attrs[(attr_start + offset) % len(attrs)]
        for offset in range(4)
    ]

    fields: list[dict[str, Any]] = []
    pairs: list[tuple[int, int]] = []
    for slot in range(4):
        pair_fields, pair_indices = pair_payload(
            slot=slot,
            subject=pair_subjects[slot],
            attribute=pair_attrs[slot],
        )
        fields.extend(pair_fields)
        pairs.append(pair_indices)

    relevant_subject = pair_subjects[relevant_pair]
    relevant_attribute = pair_attrs[relevant_pair][0]
    source_role, target_role = ROLE_LANGUAGE[relation]
    frame = SPLIT_FRAMING[split][local % len(SPLIT_FRAMING[split])]
    source_query = frame.format(
        subject=relevant_subject,
        attribute=relevant_attribute,
        role=source_role,
    )
    target_query = frame.format(
        subject=relevant_subject,
        attribute=relevant_attribute,
        role=target_role,
    )

    quad_id = f"N0V02-QEB-{relation.upper()}-{index + 1:02d}"
    rows: list[dict[str, Any]] = []

    for direction in ("A", "B"):
        main_edges: list[dict[str, Any]] = []
        relevant_edge_index = -1
        for pair_index in range(3):
            left, right = pairs[pair_index]
            source, target = (
                (left, right) if direction == "A" else (right, left)
            )
            if pair_index == relevant_pair:
                relevant_edge_index = len(main_edges)
            main_edges.append(
                edge(
                    source,
                    target,
                    relation,
                    binding_group=(
                        "relevant_same_relation"
                        if pair_index == relevant_pair
                        else "same_relation_distractor"
                    ),
                )
            )

        dleft, dright = pairs[3]
        # Distractor direction varies independently of the main counterfactual.
        if (local + relation_offset) % 2:
            dleft, dright = dright, dleft
        edges = main_edges + [
            edge(
                dleft,
                dright,
                DISTRACTOR_RELATION[relation],
                binding_group="different_relation_distractor",
            )
        ]

        relevant_edge = edges[relevant_edge_index]
        for query_role, query_text in (
            ("source", source_query),
            ("target", target_query),
        ):
            answer = int(relevant_edge[query_role])
            row_id = f"{quad_id}-{direction}-{query_role.upper()}"
            rows.append(
                {
                    "id": row_id,
                    "quad_id": quad_id,
                    "edge_direction_variant": direction,
                    "query_role": query_role,
                    "relation": relation,
                    "split": split,
                    "query_text": query_text,
                    "fields": fields,
                    "relations": edges,
                    "relevant_edge_index": relevant_edge_index,
                    "same_relation_edge_indices": [0, 1, 2],
                    "different_relation_edge_indices": [3],
                    "target_evidence_distribution": target_distribution(
                        len(fields), answer
                    ),
                    "target_field_index": answer,
                    "causal_factors": {
                        "query_role_varies": True,
                        "edge_direction_varies": True,
                        "relevant_edge_content_must_be_bound": True,
                        "three_same_relation_edges_present": True,
                        "different_relation_distractor_present": True,
                        "fields_fixed_within_quad": True,
                        "target_depends_on_query_role_and_edge_direction": True,
                    },
                    "shortcut_controls": {
                        "constant_endpoint_role_sufficient": False,
                        "relation_global_endpoint_polarity_sufficient": False,
                        "relation_type_alone_identifies_relevant_edge": False,
                        "field_position_identifies_relevant_edge": False,
                        "same_relation_candidate_edges": 3,
                        "no_content_binding_uniform_edge_guess_upper_bound": 1.0 / 3.0,
                    },
                    "routing_supervision_available_for_future_study": True,
                    "generated_text": True,
                    "data_origin": "deterministic_public_synthetic_query_edge_binding_v0_1",
                    "identity_authority": False,
                    "private_identity_content": False,
                    "intended_training_split": split == "train",
                    "training_authorized": False,
                    "heldout_opening_authorized": False,
                }
            )
    return rows


def validate_quad(quad_id: str, members: list[dict[str, Any]]) -> None:
    if len(members) != 4:
        raise SystemExit(f"quad size drift for {quad_id}: {len(members)}")
    expected = {("A", "source"), ("A", "target"), ("B", "source"), ("B", "target")}
    observed = {
        (str(row["edge_direction_variant"]), str(row["query_role"]))
        for row in members
    }
    if observed != expected:
        raise SystemExit(f"factorial coverage drift for {quad_id}: {sorted(observed)}")

    first = members[0]
    for row in members[1:]:
        for key in ("fields", "relation", "split", "quad_id", "relevant_edge_index"):
            if row[key] != first[key]:
                raise SystemExit(f"non-causal drift in {quad_id}: {key}")
        if len(row["fields"]) != 8 or len(row["relations"]) != 4:
            raise SystemExit(f"multi-edge shape drift in {quad_id}")
        same = [
            relation
            for relation in row["relations"]
            if relation["relation"] == row["relation"]
        ]
        if len(same) != 3:
            raise SystemExit(f"same-relation edge coverage drift in {quad_id}")
        if row["relations"][3]["relation"] == row["relation"]:
            raise SystemExit(f"different-relation distractor missing in {quad_id}")
        if row["shortcut_controls"]["no_content_binding_uniform_edge_guess_upper_bound"] != 1.0 / 3.0:
            raise SystemExit(f"shortcut bound drift in {quad_id}")

    by_key = {
        (str(row["edge_direction_variant"]), str(row["query_role"])): row
        for row in members
    }
    a_source = by_key[("A", "source")]
    a_target = by_key[("A", "target")]
    b_source = by_key[("B", "source")]
    b_target = by_key[("B", "target")]

    if a_source["target_field_index"] == a_target["target_field_index"]:
        raise SystemExit(f"query role failed to flip answer in {quad_id}")
    if b_source["target_field_index"] == b_target["target_field_index"]:
        raise SystemExit(f"query role failed to flip answer in {quad_id}")
    if a_source["target_field_index"] == b_source["target_field_index"]:
        raise SystemExit(f"edge reversal failed to flip source answer in {quad_id}")
    if a_target["target_field_index"] == b_target["target_field_index"]:
        raise SystemExit(f"edge reversal failed to flip target answer in {quad_id}")

    relevant = int(first["relevant_edge_index"])
    edge_a = a_source["relations"][relevant]
    edge_b = b_source["relations"][relevant]
    if edge_a["source"] != edge_b["target"] or edge_a["target"] != edge_b["source"]:
        raise SystemExit(f"relevant edge failed to reverse in {quad_id}")

    # All same-relation edges reverse together.  Direction therefore cannot
    # identify which of those edges is query-relevant.
    for edge_index in (0, 1, 2):
        ea = a_source["relations"][edge_index]
        eb = b_source["relations"][edge_index]
        if ea["source"] != eb["target"] or ea["target"] != eb["source"]:
            raise SystemExit(
                f"same-relation distractor reversal drift in {quad_id}:{edge_index}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()

    output = Path(args.output)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() or manifest.exists():
        raise SystemExit("refusing to overwrite query-edge curriculum artifact")

    rows: list[dict[str, Any]] = []
    for relation_offset, relation in enumerate(RELATIONS):
        for index in range(QUADS_PER_RELATION):
            rows.extend(
                make_quad(
                    relation=relation,
                    relation_offset=relation_offset,
                    index=index,
                )
            )

    expected_rows = len(RELATIONS) * QUADS_PER_RELATION * 4
    if len(rows) != expected_rows:
        raise SystemExit(f"row count drift: {len(rows)} != {expected_rows}")
    if len({str(row["id"]) for row in rows}) != expected_rows:
        raise SystemExit("row id collision")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["quad_id"])].append(row)
    for quad_id, members in grouped.items():
        validate_quad(quad_id, members)

    for row in rows:
        if row["training_authorized"] is not False:
            raise SystemExit("gradient authorization leaked into curriculum")
        if row["private_identity_content"] is not False:
            raise SystemExit("private identity content leaked")
        if row["identity_authority"] is not False:
            raise SystemExit("identity authority leaked")

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    split_quads: dict[str, dict[str, int]] = {}
    for relation in RELATIONS:
        counts: dict[str, int] = defaultdict(int)
        seen: set[str] = set()
        for row in rows:
            if row["relation"] != relation:
                continue
            qid = str(row["quad_id"])
            if qid in seen:
                continue
            seen.add(qid)
            counts[str(row["split"])] += 1
        split_quads[relation] = dict(sorted(counts.items()))

    payload = {
        "schema": SCHEMA,
        "status": "COMPILED_FRESH_MULTI_EDGE_QUERY_BINDING_CURRICULUM_NO_TRAINING",
        "rows": len(rows),
        "quad_count": len(grouped),
        "relations": list(RELATIONS),
        "fields_per_row": 8,
        "edges_per_row": 4,
        "same_relation_edges_per_row": 3,
        "different_relation_distractor_edges_per_row": 1,
        "train_quads_per_relation": TRAIN_QUADS_PER_RELATION,
        "dev_quads_per_relation": DEV_QUADS_PER_RELATION,
        "test_quads_per_relation": TEST_QUADS_PER_RELATION,
        "split_quad_counts_by_relation": split_quads,
        "query_to_specific_edge_binding_required": True,
        "routing_supervision_available": True,
        "relation_global_endpoint_polarity_sufficient": False,
        "no_content_binding_uniform_edge_guess_upper_bound": 1.0 / 3.0,
        "train_dev_test_subject_pools_disjoint": True,
        "train_dev_test_attribute_pools_disjoint": True,
        "train_dev_test_query_framing_disjoint": True,
        "identity_authority": False,
        "private_identity_content": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "gradient_authorized": False,
        "test_split_opening_authorized": False,
        "hard_parameter_ceiling": None,
        "compiled_sha256": sha256(output),
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
