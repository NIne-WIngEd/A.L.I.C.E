#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from alice_personality.n0.evidence_graph import relation_type_id


SUBJECTS = [
    "Unit Helix", "Unit Ion", "Unit Jasper", "Unit Kite",
    "Unit Lotus", "Unit Meridian", "Unit Nimbus", "Unit Oriel",
    "Unit Prism", "Unit Quill", "Unit Rook", "Unit Sable",
    "Unit Thistle", "Unit Umber", "Unit Vela", "Unit Wren",
]
ATTRIBUTES = [
    ("control port", "6101", "6201", "6301"),
    ("policy mode", "manual", "elastic", "strict"),
    ("deployment lane", "preview", "release", "fallback"),
    ("retry budget", "5", "9", "13"),
    ("telemetry rate", "30 Hz", "60 Hz", "90 Hz"),
    ("storage profile", "deep", "active", "burst"),
    ("routing class", "local", "weighted", "priority"),
    ("snapshot cadence", "14 minutes", "26 minutes", "38 minutes"),
]
PAIRS_PER_FAMILY = 16
TRAIN_PAIRS_PER_FAMILY = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(name: str, text: str) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": 1,
        "provenance_id": 1,
        "relation_role_id": 1,
        "temporal_scope_id": 1,
        "confidence": 1.0,
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


def parts(index: int, subject_offset: int = 0, attr_offset: int = 0) -> tuple[str, str, str, str, str]:
    subject = SUBJECTS[(index + subject_offset) % len(SUBJECTS)]
    attribute, first, second, third = ATTRIBUTES[(index + attr_offset) % len(ATTRIBUTES)]
    return subject, attribute, first, second, third


def make_row(
    *,
    family: str,
    pair_index: int,
    variant: str,
    fields: list[dict[str, Any]],
    query: str,
    relations: list[dict[str, Any]],
    target_index: int,
    target_summary: str,
) -> dict[str, Any]:
    target = [0.0] * len(fields)
    target[target_index] = 1.0
    split = "train" if pair_index < TRAIN_PAIRS_PER_FAMILY else "dev"
    pair_id = f"N0V02-RELREPAIR-{family.upper()}-{pair_index + 1:02d}"
    return {
        "id": f"{pair_id}-{variant}",
        "pair_id": pair_id,
        "variant": variant,
        "family": family,
        "split": split,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_evidence_distribution": target,
        "target_summary_text": target_summary,
        "relation_essential": True,
        "paired_text_and_query_identical": True,
        "independent_analogue": True,
        "private_identity_content": False,
    }


def pair(
    *,
    family: str,
    index: int,
    fields: list[dict[str, Any]],
    query: str,
    relations_a: list[dict[str, Any]],
    target_a: int,
    summary_a: str,
    relations_b: list[dict[str, Any]],
    target_b: int,
    summary_b: str,
) -> list[dict[str, Any]]:
    return [
        make_row(
            family=family, pair_index=index, variant="A", fields=fields, query=query,
            relations=relations_a, target_index=target_a, target_summary=summary_a,
        ),
        make_row(
            family=family, pair_index=index, variant="B", fields=fields, query=query,
            relations=relations_b, target_index=target_b, target_summary=summary_b,
        ),
    ]


def supersedes_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index)
    fields = [
        field("entry_one", f"Entry one for {subject} reports {first} for {attribute}."),
        field("entry_two", f"Entry two for {subject} reports {second} for {attribute}."),
        field("noise", f"A separate unit has an unrelated {attribute} record."),
    ]
    query = f"Use only the update links to choose the active {attribute} for {subject}."
    return pair(
        family="supersedes_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "supersedes")], target_a=0,
        summary_a=f"The active {attribute} for {subject} is {first}.",
        relations_b=[edge(1, 0, "supersedes")], target_b=1,
        summary_b=f"The active {attribute} for {subject} is {second}.",
    )


def corrects_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 1, 1)
    fields = [
        field("entry_one", f"Entry one associates {subject} with {first} for {attribute}."),
        field("entry_two", f"Entry two associates {subject} with {second} for {attribute}."),
        field("scope", f"Both entries describe the same {attribute} slot for {subject}."),
    ]
    query = f"Follow the correction link and choose the corrected {attribute} for {subject}."
    return pair(
        family="corrects_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "corrects")], target_a=0,
        summary_a=f"The corrected {attribute} for {subject} is {first}.",
        relations_b=[edge(1, 0, "corrects")], target_b=1,
        summary_b=f"The corrected {attribute} for {subject} is {second}.",
    )


def temporal_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 2, 2)
    fields = [
        field("frame_one", f"Frame one records {first} for {subject}'s {attribute}."),
        field("frame_two", f"Frame two records {second} for {subject}'s {attribute}."),
        field("noise", "A third frame belongs to a different unit."),
    ]
    query = f"From the successor link alone, identify the later {attribute} state for {subject}."
    return pair(
        family="temporal_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "temporal_successor")], target_a=0,
        summary_a=f"The later {attribute} state is {first}.",
        relations_b=[edge(1, 0, "temporal_successor")], target_b=1,
        summary_b=f"The later {attribute} state is {second}.",
    )


def causes_direction(index: int) -> list[dict[str, Any]]:
    subject, _attribute, _first, _second, _third = parts(index, 3, 3)
    fields = [
        field("event_one", f"Event one occurred inside {subject}."),
        field("event_two", f"Event two occurred inside {subject}."),
        field("noise", "Another event occurred in an unrelated unit."),
    ]
    query = f"Use the causal arrow to identify the upstream event in {subject}."
    return pair(
        family="causes_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "causes")], target_a=0,
        summary_a="Event one is upstream.",
        relations_b=[edge(1, 0, "causes")], target_b=1,
        summary_b="Event two is upstream.",
    )


def supports_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 4, 4)
    fields = [
        field("claim_one", f"Claim one assigns {first} to {subject}'s {attribute}."),
        field("claim_two", f"Claim two assigns {second} to {subject}'s {attribute}."),
        field("source", f"A neutral source concerns the same {attribute} for {subject}."),
    ]
    query = f"Follow the support arrow and select the supported {attribute} claim for {subject}."
    return pair(
        family="supports_direction", index=index, fields=fields, query=query,
        relations_a=[edge(2, 0, "supports")], target_a=0,
        summary_a=f"The supported claim gives {first} for {attribute}.",
        relations_b=[edge(2, 1, "supports")], target_b=1,
        summary_b=f"The supported claim gives {second} for {attribute}.",
    )


def conflict_resolution(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 5, 5)
    fields = [
        field("claim_one", f"Claim one assigns {first} to {subject}'s {attribute}."),
        field("claim_two", f"Claim two assigns {second} to {subject}'s {attribute}."),
        field("arbiter", f"An arbiter record concerns {subject}'s {attribute}."),
    ]
    query = f"Resolve the conflict using the arbiter support edge for {subject}'s {attribute}."
    return pair(
        family="conflict_resolution", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "conflicts_with"), edge(2, 0, "supports")], target_a=0,
        summary_a=f"The graph resolves the value to {first}.",
        relations_b=[edge(0, 1, "conflicts_with"), edge(2, 1, "supports")], target_b=1,
        summary_b=f"The graph resolves the value to {second}.",
    )


def mixed_update_support(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 6, 6)
    fields = [
        field("state_one", f"State one assigns {first} to {subject}'s {attribute}."),
        field("state_two", f"State two assigns {second} to {subject}'s {attribute}."),
        field("validator", f"A validator record concerns {subject}."),
    ]
    query = f"Combine the update and validation links to choose {subject}'s active {attribute}."
    return pair(
        family="mixed_update_support", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "supersedes"), edge(2, 0, "supports")], target_a=0,
        summary_a=f"The active {attribute} is {first}.",
        relations_b=[edge(1, 0, "supersedes"), edge(2, 1, "supports")], target_b=1,
        summary_b=f"The active {attribute} is {second}.",
    )


def temporal_chain(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, third = parts(index, 7, 7)
    fields = [
        field("frame_one", f"Frame one stores {first} for {subject}'s {attribute}."),
        field("frame_two", f"Frame two stores {second} for {subject}'s {attribute}."),
        field("frame_three", f"Frame three stores {third} for {subject}'s {attribute}."),
    ]
    query = f"Trace the successor chain and choose the latest frame for {subject}'s {attribute}."
    return pair(
        family="temporal_chain", index=index, fields=fields, query=query,
        relations_a=[edge(2, 1, "temporal_successor"), edge(1, 0, "temporal_successor")], target_a=2,
        summary_a="Frame three is latest.",
        relations_b=[edge(0, 1, "temporal_successor"), edge(1, 2, "temporal_successor")], target_b=0,
        summary_b="Frame one is latest.",
    )


def causal_chain(index: int) -> list[dict[str, Any]]:
    subject, _attribute, _first, _second, _third = parts(index)
    fields = [
        field("event_one", f"Event one happened in {subject}."),
        field("event_two", f"Event two happened in {subject}."),
        field("event_three", f"Event three happened in {subject}."),
    ]
    query = f"Trace the causal arrows and choose the root event in {subject}."
    return pair(
        family="causal_chain", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "causes"), edge(1, 2, "causes")], target_a=0,
        summary_a="Event one is the root cause.",
        relations_b=[edge(2, 1, "causes"), edge(1, 0, "causes")], target_b=2,
        summary_b="Event three is the root cause.",
    )


def support_aggregation_topology(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second, _third = parts(index, 1, 3)
    fields = [
        field("claim_one", f"Claim one assigns {first} to {subject}'s {attribute}."),
        field("claim_two", f"Claim two assigns {second} to {subject}'s {attribute}."),
        field("source_one", f"Source one concerns {subject}'s {attribute}."),
        field("source_two", f"Source two concerns {subject}'s {attribute}."),
    ]
    query = f"Use both support links to identify the jointly backed claim for {subject}."
    return pair(
        family="support_aggregation_topology", index=index, fields=fields, query=query,
        relations_a=[edge(2, 0, "supports"), edge(3, 0, "supports")], target_a=0,
        summary_a=f"The jointly backed value is {first}.",
        relations_b=[edge(2, 1, "supports"), edge(3, 1, "supports")], target_b=1,
        summary_b=f"The jointly backed value is {second}.",
    )


FAMILIES: list[tuple[str, Callable[[int], list[dict[str, Any]]]]] = [
    ("supersedes_direction", supersedes_direction),
    ("corrects_direction", corrects_direction),
    ("temporal_direction", temporal_direction),
    ("causes_direction", causes_direction),
    ("supports_direction", supports_direction),
    ("conflict_resolution", conflict_resolution),
    ("mixed_update_support", mixed_update_support),
    ("temporal_chain", temporal_chain),
    ("causal_chain", causal_chain),
    ("support_aggregation_topology", support_aggregation_topology),
]


def validate(rows: list[dict[str, Any]]) -> None:
    ids = [str(item["id"]) for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate repair curriculum id")

    by_pair: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        if item.get("private_identity_content") is not False:
            raise ValueError("private identity content crossed repair boundary")
        if item.get("independent_analogue") is not True:
            raise ValueError("repair rows must be independent analogues")
        by_pair.setdefault(str(item["pair_id"]), []).append(item)

    for pair_id, members in by_pair.items():
        if len(members) != 2:
            raise ValueError(f"{pair_id} does not contain exactly two variants")
        a, b = sorted(members, key=lambda item: item["variant"])
        if a["query_text"] != b["query_text"] or a["fields"] != b["fields"]:
            raise ValueError(f"{pair_id} changed text/query across variants")
        if a["relations"] == b["relations"]:
            raise ValueError(f"{pair_id} does not change relation structure")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise ValueError(f"{pair_id} does not flip target evidence")
        if a["split"] != b["split"]:
            raise ValueError(f"{pair_id} crosses train/dev split")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    rows: list[dict[str, Any]] = []
    for _name, builder in FAMILIES:
        for index in range(PAIRS_PER_FAMILY):
            rows.extend(builder(index))
    validate(rows)

    split_counts = Counter(str(item["split"]) for item in rows)
    family_counts = Counter(str(item["family"]) for item in rows)
    pair_splits = Counter()
    seen_pairs: set[str] = set()
    for item in rows:
        pair_id = str(item["pair_id"])
        if pair_id not in seen_pairs:
            seen_pairs.add(pair_id)
            pair_splits[str(item["split"])] += 1

    expected_rows = 2 * PAIRS_PER_FAMILY * len(FAMILIES)
    if len(rows) != expected_rows:
        raise ValueError(f"unexpected repair row count: {len(rows)}")
    if split_counts != {"train": 240, "dev": 80}:
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    if pair_splits != {"train": 120, "dev": 40}:
        raise ValueError(f"unexpected pair split counts: {dict(pair_splits)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "alice.eipm.n0.v02-relation-repair-curriculum.v0.1",
        "status": "COMPILED_NOT_TRAINED",
        "source": "deterministic_public_independent_relation_analogues",
        "compiled_file": str(output),
        "compiled_sha256": sha256_file(output),
        "compiled_rows": len(rows),
        "compiled_pairs": len(seen_pairs),
        "split_counts": dict(sorted(split_counts.items())),
        "pair_split_counts": dict(sorted(pair_splits.items())),
        "families": [name for name, _builder in FAMILIES],
        "family_counts": dict(sorted(family_counts.items())),
        "paired_text_and_query_identical": True,
        "relation_structure_flips_target": True,
        "independent_analogues": True,
        "frozen_relation_essential_challenge_rows_used_for_training": False,
        "private_identity_content": False,
        "hard_parameter_ceiling": None,
        "training_authorized": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
