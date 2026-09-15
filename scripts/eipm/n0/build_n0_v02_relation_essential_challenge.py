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
    "System Aster", "System Beacon", "System Cirrus", "System Dune",
    "System Echo", "System Frost", "System Gale", "System Haven",
]
ATTRIBUTES = [
    ("service port", "5100", "5200"),
    ("operating mode", "guarded", "dynamic"),
    ("release lane", "canary", "stable"),
    ("retry ceiling", "3", "7"),
    ("sample rate", "25 Hz", "75 Hz"),
    ("storage class", "archive", "active"),
    ("routing mode", "direct", "priority"),
    ("checkpoint cadence", "12 minutes", "28 minutes"),
]
PAIRS_PER_FAMILY = 4


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


def row(
    *,
    pair_id: str,
    variant: str,
    family: str,
    fields: list[dict[str, Any]],
    query: str,
    relations: list[dict[str, Any]],
    target_index: int,
    summary: str,
) -> dict[str, Any]:
    distribution = [0.0] * len(fields)
    distribution[target_index] = 1.0
    return {
        "id": f"{pair_id}-{variant}",
        "pair_id": pair_id,
        "variant": variant,
        "family": family,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_evidence_distribution": distribution,
        "target_summary_text": summary,
        "relation_essential": True,
        "paired_text_and_query_identical": True,
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
    pair_id = f"N0V02-RELPAIR-{family.upper()}-{index + 1:02d}"
    return [
        row(
            pair_id=pair_id,
            variant="A",
            family=family,
            fields=fields,
            query=query,
            relations=relations_a,
            target_index=target_a,
            summary=summary_a,
        ),
        row(
            pair_id=pair_id,
            variant="B",
            family=family,
            fields=fields,
            query=query,
            relations=relations_b,
            target_index=target_b,
            summary=summary_b,
        ),
    ]


def parts(index: int, subject_offset: int = 0, attr_offset: int = 0) -> tuple[str, str, str, str]:
    subject = SUBJECTS[(index + subject_offset) % len(SUBJECTS)]
    attribute, first, second = ATTRIBUTES[(index + attr_offset) % len(ATTRIBUTES)]
    return subject, attribute, first, second


def supersedes_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index)
    fields = [
        field("record_a", f"Record A for {subject} lists {first} for {attribute}."),
        field("record_b", f"Record B for {subject} lists {second} for {attribute}."),
        field("other", f"Record C belongs to another system and is irrelevant to {subject}."),
    ]
    query = f"According to the update graph, what is the current {attribute} for {subject}?"
    return pair(
        family="supersedes_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "supersedes")], target_a=0,
        summary_a=f"The current {attribute} for {subject} is {first}.",
        relations_b=[edge(1, 0, "supersedes")], target_b=1,
        summary_b=f"The current {attribute} for {subject} is {second}.",
    )


def corrects_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 1, 2)
    fields = [
        field("record_a", f"Record A for {subject} lists {first} for {attribute}."),
        field("record_b", f"Record B for {subject} lists {second} for {attribute}."),
        field("context", f"Both records concern the same {attribute} for {subject}."),
    ]
    query = f"According to the correction graph, which value should be used for {subject}'s {attribute}?"
    return pair(
        family="corrects_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "corrects")], target_a=0,
        summary_a=f"The corrected {attribute} for {subject} is {first}.",
        relations_b=[edge(1, 0, "corrects")], target_b=1,
        summary_b=f"The corrected {attribute} for {subject} is {second}.",
    )


def temporal_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 2, 3)
    fields = [
        field("snapshot_a", f"Snapshot A for {subject} records {first} for {attribute}."),
        field("snapshot_b", f"Snapshot B for {subject} records {second} for {attribute}."),
        field("other", "Snapshot C belongs to another system."),
    ]
    query = f"According to the temporal graph, what is the latest {attribute} for {subject}?"
    return pair(
        family="temporal_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "temporal_successor")], target_a=0,
        summary_a=f"The latest {attribute} for {subject} is {first}.",
        relations_b=[edge(1, 0, "temporal_successor")], target_b=1,
        summary_b=f"The latest {attribute} for {subject} is {second}.",
    )


def causes_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 3, 4)
    fields = [
        field("event_a", f"Event A occurred in {subject} while {attribute} was {first}."),
        field("event_b", f"Event B occurred in {subject} while {attribute} was {second}."),
        field("other", "Event C occurred in another system."),
    ]
    query = f"According to the causal graph, which event is the upstream cause in {subject}?"
    return pair(
        family="causes_direction", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "causes")], target_a=0, summary_a="Event A is the upstream cause.",
        relations_b=[edge(1, 0, "causes")], target_b=1, summary_b="Event B is the upstream cause.",
    )


def supports_direction(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 4, 5)
    fields = [
        field("claim_a", f"Claim A says {subject} uses {first} for {attribute}."),
        field("claim_b", f"Claim B says {subject} uses {second} for {attribute}."),
        field("note", f"A neutral note concerns {subject}'s {attribute}."),
    ]
    query = f"According to the support graph, which claim is supported for {subject}'s {attribute}?"
    return pair(
        family="supports_direction", index=index, fields=fields, query=query,
        relations_a=[edge(1, 0, "supports")], target_a=0,
        summary_a=f"Claim A is supported, so {attribute} is {first}.",
        relations_b=[edge(0, 1, "supports")], target_b=1,
        summary_b=f"Claim B is supported, so {attribute} is {second}.",
    )


def conflict_resolution(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 5, 6)
    fields = [
        field("claim_a", f"Claim A says {subject} uses {first} for {attribute}."),
        field("claim_b", f"Claim B says {subject} uses {second} for {attribute}."),
        field("authority", f"Record C is an authority record for {subject}."),
    ]
    query = f"According to the conflict-and-support graph, which value is best supported for {subject}'s {attribute}?"
    return pair(
        family="conflict_resolution", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "conflicts_with"), edge(2, 0, "supports")], target_a=0,
        summary_a=f"The best-supported {attribute} is {first}.",
        relations_b=[edge(0, 1, "conflicts_with"), edge(2, 1, "supports")], target_b=1,
        summary_b=f"The best-supported {attribute} is {second}.",
    )


def mixed_update_support(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 6, 7)
    fields = [
        field("state_a", f"State A for {subject} records {first} for {attribute}."),
        field("state_b", f"State B for {subject} records {second} for {attribute}."),
        field("verifier", f"Record C is a verifier for {subject}."),
    ]
    query = f"According to both update and support relations, what is the current {attribute} for {subject}?"
    return pair(
        family="mixed_update_support", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "supersedes"), edge(2, 0, "supports")], target_a=0,
        summary_a=f"The current {attribute} is {first}.",
        relations_b=[edge(1, 0, "supersedes"), edge(2, 1, "supports")], target_b=1,
        summary_b=f"The current {attribute} is {second}.",
    )


def temporal_chain(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 7, 0)
    third = ATTRIBUTES[(index + 1) % len(ATTRIBUTES)][2]
    fields = [
        field("node_a", f"Node A for {subject} records {first} for {attribute}."),
        field("node_b", f"Node B for {subject} records {second} for {attribute}."),
        field("node_c", f"Node C for {subject} records {third} for {attribute}."),
    ]
    query = f"According to the temporal chain, which node is latest for {subject}'s {attribute}?"
    return pair(
        family="temporal_chain", index=index, fields=fields, query=query,
        relations_a=[edge(1, 0, "temporal_successor"), edge(2, 1, "temporal_successor")], target_a=2,
        summary_a=f"Node C is latest for {subject}.",
        relations_b=[edge(1, 2, "temporal_successor"), edge(0, 1, "temporal_successor")], target_b=0,
        summary_b=f"Node A is latest for {subject}.",
    )


def causal_chain(index: int) -> list[dict[str, Any]]:
    subject, _attribute, _first, _second = parts(index)
    fields = [
        field("event_a", f"Event A occurred in {subject}."),
        field("event_b", f"Event B occurred in {subject}."),
        field("event_c", f"Event C occurred in {subject}."),
    ]
    query = f"According to the causal chain, which event is the root cause in {subject}?"
    return pair(
        family="causal_chain", index=index, fields=fields, query=query,
        relations_a=[edge(0, 1, "causes"), edge(1, 2, "causes")], target_a=0, summary_a="Event A is the root cause.",
        relations_b=[edge(2, 1, "causes"), edge(1, 0, "causes")], target_b=2, summary_b="Event C is the root cause.",
    )


def support_aggregation_topology(index: int) -> list[dict[str, Any]]:
    subject, attribute, first, second = parts(index, 1, 2)
    fields = [
        field("claim_a", f"Claim A says {subject} uses {first} for {attribute}."),
        field("claim_b", f"Claim B says {subject} uses {second} for {attribute}."),
        field("source_c", f"Source C concerns {subject}'s {attribute}."),
        field("source_d", f"Source D concerns {subject}'s {attribute}."),
    ]
    query = f"According to the support graph, which claim is jointly supported for {subject}'s {attribute}?"
    return pair(
        family="support_aggregation_topology", index=index, fields=fields, query=query,
        relations_a=[edge(2, 0, "supports"), edge(3, 0, "supports")], target_a=0,
        summary_a=f"Claim A is jointly supported, so {attribute} is {first}.",
        relations_b=[edge(2, 1, "supports"), edge(3, 1, "supports")], target_b=1,
        summary_b=f"Claim B is jointly supported, so {attribute} is {second}.",
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
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        if item.get("private_identity_content") is not False:
            raise ValueError("relation-essential challenge crossed private boundary")
        by_pair.setdefault(str(item["pair_id"]), []).append(item)
    if len(rows) != 80 or len(by_pair) != 40:
        raise ValueError("expected 80 examples / 40 pairs")
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            raise ValueError(f"pair {pair_id} is incomplete")
        a, b = sorted(members, key=lambda item: item["variant"])
        if a["query_text"] != b["query_text"] or a["fields"] != b["fields"]:
            raise ValueError(f"pair {pair_id} leaked text differences")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise ValueError(f"pair {pair_id} does not require relation-based target flip")


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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    family_counts = Counter(str(item["family"]) for item in rows)
    manifest = {
        "schema": "alice.eipm.n0.v02-relation-essential-challenge.v0.1",
        "status": "FROZEN_EVAL_ONLY",
        "compiled_file": str(output),
        "compiled_sha256": sha256_file(output),
        "examples": len(rows),
        "pairs": 40,
        "families": [name for name, _builder in FAMILIES],
        "family_example_counts": dict(sorted(family_counts.items())),
        "same_text_same_query_with_relation_only_target_flip": True,
        "training_authorized": False,
        "private_identity_content": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
