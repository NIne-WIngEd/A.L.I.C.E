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
    "Project Atlas",
    "Project Birch",
    "Project Cedar",
    "Project Delta",
    "Project Ember",
    "Project Flint",
    "Project Grove",
    "Project Harbor",
    "Project Iris",
    "Project Juniper",
    "Project Kestrel",
    "Project Lumen",
    "Project Maple",
    "Project Nova",
    "Project Orion",
    "Project Pine",
    "Project Quartz",
    "Project River",
    "Project Solace",
    "Project Tundra",
    "Project Umbra",
    "Project Vale",
    "Project Willow",
    "Project Xenon",
]

ATTRIBUTES = [
    ("service port", "4100", "4200", "4300"),
    ("operating mode", "safe", "adaptive", "precision"),
    ("release channel", "alpha", "beta", "stable"),
    ("retry limit", "2", "4", "6"),
    ("sampling rate", "20 Hz", "40 Hz", "80 Hz"),
    ("checkpoint interval", "10 minutes", "20 minutes", "30 minutes"),
    ("storage tier", "cold", "warm", "hot"),
    ("routing policy", "direct", "balanced", "priority"),
]

FIELD_TYPE_ID = 1
PROVENANCE_ID = 1
RELATION_ROLE_ID = 1
TEMPORAL_SCOPE_ID = 1
EXAMPLES_PER_FAMILY = 24
TRAIN_PER_FAMILY = 18


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(name: str, text: str, *, confidence: float = 1.0) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": FIELD_TYPE_ID,
        "provenance_id": PROVENANCE_ID,
        "relation_role_id": RELATION_ROLE_ID,
        "temporal_scope_id": TEMPORAL_SCOPE_ID,
        "confidence": confidence,
        "missing": False,
    }


def edge(source: int, target: int, relation: str, *, confidence: float = 1.0) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_type_id": relation_type_id(relation),
        "confidence": confidence,
    }


def base_parts(index: int) -> tuple[str, str, str, str, str]:
    subject = SUBJECTS[index % len(SUBJECTS)]
    attribute, old, current, alternate = ATTRIBUTES[index % len(ATTRIBUTES)]
    return subject, attribute, old, current, alternate


def make_row(
    *,
    family: str,
    index: int,
    fields: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    query: str,
    target_distribution: list[float],
    target_summary: str,
    counterfactual_required: bool = True,
) -> dict[str, Any]:
    if len(fields) != len(target_distribution):
        raise ValueError("target distribution must match field count")
    if any(value < 0.0 for value in target_distribution) or sum(target_distribution) <= 0.0:
        raise ValueError("target distribution must contain positive non-negative mass")
    split = "train" if index < TRAIN_PER_FAMILY else "dev"
    normalized = [value / sum(target_distribution) for value in target_distribution]
    return {
        "id": f"N0V02-GRAPH-{family.upper()}-{index + 1:02d}",
        "split": split,
        "family": family,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_evidence_distribution": normalized,
        "target_summary_text": target_summary,
        "counterfactual_required": counterfactual_required,
        "counterfactual_policy": "drop_first_decisive_relation" if counterfactual_required else "none",
        "synthetic_public_template": True,
        "private_identity_content": False,
    }


def supersession_current(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("historical", f"{subject} previously used {old} as its {attribute}."),
        field("current", f"{subject} now uses {current} as its {attribute}."),
        field("distractor", f"A separate system uses {alternate} as its {attribute}.", confidence=0.8),
    ]
    return make_row(
        family="supersession_current",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "supersedes")],
        query=f"What is the current {attribute} for {subject}?",
        target_distribution=[0.0, 1.0, 0.0],
        target_summary=f"The current {attribute} for {subject} is {current}.",
    )


def supersession_historical(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("historical", f"{subject} previously used {old} as its {attribute}."),
        field("current", f"{subject} now uses {current} as its {attribute}."),
        field("distractor", f"A separate system uses {alternate} as its {attribute}.", confidence=0.8),
    ]
    return make_row(
        family="supersession_historical",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "supersedes")],
        query=f"What {attribute} did {subject} use immediately before the current value?",
        target_distribution=[1.0, 0.0, 0.0],
        target_summary=f"Immediately before the current value, {subject} used {old} as its {attribute}.",
    )


def correction_current(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("incorrect_record", f"An earlier record listed {old} as the {attribute} for {subject}.", confidence=0.7),
        field("correction", f"A correction states that the {attribute} for {subject} is {current}."),
        field("source", f"The authoritative configuration for {subject} reports {current} for {attribute}."),
        field("distractor", f"A different project reports {alternate} for the same attribute.", confidence=0.8),
    ]
    return make_row(
        family="correction_current",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "corrects"), edge(2, 1, "supports")],
        query=f"After correction, what is the {attribute} for {subject}?",
        target_distribution=[0.0, 0.65, 0.35, 0.0],
        target_summary=f"After correction, the {attribute} for {subject} is {current}.",
    )


def unresolved_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("claim_a", f"One current source says {subject} uses {old} for {attribute}.", confidence=0.9),
        field("claim_b", f"Another current source says {subject} uses {current} for {attribute}.", confidence=0.9),
        field("distractor", f"A different system uses {alternate} for {attribute}.", confidence=0.8),
    ]
    return make_row(
        family="unresolved_conflict",
        index=index,
        fields=fields,
        relations=[edge(0, 1, "conflicts_with")],
        query=f"What is the {attribute} for {subject} according to the available evidence?",
        target_distribution=[0.5, 0.5, 0.0],
        target_summary=f"The available evidence is unresolved between {old} and {current} for {subject}'s {attribute}.",
    )


def resolved_conflict_support(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("claim_a", f"One source says {subject} uses {old} for {attribute}.", confidence=0.8),
        field("claim_b", f"Another source says {subject} uses {current} for {attribute}.", confidence=0.8),
        field("authority", f"The authoritative configuration confirms {current} for {subject}'s {attribute}."),
        field("distractor", f"An unrelated system uses {alternate} for {attribute}.", confidence=0.8),
    ]
    return make_row(
        family="resolved_conflict_support",
        index=index,
        fields=fields,
        relations=[edge(0, 1, "conflicts_with"), edge(2, 1, "supports")],
        query=f"Which value is best supported for {subject}'s {attribute}?",
        target_distribution=[0.0, 0.65, 0.35, 0.0],
        target_summary=f"The best-supported {attribute} for {subject} is {current}.",
    )


def support_aggregation(index: int) -> dict[str, Any]:
    subject, attribute, _old, current, alternate = base_parts(index)
    fields = [
        field("claim", f"The working claim is that {subject} uses {current} for {attribute}."),
        field("support_a", f"Configuration evidence reports {current} for {subject}'s {attribute}."),
        field("support_b", f"A verification log independently reports {current} for {subject}'s {attribute}.", confidence=0.9),
        field("distractor", f"An unrelated project reports {alternate} for {attribute}.", confidence=0.8),
    ]
    return make_row(
        family="support_aggregation",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "supports"), edge(2, 0, "supports", confidence=0.9)],
        query=f"What value is jointly supported for {subject}'s {attribute}?",
        target_distribution=[0.50, 0.25, 0.25, 0.0],
        target_summary=f"Multiple sources jointly support {current} for {subject}'s {attribute}.",
    )


def causal_explanation(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("cause", f"{subject} changed its {attribute} from {old} to {current}."),
        field("intermediate", f"That change altered the downstream configuration used by {subject}."),
        field("outcome", f"The observed output for {subject} then changed to the expected {current} behavior."),
        field("distractor", f"A separate project changed to {alternate} at the same time.", confidence=0.7),
    ]
    return make_row(
        family="causal_explanation",
        index=index,
        fields=fields,
        relations=[edge(0, 1, "causes"), edge(1, 2, "causes")],
        query=f"Why did {subject}'s observed behavior change?",
        target_distribution=[0.45, 0.35, 0.20, 0.0],
        target_summary=f"The behavior changed because {subject}'s {attribute} moved from {old} to {current}, which changed its downstream configuration.",
    )


def temporal_latest(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("oldest", f"At the first snapshot, {subject}'s {attribute} was {old}."),
        field("middle", f"At the second snapshot, {subject}'s {attribute} was {current}."),
        field("latest", f"At the latest snapshot, {subject}'s {attribute} is {alternate}."),
        field("distractor", "An unrelated snapshot belongs to another project.", confidence=0.7),
    ]
    return make_row(
        family="temporal_latest",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "temporal_successor"), edge(2, 1, "temporal_successor")],
        query=f"What is the latest recorded {attribute} for {subject}?",
        target_distribution=[0.0, 0.0, 1.0, 0.0],
        target_summary=f"The latest recorded {attribute} for {subject} is {alternate}.",
    )


def temporal_previous(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("oldest", f"At the first snapshot, {subject}'s {attribute} was {old}."),
        field("middle", f"At the second snapshot, {subject}'s {attribute} was {current}."),
        field("latest", f"At the latest snapshot, {subject}'s {attribute} is {alternate}."),
        field("distractor", "An unrelated snapshot belongs to another project.", confidence=0.7),
    ]
    return make_row(
        family="temporal_previous",
        index=index,
        fields=fields,
        relations=[edge(1, 0, "temporal_successor"), edge(2, 1, "temporal_successor")],
        query=f"What was {subject}'s {attribute} immediately before the latest snapshot?",
        target_distribution=[0.0, 1.0, 0.0, 0.0],
        target_summary=f"Immediately before the latest snapshot, {subject}'s {attribute} was {current}.",
    )


def mixed_update_support(index: int) -> dict[str, Any]:
    subject, attribute, old, current, alternate = base_parts(index)
    fields = [
        field("historical", f"The older record for {subject} lists {old} for {attribute}.", confidence=0.8),
        field("current", f"The replacement record lists {current} for {subject}'s {attribute}."),
        field("support", f"A verification source confirms {current} for {subject}'s {attribute}."),
        field("conflicting", f"A lower-confidence source instead reports {alternate} for {subject}'s {attribute}.", confidence=0.55),
    ]
    return make_row(
        family="mixed_update_support",
        index=index,
        fields=fields,
        relations=[
            edge(1, 0, "supersedes"),
            edge(2, 1, "supports"),
            edge(1, 3, "conflicts_with", confidence=0.75),
        ],
        query=f"Considering update history and support, what is the best current {attribute} for {subject}?",
        target_distribution=[0.0, 0.65, 0.35, 0.0],
        target_summary=f"Considering the update history and supporting evidence, {current} is the best current {attribute} for {subject}.",
    )


FAMILIES: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
    ("supersession_current", supersession_current),
    ("supersession_historical", supersession_historical),
    ("correction_current", correction_current),
    ("unresolved_conflict", unresolved_conflict),
    ("resolved_conflict_support", resolved_conflict_support),
    ("support_aggregation", support_aggregation),
    ("causal_explanation", causal_explanation),
    ("temporal_latest", temporal_latest),
    ("temporal_previous", temporal_previous),
    ("mixed_update_support", mixed_update_support),
]


def validate_row(row: dict[str, Any]) -> None:
    fields = row["fields"]
    relations = row["relations"]
    distribution = row["target_evidence_distribution"]
    if len(fields) < 2:
        raise ValueError(f"{row['id']} requires at least two fields")
    if len(distribution) != len(fields) or abs(sum(distribution) - 1.0) > 1e-9:
        raise ValueError(f"{row['id']} has invalid evidence distribution")
    for relation in relations:
        if relation["source"] < 0 or relation["source"] >= len(fields):
            raise ValueError(f"{row['id']} has invalid relation source")
        if relation["target"] < 0 or relation["target"] >= len(fields):
            raise ValueError(f"{row['id']} has invalid relation target")
        if relation_type_id(relation["relation"]) != relation["relation_type_id"]:
            raise ValueError(f"{row['id']} relation id drift")
    if row.get("private_identity_content") is not False:
        raise ValueError(f"{row['id']} violates public identity-neutral boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()

    rows: list[dict[str, Any]] = []
    for _family_name, builder in FAMILIES:
        for index in range(EXAMPLES_PER_FAMILY):
            row = builder(index)
            validate_row(row)
            rows.append(row)

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence graph curriculum contains duplicate ids")

    split_counts = Counter(row["split"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    split_family_counts = Counter((row["split"], row["family"]) for row in rows)
    if split_counts != {"train": 180, "dev": 60}:
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    for family_name, _builder in FAMILIES:
        if split_family_counts[("train", family_name)] != 18:
            raise ValueError(f"family {family_name} lost train coverage")
        if split_family_counts[("dev", family_name)] != 6:
            raise ValueError(f"family {family_name} lost dev coverage")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "alice.eipm.n0.v02-evidence-graph-curriculum.v0.1",
        "status": "COMPILED_NOT_ACTIVATED",
        "source": "deterministic_public_identity_neutral_relation_templates",
        "synthetic_public_templates": True,
        "private_identity_content": False,
        "compiled_file": str(output),
        "compiled_sha256": sha256_file(output),
        "compiled_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "families": [name for name, _builder in FAMILIES],
        "train_examples_per_family": 18,
        "dev_examples_per_family": 6,
        "soft_evidence_targets": True,
        "semantic_summary_targets": True,
        "relation_counterfactual_training": True,
        "historical_query_coverage": True,
        "unresolved_conflict_coverage": True,
        "multi_hop_coverage": True,
        "query_conditioned_evidence_selection": True,
        "hard_parameter_ceiling": None,
        "activation_authorized": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
