#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from alice_personality.n0.evidence_graph import relation_type_id


SUBJECTS = [
    "Array Juniper", "Array Kestrel", "Array Lumen", "Array Meridian",
    "Array Nimbus", "Array Orion", "Array Pollen", "Array Quartz",
    "Array Rill", "Array Solace", "Array Tern", "Array Umber",
    "Array Vale", "Array Willow", "Array Xenon", "Array Yarrow",
]
ATTRIBUTES = [
    ("service port", "6100", "6200"),
    ("operating mode", "guarded", "adaptive"),
    ("release lane", "preview", "stable"),
    ("retry ceiling", "4", "8"),
    ("sample rate", "30 Hz", "90 Hz"),
    ("storage class", "cold", "active"),
    ("routing mode", "direct", "priority"),
    ("checkpoint cadence", "14 minutes", "26 minutes"),
    ("replica count", "2", "5"),
    ("batch window", "6 seconds", "18 seconds"),
]
ROWS_PER_FAMILY = 32
TRAIN_PER_FAMILY = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(
    name: str,
    text: str,
    *,
    field_type_id: int = 1,
    provenance_id: int = 1,
    relation_role_id: int = 1,
    temporal_scope_id: int = 1,
    confidence: float = 1.0,
    missing: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": field_type_id,
        "provenance_id": provenance_id,
        "relation_role_id": relation_role_id,
        "temporal_scope_id": temporal_scope_id,
        "confidence": confidence,
        "missing": missing,
    }


def edge(source: int, target: int, relation: str, confidence: float = 1.0) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "relation_type_id": relation_type_id(relation),
        "confidence": confidence,
    }


def parts(index: int, offset: int = 0) -> tuple[str, str, str, str]:
    subject = SUBJECTS[(index + offset) % len(SUBJECTS)]
    attribute, old, current = ATTRIBUTES[(index + 2 * offset) % len(ATTRIBUTES)]
    return subject, attribute, old, current


def make_row(
    *,
    family: str,
    index: int,
    raw_text: str,
    query: str,
    fields: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    target_summary: str,
    target_views: list[float],
    reliability: list[float],
    available: list[bool] | None = None,
    cross_view_conflict: bool = False,
) -> dict[str, Any]:
    split = "train" if index < TRAIN_PER_FAMILY else "dev"
    if available is None:
        available = [True, True, True]
    return {
        "id": f"N0V02-FUS-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": split,
        "raw_text": raw_text,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_summary_text": target_summary,
        "target_view_distribution": target_views,
        "view_reliability": reliability,
        "view_available": available,
        "cross_view_conflict": cross_view_conflict,
        "private_identity_content": False,
        "generated_text": False,
    }


def semantic_decisive(index: int) -> dict[str, Any]:
    subject, attribute, _old, current = parts(index)
    fields = [
        field("status", f"The typed frame for {subject} leaves {attribute} unknown.", missing=True, confidence=0.2),
        field("note", f"The frame has no verified value for {subject}'s {attribute}.", confidence=0.3),
    ]
    return make_row(
        family="semantic_decisive", index=index,
        raw_text=f"The operator explicitly states that {subject}'s {attribute} is {current}.",
        query=f"What is {subject}'s {attribute}?", fields=fields, relations=[],
        target_summary=f"{subject}'s {attribute} is {current}.",
        target_views=[0.90, 0.08, 0.02], reliability=[0.98, 0.35, 0.20],
    )


def structured_decisive(index: int) -> dict[str, Any]:
    subject, attribute, _old, current = parts(index, 1)
    fields = [
        field("current_state", f"Verified typed state: {subject} has {attribute} {current}.", provenance_id=2, temporal_scope_id=2, confidence=0.99),
        field("raw_note", f"A free-form note says only that {subject}'s configuration changed.", confidence=0.5),
    ]
    return make_row(
        family="structured_decisive", index=index,
        raw_text=f"A configuration change was recorded for {subject}, but the prose does not give the new {attribute}.",
        query=f"What is the verified current {attribute} for {subject}?", fields=fields, relations=[],
        target_summary=f"The verified current {attribute} for {subject} is {current}.",
        target_views=[0.08, 0.87, 0.05], reliability=[0.45, 0.99, 0.35],
    )


def evidence_decisive(index: int) -> dict[str, Any]:
    subject, attribute, old, current = parts(index, 2)
    fields = [
        field("record_old", f"Record A lists {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("record_new", f"Record B lists {current} for {subject}'s {attribute}.", temporal_scope_id=2),
    ]
    return make_row(
        family="evidence_decisive", index=index,
        raw_text=f"Two records disagree about {subject}'s {attribute}: one says {old} and one says {current}.",
        query=f"According to the evidence relations, what is the current {attribute} for {subject}?",
        fields=fields, relations=[edge(1, 0, "supersedes")],
        target_summary=f"The current {attribute} for {subject} is {current}.",
        target_views=[0.05, 0.15, 0.80], reliability=[0.65, 0.75, 0.99],
        cross_view_conflict=True,
    )


def semantic_structured_agree(index: int) -> dict[str, Any]:
    subject, attribute, _old, current = parts(index, 3)
    fields = [field("state", f"Typed state says {subject}'s {attribute} is {current}.", confidence=0.98)]
    return make_row(
        family="semantic_structured_agree", index=index,
        raw_text=f"The current {attribute} for {subject} is {current}.",
        query=f"What is {subject}'s current {attribute}?", fields=fields, relations=[],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.48, 0.48, 0.04], reliability=[0.96, 0.96, 0.30],
    )


def structured_evidence_agree(index: int) -> dict[str, Any]:
    subject, attribute, old, current = parts(index, 4)
    fields = [
        field("old", f"Historical record lists {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("current", f"Verified typed state lists {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
    ]
    return make_row(
        family="structured_evidence_agree", index=index,
        raw_text=f"An older prose note still mentions {old} for {subject}'s {attribute}.",
        query=f"What is the current {attribute} for {subject}?", fields=fields,
        relations=[edge(1, 0, "supersedes")],
        target_summary=f"The current {attribute} for {subject} is {current}.",
        target_views=[0.06, 0.47, 0.47], reliability=[0.45, 0.97, 0.99],
        cross_view_conflict=True,
    )


def semantic_evidence_agree(index: int) -> dict[str, Any]:
    subject, attribute, old, current = parts(index, 5)
    fields = [
        field("claim_old", f"A stale typed field lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.4),
        field("claim_current", f"An evidence record lists {current} for {subject}'s {attribute}.", temporal_scope_id=2),
    ]
    return make_row(
        family="semantic_evidence_agree", index=index,
        raw_text=f"The operator states that {subject}'s current {attribute} is {current}.",
        query=f"What is the best-supported current {attribute} for {subject}?", fields=fields,
        relations=[edge(1, 0, "corrects")],
        target_summary=f"The best-supported current {attribute} for {subject} is {current}.",
        target_views=[0.47, 0.06, 0.47], reliability=[0.97, 0.40, 0.99],
        cross_view_conflict=True,
    )


def three_view_consensus(index: int) -> dict[str, Any]:
    subject, attribute, _old, current = parts(index, 6)
    fields = [field("current", f"Typed state records {current} for {subject}'s {attribute}.", confidence=0.98)]
    return make_row(
        family="three_view_consensus", index=index,
        raw_text=f"The current {attribute} for {subject} is {current}.",
        query=f"What is {subject}'s {attribute}?", fields=fields,
        relations=[edge(0, 0, "derived_from", confidence=0.9)],
        target_summary=f"{subject}'s {attribute} is {current}.",
        target_views=[1 / 3, 1 / 3, 1 / 3], reliability=[0.95, 0.95, 0.95],
    )


def unresolved_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, current = parts(index, 7)
    fields = [
        field("claim_a", f"Claim A says {subject}'s {attribute} is {old}.", confidence=0.8),
        field("claim_b", f"Claim B says {subject}'s {attribute} is {current}.", confidence=0.8),
    ]
    return make_row(
        family="unresolved_conflict", index=index,
        raw_text=f"Two equally credible reports disagree: {old} versus {current} for {subject}'s {attribute}.",
        query=f"What value can be established for {subject}'s {attribute}?", fields=fields,
        relations=[edge(0, 1, "conflicts_with")],
        target_summary=f"The available evidence is insufficient to establish one value for {subject}'s {attribute}.",
        target_views=[0.20, 0.20, 0.60], reliability=[0.75, 0.75, 0.95],
        cross_view_conflict=True,
    )


def missing_evidence(index: int) -> dict[str, Any]:
    subject, attribute, _old, current = parts(index, 8)
    fields = [field("state", f"Typed state records {current} for {subject}'s {attribute}.", confidence=0.97)]
    return make_row(
        family="missing_evidence", index=index,
        raw_text=f"The current {attribute} for {subject} is {current}.",
        query=f"What is {subject}'s current {attribute}?", fields=fields, relations=[],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.50, 0.50, 0.0], reliability=[0.95, 0.95, 0.0],
        available=[True, True, False],
    )


def missing_semantic(index: int) -> dict[str, Any]:
    subject, attribute, old, current = parts(index, 9)
    fields = [
        field("old", f"Historical state lists {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("current", f"Current typed state lists {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
    ]
    return make_row(
        family="missing_semantic", index=index,
        raw_text="No raw prose span is available for this record.",
        query=f"What is the current {attribute} for {subject}?", fields=fields,
        relations=[edge(1, 0, "supersedes")],
        target_summary=f"The current {attribute} for {subject} is {current}.",
        target_views=[0.0, 0.50, 0.50], reliability=[0.0, 0.96, 0.99],
        available=[False, True, True],
    )


FAMILIES = [
    ("semantic_decisive", semantic_decisive),
    ("structured_decisive", structured_decisive),
    ("evidence_decisive", evidence_decisive),
    ("semantic_structured_agree", semantic_structured_agree),
    ("structured_evidence_agree", structured_evidence_agree),
    ("semantic_evidence_agree", semantic_evidence_agree),
    ("three_view_consensus", three_view_consensus),
    ("unresolved_conflict", unresolved_conflict),
    ("missing_evidence", missing_evidence),
    ("missing_semantic", missing_semantic),
]


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(FAMILIES) * ROWS_PER_FAMILY:
        raise ValueError("fusion curriculum row count drift")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("fusion curriculum ids must be unique")
    family_counts = Counter(row["family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    if any(family_counts[name] != ROWS_PER_FAMILY for name, _ in FAMILIES):
        raise ValueError("fusion family coverage drift")
    if split_counts != {"train": len(FAMILIES) * TRAIN_PER_FAMILY, "dev": len(FAMILIES) * (ROWS_PER_FAMILY - TRAIN_PER_FAMILY)}:
        raise ValueError("fusion split drift")
    for row in rows:
        if row["private_identity_content"] is not False:
            raise ValueError("fusion curriculum crossed private boundary")
        if row["generated_text"] is not False:
            raise ValueError("fusion curriculum must not claim generated-source authority")
        target = row["target_view_distribution"]
        available = row["view_available"]
        if len(target) != 3 or len(available) != 3 or len(row["view_reliability"]) != 3:
            raise ValueError("fusion view tensors must have width 3")
        if abs(sum(target) - 1.0) > 1e-6:
            raise ValueError("target view distribution must sum to one")
        if any(mass > 0 and not ok for mass, ok in zip(target, available)):
            raise ValueError("target assigns mass to unavailable view")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = [builder(index) for _name, builder in FAMILIES for index in range(ROWS_PER_FAMILY)]
    validate(rows)
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    family_counts = Counter(row["family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    manifest = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-curriculum.v0.1",
        "status": "COMPILED_NOT_ACTIVATED",
        "compiled_sha256": sha256_file(output),
        "rows": len(rows),
        "train_rows": split_counts["train"],
        "dev_rows": split_counts["dev"],
        "families": dict(sorted(family_counts.items())),
        "family_count": len(family_counts),
        "view_order": ["semantic", "structured", "evidence"],
        "cross_view_conflict_rows": sum(bool(row["cross_view_conflict"]) for row in rows),
        "missing_view_rows": sum(not all(row["view_available"]) for row in rows),
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "hard_parameter_ceiling": None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
