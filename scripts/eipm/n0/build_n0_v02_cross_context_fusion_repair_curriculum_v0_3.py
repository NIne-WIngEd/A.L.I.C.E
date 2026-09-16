#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from alice_personality.n0.evidence_graph import relation_type_id


# This is a failure-driven *training* tranche.  It is intentionally authored
# with different entities, attributes, sentence forms, and row construction
# from the frozen challenge.  The failed challenge is diagnostic only and its
# rows/templates are never copied into this curriculum.
SUBJECTS = [
    "Relay Aster",
    "Relay Birch",
    "Relay Cedar",
    "Relay Delta",
    "Relay Elm",
    "Relay Fir",
    "Relay Ginkgo",
    "Relay Hazel",
    "Relay Iris",
    "Relay Juniper",
    "Relay Linden",
    "Relay Maple",
]

ATTRIBUTES = [
    ("sync policy", "deferred", "eager", "adaptive"),
    ("lease horizon", "12 minutes", "28 minutes", "44 minutes"),
    ("ingest lane", "bronze", "silver", "platinum"),
    ("verification mode", "sampled", "guarded", "strict"),
    ("cache tier", "archive", "standard", "hot"),
    ("handoff port", "8110", "8240", "8390"),
    ("worker target", "3", "6", "9"),
    ("refresh interval", "7 seconds", "19 seconds", "41 seconds"),
    ("policy band", "basic", "elevated", "critical"),
    ("dispatch rule", "round-robin", "weighted", "priority"),
    ("snapshot period", "8 minutes", "21 minutes", "34 minutes"),
    ("retention window", "brief", "normal", "long"),
]

ROWS_PER_REPAIR_FAMILY = 24
TRAIN_PER_REPAIR_FAMILY = 18


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def parts(index: int, offset: int = 0) -> tuple[str, str, str, str, str]:
    subject = SUBJECTS[(index * 5 + offset) % len(SUBJECTS)]
    attribute, old, middle, current = ATTRIBUTES[(index * 7 + 2 * offset) % len(ATTRIBUTES)]
    return subject, attribute, old, middle, current


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
    if available is None:
        available = [True, True, True]
    return {
        "id": f"N0V02-FUS-REPAIR03-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": "train" if index < TRAIN_PER_REPAIR_FAMILY else "dev",
        "raw_text": raw_text,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_summary_text": target_summary,
        "target_view_distribution": target_views,
        "view_reliability": reliability,
        "view_available": available,
        "cross_view_conflict": cross_view_conflict,
        "generated_text": True,
        "data_origin": "deterministic_public_failure_driven_repair_templates_v0.3",
        "source_authority": "public_synthetic_training_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_row": False,
    }


def temporal_correction_over_stale_pair(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 0)
    fields = [
        field(
            "stale_snapshot",
            f"An earlier typed snapshot still records {old} for {subject}'s {attribute}.",
            temporal_scope_id=1,
            confidence=0.86,
        ),
        field(
            "verified_revision",
            f"A later verified revision records {current} for {subject}'s {attribute}.",
            provenance_id=2,
            temporal_scope_id=3,
            confidence=0.995,
        ),
    ]
    return make_row(
        family="temporal_correction_over_stale_pair",
        index=index,
        raw_text=(
            f"A cached prose summary for {subject} still reports {old} as the {attribute}; "
            "it predates the verified revision."
        ),
        query=f"Which {attribute} is supported by the most current verified record for {subject}?",
        fields=fields,
        relations=[edge(1, 0, "corrects", 0.995)],
        target_summary=f"The most current verified {attribute} for {subject} is {current}.",
        target_views=[0.09, 0.17, 0.74],
        reliability=[0.58, 0.76, 0.995],
        cross_view_conflict=True,
    )


def multi_hop_revision_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 1)
    fields = [
        field("rev0", f"Revision zero records {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.90),
        field("rev1", f"Revision one changes {subject}'s {attribute} to {middle}.", temporal_scope_id=2, confidence=0.95),
        field("rev2", f"Revision two changes {subject}'s {attribute} to {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
    ]
    return make_row(
        family="multi_hop_revision_chain",
        index=index,
        raw_text=f"The prose mentions historical settings {old} and {middle} for {subject}, but omits the latest {attribute}.",
        query=f"Resolve the revision chain and report {subject}'s current {attribute}.",
        fields=fields,
        relations=[edge(1, 0, "supersedes", 0.96), edge(2, 1, "supersedes", 0.995)],
        target_summary=f"After resolving the revision chain, {subject}'s current {attribute} is {current}.",
        target_views=[0.05, 0.18, 0.77],
        reliability=[0.42, 0.82, 0.995],
        cross_view_conflict=True,
    )


def direct_semantic_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields = [
        field("imported_guess", f"An imported low-confidence state suggests {old} for {subject}'s {attribute}.", provenance_id=3, confidence=0.22),
        field("unknown_relation", f"No verified relation resolves the imported state for {subject}.", confidence=0.20, missing=True),
    ]
    return make_row(
        family="direct_semantic_authority",
        index=index,
        raw_text=f"A directly verified operator statement confirms that {subject}'s {attribute} is {current}.",
        query=f"Using the source quality, what should be treated as {subject}'s {attribute}?",
        fields=fields,
        relations=[],
        target_summary=f"{subject}'s {attribute} should be treated as {current}.",
        target_views=[0.82, 0.15, 0.03],
        reliability=[0.995, 0.26, 0.08],
        cross_view_conflict=True,
    )


def verified_structured_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 3)
    fields = [
        field("signed_state", f"A signed state record fixes {subject}'s {attribute} at {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
        field("state_witness", f"A second typed witness agrees with {current} for {subject}.", provenance_id=2, temporal_scope_id=3, confidence=0.97),
    ]
    return make_row(
        family="verified_structured_authority",
        index=index,
        raw_text=f"An informal note speculates that {subject}'s {attribute} could still be {old}; the note is explicitly unverified.",
        query=f"What verified {attribute} should be used for {subject}?",
        fields=fields,
        relations=[],
        target_summary=f"The verified {attribute} for {subject} is {current}.",
        target_views=[0.07, 0.89, 0.04],
        reliability=[0.30, 0.995, 0.18],
        cross_view_conflict=True,
    )


def missing_semantic_relational(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 4)
    fields = [
        field("prior", f"A prior state records {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.92),
        field("current", f"The latest structured record gives {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="missing_semantic_relational",
        index=index,
        raw_text="No raw prose view is available for this query.",
        query=f"What is the current {attribute} for {subject}?",
        fields=fields,
        relations=[edge(1, 0, "supersedes", 0.99)],
        target_summary=f"The current {attribute} for {subject} is {current}.",
        target_views=[0.0, 0.38, 0.62],
        reliability=[0.0, 0.92, 0.99],
        available=[False, True, True],
        cross_view_conflict=True,
    )


def missing_evidence_joint_agreement(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 5)
    fields = [
        field("typed_current", f"The typed state records {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.98),
    ]
    return make_row(
        family="missing_evidence_joint_agreement",
        index=index,
        raw_text=f"The current {attribute} for {subject} is explicitly stated as {current}.",
        query=f"What is {subject}'s current {attribute}?",
        fields=fields,
        relations=[],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.52, 0.48, 0.0],
        reliability=[0.97, 0.96, 0.0],
        available=[True, True, False],
    )


def balanced_unresolved_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 6)
    fields = [
        field("report_a", f"Verified report A gives {old} for {subject}'s {attribute}.", provenance_id=2, confidence=0.90),
        field("report_b", f"Verified report B gives {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.90),
    ]
    return make_row(
        family="balanced_unresolved_conflict",
        index=index,
        raw_text=f"Two verified sources disagree about {subject}'s {attribute}; neither is newer or stronger.",
        query=f"Can one {attribute} be established for {subject}?",
        fields=fields,
        relations=[edge(0, 1, "conflicts_with", 0.95)],
        target_summary=f"No single {attribute} can yet be established for {subject}.",
        target_views=[0.18, 0.22, 0.60],
        reliability=[0.82, 0.82, 0.97],
        cross_view_conflict=True,
    )


def consensus_with_noisy_low_authority(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 7)
    fields = [
        field("verified_state", f"Verified typed state gives {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.97),
        field("weak_duplicate", f"A low-authority imported duplicate also says {current} for {subject}.", provenance_id=3, confidence=0.35),
    ]
    return make_row(
        family="consensus_with_noisy_low_authority",
        index=index,
        raw_text=f"A direct status statement says {subject}'s {attribute} is {current}.",
        query=f"What is {subject}'s {attribute}, accounting for source quality?",
        fields=fields,
        relations=[edge(1, 0, "derived_from", 0.45)],
        target_summary=f"{subject}'s {attribute} is {current}.",
        target_views=[0.43, 0.42, 0.15],
        reliability=[0.98, 0.97, 0.42],
    )


REPAIR_FAMILIES: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
    ("temporal_correction_over_stale_pair", temporal_correction_over_stale_pair),
    ("multi_hop_revision_chain", multi_hop_revision_chain),
    ("direct_semantic_authority", direct_semantic_authority),
    ("verified_structured_authority", verified_structured_authority),
    ("missing_semantic_relational", missing_semantic_relational),
    ("missing_evidence_joint_agreement", missing_evidence_joint_agreement),
    ("balanced_unresolved_conflict", balanced_unresolved_conflict),
    ("consensus_with_noisy_low_authority", consensus_with_noisy_low_authority),
]


def validate(rows: list[dict[str, Any]], base_count: int) -> None:
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("repair curriculum ids must be unique")
    expected_added = len(REPAIR_FAMILIES) * ROWS_PER_REPAIR_FAMILY
    if len(rows) != base_count + expected_added:
        raise ValueError("repair curriculum row count drift")
    for row in rows:
        if row.get("private_identity_content") is not False:
            raise ValueError("repair curriculum crossed private identity boundary")
        target = row["target_view_distribution"]
        available = row["view_available"]
        reliability = row["view_reliability"]
        if len(target) != 3 or len(available) != 3 or len(reliability) != 3:
            raise ValueError("current N0 repair curriculum must provide three public views")
        if abs(sum(float(x) for x in target) - 1.0) > 1e-6:
            raise ValueError("target view distribution must sum to one")
        if any(float(mass) > 0.0 and not bool(ok) for mass, ok in zip(target, available)):
            raise ValueError("target assigns mass to unavailable view")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-curriculum", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_curriculum).resolve()
    base_rows = read_jsonl(base_path)
    if not base_rows:
        raise SystemExit("base fusion curriculum is empty")
    if any(row.get("source_authority") != "public_synthetic_training_only" for row in base_rows):
        raise SystemExit("base fusion curriculum authority mismatch")

    repair_rows: list[dict[str, Any]] = []
    for _name, builder in REPAIR_FAMILIES:
        for index in range(ROWS_PER_REPAIR_FAMILY):
            repair_rows.append(builder(index))

    rows = base_rows + repair_rows
    validate(rows, len(base_rows))

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    split_counts = Counter(str(row["split"]) for row in rows)
    family_counts = Counter(str(row["family"]) for row in rows)
    repair_family_counts = Counter(str(row["family"]) for row in repair_rows)
    manifest = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-repair-curriculum.v0.3",
        "status": "COMPILED_FAILURE_DRIVEN_NOT_ACTIVATED",
        "compiled_sha256": sha256_file(output),
        "base_curriculum_sha256": sha256_file(base_path),
        "rows": len(rows),
        "base_rows": len(base_rows),
        "repair_rows": len(repair_rows),
        "train_rows": split_counts["train"],
        "dev_rows": split_counts["dev"],
        "family_count": len(family_counts),
        "repair_family_count": len(repair_family_counts),
        "repair_families": dict(sorted(repair_family_counts.items())),
        "current_view_order": ["semantic", "structured", "evidence"],
        "row_count_is_training_tranche_not_cap": True,
        "hard_parameter_ceiling": None,
        "view_count_ceiling": None,
        "failure_source": "first_frozen_fusion_challenge_v0.1",
        "frozen_challenge_rows_reused": False,
        "frozen_challenge_text_templates_reused": False,
        "frozen_challenge_entities_reused": False,
        "data_origin": "base_public_curriculum_plus_independent_failure_driven_public_repair_templates",
        "source_authority": "public_synthetic_training_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
