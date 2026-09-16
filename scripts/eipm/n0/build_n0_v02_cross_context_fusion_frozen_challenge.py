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
    "Beacon Alder",
    "Beacon Brume",
    "Beacon Cinder",
    "Beacon Drift",
    "Beacon Ember",
    "Beacon Fallow",
    "Beacon Grove",
    "Beacon Harbor",
]

ATTRIBUTES = [
    ("handoff lane", "silver", "amber", "violet"),
    ("audit cadence", "11 minutes", "23 minutes", "37 minutes"),
    ("control profile", "passive", "guarded", "adaptive"),
    ("signal tier", "low", "medium", "high"),
    ("archive mode", "cold", "warm", "live"),
    ("relay port", "7310", "7420", "7530"),
    ("replica target", "2", "4", "7"),
    ("sampling window", "5 seconds", "13 seconds", "29 seconds"),
    ("priority band", "bronze", "silver", "gold"),
    ("routing policy", "direct", "guarded", "priority"),
    ("checkpoint interval", "9 minutes", "17 minutes", "31 minutes"),
    ("retention class", "short", "standard", "extended"),
]

ROWS_PER_FAMILY = 8


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


def parts(index: int, offset: int = 0) -> tuple[str, str, str, str, str]:
    subject = SUBJECTS[(index + offset) % len(SUBJECTS)]
    attribute, old, middle, current = ATTRIBUTES[(index * 3 + offset) % len(ATTRIBUTES)]
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
        "id": f"N0V02-FUS-FROZEN-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": "frozen_challenge",
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
        "data_origin": "deterministic_public_independent_fusion_challenge_templates",
        "source_authority": "public_synthetic_evaluation_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_training_use_forbidden": True,
    }


def semantic_explicit_over_unknown_state(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 0)
    fields = [
        field("state_unknown", f"The typed state for {subject} does not resolve the {attribute}.", confidence=0.15, missing=True),
        field("state_note", f"A structural note only confirms that {subject} has an active configuration.", confidence=0.35),
    ]
    return make_row(
        family="semantic_explicit_over_unknown_state",
        index=index,
        raw_text=f"The operator explicitly confirms: {subject} now uses {current} for its {attribute}.",
        query=f"What {attribute} does {subject} now use?",
        fields=fields,
        relations=[],
        target_summary=f"{subject} now uses {current} for its {attribute}.",
        target_views=[0.84, 0.13, 0.03],
        reliability=[0.99, 0.30, 0.10],
    )


def structured_verified_over_hedged_prose(index: int) -> dict[str, Any]:
    subject, attribute, _old, middle, current = parts(index, 1)
    fields = [
        field("verified_state", f"Verified state: {subject}'s {attribute} is {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
        field("supporting_state", f"The active state record for {subject} is internally consistent with {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.96),
    ]
    return make_row(
        family="structured_verified_over_hedged_prose",
        index=index,
        raw_text=f"A preliminary note says {subject}'s {attribute} might still be {middle}, but the note is explicitly unverified.",
        query=f"What is the verified {attribute} for {subject}?",
        fields=fields,
        relations=[],
        target_summary=f"The verified {attribute} for {subject} is {current}.",
        target_views=[0.08, 0.88, 0.04],
        reliability=[0.35, 0.995, 0.20],
        cross_view_conflict=True,
    )


def evidence_multihop_supersession(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 2)
    fields = [
        field("old_record", f"Old record: {subject}'s {attribute} was {old}.", temporal_scope_id=1, confidence=0.95),
        field("middle_record", f"Intermediate record: {subject}'s {attribute} changed to {middle}.", temporal_scope_id=2, confidence=0.97),
        field("current_record", f"Latest record: {subject}'s {attribute} changed to {current}.", temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="evidence_multihop_supersession",
        index=index,
        raw_text=f"Several archived notes mention {old} and {middle} for {subject}'s {attribute}; the current value is not stated in the prose.",
        query=f"Following the evidence update chain, what is {subject}'s current {attribute}?",
        fields=fields,
        relations=[edge(1, 0, "supersedes"), edge(2, 1, "supersedes")],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.04, 0.16, 0.80],
        reliability=[0.45, 0.80, 0.995],
        cross_view_conflict=True,
    )


def evidence_corrects_stale_consensus(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 3)
    fields = [
        field("stale_state", f"The older typed state still lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.86),
        field("correction", f"A later verified evidence record lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
    ]
    return make_row(
        family="evidence_corrects_stale_consensus",
        index=index,
        raw_text=f"A cached status sentence still says {subject}'s {attribute} is {old}.",
        query=f"What is the best-supported current {attribute} for {subject}?",
        fields=fields,
        relations=[edge(1, 0, "corrects")],
        target_summary=f"The best-supported current {attribute} for {subject} is {current}.",
        target_views=[0.10, 0.18, 0.72],
        reliability=[0.62, 0.72, 0.995],
        cross_view_conflict=True,
    )


def unresolved_equal_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 4)
    fields = [
        field("claim_left", f"Verified report L says {subject}'s {attribute} is {old}.", provenance_id=2, confidence=0.91),
        field("claim_right", f"Verified report R says {subject}'s {attribute} is {current}.", provenance_id=2, confidence=0.91),
    ]
    return make_row(
        family="unresolved_equal_conflict",
        index=index,
        raw_text=f"Two equally credible records disagree about {subject}'s {attribute}: one gives {old}, the other {current}.",
        query=f"What value can currently be established for {subject}'s {attribute}?",
        fields=fields,
        relations=[edge(0, 1, "conflicts_with")],
        target_summary=f"The current evidence does not establish a single value for {subject}'s {attribute}.",
        target_views=[0.18, 0.20, 0.62],
        reliability=[0.82, 0.82, 0.97],
        cross_view_conflict=True,
    )


def semantic_reliability_wins(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 5)
    fields = [
        field("low_conf_state", f"Low-confidence imported state lists {old} for {subject}'s {attribute}.", provenance_id=3, confidence=0.25),
    ]
    return make_row(
        family="semantic_reliability_wins",
        index=index,
        raw_text=f"A directly verified operator statement says {subject}'s {attribute} is {current}.",
        query=f"Given the reliability of the available views, what {attribute} should be used for {subject}?",
        fields=fields,
        relations=[],
        target_summary=f"{subject}'s {attribute} should be treated as {current}.",
        target_views=[0.78, 0.18, 0.04],
        reliability=[0.995, 0.28, 0.10],
        cross_view_conflict=True,
    )


def structured_reliability_wins(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 6)
    fields = [
        field("high_conf_state", f"Signed typed state lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
    ]
    return make_row(
        family="structured_reliability_wins",
        index=index,
        raw_text=f"An informal unverified note says {subject}'s {attribute} may be {old}.",
        query=f"Using the reliability of the sources, what is {subject}'s {attribute}?",
        fields=fields,
        relations=[],
        target_summary=f"{subject}'s {attribute} is {current}.",
        target_views=[0.15, 0.81, 0.04],
        reliability=[0.30, 0.995, 0.10],
        cross_view_conflict=True,
    )


def missing_structured_semantic_evidence_agree(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 7)
    fields = [
        field("old_evidence", f"Old evidence lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.75),
        field("current_evidence", f"Current evidence lists {current} for {subject}'s {attribute}.", temporal_scope_id=3, confidence=0.98),
    ]
    return make_row(
        family="missing_structured_semantic_evidence_agree",
        index=index,
        raw_text=f"The current {attribute} for {subject} is {current}.",
        query=f"What is {subject}'s current {attribute}?",
        fields=fields,
        relations=[edge(1, 0, "supersedes")],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.50, 0.0, 0.50],
        reliability=[0.97, 0.0, 0.98],
        available=[True, False, True],
    )


def missing_evidence_semantic_structured_agree(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 8)
    fields = [
        field("verified_state", f"Verified typed state lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.98),
    ]
    return make_row(
        family="missing_evidence_semantic_structured_agree",
        index=index,
        raw_text=f"The active configuration states that {subject}'s {attribute} is {current}.",
        query=f"What is {subject}'s active {attribute}?",
        fields=fields,
        relations=[],
        target_summary=f"{subject}'s active {attribute} is {current}.",
        target_views=[0.50, 0.50, 0.0],
        reliability=[0.97, 0.98, 0.0],
        available=[True, True, False],
    )


def missing_semantic_structured_evidence_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 9)
    fields = [
        field("old_state", f"Historical typed state lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.82),
        field("middle_state", f"Intermediate verified state lists {middle} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=2, confidence=0.94),
        field("current_evidence", f"Newest evidence lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="missing_semantic_structured_evidence_chain",
        index=index,
        raw_text="No free-form semantic record is available for this item.",
        query=f"What is the current {attribute} for {subject}?",
        fields=fields,
        relations=[edge(1, 0, "supersedes"), edge(2, 1, "supersedes")],
        target_summary=f"The current {attribute} for {subject} is {current}.",
        target_views=[0.0, 0.42, 0.58],
        reliability=[0.0, 0.94, 0.99],
        available=[False, True, True],
        cross_view_conflict=True,
    )


def consensus_with_low_confidence_distractor(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 10)
    fields = [
        field("current_state", f"Current typed state lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.98),
        field("weak_archive", f"A weak archival fragment mentions {old} for {subject}'s {attribute}.", provenance_id=3, temporal_scope_id=1, confidence=0.18),
    ]
    return make_row(
        family="consensus_with_low_confidence_distractor",
        index=index,
        raw_text=f"The active operator note says {subject}'s {attribute} is {current}.",
        query=f"What is the best-supported {attribute} for {subject}?",
        fields=fields,
        relations=[edge(0, 1, "supersedes", confidence=0.95)],
        target_summary=f"The best-supported {attribute} for {subject} is {current}.",
        target_views=[0.44, 0.44, 0.12],
        reliability=[0.96, 0.98, 0.70],
        cross_view_conflict=True,
    )


def three_way_temporal_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 11)
    fields = [
        field("typed_middle", f"The typed state snapshot lists {middle} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.88),
        field("evidence_old", f"An older evidence record lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.90),
        field("evidence_current", f"A newer evidence record lists {current} for {subject}'s {attribute}.", temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="three_way_temporal_conflict",
        index=index,
        raw_text=f"A stale narrative still reports {old} for {subject}'s {attribute}.",
        query=f"Resolve the temporal conflict: what is {subject}'s current {attribute}?",
        fields=fields,
        relations=[edge(2, 1, "supersedes"), edge(2, 0, "corrects")],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.08, 0.22, 0.70],
        reliability=[0.50, 0.82, 0.995],
        cross_view_conflict=True,
    )


FAMILIES: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
    ("semantic_explicit_over_unknown_state", semantic_explicit_over_unknown_state),
    ("structured_verified_over_hedged_prose", structured_verified_over_hedged_prose),
    ("evidence_multihop_supersession", evidence_multihop_supersession),
    ("evidence_corrects_stale_consensus", evidence_corrects_stale_consensus),
    ("unresolved_equal_conflict", unresolved_equal_conflict),
    ("semantic_reliability_wins", semantic_reliability_wins),
    ("structured_reliability_wins", structured_reliability_wins),
    ("missing_structured_semantic_evidence_agree", missing_structured_semantic_evidence_agree),
    ("missing_evidence_semantic_structured_agree", missing_evidence_semantic_structured_agree),
    ("missing_semantic_structured_evidence_chain", missing_semantic_structured_evidence_chain),
    ("consensus_with_low_confidence_distractor", consensus_with_low_confidence_distractor),
    ("three_way_temporal_conflict", three_way_temporal_conflict),
]


def validate(rows: list[dict[str, Any]]) -> None:
    expected_rows = len(FAMILIES) * ROWS_PER_FAMILY
    if len(rows) != expected_rows:
        raise ValueError(f"challenge row count drift: expected {expected_rows}, observed {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("challenge ids must be unique")
    counts = Counter(row["family"] for row in rows)
    if any(counts[name] != ROWS_PER_FAMILY for name, _builder in FAMILIES):
        raise ValueError("challenge family count drift")
    for row in rows:
        target = row["target_view_distribution"]
        reliability = row["view_reliability"]
        available = row["view_available"]
        if len(target) != 3 or len(reliability) != 3 or len(available) != 3:
            raise ValueError("current frozen challenge requires three public N0 views")
        if abs(sum(target) - 1.0) > 1e-6:
            raise ValueError("target view distribution must sum to one")
        if any(mass > 0.0 and not ok for mass, ok in zip(target, available)):
            raise ValueError("challenge target assigns mass to an unavailable view")
        if row["training_authorized"] is not False:
            raise ValueError("frozen challenge may not authorize training")
        if row["frozen_challenge_training_use_forbidden"] is not True:
            raise ValueError("frozen challenge training-use prohibition missing")
        if row["private_identity_content"] is not False:
            raise ValueError("frozen challenge crossed private identity boundary")
        if row["identity_authority"] is not False:
            raise ValueError("synthetic public challenge may not claim identity authority")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("frozen challenge spec status mismatch")
    if spec.get("training_use") != "FORBIDDEN":
        raise SystemExit("frozen challenge training-use rule missing")

    rows = [builder(index) for _name, builder in FAMILIES for index in range(ROWS_PER_FAMILY)]
    validate(rows)

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    counts = Counter(row["family"] for row in rows)
    manifest = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-manifest.v0.1",
        "status": "FROZEN_UNTOUCHED_NOT_EVALUATED",
        "challenge_sha256": sha256_file(output),
        "spec_sha256": sha256_file(spec_path),
        "rows": len(rows),
        "family_count": len(counts),
        "families": dict(sorted(counts.items())),
        "rows_per_family": ROWS_PER_FAMILY,
        "data_origin": "deterministic_public_independent_fusion_challenge_templates",
        "training_curriculum_rows_reused": False,
        "training_entities_reused": False,
        "training_text_templates_reused": False,
        "generated_text": True,
        "source_authority": "public_synthetic_evaluation_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_training_use_forbidden": True,
        "gradient_performed": False,
        "results_observed_at_compile_time": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
