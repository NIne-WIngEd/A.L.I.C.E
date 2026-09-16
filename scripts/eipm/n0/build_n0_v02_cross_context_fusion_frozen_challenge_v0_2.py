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
    "Cipher Ash", "Cipher Briar", "Cipher Cove", "Cipher Dune",
    "Cipher Elm", "Cipher Fjord", "Cipher Glade", "Cipher Heath",
]
ATTRIBUTES = [
    ("handover profile", "quiet", "guarded", "adaptive"),
    ("review interval", "8 minutes", "19 minutes", "41 minutes"),
    ("relay channel", "5100", "6400", "7700"),
    ("storage posture", "archive", "warm", "active"),
    ("replica policy", "single", "dual", "quorum"),
    ("priority class", "bronze", "silver", "platinum"),
    ("sampling cadence", "12 Hz", "48 Hz", "144 Hz"),
    ("routing posture", "direct", "guarded", "priority"),
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
    attribute, old, middle, current = ATTRIBUTES[(index * 5 + offset) % len(ATTRIBUTES)]
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
        "id": f"N0V02-FUS-CONF2-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": "frozen_confirmatory_v0.2",
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
        "data_origin": "deterministic_public_independent_confirmatory_fusion_templates_v0.2",
        "source_authority": "public_synthetic_evaluation_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_training_use_forbidden": True,
    }


def historical_before_supersession(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 0)
    fields = [
        field("earlier", f"Earlier verified record: {subject} used {old} for {attribute}.", temporal_scope_id=1, confidence=0.98),
        field("current", f"Current verified record: {subject} uses {current} for {attribute}.", temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="historical_before_supersession", index=index,
        raw_text=f"A status summary only mentions the current {attribute} for {subject}: {current}.",
        query=f"What {attribute} did {subject} use immediately before the current value?",
        fields=fields, relations=[edge(1, 0, "supersedes")],
        target_summary=f"Immediately before the current value, {subject} used {old} for {attribute}.",
        target_views=[0.08, 0.27, 0.65], reliability=[0.60, 0.90, 0.99], cross_view_conflict=True,
    )


def current_three_update_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 1)
    fields = [
        field("v1", f"Version one lists {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("v2", f"Version two lists {middle} for {subject}'s {attribute}.", temporal_scope_id=2),
        field("v3", f"Version three lists {current} for {subject}'s {attribute}.", temporal_scope_id=3),
    ]
    return make_row(
        family="current_three_update_chain", index=index,
        raw_text=f"Archived prose for {subject} mentions {old} and {middle}, but does not state the latest {attribute}.",
        query=f"Following the update chain, what is {subject}'s current {attribute}?",
        fields=fields, relations=[edge(1, 0, "supersedes"), edge(2, 1, "supersedes")],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        target_views=[0.04, 0.16, 0.80], reliability=[0.42, 0.82, 0.995], cross_view_conflict=True,
    )


def temporal_successor_current(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 2)
    fields = [
        field("snap_a", f"Snapshot A: {subject} had {old} for {attribute}.", temporal_scope_id=1),
        field("snap_b", f"Snapshot B: {subject} had {middle} for {attribute}.", temporal_scope_id=2),
        field("snap_c", f"Snapshot C: {subject} has {current} for {attribute}.", temporal_scope_id=3),
    ]
    return make_row(
        family="temporal_successor_current", index=index,
        raw_text=f"Three snapshots exist for {subject}'s {attribute}.",
        query=f"Which value is in the latest snapshot for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1, 0, "temporal_successor"), edge(2, 1, "temporal_successor")],
        target_summary=f"The latest snapshot gives {current} for {subject}'s {attribute}.",
        target_views=[0.08, 0.22, 0.70], reliability=[0.55, 0.90, 0.99],
    )


def temporal_successor_previous(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 3)
    fields = [
        field("snap_a", f"Snapshot A: {subject} had {old} for {attribute}.", temporal_scope_id=1),
        field("snap_b", f"Snapshot B: {subject} had {middle} for {attribute}.", temporal_scope_id=2),
        field("snap_c", f"Snapshot C: {subject} has {current} for {attribute}.", temporal_scope_id=3),
    ]
    return make_row(
        family="temporal_successor_previous", index=index,
        raw_text=f"The prose highlights only the latest value {current} for {subject}'s {attribute}.",
        query=f"What value appears in the snapshot immediately before the latest one for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1, 0, "temporal_successor"), edge(2, 1, "temporal_successor")],
        target_summary=f"The snapshot immediately before the latest one gives {middle} for {subject}'s {attribute}.",
        target_views=[0.06, 0.24, 0.70], reliability=[0.50, 0.88, 0.99], cross_view_conflict=True,
    )


def correction_against_stale_majority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 4)
    fields = [
        field("stale_state", f"Cached typed state still lists {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.84),
        field("correction", f"Later signed correction lists {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
    ]
    return make_row(
        family="correction_against_stale_majority", index=index,
        raw_text=f"Two old summaries repeat {old} for {subject}'s {attribute}.",
        query=f"After applying the signed correction, what is {subject}'s current {attribute}?",
        fields=fields, relations=[edge(1, 0, "corrects")],
        target_summary=f"After the correction, {subject}'s current {attribute} is {current}.",
        target_views=[0.10, 0.18, 0.72], reliability=[0.58, 0.70, 0.995], cross_view_conflict=True,
    )


def support_aggregation(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 5)
    fields = [
        field("claim", f"Claim: {subject}'s {attribute} is {current}.", confidence=0.72),
        field("source_a", f"Independent signed source A verifies {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.96),
        field("source_b", f"Independent signed source B verifies {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.97),
    ]
    return make_row(
        family="support_aggregation", index=index,
        raw_text=f"A prose note says the {attribute} for {subject} was recently reviewed.",
        query=f"What value is jointly supported for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1, 0, "supports"), edge(2, 0, "supports")],
        target_summary=f"The jointly supported value for {subject}'s {attribute} is {current}.",
        target_views=[0.06, 0.32, 0.62], reliability=[0.48, 0.91, 0.98],
    )


def derived_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 6)
    fields = [
        field("source", f"Primary source records {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.99),
        field("derived", f"Derived record reports {current} for {subject}'s {attribute}.", confidence=0.86),
        field("legacy", f"Legacy prose mentions {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.45),
    ]
    return make_row(
        family="derived_chain", index=index,
        raw_text=f"An old narrative still contains {old} for {subject}'s {attribute}.",
        query=f"Tracing derivation to the primary source, what value is supported for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1, 0, "derived_from"), edge(0, 2, "corrects")],
        target_summary=f"The primary-source-backed value for {subject}'s {attribute} is {current}.",
        target_views=[0.05, 0.25, 0.70], reliability=[0.40, 0.83, 0.99], cross_view_conflict=True,
    )


def causal_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 7)
    fields = [
        field("cause", f"Changing {subject}'s control from {old} to {middle} triggered a policy update.", confidence=0.96),
        field("effect", f"The policy update changed {subject}'s {attribute} to {current}.", confidence=0.98),
        field("result", f"Current state records {current} for {subject}'s {attribute}.", temporal_scope_id=3, confidence=0.99),
    ]
    return make_row(
        family="causal_chain", index=index,
        raw_text=f"The prose only says that a control change affected {subject}.",
        query=f"According to the causal evidence chain, what {attribute} resulted for {subject}?",
        fields=fields, relations=[edge(0, 1, "causes"), edge(1, 2, "causes")],
        target_summary=f"The causal chain results in {current} for {subject}'s {attribute}.",
        target_views=[0.06, 0.24, 0.70], reliability=[0.45, 0.88, 0.99],
    )


def unresolved_three_source_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 0)
    fields = [
        field("claim_a", f"Source A says {old} for {subject}'s {attribute}.", confidence=0.88),
        field("claim_b", f"Source B says {middle} for {subject}'s {attribute}.", confidence=0.88),
        field("claim_c", f"Source C says {current} for {subject}'s {attribute}.", confidence=0.88),
    ]
    return make_row(
        family="unresolved_three_source_conflict", index=index,
        raw_text=f"Three equally credible reports disagree on {subject}'s {attribute}.",
        query=f"Can one current value be established for {subject}'s {attribute}?",
        fields=fields, relations=[edge(0, 1, "conflicts_with"), edge(1, 2, "conflicts_with")],
        target_summary=f"No single current value can be established for {subject}'s {attribute}.",
        target_views=[0.16, 0.20, 0.64], reliability=[0.78, 0.80, 0.97], cross_view_conflict=True,
    )


def semantic_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 1)
    fields = [field("stale_import", f"An imported low-confidence state lists {old} for {subject}'s {attribute}.", provenance_id=3, confidence=0.22)]
    return make_row(
        family="semantic_authority", index=index,
        raw_text=f"A directly verified operator statement says {subject}'s {attribute} is {current}.",
        query=f"Which value should be used for {subject}'s {attribute}?",
        fields=fields, relations=[],
        target_summary=f"{subject}'s {attribute} should be treated as {current}.",
        target_views=[0.82, 0.14, 0.04], reliability=[0.995, 0.25, 0.10], cross_view_conflict=True,
    )


def structured_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields = [field("signed_state", f"Signed current state: {subject}'s {attribute} is {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.995)]
    return make_row(
        family="structured_authority", index=index,
        raw_text=f"An ambiguous informal note says {subject}'s {attribute} may still be {old}.",
        query=f"What is the signed current {attribute} for {subject}?",
        fields=fields, relations=[],
        target_summary=f"The signed current {attribute} for {subject} is {current}.",
        target_views=[0.08, 0.87, 0.05], reliability=[0.34, 0.995, 0.20], cross_view_conflict=True,
    )


def reliability_reversal(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 3)
    fields = [field("structured_note", f"Unsigned state claims {current} for {subject}'s {attribute}.", confidence=0.35)]
    return make_row(
        family="reliability_reversal", index=index,
        raw_text=f"A signed operator statement gives {old} for {subject}'s {attribute}, while an unsigned state gives {current}.",
        query=f"Using source reliability, which {attribute} should be trusted for {subject}?",
        fields=fields, relations=[],
        target_summary=f"The signed value {old} should be trusted for {subject}'s {attribute}.",
        target_views=[0.76, 0.20, 0.04], reliability=[0.99, 0.32, 0.10], cross_view_conflict=True,
    )


def missing_structured_evidence_resolves(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 4)
    fields = [
        field("old", f"Evidence record A lists {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("new", f"Evidence record B lists {current} for {subject}'s {attribute}.", temporal_scope_id=3),
    ]
    return make_row(
        family="missing_structured_evidence_resolves", index=index,
        raw_text=f"A stale prose line says {subject}'s {attribute} is {old}.",
        query=f"With structured state unavailable, what is the relation-supported current {attribute} for {subject}?",
        fields=fields, relations=[edge(1, 0, "supersedes")],
        target_summary=f"The relation-supported current {attribute} for {subject} is {current}.",
        target_views=[0.18, 0.0, 0.82], reliability=[0.48, 0.0, 0.995], available=[True, False, True], cross_view_conflict=True,
    )


def missing_evidence_semantic_wins(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 5)
    fields = [field("weak_state", f"Low-confidence state lists {old} for {subject}'s {attribute}.", confidence=0.30)]
    return make_row(
        family="missing_evidence_semantic_wins", index=index,
        raw_text=f"A signed current statement says {subject}'s {attribute} is {current}.",
        query=f"With relation evidence unavailable, what value should be used for {subject}'s {attribute}?",
        fields=fields, relations=[],
        target_summary=f"{subject}'s {attribute} should be treated as {current}.",
        target_views=[0.82, 0.18, 0.0], reliability=[0.995, 0.32, 0.0], available=[True, True, False], cross_view_conflict=True,
    )


def missing_semantic_historical(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 6)
    fields = [
        field("old", f"Old state: {old} for {subject}'s {attribute}.", temporal_scope_id=1),
        field("middle", f"Middle state: {middle} for {subject}'s {attribute}.", temporal_scope_id=2),
        field("current", f"Current state: {current} for {subject}'s {attribute}.", temporal_scope_id=3),
    ]
    return make_row(
        family="missing_semantic_historical", index=index,
        raw_text="No raw prose is available for this item.",
        query=f"What was {subject}'s {attribute} one revision before the current state?",
        fields=fields, relations=[edge(1, 0, "supersedes"), edge(2, 1, "supersedes")],
        target_summary=f"One revision before the current state, {subject}'s {attribute} was {middle}.",
        target_views=[0.0, 0.28, 0.72], reliability=[0.0, 0.90, 0.995], available=[False, True, True],
    )


def lexical_distractor_low_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 7)
    fields = [
        field("verified", f"Verified state records {current} for {subject}'s {attribute}.", provenance_id=2, temporal_scope_id=3, confidence=0.995),
        field("distractor", f"A copied low-authority note repeatedly says {old}, {old}, {old} for {subject}'s {attribute}.", provenance_id=3, confidence=0.15),
    ]
    return make_row(
        family="lexical_distractor_low_authority", index=index,
        raw_text=f"An old memo repeatedly emphasizes {old} for {subject}'s {attribute} but is marked obsolete.",
        query=f"Ignoring obsolete low-authority repetition, what is the verified {attribute} for {subject}?",
        fields=fields, relations=[edge(0, 1, "corrects")],
        target_summary=f"The verified {attribute} for {subject} is {current}.",
        target_views=[0.08, 0.48, 0.44], reliability=[0.28, 0.98, 0.96], cross_view_conflict=True,
    )


def consensus_one_stale(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 0)
    fields = [
        field("current_state", f"Current typed state gives {current} for {subject}'s {attribute}.", temporal_scope_id=3, confidence=0.98),
        field("stale_evidence", f"Archived evidence gives {old} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.85),
        field("current_evidence", f"Later evidence gives {current} for {subject}'s {attribute}.", temporal_scope_id=3, confidence=0.98),
    ]
    return make_row(
        family="consensus_one_stale", index=index,
        raw_text=f"The current prose says {subject}'s {attribute} is {current}.",
        query=f"What current value is supported across the available views for {subject}'s {attribute}?",
        fields=fields, relations=[edge(2, 1, "supersedes")],
        target_summary=f"The current supported value for {subject}'s {attribute} is {current}.",
        target_views=[0.34, 0.33, 0.33], reliability=[0.97, 0.97, 0.98], cross_view_conflict=True,
    )


FAMILIES: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
    ("historical_before_supersession", historical_before_supersession),
    ("current_three_update_chain", current_three_update_chain),
    ("temporal_successor_current", temporal_successor_current),
    ("temporal_successor_previous", temporal_successor_previous),
    ("correction_against_stale_majority", correction_against_stale_majority),
    ("support_aggregation", support_aggregation),
    ("derived_chain", derived_chain),
    ("causal_chain", causal_chain),
    ("unresolved_three_source_conflict", unresolved_three_source_conflict),
    ("semantic_authority", semantic_authority),
    ("structured_authority", structured_authority),
    ("reliability_reversal", reliability_reversal),
    ("missing_structured_evidence_resolves", missing_structured_evidence_resolves),
    ("missing_evidence_semantic_wins", missing_evidence_semantic_wins),
    ("missing_semantic_historical", missing_semantic_historical),
    ("lexical_distractor_low_authority", lexical_distractor_low_authority),
    ("consensus_one_stale", consensus_one_stale),
]


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len(FAMILIES) * ROWS_PER_FAMILY:
        raise ValueError("confirmatory challenge row count drift")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("confirmatory challenge ids must be unique")
    counts = Counter(row["family"] for row in rows)
    if any(counts[name] != ROWS_PER_FAMILY for name, _ in FAMILIES):
        raise ValueError("confirmatory challenge family coverage drift")
    for row in rows:
        if row["split"] != "frozen_confirmatory_v0.2":
            raise ValueError("confirmatory split drift")
        if row.get("training_authorized") is not False:
            raise ValueError("confirmatory challenge may not authorize training")
        if row.get("frozen_challenge_training_use_forbidden") is not True:
            raise ValueError("confirmatory challenge training-use prohibition missing")
        if row.get("private_identity_content") is not False or row.get("private_identity_gradient") is not False:
            raise ValueError("confirmatory challenge crossed private boundary")
        if row.get("identity_authority") is not False:
            raise ValueError("synthetic confirmatory challenge cannot be identity authority")
        if row.get("data_origin") != "deterministic_public_independent_confirmatory_fusion_templates_v0.2":
            raise ValueError("confirmatory challenge origin drift")
        target = row["target_view_distribution"]
        available = row["view_available"]
        if len(target) != 3 or len(available) != 3 or len(row["view_reliability"]) != 3:
            raise ValueError("current confirmatory challenge must use the three instantiated N0 views")
        if abs(sum(target) - 1.0) > 1e-6:
            raise ValueError("target view distribution must sum to one")
        if any(mass > 0.0 and not ok for mass, ok in zip(target, available)):
            raise ValueError("target assigns mass to unavailable view")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CONFIRMATORY_CHALLENGE_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("confirmatory challenge spec status drift")

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
        "schema": "alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-manifest.v0.2",
        "status": "FROZEN_UNTOUCHED_CONFIRMATORY_NOT_EVALUATED",
        "challenge_sha256": sha256_file(output),
        "spec_sha256": sha256_file(spec_path),
        "rows": len(rows),
        "family_count": len(counts),
        "rows_per_family": ROWS_PER_FAMILY,
        "families": dict(sorted(counts.items())),
        "candidate_preselected_before_challenge": True,
        "candidate_repair_step": int(spec["candidate_selection"]["repair_step"]),
        "candidate_fusion_sha256": spec["candidate_selection"]["fusion_sha256"],
        "data_origin": "deterministic_public_independent_confirmatory_fusion_templates_v0.2",
        "source_authority": "public_synthetic_evaluation_only",
        "identity_authority": False,
        "generated_text": True,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_training_use_forbidden": True,
        "results_observed_at_compile_time": False,
        "original_training_rows_reused": False,
        "repair_training_rows_reused": False,
        "retired_challenge_rows_reused": False,
        "training_entities_reused": False,
        "training_text_templates_reused": False,
        "retired_challenge_entities_reused": False,
        "retired_challenge_text_templates_reused": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
