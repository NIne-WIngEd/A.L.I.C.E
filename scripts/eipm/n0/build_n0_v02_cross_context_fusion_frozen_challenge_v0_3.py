#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from alice_personality.n0.cross_context_fusion_routing_constraints import validate_routing_constraints
from alice_personality.n0.evidence_graph import relation_type_id


SUBJECTS = [
    "Atlas Kestrel", "Beacon Larch", "Crux Mica", "Delta Nori",
    "Ember Quartz", "Fable Rowan", "Glyph Sable", "Helix Tern",
]
ATTRIBUTES = [
    ("handoff mode", "quiet", "guarded", "adaptive"),
    ("review cycle", "11 minutes", "23 minutes", "47 minutes"),
    ("relay band", "alpha", "gamma", "omega"),
    ("storage class", "cold", "warm", "live"),
    ("replication rule", "single", "paired", "quorum"),
    ("priority tier", "copper", "jade", "violet"),
    ("sampling rate", "14 Hz", "52 Hz", "156 Hz"),
    ("dispatch mode", "direct", "weighted", "priority"),
]
ROWS_PER_FAMILY = 8
VIEW_COUNT = 3


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
    subject = SUBJECTS[(index * 3 + offset) % len(SUBJECTS)]
    attribute, old, middle, current = ATTRIBUTES[(index * 5 + offset) % len(ATTRIBUTES)]
    return subject, attribute, old, middle, current


def compatibility_distribution(
    constraints: dict[str, Any],
    reliability: list[float],
    available: list[bool],
) -> list[float]:
    """Legacy parent-cache field only; never used for v0.3 ratification."""
    allowed = constraints.get("allowed_top_views")
    values = [0.0] * VIEW_COUNT
    active = [index for index, flag in enumerate(available) if flag]
    if allowed:
        allowed_active = [int(index) for index in allowed if available[int(index)]]
        if len(allowed_active) == 1:
            top = allowed_active[0]
            values[top] = 0.80
            others = [index for index in active if index != top]
            for index in others:
                values[index] = 0.20 / max(len(others), 1)
        else:
            for index in allowed_active:
                values[index] = 1.0 / len(allowed_active)
    else:
        total = sum(reliability[index] for index in active)
        for index in active:
            values[index] = reliability[index] / total if total > 0 else 1.0 / len(active)
    total = sum(values)
    if total <= 0:
        raise ValueError("compatibility distribution has no active mass")
    return [value / total for value in values]


def make_row(
    *,
    family: str,
    index: int,
    raw_text: str,
    query: str,
    fields: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    target_summary: str,
    constraints: dict[str, Any],
    reliability: list[float],
    available: list[bool] | None = None,
    cross_view_conflict: bool = False,
) -> dict[str, Any]:
    if available is None:
        available = [True, True, True]
    validate_routing_constraints(constraints, view_count=VIEW_COUNT)
    if len(reliability) != VIEW_COUNT or len(available) != VIEW_COUNT:
        raise ValueError("fusion view metadata must have three entries")
    return {
        "id": f"N0V02-FUS-CONF3-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": "frozen_constraint_confirmatory_v0.3",
        "raw_text": raw_text,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_summary_text": target_summary,
        "routing_constraints": constraints,
        "target_view_distribution": compatibility_distribution(constraints, reliability, available),
        "target_view_distribution_role": "parent_cache_compatibility_only_not_used_for_ratification",
        "view_reliability": reliability,
        "view_available": available,
        "cross_view_conflict": cross_view_conflict,
        "generated_text": True,
        "data_origin": "deterministic_public_independent_constraint_confirmatory_templates_v0.3",
        "source_authority": "public_synthetic_evaluation_only",
        "identity_authority": False,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "frozen_challenge_training_use_forbidden": True,
        "exact_target_routing_distribution_used_for_ratification": False,
    }


def semantic_signed_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 0)
    return make_row(
        family="semantic_signed_authority", index=index,
        raw_text=f"A signed operator statement confirms {subject}'s {attribute} is {current}.",
        query=f"What verified {attribute} should be used for {subject}?",
        fields=[field("stale", f"An old imported state lists {old} for {subject}'s {attribute}.", provenance_id=3, confidence=0.28)],
        relations=[], target_summary=f"{subject}'s verified {attribute} is {current}.",
        constraints={"allowed_top_views":[0],"min_view_mass":{"0":0.45},"pairwise_margins":[{"higher":0,"lower":1,"margin":0.10},{"higher":0,"lower":2,"margin":0.10}]},
        reliability=[0.995,0.32,0.18], cross_view_conflict=True,
    )


def structured_verified_authority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 1)
    return make_row(
        family="structured_verified_authority", index=index,
        raw_text=f"An unsigned note speculates {subject}'s {attribute} may still be {old}.",
        query=f"What signed structured value should be used for {subject}'s {attribute}?",
        fields=[field("signed", f"Signed structured record: {subject}'s {attribute} is {current}.", provenance_id=2, temporal_scope_id=3, confidence=0.995)],
        relations=[], target_summary=f"The signed structured value is {current}.",
        constraints={"allowed_top_views":[1],"min_view_mass":{"1":0.45},"pairwise_margins":[{"higher":1,"lower":0,"margin":0.10}]},
        reliability=[0.35,0.995,0.30], cross_view_conflict=True,
    )


def current_supersession(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields=[field("old",f"Older record: {subject}'s {attribute} was {old}.",temporal_scope_id=1,confidence=0.92),field("new",f"Replacement record: {subject}'s {attribute} is {current}.",provenance_id=2,temporal_scope_id=3,confidence=0.995)]
    return make_row(
        family="current_supersession", index=index, raw_text=f"Legacy prose still repeats {old} for {subject}.",
        query=f"After supersession, what is {subject}'s current {attribute}?", fields=fields, relations=[edge(1,0,"supersedes")],
        target_summary=f"{subject}'s current {attribute} is {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.45},"max_view_mass":{"0":0.30},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.15}]},
        reliability=[0.48,0.88,0.995], cross_view_conflict=True,
    )


def historical_superseded_state(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 3)
    fields=[field("old",f"Earlier verified record: {subject}'s {attribute} was {old}.",temporal_scope_id=1,confidence=0.98),field("new",f"Later record: {subject}'s {attribute} became {current}.",temporal_scope_id=3,confidence=0.99)]
    return make_row(
        family="historical_superseded_state", index=index, raw_text=f"Current prose only states {current} for {subject}'s {attribute}.",
        query=f"What was {subject}'s {attribute} before the later record replaced it?", fields=fields, relations=[edge(1,0,"supersedes")],
        target_summary=f"Before replacement, {subject}'s {attribute} was {old}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.40},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.08}]},
        reliability=[0.55,0.90,0.99], cross_view_conflict=True,
    )


def temporal_current(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 4)
    fields=[field("a",f"Snapshot A: {old}.",temporal_scope_id=1),field("b",f"Snapshot B: {middle}.",temporal_scope_id=2),field("c",f"Snapshot C: {current}.",temporal_scope_id=3)]
    return make_row(
        family="temporal_current", index=index, raw_text=f"Three snapshots exist for {subject}'s {attribute}.", query=f"What value is in the latest snapshot for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1,0,"temporal_successor"),edge(2,1,"temporal_successor")], target_summary=f"The latest value is {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.40},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.08}]}, reliability=[0.50,0.88,0.99],
    )


def temporal_previous(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 5)
    fields=[field("a",f"Snapshot A: {old}.",temporal_scope_id=1),field("b",f"Snapshot B: {middle}.",temporal_scope_id=2),field("c",f"Snapshot C: {current}.",temporal_scope_id=3)]
    return make_row(
        family="temporal_previous", index=index, raw_text=f"The latest prose highlights {current} for {subject}'s {attribute}.", query=f"What was {subject}'s {attribute} immediately before the latest snapshot?",
        fields=fields, relations=[edge(1,0,"temporal_successor"),edge(2,1,"temporal_successor")], target_summary=f"Immediately before the latest snapshot it was {middle}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.40},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.08}]}, reliability=[0.48,0.88,0.99], cross_view_conflict=True,
    )


def correction_stale_majority(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 6)
    fields=[field("stale",f"Cached state lists {old} for {subject}'s {attribute}.",confidence=0.80),field("fix",f"Signed correction lists {current} for {subject}'s {attribute}.",provenance_id=2,temporal_scope_id=3,confidence=0.995)]
    return make_row(
        family="correction_stale_majority", index=index, raw_text=f"Several old prose summaries repeat {old} for {subject}.", query=f"After the signed correction, what is the current {attribute}?",
        fields=fields, relations=[edge(1,0,"corrects")], target_summary=f"The corrected current {attribute} is {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.45},"max_view_mass":{"0":0.30},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.15}]}, reliability=[0.52,0.72,0.995], cross_view_conflict=True,
    )


def support_aggregation(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 7)
    fields=[field("claim",f"Claim: {subject}'s {attribute} is {current}.",confidence=0.76),field("s1",f"Signed source one supports {current}.",provenance_id=2,confidence=0.97),field("s2",f"Signed source two supports {current}.",provenance_id=2,confidence=0.98)]
    return make_row(
        family="support_aggregation", index=index, raw_text=f"A prose note says {subject}'s {attribute} was verified.", query=f"What value is jointly supported for {subject}'s {attribute}?",
        fields=fields, relations=[edge(1,0,"supports"),edge(2,0,"supports")], target_summary=f"The jointly supported value is {current}.",
        constraints={"allowed_top_views":[1,2],"min_set_mass":[{"views":[1,2],"min":0.65}],"max_view_mass":{"0":0.35}}, reliability=[0.60,0.93,0.99],
    )


def derived_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 0)
    fields=[field("primary",f"Primary signed source records {current} for {subject}'s {attribute}.",provenance_id=2,confidence=0.99),field("derived",f"Derived state reports {current}.",confidence=0.88),field("old",f"Old prose reports {old}.",confidence=0.40)]
    return make_row(
        family="derived_chain", index=index, raw_text=f"A legacy paragraph still says {old} for {subject}.", query=f"Tracing derivation to source authority, what {attribute} is supported?",
        fields=fields, relations=[edge(1,0,"derived_from"),edge(0,2,"corrects")], target_summary=f"The source-backed value is {current}.",
        constraints={"allowed_top_views":[1,2],"min_set_mass":[{"views":[1,2],"min":0.70}],"max_view_mass":{"0":0.25}}, reliability=[0.40,0.88,0.99], cross_view_conflict=True,
    )


def causal_chain(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 1)
    fields=[field("cause",f"Moving control from {old} to {middle} triggered a policy change.",confidence=0.96),field("effect",f"The policy change set {subject}'s {attribute} to {current}.",confidence=0.98),field("state",f"Current state confirms {current}.",confidence=0.99)]
    return make_row(
        family="causal_chain", index=index, raw_text=f"A prose note only says a control change affected {subject}.", query=f"What {attribute} resulted from the causal chain?",
        fields=fields, relations=[edge(0,1,"causes"),edge(1,2,"causes")], target_summary=f"The causal chain resulted in {current}.",
        constraints={"allowed_top_views":[1,2],"min_set_mass":[{"views":[1,2],"min":0.65}],"max_view_mass":{"0":0.35}}, reliability=[0.50,0.90,0.99],
    )


def missing_structured_evidence(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields=[field("old",f"Historical record says {old}.",temporal_scope_id=1),field("new",f"Evidence correction says {current}.",provenance_id=2,temporal_scope_id=3,confidence=0.995)]
    return make_row(
        family="missing_structured_evidence", index=index, raw_text=f"An old note says {old} for {subject}'s {attribute}.", query=f"With structured state unavailable, what does correction evidence establish?",
        fields=fields, relations=[edge(1,0,"corrects")], target_summary=f"Correction evidence establishes {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.50},"max_view_mass":{"0":0.35}}, reliability=[0.50,0.0,0.995], available=[True,False,True], cross_view_conflict=True,
    )


def missing_evidence_semantic(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 3)
    fields=[field("state",f"Signed structured state records {current} for {subject}'s {attribute}.",provenance_id=2,confidence=0.98)]
    return make_row(
        family="missing_evidence_semantic", index=index, raw_text=f"A signed direct statement also gives {current} for {subject}'s {attribute}.", query=f"What is {subject}'s current {attribute}?",
        fields=fields, relations=[], target_summary=f"{subject}'s current {attribute} is {current}.",
        constraints={"allowed_top_views":[0,1],"min_set_mass":[{"views":[0,1],"min":0.99}]}, reliability=[0.98,0.98,0.0], available=[True,True,False],
    )


def missing_semantic_historical(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 4)
    fields=[field("old",f"Earlier state records {old} for {subject}'s {attribute}.",temporal_scope_id=1,confidence=0.98),field("new",f"Later state records {current}.",temporal_scope_id=3,confidence=0.99)]
    return make_row(
        family="missing_semantic_historical", index=index, raw_text="No raw prose view is available.", query=f"What was {subject}'s {attribute} before the later supersession?",
        fields=fields, relations=[edge(1,0,"supersedes")], target_summary=f"Before supersession it was {old}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.45}}, reliability=[0.0,0.90,0.99], available=[False,True,True], cross_view_conflict=True,
    )


def reliability_flip_semantic(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 5)
    fields=[field("weak",f"Low-confidence imported state says {old}.",provenance_id=3,confidence=0.20)]
    return make_row(
        family="reliability_flip_semantic", index=index, raw_text=f"High-authority signed prose says {subject}'s {attribute} is {current}.", query=f"Which value should win under source reliability?",
        fields=fields, relations=[], target_summary=f"The high-authority value is {current}.",
        constraints={"allowed_top_views":[0],"min_view_mass":{"0":0.50},"pairwise_margins":[{"higher":0,"lower":1,"margin":0.15},{"higher":0,"lower":2,"margin":0.15}]}, reliability=[0.995,0.25,0.15], cross_view_conflict=True,
    )


def reliability_flip_evidence(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 6)
    fields=[field("stale",f"Typed state says {old}.",confidence=0.70),field("signed",f"Signed evidence correction says {current}.",provenance_id=2,confidence=0.995)]
    return make_row(
        family="reliability_flip_evidence", index=index, raw_text=f"An informal sentence repeats {old} for {subject}.", query=f"Which value should win after the signed correction?",
        fields=fields, relations=[edge(1,0,"corrects")], target_summary=f"The signed correction establishes {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.50},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.15},{"higher":2,"lower":1,"margin":0.10}]}, reliability=[0.35,0.72,0.995], cross_view_conflict=True,
    )


def lexical_stale_repetition(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 7)
    fields=[field("current",f"Verified structured value is {current}.",provenance_id=2,confidence=0.98),field("proof",f"Evidence confirms {current} for {subject}.",provenance_id=2,confidence=0.99)]
    raw=(f"{old} {old} {old}. A verbose obsolete memo repeatedly calls {old} the {attribute} for {subject}; it is marked stale.")
    return make_row(
        family="lexical_stale_repetition", index=index, raw_text=raw, query=f"Ignoring stale repetition, what current {attribute} is supported?",
        fields=fields, relations=[edge(1,0,"supports")], target_summary=f"The supported current value is {current}.",
        constraints={"allowed_top_views":[1,2],"min_set_mass":[{"views":[1,2],"min":0.70}],"max_view_mass":{"0":0.30}}, reliability=[0.25,0.98,0.99], cross_view_conflict=True,
    )


def consensus_all_aligned(index: int) -> dict[str, Any]:
    subject, attribute, _old, _middle, current = parts(index, 0)
    fields=[field("state",f"Structured state says {current}.",confidence=0.98),field("proof",f"Evidence source says {current}.",confidence=0.98)]
    return make_row(
        family="consensus_all_aligned", index=index, raw_text=f"Direct prose says {subject}'s {attribute} is {current}.", query=f"What is {subject}'s {attribute}?",
        fields=fields, relations=[edge(1,0,"supports")], target_summary=f"All available views support {current}.",
        constraints={"min_set_mass":[{"views":[0,1,2],"min":0.999}]}, reliability=[0.98,0.98,0.98],
    )


def unresolved_conflict(index: int) -> dict[str, Any]:
    subject, attribute, old, middle, current = parts(index, 1)
    fields=[field("a",f"Source A says {old}.",confidence=0.90),field("b",f"Source B says {middle}.",confidence=0.90),field("c",f"Source C says {current}.",confidence=0.90)]
    return make_row(
        family="unresolved_conflict", index=index, raw_text=f"Three equally credible reports disagree on {subject}'s {attribute}.", query=f"Can one current value be established?",
        fields=fields, relations=[edge(0,1,"conflicts_with"),edge(1,2,"conflicts_with")], target_summary=f"No single current {attribute} can be established for {subject}.",
        constraints={"min_view_mass":{"2":0.30},"max_view_mass":{"0":0.50}}, reliability=[0.80,0.85,0.97], cross_view_conflict=True,
    )


def paired_current_query(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields=[field("prior",f"Prior verified state: {old}.",temporal_scope_id=1,confidence=0.98),field("current",f"Current verified state: {current}.",temporal_scope_id=3,confidence=0.99)]
    return make_row(
        family="paired_current_query", index=index, raw_text=f"Both historical and current records exist for {subject}'s {attribute}.", query=f"What is {subject}'s {attribute} now?",
        fields=fields, relations=[edge(1,0,"supersedes")], target_summary=f"Now it is {current}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.40},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.08}]}, reliability=[0.62,0.90,0.99], cross_view_conflict=True,
    )


def paired_historical_query(index: int) -> dict[str, Any]:
    subject, attribute, old, _middle, current = parts(index, 2)
    fields=[field("prior",f"Prior verified state: {old}.",temporal_scope_id=1,confidence=0.98),field("current",f"Current verified state: {current}.",temporal_scope_id=3,confidence=0.99)]
    return make_row(
        family="paired_historical_query", index=index, raw_text=f"Both historical and current records exist for {subject}'s {attribute}.", query=f"What was {subject}'s {attribute} before the current record?",
        fields=fields, relations=[edge(1,0,"supersedes")], target_summary=f"Before the current record it was {old}.",
        constraints={"allowed_top_views":[2],"min_view_mass":{"2":0.40},"pairwise_margins":[{"higher":2,"lower":0,"margin":0.08}]}, reliability=[0.62,0.90,0.99], cross_view_conflict=True,
    )


FAMILIES: list[tuple[str, Callable[[int], dict[str, Any]]]] = [
    ("semantic_signed_authority", semantic_signed_authority),
    ("structured_verified_authority", structured_verified_authority),
    ("current_supersession", current_supersession),
    ("historical_superseded_state", historical_superseded_state),
    ("temporal_current", temporal_current),
    ("temporal_previous", temporal_previous),
    ("correction_stale_majority", correction_stale_majority),
    ("support_aggregation", support_aggregation),
    ("derived_chain", derived_chain),
    ("causal_chain", causal_chain),
    ("missing_structured_evidence", missing_structured_evidence),
    ("missing_evidence_semantic", missing_evidence_semantic),
    ("missing_semantic_historical", missing_semantic_historical),
    ("reliability_flip_semantic", reliability_flip_semantic),
    ("reliability_flip_evidence", reliability_flip_evidence),
    ("lexical_stale_repetition", lexical_stale_repetition),
    ("consensus_all_aligned", consensus_all_aligned),
    ("unresolved_conflict", unresolved_conflict),
    ("paired_current_query", paired_current_query),
    ("paired_historical_query", paired_historical_query),
]


def validate_rows(rows: list[dict[str, Any]]) -> None:
    ids=[str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("v0.3 challenge ids must be unique")
    if len(rows) != len(FAMILIES) * ROWS_PER_FAMILY:
        raise ValueError("v0.3 challenge row count drift")
    counts=Counter(str(row["family"]) for row in rows)
    if any(counts[name] != ROWS_PER_FAMILY for name,_ in FAMILIES):
        raise ValueError("v0.3 family coverage drift")
    for row in rows:
        if row.get("training_authorized") is not False:
            raise ValueError("v0.3 row unexpectedly authorizes training")
        if row.get("exact_target_routing_distribution_used_for_ratification") is not False:
            raise ValueError("v0.3 exact routing distribution must not be a ratification target")
        if row.get("target_view_distribution_role") != "parent_cache_compatibility_only_not_used_for_ratification":
            raise ValueError("v0.3 compatibility routing role drift")
        validate_routing_constraints(row["routing_constraints"], view_count=VIEW_COUNT)
        for relation in row["relations"]:
            if relation_type_id(relation["relation"]) != int(relation["relation_type_id"]):
                raise ValueError("relation id drift")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    args=parser.parse_args()
    output=Path(args.output).resolve(); manifest_path=Path(args.manifest).resolve(); spec_path=Path(args.spec).resolve()
    spec=json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CONSTRAINT_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("v0.3 challenge spec status mismatch")
    rows=[builder(index) for _name,builder in FAMILIES for index in range(ROWS_PER_FAMILY)]
    validate_rows(rows)
    if len(rows) != int(spec["challenge"]["rows"]) or len(FAMILIES) != int(spec["challenge"]["families"]):
        raise SystemExit("v0.3 spec coverage mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w",encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    manifest={
        "schema":"alice.eipm.n0.v02-cross-context-fusion-frozen-challenge-manifest.v0.3",
        "status":"FROZEN_UNTOUCHED_CONSTRAINT_CONFIRMATORY_NOT_EVALUATED",
        "rows":len(rows),
        "family_count":len(FAMILIES),
        "rows_per_family":ROWS_PER_FAMILY,
        "families":dict(sorted(Counter(row["family"] for row in rows).items())),
        "challenge_sha256":sha256_file(output),
        "spec_sha256":sha256_file(spec_path),
        "candidate_preselected_before_challenge":True,
        "candidate_repair_step":int(spec["candidate_selection"]["repair_step"]),
        "candidate_fusion_sha256":str(spec["candidate_selection"]["fusion_sha256"]),
        "routing_validation":"constraint_based_not_exact_distribution_similarity",
        "exact_target_routing_distribution_used_for_ratification":False,
        "generated_text":True,
        "data_origin":"deterministic_public_independent_constraint_confirmatory_templates_v0.3",
        "source_authority":"public_synthetic_evaluation_only",
        "training_authorized":False,
        "frozen_challenge_training_use_forbidden":True,
        "results_observed_at_compile_time":False,
        "original_training_rows_reused":False,
        "repair_training_rows_reused":False,
        "retired_challenge_v0_1_rows_reused":False,
        "retired_challenge_v0_2_rows_reused":False,
        "training_entities_reused":False,
        "retired_challenge_entities_reused":False,
        "private_identity_content":False,
        "private_identity_gradient":False,
    }
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
