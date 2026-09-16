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
    "Constellation Arbor", "Constellation Brine", "Constellation Cinder", "Constellation Delta",
    "Constellation Ember", "Constellation Fallow", "Constellation Grove", "Constellation Harbor",
    "Constellation Indigo", "Constellation Jasper", "Constellation Kelp", "Constellation Lattice",
    "Constellation Mica", "Constellation North", "Constellation Oriel", "Constellation Prairie",
]
ATTRIBUTES = [
    ("control channel", "gamma", "lambda"),
    ("service tier", "bronze", "platinum"),
    ("sync interval", "11 seconds", "37 seconds"),
    ("routing policy", "static", "adaptive"),
    ("worker count", "3", "7"),
    ("archive mode", "sealed", "rolling"),
    ("inspection cadence", "9 minutes", "21 minutes"),
    ("uplink band", "low", "wide"),
    ("replication mode", "single", "mirrored"),
    ("sampling profile", "coarse", "fine"),
]
FAMILIES = [
    "novel_semantic_authority",
    "novel_structured_authority",
    "supersession_current",
    "supersession_historical",
    "correction_chain_current",
    "correction_chain_historical",
    "temporal_successor_current",
    "temporal_successor_previous",
    "derived_support_chain",
    "causal_chain",
    "missing_semantic_relational",
    "missing_structured_evidence",
    "missing_evidence_semantic_structured",
    "unresolved_balanced_conflict",
    "noisy_consensus",
    "semantic_reliability_reversal",
    "evidence_reliability_reversal",
    "lexical_high_overlap_distractor",
    "paired_current_state",
    "paired_historical_state",
]
ROWS_PER_FAMILY = 8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    attribute, previous, current = ATTRIBUTES[(index + 3 * offset) % len(ATTRIBUTES)]
    return subject, attribute, previous, current


def compat_distribution(available: list[bool]) -> list[float]:
    count = sum(1 for value in available if value)
    return [1.0 / count if value else 0.0 for value in available]


def row(
    *,
    family: str,
    index: int,
    raw_text: str,
    query: str,
    fields: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    target_summary: str,
    reliability: list[float],
    available: list[bool] | None = None,
    conflict: bool = False,
    counterfactual_view: int | None = None,
) -> dict[str, Any]:
    available = available or [True, True, True]
    return {
        "id": f"N0V02-LATCHAL1-{family.upper()}-{index + 1:02d}",
        "family": family,
        "split": "challenge",
        "raw_text": raw_text,
        "query_text": query,
        "fields": fields,
        "relations": relations,
        "target_summary_text": target_summary,
        "target_view_distribution": compat_distribution(available),
        "target_view_distribution_role": "parent_cache_compatibility_only_not_used_for_latent_ratification",
        "view_reliability": reliability,
        "view_available": available,
        "cross_view_conflict": conflict,
        "counterfactual_required": counterfactual_view is not None,
        "counterfactual_view": counterfactual_view,
        "private_identity_content": False,
        "generated_text": True,
        "data_origin": "deterministic_public_synthetic_challenge_template",
        "identity_authority": False,
        "training_authorized": False,
    }


def build_row(family: str, index: int) -> dict[str, Any]:
    subject, attribute, previous, current = parts(index, FAMILIES.index(family))

    if family == "novel_semantic_authority":
        return row(
            family=family, index=index,
            raw_text=f"A signed operator note states that {subject}'s {attribute} is {current}.",
            query=f"What is the signed current {attribute} for {subject}?",
            fields=[field("unknown", f"The typed frame leaves {subject}'s {attribute} unset.", missing=True, confidence=0.15)],
            relations=[], target_summary=f"{subject}'s current {attribute} is {current}.",
            reliability=[0.99, 0.20, 0.10], counterfactual_view=0,
        )

    if family == "novel_structured_authority":
        return row(
            family=family, index=index,
            raw_text=f"A prose note says {subject} was reconfigured but omits the new {attribute}.",
            query=f"What verified {attribute} is recorded for {subject}?",
            fields=[field("verified", f"Verified typed state records {subject}'s {attribute} as {current}.", provenance_id=2, temporal_scope_id=2, confidence=0.995)],
            relations=[], target_summary=f"The verified {attribute} for {subject} is {current}.",
            reliability=[0.35, 0.995, 0.25], counterfactual_view=1,
        )

    if family == "supersession_current":
        fields = [
            field("old", f"Historical record lists {previous} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.95),
            field("new", f"Replacement record lists {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
        ]
        return row(
            family=family, index=index,
            raw_text=f"Two records mention {subject}'s {attribute}: {previous} and {current}.",
            query=f"What is the current {attribute} for {subject}?", fields=fields,
            relations=[edge(1, 0, "supersedes")], target_summary=f"The current {attribute} for {subject} is {current}.",
            reliability=[0.55, 0.85, 0.995], conflict=True, counterfactual_view=2,
        )

    if family == "supersession_historical":
        fields = [
            field("old", f"Earlier record lists {previous} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.96),
            field("new", f"Later replacement lists {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
        ]
        return row(
            family=family, index=index,
            raw_text=f"The record changed from {previous} to {current} for {subject}'s {attribute}.",
            query=f"Before the replacement, what {attribute} was recorded for {subject}?", fields=fields,
            relations=[edge(1, 0, "supersedes")], target_summary=f"Before replacement, {subject}'s {attribute} was {previous}.",
            reliability=[0.55, 0.85, 0.995], conflict=True, counterfactual_view=2,
        )

    if family in {"correction_chain_current", "correction_chain_historical"}:
        third = f"legacy-{previous}"
        fields = [
            field("legacy", f"Legacy record says {subject}'s {attribute} is {third}.", temporal_scope_id=1, confidence=0.75),
            field("middle", f"Intermediate record corrected the legacy value to {previous}.", temporal_scope_id=1, confidence=0.90),
            field("current", f"Final verified record corrected the intermediate value to {current}.", temporal_scope_id=2, confidence=0.99),
        ]
        historical = family.endswith("historical")
        query = (
            f"Before the final correction, what {attribute} did the intermediate record give for {subject}?"
            if historical else f"After all corrections, what is {subject}'s {attribute}?"
        )
        target = previous if historical else current
        return row(
            family=family, index=index,
            raw_text=f"Successive records list {third}, then {previous}, then {current} for {subject}'s {attribute}.",
            query=query, fields=fields,
            relations=[edge(1, 0, "corrects"), edge(2, 1, "corrects")],
            target_summary=f"The requested {attribute} for {subject} is {target}.",
            reliability=[0.50, 0.82, 0.995], conflict=True, counterfactual_view=2,
        )

    if family in {"temporal_successor_current", "temporal_successor_previous"}:
        fields = [
            field("previous", f"Earlier state records {previous} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.97),
            field("current", f"Successor state records {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
        ]
        previous_query = family.endswith("previous")
        target = previous if previous_query else current
        query = (
            f"What {attribute} immediately preceded the successor state for {subject}?"
            if previous_query else f"What {attribute} is in the successor state for {subject}?"
        )
        return row(
            family=family, index=index,
            raw_text=f"{subject} moved from {previous} to {current} for {attribute}.", query=query, fields=fields,
            relations=[edge(1, 0, "temporal_successor")], target_summary=f"The requested {attribute} for {subject} is {target}.",
            reliability=[0.65, 0.88, 0.99], counterfactual_view=2,
        )

    if family == "derived_support_chain":
        fields = [
            field("source", f"Primary observation establishes {subject}'s {attribute} as {current}.", provenance_id=2, confidence=0.99),
            field("derived", f"Derived summary reports {current} for {subject}'s {attribute}.", provenance_id=3, confidence=0.90),
            field("support", f"Independent check supports the derived summary for {subject}.", provenance_id=2, confidence=0.92),
        ]
        return row(
            family=family, index=index,
            raw_text=f"A terse note says the {attribute} check for {subject} completed.",
            query=f"What {attribute} is supported by the evidence chain for {subject}?", fields=fields,
            relations=[edge(1, 0, "derived_from"), edge(2, 1, "supports")],
            target_summary=f"The evidence chain supports {current} for {subject}'s {attribute}.",
            reliability=[0.45, 0.90, 0.99], counterfactual_view=2,
        )

    if family == "causal_chain":
        fields = [
            field("cause", f"Setting {subject}'s controller to {current} initiated the downstream change.", confidence=0.98),
            field("effect", f"The downstream monitor entered the expected state after {current} was applied.", confidence=0.95),
        ]
        return row(
            family=family, index=index,
            raw_text=f"The monitor changed after a controller update on {subject}.",
            query=f"Which {attribute} setting caused the documented downstream change for {subject}?", fields=fields,
            relations=[edge(0, 1, "causes")], target_summary=f"The causal evidence identifies {current} as the relevant {attribute} setting for {subject}.",
            reliability=[0.50, 0.86, 0.99], counterfactual_view=2,
        )

    if family == "missing_semantic_relational":
        fields = [
            field("old", f"Old record lists {previous} for {subject}'s {attribute}.", temporal_scope_id=1),
            field("new", f"New record lists {current} for {subject}'s {attribute}.", temporal_scope_id=2),
        ]
        return row(
            family=family, index=index,
            raw_text="Semantic prose intentionally unavailable.", query=f"What is the current {attribute} for {subject}?",
            fields=fields, relations=[edge(1, 0, "supersedes")], target_summary=f"The current {attribute} for {subject} is {current}.",
            reliability=[0.0, 0.88, 0.99], available=[False, True, True], conflict=True, counterfactual_view=2,
        )

    if family == "missing_structured_evidence":
        fields = [
            field("old", f"Evidence record A lists {previous} for {subject}'s {attribute}.", temporal_scope_id=1),
            field("new", f"Evidence record B lists {current} for {subject}'s {attribute}.", temporal_scope_id=2),
        ]
        return row(
            family=family, index=index,
            raw_text=f"The operator says the current {attribute} for {subject} is {current}.", query=f"What is {subject}'s current {attribute}?",
            fields=fields, relations=[edge(1, 0, "corrects")], target_summary=f"{subject}'s current {attribute} is {current}.",
            reliability=[0.95, 0.0, 0.99], available=[True, False, True], counterfactual_view=2,
        )

    if family == "missing_evidence_semantic_structured":
        fields = [field("verified", f"Verified state records {current} for {subject}'s {attribute}.", confidence=0.99)]
        return row(
            family=family, index=index,
            raw_text=f"Current operator text states {subject}'s {attribute} is {current}.", query=f"What is the current {attribute} for {subject}?",
            fields=fields, relations=[], target_summary=f"The current {attribute} for {subject} is {current}.",
            reliability=[0.97, 0.99, 0.0], available=[True, True, False], counterfactual_view=None,
        )

    if family == "unresolved_balanced_conflict":
        fields = [
            field("claim_a", f"Equally credible source A says {previous} for {subject}'s {attribute}.", confidence=0.85),
            field("claim_b", f"Equally credible source B says {current} for {subject}'s {attribute}.", confidence=0.85),
        ]
        return row(
            family=family, index=index,
            raw_text=f"Two peer reports disagree about {subject}'s {attribute}: {previous} versus {current}.",
            query=f"Can one {attribute} value be established for {subject}?", fields=fields,
            relations=[edge(0, 1, "conflicts_with")],
            target_summary=f"The available evidence does not establish a single {attribute} value for {subject}.",
            reliability=[0.80, 0.80, 0.95], conflict=True,
        )

    if family == "noisy_consensus":
        fields = [
            field("verified", f"Verified state gives {current} for {subject}'s {attribute}.", confidence=0.98),
            field("noise", f"A low-authority duplicate still repeats {previous}.", confidence=0.20),
        ]
        return row(
            family=family, index=index,
            raw_text=f"Most current notes agree that {subject}'s {attribute} is {current}; one stale note repeats {previous}.",
            query=f"What is the best-supported current {attribute} for {subject}?", fields=fields,
            relations=[edge(0, 1, "corrects")], target_summary=f"The best-supported current {attribute} for {subject} is {current}.",
            reliability=[0.94, 0.97, 0.94], conflict=True,
        )

    if family == "semantic_reliability_reversal":
        fields = [field("uncertain", f"A low-confidence frame suggests {previous} for {subject}'s {attribute}.", confidence=0.25)]
        return row(
            family=family, index=index,
            raw_text=f"A signed high-reliability statement gives {current} for {subject}'s {attribute}.",
            query=f"Which {attribute} should be used for {subject}?", fields=fields, relations=[],
            target_summary=f"The high-reliability statement supports {current} for {subject}'s {attribute}.",
            reliability=[0.995, 0.25, 0.20], counterfactual_view=0,
        )

    if family == "evidence_reliability_reversal":
        fields = [
            field("typed_guess", f"A provisional frame guesses {previous} for {subject}'s {attribute}.", confidence=0.30),
            field("signed_record", f"A signed evidence record establishes {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.995),
        ]
        return row(
            family=family, index=index,
            raw_text=f"An old prose note mentions {previous} for {subject}'s {attribute}.",
            query=f"Which {attribute} is supported by the highest-authority evidence for {subject}?", fields=fields,
            relations=[edge(1, 0, "corrects")], target_summary=f"The highest-authority evidence supports {current} for {subject}'s {attribute}.",
            reliability=[0.25, 0.35, 0.995], conflict=True, counterfactual_view=2,
        )

    if family == "lexical_high_overlap_distractor":
        fields = [
            field("distractor", f"Low-authority copied note repeats {previous} {previous} {previous} for {subject}'s {attribute}.", confidence=0.15),
            field("verified", f"Compact verified record gives {current} for {subject}'s {attribute}.", provenance_id=2, confidence=0.99),
        ]
        return row(
            family=family, index=index,
            raw_text=f"A verbose stale memo repeats {previous} many times for {subject}; the verified update is concise.",
            query=f"What verified {attribute} should be retained for {subject}?", fields=fields,
            relations=[edge(1, 0, "corrects")], target_summary=f"The verified {attribute} for {subject} is {current}.",
            reliability=[0.35, 0.72, 0.99], conflict=True, counterfactual_view=2,
        )

    if family in {"paired_current_state", "paired_historical_state"}:
        fields = [
            field("previous", f"Earlier state lists {previous} for {subject}'s {attribute}.", temporal_scope_id=1, confidence=0.97),
            field("current", f"Current state lists {current} for {subject}'s {attribute}.", temporal_scope_id=2, confidence=0.99),
        ]
        historical = family.endswith("historical_state")
        query = (
            f"What {attribute} did {subject} have immediately before the current state?"
            if historical else f"What {attribute} does {subject} have now?"
        )
        target = previous if historical else current
        return row(
            family=family, index=index,
            raw_text=f"{subject} changed its {attribute} from {previous} to {current}.", query=query, fields=fields,
            relations=[edge(1, 0, "temporal_successor")], target_summary=f"The requested {attribute} for {subject} is {target}.",
            reliability=[0.70, 0.90, 0.99], conflict=True, counterfactual_view=2,
        )

    raise KeyError(family)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    spec_path = Path(args.spec).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_CONFIRMATORY_SPEC_NO_RESULTS_OBSERVED":
        raise SystemExit("latent challenge spec status drift")

    rows = [build_row(family, index) for family in FAMILIES for index in range(ROWS_PER_FAMILY)]
    if len(rows) != 160 or len({row["id"] for row in rows}) != 160:
        raise SystemExit("latent challenge row count/id uniqueness drift")
    counts = Counter(str(row["family"]) for row in rows)
    if set(counts) != set(FAMILIES) or any(value != ROWS_PER_FAMILY for value in counts.values()):
        raise SystemExit("latent challenge family coverage drift")
    if any(row["private_identity_content"] for row in rows):
        raise SystemExit("latent challenge crossed private identity boundary")
    if any(row["training_authorized"] for row in rows):
        raise SystemExit("latent challenge unexpectedly authorizes training")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": "alice.eipm.n0.v02-adaptive-multi-view-latent-pool-frozen-challenge-manifest.v0.1",
        "status": "FROZEN_UNTOUCHED_NOT_EVALUATED",
        "rows": len(rows),
        "family_count": len(counts),
        "family_counts": dict(sorted(counts.items())),
        "challenge_sha256": sha(output),
        "spec_sha256": sha(spec_path),
        "data_origin": "deterministic_public_synthetic_challenge_template",
        "generated_text": True,
        "identity_authority": False,
        "private_identity_content": False,
        "training_authorized": False,
        "results_observed_at_compile_time": False,
        "exact_routing_percentage_supervision": False,
        "fixed_slot_trait_labels": False,
        "slot_permutation_invariant_evaluation": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
