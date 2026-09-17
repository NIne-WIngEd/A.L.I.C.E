#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from alice_personality.n0.evidence_graph import relation_type_id

SUBJECTS = [f"Selector {name}" for name in [
    "Atlas","Birch","Cobalt","Dune","Ember","Fjord","Grove","Harbor",
    "Indigo","Juniper","Kestrel","Lumen","Mica","Nimbus","Osprey","Quartz",
    "Reed","Saffron","Tundra","Umber","Vale","Willow","Xenon","Yarrow",
    "Zephyr","Aster","Bramble","Cirrus","Delta","Elm","Flint","Garnet",
]]
ATTRIBUTES = [
    ("handoff class", "amber", "violet"),
    ("control token", "314", "927"),
    ("service lane", "quiet", "burst"),
    ("refresh cadence", "17 seconds", "43 seconds"),
    ("replica mode", "single", "paired"),
    ("telemetry tier", "basic", "extended"),
    ("routing channel", "west", "east"),
    ("snapshot interval", "19 minutes", "47 minutes"),
    ("worker quota", "6", "11"),
    ("sampling profile", "coarse", "fine"),
    ("archive policy", "sealed", "rolling"),
    ("sync band", "narrow", "wide"),
]
PAIRS_PER_FAMILY = 32
TRAIN_PAIRS_PER_FAMILY = 20
DEV_PAIRS_PER_FAMILY = 6
TEST_PAIRS_PER_FAMILY = 6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(name: str, text: str, *, temporal_scope_id: int, confidence: float, provenance_id: int = 1) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "field_type_id": 1,
        "provenance_id": provenance_id,
        "relation_role_id": 1,
        "temporal_scope_id": temporal_scope_id,
        "confidence": confidence,
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


def split_for(index: int) -> str:
    if index < TRAIN_PAIRS_PER_FAMILY:
        return "train"
    if index < TRAIN_PAIRS_PER_FAMILY + DEV_PAIRS_PER_FAMILY:
        return "dev"
    return "test"


def parts(index: int, offset: int) -> tuple[str, str, str, str]:
    subject = SUBJECTS[(index + 3 * offset) % len(SUBJECTS)]
    attribute, first, second = ATTRIBUTES[(index + offset) % len(ATTRIBUTES)]
    return subject, attribute, first, second


def two_records(subject: str, attribute: str, first: str, second: str) -> list[dict[str, Any]]:
    # Deliberately asymmetric metadata recreates a strong structured prior.
    # The paired relation flip makes that prior non-authoritative.
    return [
        field("record_alpha", f"Record alpha for {subject} reports {first} for {attribute}.", temporal_scope_id=1, confidence=0.96),
        field("record_beta", f"Record beta for {subject} reports {second} for {attribute}.", temporal_scope_id=2, confidence=0.99),
    ]


def make_pair(*, family: str, index: int, fields: list[dict[str, Any]], query: str,
              relations_a: list[dict[str, Any]], target_a: int, summary_a: str,
              relations_b: list[dict[str, Any]], target_b: int, summary_b: str) -> list[dict[str, Any]]:
    if target_a == target_b:
        raise ValueError("paired selector-repair target must flip")
    split = split_for(index)
    pair_id = f"N0V02-SELREPAIR-{family.upper()}-{index + 1:02d}"
    def row(variant: str, relations: list[dict[str, Any]], target: int, summary: str) -> dict[str, Any]:
        distribution = [0.0] * len(fields)
        distribution[target] = 1.0
        return {
            "id": f"{pair_id}-{variant}",
            "pair_id": pair_id,
            "variant": variant,
            "family": family,
            "split": split,
            "query_text": query,
            "fields": fields,
            "relations": relations,
            "target_evidence_distribution": distribution,
            "target_summary_text": summary,
            "relation_essential": True,
            "paired_adapter_inputs_identical": True,
            "target_flips_only_with_relation_graph": True,
            "generated_text": True,
            "data_origin": "deterministic_public_synthetic_selector_repair_template",
            "identity_authority": False,
            "private_identity_content": False,
            "training_authorized": split == "train",
        }
    return [row("A", relations_a, target_a, summary_a), row("B", relations_b, target_b, summary_b)]


def corrects_source(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 0); fields = two_records(s, a, x, y)
    q = f"Using only the correction edge, which record supplies the corrected {a} for {s}?"
    return make_pair(family="corrects_source", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"corrects")], target_a=1, summary_a=f"The correcting record supplies {y}.",
        relations_b=[edge(0,1,"corrects")], target_b=0, summary_b=f"The correcting record supplies {x}.")


def corrects_target(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 1); fields = two_records(s, a, x, y)
    q = f"Using only the correction edge, which {a} record for {s} is being corrected?"
    return make_pair(family="corrects_target", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"corrects")], target_a=0, summary_a=f"The corrected record contains {x}.",
        relations_b=[edge(0,1,"corrects")], target_b=1, summary_b=f"The corrected record contains {y}.")


def supersedes_source(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 2); fields = two_records(s, a, x, y)
    q = f"According to the supersession edge, which record is the replacement for {s}'s {a}?"
    return make_pair(family="supersedes_source", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"supersedes")], target_a=1, summary_a=f"The replacement record contains {y}.",
        relations_b=[edge(0,1,"supersedes")], target_b=0, summary_b=f"The replacement record contains {x}.")


def supersedes_target(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 3); fields = two_records(s, a, x, y)
    q = f"According to the supersession edge, which {a} record for {s} was superseded?"
    return make_pair(family="supersedes_target", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"supersedes")], target_a=0, summary_a=f"The superseded record contains {x}.",
        relations_b=[edge(0,1,"supersedes")], target_b=1, summary_b=f"The superseded record contains {y}.")


def temporal_source(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 4); fields = two_records(s, a, x, y)
    q = f"Using the temporal-successor edge alone, which record is the successor for {s}'s {a}?"
    return make_pair(family="temporal_successor_source", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"temporal_successor")], target_a=1, summary_a=f"The successor record contains {y}.",
        relations_b=[edge(0,1,"temporal_successor")], target_b=0, summary_b=f"The successor record contains {x}.")


def supports_target(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 5)
    fields = [
        field("claim_alpha", f"Claim alpha for {s} assigns {x} to {a}.", temporal_scope_id=1, confidence=0.96),
        field("claim_beta", f"Claim beta for {s} assigns {y} to {a}.", temporal_scope_id=2, confidence=0.99),
        field("witness", f"A witness record concerns {s}'s {a} but does not state its value.", temporal_scope_id=2, confidence=0.98, provenance_id=2),
    ]
    q = f"Follow the support edge and choose the supported {a} claim for {s}."
    return make_pair(family="supports_target", index=index, fields=fields, query=q,
        relations_a=[edge(2,0,"supports")], target_a=0, summary_a=f"The supported claim gives {x}.",
        relations_b=[edge(2,1,"supports")], target_b=1, summary_b=f"The supported claim gives {y}.")


def conflict_support_target(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 6)
    fields = [
        field("claim_alpha", f"Claim alpha for {s} reports {x} for {a}.", temporal_scope_id=1, confidence=0.97),
        field("claim_beta", f"Claim beta for {s} reports {y} for {a}.", temporal_scope_id=2, confidence=0.99),
        field("arbiter", f"An arbiter record concerns the disputed {a} for {s}.", temporal_scope_id=2, confidence=0.99, provenance_id=2),
    ]
    q = f"Resolve the conflict using the arbiter support edge for {s}'s {a}."
    common = [edge(0,1,"conflicts_with")]
    return make_pair(family="conflict_support_target", index=index, fields=fields, query=q,
        relations_a=common+[edge(2,0,"supports")], target_a=0, summary_a=f"The arbiter supports {x}.",
        relations_b=common+[edge(2,1,"supports")], target_b=1, summary_b=f"The arbiter supports {y}.")


def mixed_correct_support(index: int) -> list[dict[str, Any]]:
    s, a, x, y = parts(index, 7)
    fields = two_records(s, a, x, y) + [
        field("validator", f"A validator record concerns {s}'s {a}.", temporal_scope_id=2, confidence=0.98, provenance_id=2)
    ]
    q = f"Combine the correction and validator-support edges to choose the corrected {a} for {s}."
    return make_pair(family="mixed_correct_support_source", index=index, fields=fields, query=q,
        relations_a=[edge(1,0,"corrects"),edge(2,1,"supports")], target_a=1, summary_a=f"The corrected supported record gives {y}.",
        relations_b=[edge(0,1,"corrects"),edge(2,0,"supports")], target_b=0, summary_b=f"The corrected supported record gives {x}.")


FAMILIES: list[tuple[str, Callable[[int], list[dict[str, Any]]]]] = [
    ("corrects_source", corrects_source),
    ("corrects_target", corrects_target),
    ("supersedes_source", supersedes_source),
    ("supersedes_target", supersedes_target),
    ("temporal_successor_source", temporal_source),
    ("supports_target", supports_target),
    ("conflict_support_target", conflict_support_target),
    ("mixed_correct_support_source", mixed_correct_support),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()

    rows: list[dict[str, Any]] = []
    for _name, factory in FAMILIES:
        for index in range(PAIRS_PER_FAMILY):
            rows.extend(factory(index))

    if len(rows) != len(FAMILIES) * PAIRS_PER_FAMILY * 2:
        raise SystemExit("selector-repair row count drift")
    if len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("selector-repair id collision")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(row)
    for pair_id, members in grouped.items():
        members = sorted(members, key=lambda item: item["variant"])
        if len(members) != 2 or {m["variant"] for m in members} != {"A", "B"}:
            raise SystemExit(f"malformed selector pair: {pair_id}")
        a, b = members
        if a["fields"] != b["fields"] or a["query_text"] != b["query_text"]:
            raise SystemExit(f"adapter input drift inside pair: {pair_id}")
        if a["target_evidence_distribution"] == b["target_evidence_distribution"]:
            raise SystemExit(f"target did not flip inside pair: {pair_id}")
        if a["split"] != b["split"]:
            raise SystemExit(f"split drift inside pair: {pair_id}")

    expected = {"train": TRAIN_PAIRS_PER_FAMILY, "dev": DEV_PAIRS_PER_FAMILY, "test": TEST_PAIRS_PER_FAMILY}
    for family, _factory in FAMILIES:
        counts = defaultdict(int)
        seen = set()
        for row in rows:
            if row["family"] != family or row["pair_id"] in seen:
                continue
            seen.add(row["pair_id"])
            counts[row["split"]] += 1
        if dict(counts) != expected:
            raise SystemExit(f"split count drift for {family}: {dict(counts)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema": "alice.eipm.n0.v02-evidence-selector-repair-curriculum.v0.1",
        "status": "TRAIN_DEV_TEST_COMPILED_TEST_UNTOUCHED",
        "rows": len(rows),
        "family_count": len(FAMILIES),
        "families": [name for name, _factory in FAMILIES],
        "pairs_per_family": PAIRS_PER_FAMILY,
        "train_pairs_per_family": TRAIN_PAIRS_PER_FAMILY,
        "dev_pairs_per_family": DEV_PAIRS_PER_FAMILY,
        "test_pairs_per_family": TEST_PAIRS_PER_FAMILY,
        "compiled_sha256": sha(output),
        "generated_text": True,
        "data_origin": "deterministic_public_synthetic_selector_repair_template",
        "identity_authority": False,
        "private_identity_content": False,
        "train_split_training_authorized": True,
        "dev_split_training_authorized": False,
        "test_split_training_authorized": False,
        "frozen_latent_challenge_rows_used_for_training": False,
        "parent_value_path_diagnostic_rows_used_for_training": False,
        "paired_adapter_inputs_identical": True,
        "target_flips_only_with_relation_graph": True,
        "graph_pooled_cosine_is_optimization_target": False,
        "exact_routing_percentage_supervision": False,
        "hard_parameter_ceiling": None,
        "curriculum_counts_are_operating_budget_not_architecture_ceiling": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
