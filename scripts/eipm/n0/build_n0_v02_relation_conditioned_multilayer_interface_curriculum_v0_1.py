#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "alice.eipm.n0.v02-relation-conditioned-multilayer-interface-curriculum.v0.1"

TRAIN_QUADS_PER_RELATION = 18
DEV_QUADS_PER_RELATION = 6
TEST_QUADS_PER_RELATION = 6
QUADS_PER_RELATION = (
    TRAIN_QUADS_PER_RELATION + DEV_QUADS_PER_RELATION + TEST_QUADS_PER_RELATION
)

SUBJECTS = {
    "train": [
        "Aster", "Brook", "Cedar", "Dune", "Ember", "Flint",
        "Grove", "Harbor", "Iris", "Juniper", "Kestrel", "Lagoon",
        "Maple", "Nimbus", "Opal", "Prairie", "Quartz", "Ridge",
    ],
    "dev": ["Beacon", "Coral", "Delta", "Fennel", "Granite", "Linden"],
    "test": ["Aurora", "Birch", "Cobalt", "Fern", "Glacier", "Laurel"],
}

ATTRIBUTES = {
    "train": [
        ("handoff mode", "manual", "coordinated"),
        ("control epoch", "23", "71"),
        ("service posture", "standby", "active"),
        ("refresh window", "11 seconds", "31 seconds"),
        ("replica policy", "single", "paired"),
        ("telemetry tier", "basic", "extended"),
    ],
    "dev": [
        ("worker quota", "6", "14"),
        ("sampling regime", "sparse", "dense"),
        ("archive tier", "cold", "warm"),
    ],
    "test": [
        ("dispatch plan", "serial", "parallel"),
        ("retention window", "18 hours", "42 hours"),
        ("failover mode", "passive", "automatic"),
    ],
}

RELATIONS: dict[str, dict[str, Any]] = {
    "corrects": {
        "source_role": "correcting_record",
        "target_role": "corrected_record",
        "source_queries": {
            "train": [
                "Which record gives the corrected {attribute} for {subject}?",
                "After the mistake is fixed, which record should determine {subject}'s {attribute}?",
                "Which record contains the valid replacement value for {subject}'s {attribute}?",
            ],
            "dev": [
                "Once the error is amended, which record is authoritative for {subject}'s {attribute}?",
                "Which record carries the rectified {attribute} for {subject}?",
            ],
            "test": [
                "After remediation, which record supplies the operative {attribute} for {subject}?",
                "Which record is the correcting side of the {attribute} update for {subject}?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record contains the {attribute} that was corrected for {subject}?",
                "Which record held the mistaken {attribute} before the fix for {subject}?",
                "Which record is the one being replaced by the correction to {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record lost authority when {subject}'s {attribute} was rectified?",
                "Which record is the corrected side rather than the correcting side for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record contains the superseded error in {subject}'s {attribute}?",
                "Which record was acted on by the correction to {subject}'s {attribute}?",
            ],
        },
    },
    "supersedes": {
        "source_role": "replacement_record",
        "target_role": "superseded_record",
        "source_queries": {
            "train": [
                "Which record now takes precedence for {subject}'s {attribute}?",
                "Which record is the replacement state for {subject}'s {attribute}?",
                "Which record should be used after the older {attribute} state is superseded for {subject}?",
            ],
            "dev": [
                "Which record remains operative after replacement for {subject}'s {attribute}?",
                "Which record is the prevailing successor state for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record controls after the prior {attribute} state is displaced for {subject}?",
                "Which record is the superseding side of {subject}'s {attribute} transition?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record is the older {attribute} state that lost precedence for {subject}?",
                "Which record was superseded in {subject}'s {attribute} history?",
                "Which record is displaced by the replacement {attribute} state for {subject}?",
            ],
            "dev": [
                "Which record ceased to be operative when {subject}'s {attribute} was replaced?",
                "Which record is the superseded predecessor for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record belongs to the displaced {attribute} state for {subject}?",
                "Which record is the target of the supersession affecting {subject}'s {attribute}?",
            ],
        },
    },
    "derived_from": {
        "source_role": "derived_record",
        "target_role": "basis_record",
        "source_queries": {
            "train": [
                "Which record is derived from the other for {subject}'s {attribute}?",
                "Which record is the resulting item in the derivation of {subject}'s {attribute}?",
                "Which record depends on another record as its derivation basis for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is the derived side of the provenance relation for {subject}'s {attribute}?",
                "Which record results from the basis record for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record is downstream in the derivation of {subject}'s {attribute}?",
                "Which record is the derived artifact rather than its basis for {subject}'s {attribute}?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record is the basis from which the other record is derived for {subject}'s {attribute}?",
                "Which record supplies the derivation source for {subject}'s {attribute}?",
                "Which record is upstream of the derived record for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is the provenance basis for the derived {attribute} record for {subject}?",
                "Which record is the source side of the derivation for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record underlies the derived {attribute} item for {subject}?",
                "Which record is the derivation basis rather than the derived artifact for {subject}'s {attribute}?",
            ],
        },
    },
    "causes": {
        "source_role": "cause_record",
        "target_role": "effect_record",
        "source_queries": {
            "train": [
                "Which record describes the cause of the other record for {subject}'s {attribute}?",
                "Which record is upstream in the causal relation affecting {subject}'s {attribute}?",
                "Which record gives the event that produces the other event for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is the causal antecedent for {subject}'s {attribute}?",
                "Which record is the producing side of the causal relation for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record is the cause rather than the consequence for {subject}'s {attribute}?",
                "Which record initiates the causal chain involving {subject}'s {attribute}?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record describes the effect caused by the other record for {subject}'s {attribute}?",
                "Which record is downstream in the causal relation affecting {subject}'s {attribute}?",
                "Which record gives the consequence produced by the other event for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is the causal consequence for {subject}'s {attribute}?",
                "Which record is the produced side of the causal relation for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record is the consequence rather than the cause for {subject}'s {attribute}?",
                "Which record is produced by the causal antecedent involving {subject}'s {attribute}?",
            ],
        },
    },
    "supports": {
        "source_role": "supporting_record",
        "target_role": "supported_record",
        "source_queries": {
            "train": [
                "Which record provides support for the other record about {subject}'s {attribute}?",
                "Which record is the supporting evidence in the relation about {subject}'s {attribute}?",
                "Which record strengthens the other record concerning {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is the evidentiary support for the claim about {subject}'s {attribute}?",
                "Which record is the supporting side of the relation for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record supplies the support rather than receiving it for {subject}'s {attribute}?",
                "Which record acts as evidence for the other record concerning {subject}'s {attribute}?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record is supported by the other record about {subject}'s {attribute}?",
                "Which record is the claim receiving evidentiary support for {subject}'s {attribute}?",
                "Which record is strengthened by the other record concerning {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record receives the evidence supporting {subject}'s {attribute}?",
                "Which record is the supported side of the relation for {subject}'s {attribute}?",
            ],
            "test": [
                "Which record receives support rather than supplying it for {subject}'s {attribute}?",
                "Which record is backed by the other record concerning {subject}'s {attribute}?",
            ],
        },
    },
    "temporal_successor": {
        "source_role": "later_record",
        "target_role": "earlier_record",
        "source_queries": {
            "train": [
                "Which record is the later state for {subject}'s {attribute}?",
                "Which record comes after the other in {subject}'s {attribute} history?",
                "Which record is the temporal successor for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is later in time for {subject}'s {attribute}?",
                "Which record follows the earlier {attribute} state for {subject}?",
            ],
            "test": [
                "Which record is the subsequent state for {subject}'s {attribute}?",
                "Which record is the successor rather than predecessor for {subject}'s {attribute}?",
            ],
        },
        "target_queries": {
            "train": [
                "Which record is the earlier state for {subject}'s {attribute}?",
                "Which record comes before the other in {subject}'s {attribute} history?",
                "Which record is the temporal predecessor for {subject}'s {attribute}?",
            ],
            "dev": [
                "Which record is earlier in time for {subject}'s {attribute}?",
                "Which record precedes the later {attribute} state for {subject}?",
            ],
            "test": [
                "Which record is the prior state for {subject}'s {attribute}?",
                "Which record is the predecessor rather than successor for {subject}'s {attribute}?",
            ],
        },
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_for(index: int) -> str:
    if index < TRAIN_QUADS_PER_RELATION:
        return "train"
    if index < TRAIN_QUADS_PER_RELATION + DEV_QUADS_PER_RELATION:
        return "dev"
    return "test"


def local_index(index: int, split: str) -> int:
    if split == "train":
        return index
    if split == "dev":
        return index - TRAIN_QUADS_PER_RELATION
    return index - TRAIN_QUADS_PER_RELATION - DEV_QUADS_PER_RELATION


def field(name: str, text: str) -> dict[str, Any]:
    return {
        "name": name,
        "text": text,
        "provenance": "PUBLIC_SYNTHETIC",
        "confidence": 1.0,
    }


def edge(source: int, target: int, relation: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": 1.0,
    }


def target_distribution(index: int) -> list[float]:
    values = [0.0, 0.0]
    values[index] = 1.0
    return values


def make_quad(
    *,
    relation: str,
    relation_spec: dict[str, Any],
    index: int,
    relation_offset: int,
) -> list[dict[str, Any]]:
    split = split_for(index)
    local = local_index(index, split)
    subjects = SUBJECTS[split]
    attrs = ATTRIBUTES[split]
    subject = subjects[(local + relation_offset) % len(subjects)]
    attribute, value_a, value_b = attrs[(local + 2 * relation_offset) % len(attrs)]

    fields = [
        field("record_one", f"Record one reports {value_a} for {subject}'s {attribute}."),
        field("record_two", f"Record two reports {value_b} for {subject}'s {attribute}."),
    ]

    source_templates = relation_spec["source_queries"][split]
    target_templates = relation_spec["target_queries"][split]
    source_query = source_templates[local % len(source_templates)].format(
        subject=subject,
        attribute=attribute,
    )
    target_query = target_templates[local % len(target_templates)].format(
        subject=subject,
        attribute=attribute,
    )

    quad_id = f"N0V02-RCMLI-{relation.upper()}-{index + 1:02d}"
    rows: list[dict[str, Any]] = []
    for direction, source, target in (("A", 0, 1), ("B", 1, 0)):
        for query_role, query_text in (
            ("source", source_query),
            ("target", target_query),
        ):
            answer_index = source if query_role == "source" else target
            row_id = f"{quad_id}-{direction}-{query_role.upper()}"
            rows.append(
                {
                    "id": row_id,
                    "quad_id": quad_id,
                    "edge_direction_variant": direction,
                    "query_role": query_role,
                    "semantic_role_name": relation_spec[f"{query_role}_role"],
                    "relation": relation,
                    "split": split,
                    "query_text": query_text,
                    "fields": fields,
                    "relations": [edge(source, target, relation)],
                    "target_evidence_distribution": target_distribution(answer_index),
                    "target_field_index": answer_index,
                    "causal_factors": {
                        "query_role_varies": True,
                        "edge_direction_varies": True,
                        "field_text_fixed_within_quad": True,
                        "relation_type_fixed_within_quad": True,
                        "target_depends_on_query_role_and_edge_direction": True,
                    },
                    "interface_hypothesis": (
                        "relation-conditioned intermediate token access must preserve "
                        "query-role meaning before evidence-field selection"
                    ),
                    "generated_text": True,
                    "data_origin": (
                        "deterministic_public_synthetic_relation_conditioned_"
                        "multilayer_interface_v0_1"
                    ),
                    "identity_authority": False,
                    "private_identity_content": False,
                    "intended_training_split": split == "train",
                    "training_authorized": False,
                    "heldout_opening_authorized": False,
                    "frozen_latent_challenge_row_reused": False,
                    "missing_evidence_localization_row_reused": False,
                    "relation_semantic_v0_1_row_reused": False,
                    "relation_semantic_role_residual_v0_2_row_reused": False,
                    "query_relation_role_router_v0_3_row_reused": False,
                    "endpoint_repair_v0_2_heldout_row_reused": False,
                }
            )
    return rows


def validate_quad(quad_id: str, members: list[dict[str, Any]]) -> None:
    if len(members) != 4:
        raise SystemExit(f"quad size drift for {quad_id}: {len(members)}")
    keys = {
        (str(row["edge_direction_variant"]), str(row["query_role"]))
        for row in members
    }
    expected = {("A", "source"), ("A", "target"), ("B", "source"), ("B", "target")}
    if keys != expected:
        raise SystemExit(f"factorial coverage drift for {quad_id}: {sorted(keys)}")

    first = members[0]
    for row in members[1:]:
        for key in ("fields", "relation", "split", "quad_id"):
            if row[key] != first[key]:
                raise SystemExit(f"non-causal drift in {quad_id}: {key}")

    by_key = {
        (str(row["edge_direction_variant"]), str(row["query_role"])): row
        for row in members
    }

    a_source = by_key[("A", "source")]
    a_target = by_key[("A", "target")]
    b_source = by_key[("B", "source")]
    b_target = by_key[("B", "target")]

    if a_source["relations"] != a_target["relations"]:
        raise SystemExit(f"query-role comparison changed relation in {quad_id}")
    if b_source["relations"] != b_target["relations"]:
        raise SystemExit(f"query-role comparison changed relation in {quad_id}")

    edge_a = a_source["relations"][0]
    edge_b = b_source["relations"][0]
    if edge_a["source"] != edge_b["target"] or edge_a["target"] != edge_b["source"]:
        raise SystemExit(f"edge direction did not reverse in {quad_id}")

    if a_source["target_field_index"] == a_target["target_field_index"]:
        raise SystemExit(f"query role failed to flip target in {quad_id}")
    if b_source["target_field_index"] == b_target["target_field_index"]:
        raise SystemExit(f"query role failed to flip target in {quad_id}")
    if a_source["target_field_index"] == b_source["target_field_index"]:
        raise SystemExit(f"edge direction failed to flip source-role target in {quad_id}")
    if a_target["target_field_index"] == b_target["target_field_index"]:
        raise SystemExit(f"edge direction failed to flip target-role target in {quad_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for relation_offset, (relation, spec) in enumerate(RELATIONS.items()):
        for index in range(QUADS_PER_RELATION):
            rows.extend(
                make_quad(
                    relation=relation,
                    relation_spec=spec,
                    index=index,
                    relation_offset=relation_offset,
                )
            )

    expected_rows = len(RELATIONS) * QUADS_PER_RELATION * 4
    if len(rows) != expected_rows:
        raise SystemExit(f"curriculum row count drift: {len(rows)} != {expected_rows}")
    if len({str(row["id"]) for row in rows}) != expected_rows:
        raise SystemExit("curriculum row id collision")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["quad_id"])].append(row)
    for quad_id, members in grouped.items():
        validate_quad(quad_id, members)

    expected_quads = {
        "train": TRAIN_QUADS_PER_RELATION,
        "dev": DEV_QUADS_PER_RELATION,
        "test": TEST_QUADS_PER_RELATION,
    }
    counts: dict[str, dict[str, int]] = {}
    for relation in RELATIONS:
        relation_counts: dict[str, int] = defaultdict(int)
        seen: set[str] = set()
        for row in rows:
            if row["relation"] != relation:
                continue
            quad_id = str(row["quad_id"])
            if quad_id in seen:
                continue
            seen.add(quad_id)
            relation_counts[str(row["split"])] += 1
        if dict(relation_counts) != expected_quads:
            raise SystemExit(
                f"split count drift for {relation}: {dict(relation_counts)}"
            )
        counts[relation] = dict(relation_counts)

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite relation-conditioned curriculum artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = {
        "schema": SCHEMA,
        "status": "COMPILED_FRESH_CAUSAL_CURRICULUM_TRAINING_STILL_UNAUTHORIZED",
        "rows": len(rows),
        "quad_count": len(grouped),
        "relations": list(RELATIONS),
        "quads_per_relation": QUADS_PER_RELATION,
        "train_quads_per_relation": TRAIN_QUADS_PER_RELATION,
        "dev_quads_per_relation": DEV_QUADS_PER_RELATION,
        "test_quads_per_relation": TEST_QUADS_PER_RELATION,
        "split_quad_counts_by_relation": counts,
        "factorial_design": {
            "query_role": ["source", "target"],
            "edge_direction": ["A", "B"],
            "rows_per_quad": 4,
            "field_text_fixed_within_quad": True,
            "relation_type_fixed_within_quad": True,
            "target_must_flip_with_query_role": True,
            "target_must_flip_with_edge_direction": True,
        },
        "causal_question": (
            "can relation-conditioned access to intermediate semantic token layers "
            "recover query-role-dependent evidence selection without changing the "
            "semantic backbone or proven parent graph"
        ),
        "train_dev_test_query_templates_lexically_disjoint": True,
        "train_dev_test_subject_pools_disjoint": True,
        "train_dev_test_attribute_pools_disjoint": True,
        "generated_text": True,
        "data_origin": (
            "deterministic_public_synthetic_relation_conditioned_"
            "multilayer_interface_v0_1"
        ),
        "identity_authority": False,
        "private_identity_content": False,
        "train_rows_intended_for_future_training_if_authorized": True,
        "interface_training_authorized": False,
        "optimizer_authorized": False,
        "gradient_authorized": False,
        "dev_split_training_authorized": False,
        "test_split_training_authorized": False,
        "test_split_opening_authorized": False,
        "frozen_latent_challenge_rows_reused": False,
        "missing_evidence_localization_rows_reused": False,
        "relation_semantic_v0_1_rows_reused": False,
        "relation_semantic_role_residual_v0_2_rows_reused": False,
        "query_relation_role_router_v0_3_rows_reused": False,
        "endpoint_repair_v0_2_heldout_rows_reused": False,
        "curriculum_size_is_permanent_capability_limit": False,
        "hard_parameter_ceiling": None,
        "compiled_sha256": sha256(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
