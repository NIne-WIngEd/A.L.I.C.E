from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCHEMA = "alice.eipm.n0.qsre-t1-row.v0.1"
FAMILIES = (
    "endpoint_role",
    "relation_filter",
    "multi_support_aggregate",
    "ordered_path",
    "reliability_arbitration",
    "temporal_arbitration",
    "ambiguity_plurality",
    "fallback_defer",
    "outside_support_distractor",
)


def field(split: str, family: str, row_index: int, slot: int) -> dict:
    token = f"{split}_{family}_{row_index:04d}_{slot:02d}"
    return {
        "id": f"f_{token}",
        "text": f"neutral evidence record {token}",
        "metadata": {
            "reliability": 0.5,
            "normalized_time": 0.5,
            "is_focus_hint_reserved_zero": 0.0,
        },
    }


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation_id: int,
    *,
    support_weight: float,
    reliability: float = 0.5,
    normalized_time: float = 0.5,
) -> dict:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation_id": relation_id,
        "support_weight": support_weight,
        "metadata": {
            "reliability": reliability,
            "normalized_time": normalized_time,
        },
    }


def one_hot_target(field_id: str) -> dict[str, float]:
    return {field_id: 1.0}


def make_row(
    *,
    split: str,
    family: str,
    row_index: int,
    relation_count: int,
    rng: random.Random,
) -> dict:
    rid = f"t1_{split}_{family}_{row_index:04d}"
    relation = rng.randrange(relation_count)
    role_source = row_index % 2 == 0
    role_id = 0 if role_source else 1
    applicability = 0.9
    expected_control = "RELATIONAL"
    operation_id = 0
    relation_sequence = [relation]
    focus_field_id = None
    fields = [field(split, family, row_index, i) for i in range(6)]
    ids = [x["id"] for x in fields]
    edges = []
    target = {}
    field_support = {}

    if family == "endpoint_role":
        edges = [edge("e0", ids[0], ids[1], relation, support_weight=1.0)]
        target = one_hot_target(ids[0] if role_source else ids[1])

    elif family == "relation_filter":
        other = (relation + 1) % relation_count
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0),
            edge("e1", ids[0], ids[2], other, support_weight=1.0),
        ]
        role_id = 1
        target = one_hot_target(ids[1])

    elif family == "multi_support_aggregate":
        operation_id = 2
        role_id = 1
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0),
            edge("e1", ids[2], ids[3], relation, support_weight=1.0),
        ]
        target = {ids[1]: 0.5, ids[3]: 0.5}

    elif family == "ordered_path":
        operation_id = 1
        role_id = 1
        r2 = (relation + 1 + (row_index % max(relation_count - 1, 1))) % relation_count
        if r2 == relation:
            r2 = (relation + 1) % relation_count
        relation_sequence = [relation, r2]
        focus_field_id = ids[0]
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0),
            edge("e1", ids[1], ids[2], r2, support_weight=1.0),
            edge("e2", ids[0], ids[3], r2, support_weight=1.0),
            edge("e3", ids[3], ids[4], relation, support_weight=1.0),
        ]
        target = one_hot_target(ids[2])

    elif family == "reliability_arbitration":
        operation_id = 3
        role_id = 1
        hi_first = row_index % 2 == 0
        a = 0.9 if hi_first else 0.2
        b = 0.2 if hi_first else 0.9
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0, reliability=a),
            edge("e1", ids[2], ids[3], relation, support_weight=1.0, reliability=b),
        ]
        target = one_hot_target(ids[1] if hi_first else ids[3])

    elif family == "temporal_arbitration":
        operation_id = 4
        role_id = 1
        first_latest = row_index % 2 == 0
        a = 0.9 if first_latest else 0.1
        b = 0.1 if first_latest else 0.9
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0, normalized_time=a),
            edge("e1", ids[2], ids[3], relation, support_weight=1.0, normalized_time=b),
        ]
        target = one_hot_target(ids[1] if first_latest else ids[3])

    elif family == "ambiguity_plurality":
        operation_id = 2
        role_id = 1
        edges = [
            edge("e0", ids[0], ids[1], relation, support_weight=1.0),
            edge("e1", ids[2], ids[3], relation, support_weight=1.0),
        ]
        target = {ids[1]: 0.5, ids[3]: 0.5}

    elif family == "fallback_defer":
        if row_index % 2 == 0:
            applicability = 0.1
            expected_control = "FALLBACK"
        else:
            applicability = 0.5
            expected_control = "DEFER"
        role_id = 3
        relation_sequence = [relation]
        edges = []
        target = {}

    elif family == "outside_support_distractor":
        role_id = 1
        edges = [edge("e0", ids[0], ids[1], relation, support_weight=1.0)]
        fields[5]["metadata"]["reliability"] = 1.0
        fields[5]["metadata"]["normalized_time"] = 1.0
        target = one_hot_target(ids[1])

    else:
        raise ValueError(f"unknown family: {family}")

    if focus_field_id is None and family in {
        "endpoint_role",
        "relation_filter",
        "reliability_arbitration",
        "temporal_arbitration",
        "outside_support_distractor",
    }:
        focus_field_id = ids[0]

    return {
        "schema": SCHEMA,
        "id": rid,
        "split": split,
        "family": family,
        "fields": fields,
        "edges": edges,
        "field_support": field_support,
        "operator": {
            "relation_sequence_id": relation_sequence,
            "role_id": role_id,
            "operation_id": operation_id,
            "focus_field_id": focus_field_id,
            "context": [0.0, 0.0, 0.0, 0.0],
            "applicability": applicability,
        },
        "target_distribution": target,
        "expected_control": expected_control,
        "private_identity_data": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if contract.get("schema") != "alice.eipm.n0.qsre-t1-curriculum-contract.v0.1":
        raise SystemExit("T1 curriculum contract schema drift")

    rng = random.Random(int(contract["seed"]))
    rows = []
    for split, count in (
        ("train", int(contract["train_rows_per_family"])),
        ("dev", int(contract["dev_rows_per_family"])),
    ):
        for family in contract["families"]:
            for row_index in range(count):
                rows.append(
                    make_row(
                        split=split,
                        family=family,
                        row_index=row_index,
                        relation_count=int(contract["relation_seed_count"]),
                        rng=rng,
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"rows={len(rows)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
