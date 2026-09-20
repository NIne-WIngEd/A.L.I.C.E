from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SCHEMA = "alice.eipm.n0.qsre-t1-row.v0.2"
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


def opaque_text(split: str, family: str, group: int, slot: int) -> str:
    raw = f"{split}|{family}|{group}|{slot}"
    opaque = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"neutral evidence record {opaque}"


def make_fields(split: str, family: str, group: int) -> list[dict]:
    fields = []
    for slot in range(6):
        fid = f"f_{split}_{family}_{group:04d}_{slot:02d}"
        fields.append(
            {
                "id": fid,
                "text": opaque_text(split, family, group, slot),
                "metadata": {
                    "reliability": 0.5,
                    "normalized_time": 0.5,
                    "is_focus_hint_reserved_zero": 0.0,
                },
            }
        )
    return fields


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation_id: int,
    *,
    support_weight: float = 1.0,
    reliability: float = 0.5,
    normalized_time: float = 0.5,
) -> dict:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "relation_id": int(relation_id),
        "support_weight": float(support_weight),
        "metadata": {
            "reliability": float(reliability),
            "normalized_time": float(normalized_time),
        },
    }


def permute_items(items: list[dict], seed_key: str) -> list[dict]:
    rng = random.Random(
        int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16)
    )
    out = list(items)
    rng.shuffle(out)
    return out


def make_pair(
    *,
    split: str,
    family: str,
    group: int,
    relation_count: int,
) -> list[dict]:
    fields = make_fields(split, family, group)
    ids = [f["id"] for f in fields]
    seed = int(
        hashlib.sha256(
            f"{split}|{family}|{group}".encode()
        ).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    relation = rng.randrange(relation_count)
    other = (
        relation
        + 1
        + rng.randrange(max(relation_count - 1, 1))
    ) % relation_count
    if other == relation:
        other = (relation + 1) % relation_count

    base_edges: list[dict] = []
    variants: list[dict] = []

    if family == "endpoint_role":
        src, tgt = (
            (ids[0], ids[1])
            if group % 2 == 0
            else (ids[1], ids[0])
        )
        base_edges = [edge("e0", src, tgt, relation)]
        variants = [
            {
                "name": "SOURCE",
                "role": 0,
                "op": 0,
                "seq": [relation],
                "focus": src,
                "app": 0.9,
                "target": {src: 1.0},
                "control": "RELATIONAL",
            },
            {
                "name": "TARGET",
                "role": 1,
                "op": 0,
                "seq": [relation],
                "focus": src,
                "app": 0.9,
                "target": {tgt: 1.0},
                "control": "RELATIONAL",
            },
        ]

    elif family == "relation_filter":
        base_edges = [
            edge("e0", ids[0], ids[1], relation),
            edge("e1", ids[0], ids[2], other),
        ]
        variants = [
            {
                "name": "REL_A",
                "role": 1,
                "op": 0,
                "seq": [relation],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[1]: 1.0},
                "control": "RELATIONAL",
            },
            {
                "name": "REL_B",
                "role": 1,
                "op": 0,
                "seq": [other],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[2]: 1.0},
                "control": "RELATIONAL",
            },
        ]

    elif family == "multi_support_aggregate":
        base_edges = [
            edge("e0", ids[0], ids[1], relation),
            edge("e1", ids[2], ids[3], relation),
        ]
        variants = [
            {
                "name": "SOURCE_PAIR",
                "role": 0,
                "op": 2,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[0]: 0.5, ids[2]: 0.5},
                "control": "RELATIONAL",
            },
            {
                "name": "TARGET_PAIR",
                "role": 1,
                "op": 2,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[1]: 0.5, ids[3]: 0.5},
                "control": "RELATIONAL",
            },
        ]

    elif family == "ordered_path":
        base_edges = [
            edge("e0", ids[0], ids[1], relation),
            edge("e1", ids[1], ids[2], other),
            edge("e2", ids[0], ids[3], other),
            edge("e3", ids[3], ids[4], relation),
        ]
        variants = [
            {
                "name": "R1_R2",
                "role": 1,
                "op": 1,
                "seq": [relation, other],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[2]: 1.0},
                "control": "RELATIONAL",
            },
            {
                "name": "R2_R1",
                "role": 1,
                "op": 1,
                "seq": [other, relation],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[4]: 1.0},
                "control": "RELATIONAL",
            },
        ]

    elif family == "reliability_arbitration":
        variants = [
            {
                "name": "LEFT_HIGH",
                "role": 1,
                "op": 3,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[1]: 1.0},
                "control": "RELATIONAL",
                "edges": [
                    edge(
                        "e0",
                        ids[0],
                        ids[1],
                        relation,
                        reliability=0.9,
                    ),
                    edge(
                        "e1",
                        ids[2],
                        ids[3],
                        relation,
                        reliability=0.2,
                    ),
                ],
            },
            {
                "name": "RIGHT_HIGH",
                "role": 1,
                "op": 3,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[3]: 1.0},
                "control": "RELATIONAL",
                "edges": [
                    edge(
                        "e0",
                        ids[0],
                        ids[1],
                        relation,
                        reliability=0.2,
                    ),
                    edge(
                        "e1",
                        ids[2],
                        ids[3],
                        relation,
                        reliability=0.9,
                    ),
                ],
            },
        ]

    elif family == "temporal_arbitration":
        variants = [
            {
                "name": "LEFT_LATEST",
                "role": 1,
                "op": 4,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[1]: 1.0},
                "control": "RELATIONAL",
                "edges": [
                    edge(
                        "e0",
                        ids[0],
                        ids[1],
                        relation,
                        normalized_time=0.9,
                    ),
                    edge(
                        "e1",
                        ids[2],
                        ids[3],
                        relation,
                        normalized_time=0.1,
                    ),
                ],
            },
            {
                "name": "RIGHT_LATEST",
                "role": 1,
                "op": 4,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[3]: 1.0},
                "control": "RELATIONAL",
                "edges": [
                    edge(
                        "e0",
                        ids[0],
                        ids[1],
                        relation,
                        normalized_time=0.1,
                    ),
                    edge(
                        "e1",
                        ids[2],
                        ids[3],
                        relation,
                        normalized_time=0.9,
                    ),
                ],
            },
        ]

    elif family == "ambiguity_plurality":
        base_edges = [
            edge("e0", ids[0], ids[1], relation),
            edge("e1", ids[2], ids[3], relation),
        ]
        variants = [
            {
                "name": "SOURCE_PLURAL",
                "role": 0,
                "op": 2,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[0]: 0.5, ids[2]: 0.5},
                "control": "RELATIONAL",
            },
            {
                "name": "TARGET_PLURAL",
                "role": 1,
                "op": 2,
                "seq": [relation],
                "focus": None,
                "app": 0.9,
                "target": {ids[1]: 0.5, ids[3]: 0.5},
                "control": "RELATIONAL",
            },
        ]

    elif family == "fallback_defer":
        variants = [
            {
                "name": "FALLBACK",
                "role": 3,
                "op": 0,
                "seq": [relation],
                "focus": None,
                "app": 0.1,
                "target": {},
                "control": "FALLBACK",
            },
            {
                "name": "DEFER",
                "role": 3,
                "op": 0,
                "seq": [relation],
                "focus": None,
                "app": 0.5,
                "target": {},
                "control": "DEFER",
            },
        ]

    elif family == "outside_support_distractor":
        base_edges = [
            edge("e0", ids[0], ids[1], relation)
        ]
        variants = [
            {
                "name": "DISTRACTOR_LOW",
                "role": 1,
                "op": 0,
                "seq": [relation],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[1]: 1.0},
                "control": "RELATIONAL",
                "outside_meta": (0.1, 0.1),
            },
            {
                "name": "DISTRACTOR_HIGH",
                "role": 1,
                "op": 0,
                "seq": [relation],
                "focus": ids[0],
                "app": 0.9,
                "target": {ids[1]: 1.0},
                "control": "RELATIONAL",
                "outside_meta": (1.0, 1.0),
            },
        ]

    else:
        raise ValueError(family)

    rows: list[dict] = []
    field_order = permute_items(
        fields,
        f"{split}|{family}|{group}|fields",
    )

    for variant_index, variant in enumerate(variants):
        row_fields = [
            {
                **field_item,
                "metadata": dict(field_item["metadata"]),
            }
            for field_item in field_order
        ]

        if family == "outside_support_distractor":
            reliability, normalized_time = variant["outside_meta"]
            for field_item in row_fields:
                if field_item["id"] == ids[5]:
                    field_item["metadata"]["reliability"] = reliability
                    field_item["metadata"]["normalized_time"] = (
                        normalized_time
                    )

        edges = variant.get("edges", base_edges)
        edge_order = permute_items(
            edges,
            f"{split}|{family}|{group}|edges",
        )

        rows.append(
            {
                "schema": SCHEMA,
                "id": (
                    f"t1v2_{split}_{family}_"
                    f"{group:04d}_{variant_index}"
                ),
                "split": split,
                "family": family,
                "causal_group": (
                    f"{split}:{family}:{group:04d}"
                ),
                "variant": variant["name"],
                "fields": row_fields,
                "edges": edge_order,
                "field_support": {},
                "operator": {
                    "relation_sequence_id": variant["seq"],
                    "role_id": variant["role"],
                    "operation_id": variant["op"],
                    "focus_field_id": variant["focus"],
                    "context": [0.0, 0.0, 0.0, 0.0],
                    "applicability": variant["app"],
                },
                "target_distribution": variant["target"],
                "expected_control": variant["control"],
                "private_identity_data": False,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = json.loads(
        Path(args.contract).read_text(encoding="utf-8")
    )
    if (
        contract.get("schema")
        != "alice.eipm.n0.qsre-t1-curriculum-contract.v0.2"
    ):
        raise SystemExit("T1 v0.2 curriculum contract schema drift")

    rows: list[dict] = []
    for split, count in (
        ("train", int(contract["train_rows_per_family"])),
        ("dev", int(contract["dev_rows_per_family"])),
    ):
        if count % 2:
            raise SystemExit(
                "rows per family must be even for paired curriculum"
            )
        groups = count // 2
        for family in contract["families"]:
            for group in range(groups):
                rows.extend(
                    make_pair(
                        split=split,
                        family=family,
                        group=group,
                        relation_count=int(
                            contract["relation_seed_count"]
                        ),
                    )
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    print(f"rows={len(rows)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
