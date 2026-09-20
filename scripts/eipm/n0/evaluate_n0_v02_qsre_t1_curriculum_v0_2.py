from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DIFF_TARGET_FAMILIES = {
    "endpoint_role",
    "relation_filter",
    "multi_support_aggregate",
    "ordered_path",
    "reliability_arbitration",
    "temporal_arbitration",
    "ambiguity_plurality",
}
SAME_TARGET_FAMILIES = {
    "outside_support_distractor",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def topology_signature(row: dict) -> list[tuple]:
    return sorted(
        (
            edge["id"],
            edge["source"],
            edge["target"],
            int(edge["relation_id"]),
            float(edge["support_weight"]),
        )
        for edge in row["edges"]
    )


def edge_meta(row: dict, key: str) -> dict[str, float]:
    return {
        edge["id"]: float(edge["metadata"][key])
        for edge in row["edges"]
    }


def target_positions(row: dict) -> list[int]:
    position = {
        field["id"]: index
        for index, field in enumerate(row["fields"])
    }
    return sorted(
        position[field_id]
        for field_id in row["target_distribution"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    curriculum_path = Path(args.curriculum)

    contract = json.loads(
        contract_path.read_text(encoding="utf-8")
    )
    rows = read_jsonl(curriculum_path)

    if (
        contract.get("schema")
        != "alice.eipm.n0.qsre-t1-curriculum-contract.v0.2"
    ):
        raise SystemExit("T1 v0.2 contract drift")
    if not rows:
        raise SystemExit("empty T1 v0.2 curriculum")

    family_split = Counter()
    controls = Counter()
    support_sizes: set[int] = set()
    path_lengths: set[int] = set()
    target_sizes: set[int] = set()
    relation_ids: set[int] = set()
    train_text: set[str] = set()
    dev_text: set[str] = set()
    groups: dict[str, list[dict]] = defaultdict(list)
    target_positions_by_family: dict[
        str, set[int]
    ] = defaultdict(set)

    for row in rows:
        if (
            row.get("schema")
            != "alice.eipm.n0.qsre-t1-row.v0.2"
        ):
            raise SystemExit(
                f"row schema drift: {row.get('id')}"
            )
        if row["split"] not in {"train", "dev"}:
            raise SystemExit(
                "TEST/unknown split entered T1 curriculum"
            )
        if row.get("private_identity_data") is not False:
            raise SystemExit(
                "private identity data entered T1 curriculum"
            )

        family_split[
            (row["family"], row["split"])
        ] += 1
        controls[row["expected_control"]] += 1
        groups[row["causal_group"]].append(row)

        field_ids = [
            field["id"]
            for field in row["fields"]
        ]
        if len(field_ids) != len(set(field_ids)):
            raise SystemExit(
                f"duplicate field id: {row['id']}"
            )

        field_text = {
            field["text"]
            for field in row["fields"]
        }
        if row["split"] == "train":
            train_text.update(field_text)
        else:
            dev_text.update(field_text)

        supported_edges = [
            edge
            for edge in row["edges"]
            if float(edge["support_weight"]) > 0
        ]
        support_sizes.add(len(supported_edges))
        path_lengths.add(
            len(
                row["operator"][
                    "relation_sequence_id"
                ]
            )
        )
        target_sizes.add(
            len(row["target_distribution"])
        )

        supported_fields: set[str] = set()
        for edge in supported_edges:
            if (
                edge["source"] not in field_ids
                or edge["target"] not in field_ids
            ):
                raise SystemExit(
                    f"edge endpoint missing: {row['id']}"
                )
            if float(edge["support_weight"]) not in {
                0.0,
                1.0,
            }:
                raise SystemExit(
                    f"T1 support is not membership: {row['id']}"
                )
            supported_fields.add(edge["source"])
            supported_fields.add(edge["target"])
            relation_ids.add(
                int(edge["relation_id"])
            )

        relation_ids.update(
            int(value)
            for value in row["operator"][
                "relation_sequence_id"
            ]
        )

        if row["field_support"]:
            raise SystemExit(
                f"direct field support entered T1: {row['id']}"
            )

        control = row["expected_control"]
        target = row["target_distribution"]

        if control == "RELATIONAL":
            if not target:
                raise SystemExit(
                    f"relational row has empty target: {row['id']}"
                )
            total = sum(
                float(value)
                for value in target.values()
            )
            if abs(total - 1.0) > 1e-9:
                raise SystemExit(
                    f"target does not sum to one: {row['id']}"
                )
            if not set(target).issubset(
                supported_fields
            ):
                raise SystemExit(
                    f"target escaped oracle support: {row['id']}"
                )
            target_positions_by_family[
                row["family"]
            ].update(target_positions(row))
        elif target:
            raise SystemExit(
                f"fallback/defer row asserted target: {row['id']}"
            )

    if train_text & dev_text:
        raise SystemExit(
            "TRAIN/DEV field text overlap"
        )

    for family in contract["families"]:
        if (
            family_split[(family, "train")]
            != int(
                contract[
                    "train_rows_per_family"
                ]
            )
        ):
            raise SystemExit(
                f"TRAIN count drift for {family}"
            )
        if (
            family_split[(family, "dev")]
            != int(
                contract[
                    "dev_rows_per_family"
                ]
            )
        ):
            raise SystemExit(
                f"DEV count drift for {family}"
            )

    if not {0, 1, 2, 4}.issubset(
        support_sizes
    ):
        raise SystemExit(
            f"support-cardinality coverage weak: "
            f"{sorted(support_sizes)}"
        )
    if not {1, 2}.issubset(path_lengths):
        raise SystemExit(
            "path-length coverage weak"
        )
    if not {0, 1, 2}.issubset(target_sizes):
        raise SystemExit(
            "target plurality coverage weak"
        )
    if any(
        controls[name] == 0
        for name in (
            "RELATIONAL",
            "FALLBACK",
            "DEFER",
        )
    ):
        raise SystemExit(
            "control-state coverage incomplete"
        )

    pair_counts = Counter()

    for group_id, pair in groups.items():
        if len(pair) != 2:
            raise SystemExit(
                f"{group_id}: expected pair of 2"
            )
        first, second = pair
        family = first["family"]

        if (
            second["family"] != family
            or second["split"] != first["split"]
        ):
            raise SystemExit(
                f"{group_id}: pair family/split drift"
            )

        if {
            field["id"]
            for field in first["fields"]
        } != {
            field["id"]
            for field in second["fields"]
        }:
            raise SystemExit(
                f"{group_id}: field graph drift"
            )

        if [
            field["id"]
            for field in first["fields"]
        ] != [
            field["id"]
            for field in second["fields"]
        ]:
            raise SystemExit(
                f"{group_id}: paired field permutation drift"
            )

        if (
            topology_signature(first)
            != topology_signature(second)
        ):
            raise SystemExit(
                f"{group_id}: topology drift"
            )

        pair_counts[family] += 1

        if (
            family in DIFF_TARGET_FAMILIES
            and first["target_distribution"]
            == second["target_distribution"]
        ):
            raise SystemExit(
                f"{group_id}: intervention failed "
                "to change target"
            )

        if (
            family in SAME_TARGET_FAMILIES
            and first["target_distribution"]
            != second["target_distribution"]
        ):
            raise SystemExit(
                f"{group_id}: outside-support "
                "intervention changed target"
            )

        if family == "endpoint_role":
            if (
                first["operator"]["role_id"]
                == second["operator"]["role_id"]
            ):
                raise SystemExit(
                    "endpoint role did not intervene"
                )
            if (
                first["operator"][
                    "relation_sequence_id"
                ]
                != second["operator"][
                    "relation_sequence_id"
                ]
            ):
                raise SystemExit(
                    "endpoint relation drift"
                )

        elif family == "relation_filter":
            if (
                first["operator"][
                    "relation_sequence_id"
                ]
                == second["operator"][
                    "relation_sequence_id"
                ]
            ):
                raise SystemExit(
                    "relation filter did not intervene"
                )

        elif family in {
            "multi_support_aggregate",
            "ambiguity_plurality",
        }:
            if (
                first["operator"]["role_id"]
                == second["operator"]["role_id"]
            ):
                raise SystemExit(
                    "plural role did not intervene"
                )

        elif family == "ordered_path":
            first_seq = first["operator"][
                "relation_sequence_id"
            ]
            second_seq = second["operator"][
                "relation_sequence_id"
            ]
            if first_seq != list(
                reversed(second_seq)
            ):
                raise SystemExit(
                    "ordered path is not relation-order "
                    "reversal intervention"
                )

        elif family == "reliability_arbitration":
            if (
                edge_meta(first, "reliability")
                == edge_meta(second, "reliability")
            ):
                raise SystemExit(
                    "reliability metadata did not intervene"
                )
            if (
                edge_meta(first, "normalized_time")
                != edge_meta(second, "normalized_time")
            ):
                raise SystemExit(
                    "reliability pair changed time"
                )

        elif family == "temporal_arbitration":
            if (
                edge_meta(first, "normalized_time")
                == edge_meta(second, "normalized_time")
            ):
                raise SystemExit(
                    "temporal metadata did not intervene"
                )
            if (
                edge_meta(first, "reliability")
                != edge_meta(second, "reliability")
            ):
                raise SystemExit(
                    "temporal pair changed reliability"
                )

        elif family == "fallback_defer":
            if (
                first["operator"]["applicability"]
                == second["operator"][
                    "applicability"
                ]
                or first["expected_control"]
                == second["expected_control"]
            ):
                raise SystemExit(
                    "applicability pair failed"
                )

        elif family == "outside_support_distractor":
            if first["operator"] != second["operator"]:
                raise SystemExit(
                    "outside-support operator drift"
                )

    weak_positions = {
        family: sorted(positions)
        for family, positions
        in target_positions_by_family.items()
        if len(positions) < 4
    }
    if weak_positions:
        raise SystemExit(
            "target-position coverage weak: "
            f"{weak_positions}"
        )

    receipt = {
        "schema": (
            "alice.eipm.n0."
            "qsre-t1-curriculum-result.v0.2"
        ),
        "status": (
            "PASS_QSRE_T1_PAIRED_CAUSAL_"
            "CURRICULUM_CONTRACT"
        ),
        "rows": len(rows),
        "sha256": hashlib.sha256(
            curriculum_path.read_bytes()
        ).hexdigest(),
        "family_split_counts": {
            f"{family}:{split}": count
            for (family, split), count
            in sorted(family_split.items())
        },
        "causal_pair_counts": dict(
            sorted(pair_counts.items())
        ),
        "support_cardinalities": sorted(
            support_sizes
        ),
        "relation_sequence_lengths": sorted(
            path_lengths
        ),
        "target_cardinalities": sorted(
            target_sizes
        ),
        "control_counts": dict(
            sorted(controls.items())
        ),
        "relation_ids": sorted(relation_ids),
        "target_positions_by_family": {
            key: sorted(value)
            for key, value
            in sorted(
                target_positions_by_family.items()
            )
        },
        "train_dev_field_text_overlap": False,
        "private_identity_data": False,
        "test_split_present": False,
        "gradient_authorized": False,
        "gpu_training_authorized": False,
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
