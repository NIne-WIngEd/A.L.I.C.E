from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


EXPECTED_SOURCE_CACHE_SHA256 = (
    "5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823"
)
CONTROL = {
    "FALLBACK": 0,
    "RELATIONAL": 1,
    "DEFER": 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def source_pool(
    cache: dict,
    split: str,
) -> tuple[torch.Tensor, list[int]]:
    field_semantic = cache[
        "field_semantic"
    ].float()
    valid_mask = cache["valid_mask"].bool()
    splits = [
        str(value)
        for value in cache["splits"]
    ]
    row_indices = [
        index
        for index, value in enumerate(splits)
        if value == split
    ]
    if not row_indices:
        raise SystemExit(
            f"source cache has no {split} rows"
        )

    values = [
        field_semantic[index][
            valid_mask[index]
        ]
        for index in row_indices
    ]
    return torch.cat(values, dim=0), row_indices


def build_split(
    rows: list[dict],
    mapping: dict[str, torch.Tensor],
) -> dict:
    count = len(rows)
    fields = max(
        len(row["fields"])
        for row in rows
    )
    edges = max(
        (
            len(row["edges"])
            for row in rows
        ),
        default=0,
    )
    steps = max(
        len(
            row["operator"][
                "relation_sequence_id"
            ]
        )
        for row in rows
    )
    field_width = next(
        iter(mapping.values())
    ).numel()

    field_state = torch.zeros(
        count,
        fields,
        field_width,
    )
    field_metadata = torch.zeros(
        count,
        fields,
        3,
    )
    field_valid_mask = torch.zeros(
        count,
        fields,
        dtype=torch.bool,
    )

    edge_index = torch.zeros(
        count,
        edges,
        2,
        dtype=torch.long,
    )
    edge_relation_id = torch.zeros(
        count,
        edges,
        dtype=torch.long,
    )
    edge_metadata = torch.zeros(
        count,
        edges,
        2,
    )
    edge_valid_mask = torch.zeros(
        count,
        edges,
        dtype=torch.bool,
    )

    edge_support_weight = torch.zeros(
        count,
        edges,
    )
    field_support_weight = torch.zeros(
        count,
        fields,
    )

    relation_sequence_id = torch.zeros(
        count,
        steps,
        dtype=torch.long,
    )
    relation_sequence_mask = torch.zeros(
        count,
        steps,
        dtype=torch.bool,
    )
    role_id = torch.zeros(
        count,
        dtype=torch.long,
    )
    operation_id = torch.zeros(
        count,
        dtype=torch.long,
    )
    focus_field_weight = torch.zeros(
        count,
        fields,
    )
    operator_context = torch.zeros(
        count,
        4,
    )
    applicability = torch.zeros(count)

    target_distribution = torch.zeros(
        count,
        fields,
    )
    expected_control = torch.zeros(
        count,
        dtype=torch.long,
    )

    ids: list[str] = []
    families: list[str] = []
    causal_groups: list[str] = []
    variants: list[str] = []
    field_ids: list[list[str]] = []

    for row_index, row in enumerate(rows):
        ids.append(row["id"])
        families.append(row["family"])
        causal_groups.append(
            row["causal_group"]
        )
        variants.append(row["variant"])

        position: dict[str, int] = {}
        row_field_ids = [
            field["id"]
            for field in row["fields"]
        ]
        field_ids.append(row_field_ids)

        for field_index, field in enumerate(
            row["fields"]
        ):
            field_id = field["id"]
            position[field_id] = field_index
            field_state[
                row_index,
                field_index,
            ] = mapping[field_id]
            field_metadata[
                row_index,
                field_index,
            ] = torch.tensor(
                [
                    float(
                        field["metadata"][
                            "reliability"
                        ]
                    ),
                    float(
                        field["metadata"][
                            "normalized_time"
                        ]
                    ),
                    float(
                        field["metadata"][
                            "is_focus_hint_reserved_zero"
                        ]
                    ),
                ]
            )
            field_valid_mask[
                row_index,
                field_index,
            ] = True

        for edge_i, edge in enumerate(
            row["edges"]
        ):
            edge_index[
                row_index,
                edge_i,
                0,
            ] = position[edge["source"]]
            edge_index[
                row_index,
                edge_i,
                1,
            ] = position[edge["target"]]
            edge_relation_id[
                row_index,
                edge_i,
            ] = int(edge["relation_id"])
            edge_metadata[
                row_index,
                edge_i,
            ] = torch.tensor(
                [
                    float(
                        edge["metadata"][
                            "reliability"
                        ]
                    ),
                    float(
                        edge["metadata"][
                            "normalized_time"
                        ]
                    ),
                ]
            )
            edge_valid_mask[
                row_index,
                edge_i,
            ] = True

            support = float(
                edge["support_weight"]
            )
            if support not in (0.0, 1.0):
                raise SystemExit(
                    f"{row['id']}: "
                    "T1 support is not membership"
                )
            edge_support_weight[
                row_index,
                edge_i,
            ] = support

        if row["field_support"]:
            raise SystemExit(
                f"{row['id']}: "
                "direct field support entered T1"
            )

        sequence = row["operator"][
            "relation_sequence_id"
        ]
        relation_sequence_id[
            row_index,
            : len(sequence),
        ] = torch.tensor(
            sequence,
            dtype=torch.long,
        )
        relation_sequence_mask[
            row_index,
            : len(sequence),
        ] = True

        role_id[row_index] = int(
            row["operator"]["role_id"]
        )
        operation_id[row_index] = int(
            row["operator"][
                "operation_id"
            ]
        )

        focus_id = row["operator"][
            "focus_field_id"
        ]
        if focus_id is not None:
            focus_field_weight[
                row_index,
                position[focus_id],
            ] = 1.0

        context = row["operator"]["context"]
        if len(context) != 4:
            raise SystemExit(
                "operator context width drift"
            )
        operator_context[row_index] = (
            torch.tensor(
                context,
                dtype=torch.float32,
            )
        )
        applicability[row_index] = float(
            row["operator"][
                "applicability"
            ]
        )

        for field_id, weight in row[
            "target_distribution"
        ].items():
            target_distribution[
                row_index,
                position[field_id],
            ] = float(weight)

        expected_control[row_index] = (
            CONTROL[
                row["expected_control"]
            ]
        )

    return {
        "field_state": field_state,
        "field_metadata": field_metadata,
        "field_valid_mask": field_valid_mask,
        "edge_index": edge_index,
        "edge_relation_id": edge_relation_id,
        "edge_metadata": edge_metadata,
        "edge_valid_mask": edge_valid_mask,
        "field_support_weight": (
            field_support_weight
        ),
        "edge_support_weight": (
            edge_support_weight
        ),
        "relation_sequence_id": (
            relation_sequence_id
        ),
        "relation_sequence_mask": (
            relation_sequence_mask
        ),
        "role_id": role_id,
        "operation_id": operation_id,
        "focus_field_weight": (
            focus_field_weight
        ),
        "operator_context": (
            operator_context
        ),
        "applicability": applicability,
        "target_distribution": (
            target_distribution
        ),
        "expected_control": expected_control,
        "ids": ids,
        "families": families,
        "causal_groups": causal_groups,
        "variants": variants,
        "field_ids": field_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curriculum",
        required=True,
    )
    parser.add_argument(
        "--source-cache",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--receipt",
        required=True,
    )
    parser.add_argument(
        "--expected-source-cache-sha256",
        default=EXPECTED_SOURCE_CACHE_SHA256,
    )
    args = parser.parse_args()

    curriculum_path = Path(
        args.curriculum
    )
    source_path = Path(
        args.source_cache
    )
    output_path = Path(
        args.output
    )
    receipt_path = Path(
        args.receipt
    )

    if (
        not curriculum_path.is_file()
        or not source_path.is_file()
    ):
        raise SystemExit(
            "missing T1 preparation input"
        )

    source_sha = sha256(source_path)
    if (
        args.expected_source_cache_sha256
        and source_sha
        != args.expected_source_cache_sha256
    ):
        raise SystemExit(
            "source cache hash drift: "
            f"{source_sha}"
        )

    rows = read_jsonl(curriculum_path)
    if not rows:
        raise SystemExit(
            "empty T1 curriculum"
        )
    if any(
        row["split"] not in {
            "train",
            "dev",
        }
        for row in rows
    ):
        raise SystemExit(
            "T1 curriculum split drift"
        )
    if any(
        row.get("private_identity_data")
        is not False
        for row in rows
    ):
        raise SystemExit(
            "private identity data entered T1"
        )

    cache = torch.load(
        source_path,
        map_location="cpu",
    )
    for key in (
        "field_semantic",
        "valid_mask",
        "splits",
    ):
        if key not in cache:
            raise SystemExit(
                f"source cache missing {key}"
            )

    mappings: dict[
        str,
        dict[str, torch.Tensor],
    ] = {}
    pool_stats: dict[str, dict] = {}

    for split, seed in (
        ("train", 20260920),
        ("dev", 20260921),
    ):
        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]
        unique_ids = sorted(
            {
                field["id"]
                for row in split_rows
                for field in row["fields"]
            }
        )

        pool, source_rows = source_pool(
            cache,
            split,
        )

        if pool.size(0) < len(unique_ids):
            raise SystemExit(
                f"{split}: insufficient source "
                "representations "
                f"{pool.size(0)} < {len(unique_ids)}"
            )

        generator = (
            torch.Generator()
            .manual_seed(seed)
        )
        permutation = torch.randperm(
            pool.size(0),
            generator=generator,
        )
        selected = pool[
            permutation[: len(unique_ids)]
        ].clone()

        mappings[split] = {
            field_id: selected[index]
            for index, field_id
            in enumerate(unique_ids)
        }

        pool_stats[split] = {
            "available": int(
                pool.size(0)
            ),
            "used": len(unique_ids),
            "source_rows": len(
                source_rows
            ),
        }

    if (
        set(mappings["train"])
        & set(mappings["dev"])
    ):
        raise SystemExit(
            "T1 field id split overlap"
        )

    payload = {
        "schema": (
            "alice.eipm.n0."
            "qsre-t1-real-cache.v0.1"
        ),
        "source_cache_sha256": source_sha,
        "curriculum_sha256": sha256(
            curriculum_path
        ),
        "field_state_source": (
            "field_semantic"
        ),
        "field_state_dim": int(
            cache["field_semantic"].shape[
                -1
            ]
        ),
        "train": build_split(
            [
                row
                for row in rows
                if row["split"] == "train"
            ],
            mappings["train"],
        ),
        "dev": build_split(
            [
                row
                for row in rows
                if row["split"] == "dev"
            ],
            mappings["dev"],
        ),
        "test_present": False,
        "private_identity_data": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        payload,
        output_path,
    )

    for split in ("train", "dev"):
        prepared = payload[split]
        by_group: dict[
            str,
            list[int],
        ] = defaultdict(list)

        for index, group in enumerate(
            prepared["causal_groups"]
        ):
            by_group[group].append(index)

        for group, indices in by_group.items():
            if len(indices) != 2:
                raise SystemExit(
                    f"{group}: "
                    "prepared pair count drift"
                )
            first, second = indices

            if (
                prepared["field_ids"][first]
                != prepared["field_ids"][
                    second
                ]
            ):
                raise SystemExit(
                    f"{group}: "
                    "paired field order drift"
                )

            if not torch.equal(
                prepared["field_state"][
                    first
                ],
                prepared["field_state"][
                    second
                ],
            ):
                raise SystemExit(
                    f"{group}: "
                    "paired representations drift"
                )

    receipt = {
        "schema": (
            "alice.eipm.n0."
            "qsre-t1-real-cache-result.v0.1"
        ),
        "status": (
            "PASS_QSRE_T1_REAL_CACHE_PREPARATION"
        ),
        "source_cache_sha256": (
            source_sha
        ),
        "curriculum_sha256": sha256(
            curriculum_path
        ),
        "prepared_cache_sha256": (
            sha256(output_path)
        ),
        "field_state_dim": (
            payload["field_state_dim"]
        ),
        "rows": {
            "train": len(
                payload["train"]["ids"]
            ),
            "dev": len(
                payload["dev"]["ids"]
            ),
        },
        "representation_pool": pool_stats,
        "paired_representation_identity": (
            True
        ),
        "train_dev_representation_pool_isolated": (
            True
        ),
        "field_support_all_zero": bool(
            (
                payload["train"][
                    "field_support_weight"
                ]
                == 0
            ).all()
            and (
                payload["dev"][
                    "field_support_weight"
                ]
                == 0
            ).all()
        ),
        "edge_support_membership_only": (
            True
        ),
        "test_present": False,
        "private_identity_data": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
    }

    receipt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    receipt_path.write_text(
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
