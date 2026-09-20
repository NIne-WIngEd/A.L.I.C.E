from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.curriculum))
    if not rows:
        raise SystemExit("empty T1 curriculum")

    family_split = Counter()
    support_sizes = set()
    path_lengths = set()
    target_sizes = set()
    controls = Counter()
    relation_ids = set()
    train_field_tokens = set()
    dev_field_tokens = set()
    family_targets = defaultdict(int)

    for row in rows:
        if row.get("schema") != "alice.eipm.n0.qsre-t1-row.v0.1":
            raise SystemExit(f"row schema drift: {row.get('id')}")
        if row["split"] not in {"train", "dev"}:
            raise SystemExit("TEST/unknown split entered T1 curriculum")
        if row.get("private_identity_data") is not False:
            raise SystemExit("private identity data entered T1 curriculum")

        family_split[(row["family"], row["split"])] += 1
        controls[row["expected_control"]] += 1

        field_ids = {f["id"] for f in row["fields"]}
        if len(field_ids) != len(row["fields"]):
            raise SystemExit(f"duplicate field id: {row['id']}")
        for f in row["fields"]:
            token = f["text"]
            (train_field_tokens if row["split"] == "train" else dev_field_tokens).add(token)

        supported_edges = [e for e in row["edges"] if float(e["support_weight"]) > 0]
        support_sizes.add(len(supported_edges))
        path_lengths.add(len(row["operator"]["relation_sequence_id"]))
        target_sizes.add(len(row["target_distribution"]))

        supported_fields = set()
        for e in supported_edges:
            if e["source"] not in field_ids or e["target"] not in field_ids:
                raise SystemExit(f"edge endpoint missing: {row['id']}")
            supported_fields.add(e["source"])
            supported_fields.add(e["target"])
            relation_ids.add(int(e["relation_id"]))

        relation_ids.update(int(x) for x in row["operator"]["relation_sequence_id"])

        control = row["expected_control"]
        target = row["target_distribution"]
        if control == "RELATIONAL":
            if not target:
                raise SystemExit(f"relational row has empty target: {row['id']}")
            total = sum(float(v) for v in target.values())
            if abs(total - 1.0) > 1e-9:
                raise SystemExit(f"target does not sum to one: {row['id']}")
            if not set(target).issubset(supported_fields):
                raise SystemExit(f"target escaped oracle support: {row['id']}")
            family_targets[row["family"]] += 1
        else:
            if target:
                raise SystemExit(f"fallback/defer row asserted relational target: {row['id']}")

        if row["family"] == "outside_support_distractor":
            outside = field_ids - supported_fields
            if not outside:
                raise SystemExit("outside-support distractor family lost distractor")

    if train_field_tokens & dev_field_tokens:
        raise SystemExit("TRAIN/DEV field text overlap")

    expected_families = set(contract["families"])
    if {family for family, _split in family_split} != expected_families:
        raise SystemExit("family coverage drift")

    for family in contract["families"]:
        if family_split[(family, "train")] != int(contract["train_rows_per_family"]):
            raise SystemExit(f"TRAIN count drift for {family}")
        if family_split[(family, "dev")] != int(contract["dev_rows_per_family"]):
            raise SystemExit(f"DEV count drift for {family}")

    if not {0, 1, 2, 4}.issubset(support_sizes):
        raise SystemExit(f"support-cardinality coverage weak: {sorted(support_sizes)}")
    if not {1, 2}.issubset(path_lengths):
        raise SystemExit(f"path-length coverage weak: {sorted(path_lengths)}")
    if not {0, 1, 2}.issubset(target_sizes):
        raise SystemExit(f"target plurality coverage weak: {sorted(target_sizes)}")
    if controls["FALLBACK"] == 0 or controls["DEFER"] == 0 or controls["RELATIONAL"] == 0:
        raise SystemExit("control-state coverage incomplete")

    receipt = {
        "schema": "alice.eipm.n0.qsre-t1-curriculum-result.v0.1",
        "status": "PASS_QSRE_T1_PUBLIC_CURRICULUM_CONTRACT",
        "rows": len(rows),
        "family_split_counts": {
            f"{family}:{split}": count
            for (family, split), count in sorted(family_split.items())
        },
        "support_cardinalities": sorted(support_sizes),
        "relation_sequence_lengths": sorted(path_lengths),
        "target_cardinalities": sorted(target_sizes),
        "control_counts": dict(sorted(controls.items())),
        "relation_ids": sorted(relation_ids),
        "train_dev_field_text_overlap": False,
        "private_identity_data": False,
        "test_split_present": False,
        "gradient_authorized": False,
        "gpu_training_authorized": False,
    }
    Path(args.output).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
