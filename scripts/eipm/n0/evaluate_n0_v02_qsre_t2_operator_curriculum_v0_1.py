from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    curriculum = Path(args.curriculum)
    output = Path(args.output)
    data = rows(curriculum)

    split_counts = collections.Counter(row["split"] for row in data)
    family_split = collections.Counter(
        (row["family"], row["split"]) for row in data
    )
    controls = collections.Counter(
        int(row["operator_target"]["control_id"]) for row in data
    )

    train_text = {row["query_text"] for row in data if row["split"] == "train"}
    dev_text = {row["query_text"] for row in data if row["split"] == "dev"}
    if train_text & dev_text:
        raise SystemExit("TRAIN/DEV query text overlap")

    for row in data:
        if row.get("schema") != "alice.eipm.n0.qsre-t2-operator-row.v0.1":
            raise SystemExit("row schema drift")
        if row.get("private_identity_data") is not False:
            raise SystemExit("private identity data present")
        if row["split"] not in ("train", "dev"):
            raise SystemExit("non TRAIN/DEV split present")
        text = str(row["query_text"])
        if "f_train_" in text or "f_dev_" in text or "t1v2_" in text:
            raise SystemExit("field/row identifier leaked into query")
        target = row["operator_target"]
        relations = [int(x) for x in target["relation_sequence_id"]]
        if len(relations) > 2:
            raise SystemExit("current T2 fixture relation-slot drift")
        if any(x < 0 or x >= 6 for x in relations):
            raise SystemExit("relation target outside current seed vocabulary")
        if int(target["role_id"]) not in range(4):
            raise SystemExit("role target drift")
        if int(target["operation_id"]) not in range(5):
            raise SystemExit("operation target drift")
        if int(target["control_id"]) not in range(3):
            raise SystemExit("control target drift")
        if int(target["control_id"]) != 1 and relations:
            raise SystemExit("non-relational row must use NONE relation target")

    # Two query views per original row: 360 T1 TRAIN and 144 T1 DEV.
    if split_counts != {"train": 720, "dev": 288}:
        raise SystemExit(f"unexpected split counts: {dict(split_counts)}")

    for family in sorted({row["family"] for row in data}):
        if family_split[(family, "train")] != 80:
            raise SystemExit(f"TRAIN family count drift: {family}")
        if family_split[(family, "dev")] != 32:
            raise SystemExit(f"DEV family count drift: {family}")

    outside = collections.defaultdict(list)
    for row in data:
        if row["family"] == "outside_support_distractor":
            key = (
                row["split"],
                row["source_t1_causal_group"],
                int(row["query_view"]),
            )
            outside[key].append(row["query_text"])
    for key, texts in outside.items():
        if len(texts) != 2 or len(set(texts)) != 1:
            raise SystemExit(
                f"outside-support counterfactual query drift for {key}"
            )

    result = {
        "schema": "alice.eipm.n0.qsre-t2-curriculum-result.v0.1",
        "status": "PASS_QSRE_T2_PUBLIC_OPERATOR_CURRICULUM_CONTRACT",
        "sha256": sha256(curriculum),
        "rows": len(data),
        "split_counts": dict(split_counts),
        "family_split_counts": {
            f"{family}:{split}": count
            for (family, split), count in sorted(family_split.items())
        },
        "control_counts": {str(k): v for k, v in sorted(controls.items())},
        "train_dev_query_overlap": False,
        "outside_support_same_query_counterfactual": True,
        "private_identity_data": False,
        "test_split_present": False,
        "gradient_authorized": False,
        "gpu_training_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
