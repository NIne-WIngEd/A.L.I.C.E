#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "alice.eipm.n0.query-edge-preservation-train-anchors.v0.1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_name(row: dict[str, Any]) -> str:
    value = row.get("split")
    if value is None:
        value = row.get("intended_split")
    if value is None:
        raise SystemExit(f"source row lacks split: {row.get('id')}")
    return str(value)


def row_id(row: dict[str, Any]) -> str:
    value = row.get("id")
    if value is None:
        raise SystemExit("source row lacks id")
    return str(value)


def compile_source(path: Path, label: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"empty preservation source: {label}")

    train_ids = [row_id(row) for row in rows if split_name(row) == "train"]
    dev_ids = [row_id(row) for row in rows if split_name(row) == "dev"]
    test_ids = [
        row_id(row)
        for row in rows
        if split_name(row) not in {"train", "dev"}
    ]
    if not train_ids:
        raise SystemExit(f"no train anchors available: {label}")
    if len(train_ids) != len(set(train_ids)):
        raise SystemExit(f"duplicate train anchor ids: {label}")

    return {
        "label": label,
        "source_path_basename": path.name,
        "source_sha256": sha256(path),
        "source_rows": len(rows),
        "train_anchor_ids": train_ids,
        "train_anchor_count": len(train_ids),
        "dev_ids_explicitly_excluded": dev_ids,
        "dev_excluded_count": len(dev_ids),
        "other_nontrain_ids_explicitly_excluded": test_ids,
        "other_nontrain_excluded_count": len(test_ids),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ordinary-source", required=True)
    p.add_argument("--endpoint-source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ordinary = Path(args.ordinary_source).resolve()
    endpoint = Path(args.endpoint_source).resolve()
    output = Path(args.output).resolve()
    for path in (ordinary, endpoint):
        if not path.is_file():
            raise SystemExit(f"missing preservation source: {path}")
    if output.exists():
        raise SystemExit("refusing to overwrite preservation anchor manifest")
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": SCHEMA,
        "status": "COMPILED_TRAIN_ONLY_PRESERVATION_ANCHORS_NO_GRADIENT_AUTHORIZATION",
        "ordinary": compile_source(ordinary, "ordinary_graph"),
        "endpoint": compile_source(endpoint, "endpoint_role"),
        "gradient_eligibility": {
            "train_anchors_may_be_used_only_after_separate_training_decision": True,
            "dev_rows_may_be_used_for_gradient": False,
            "heldout_or_test_rows_may_be_used_for_gradient": False,
            "frozen_challenge_rows_may_be_used_for_gradient": False,
            "private_identity_rows_may_be_used_for_gradient": False,
        },
        "intended_future_role": (
            "preservation training anchors for frozen-parent distillation or "
            "routing stability; mechanism not yet selected"
        ),
        "optimizer_authorized": False,
        "gradient_authorized": False,
        "gpu_training_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "private_identity_gradient": False,
        "n0_complete": False,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
