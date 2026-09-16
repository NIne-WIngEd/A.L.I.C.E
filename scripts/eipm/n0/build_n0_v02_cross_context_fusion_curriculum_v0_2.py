#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_n0_v02_cross_context_fusion_curriculum import (
    FAMILIES,
    ROWS_PER_FAMILY,
    TRAIN_PER_FAMILY,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(rows: list[dict]) -> None:
    if len(rows) != len(FAMILIES) * ROWS_PER_FAMILY:
        raise ValueError("fusion curriculum row count drift")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("fusion curriculum ids must be unique")
    family_counts = Counter(row["family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    if any(family_counts[name] != ROWS_PER_FAMILY for name, _ in FAMILIES):
        raise ValueError("fusion family coverage drift")
    expected_split = {
        "train": len(FAMILIES) * TRAIN_PER_FAMILY,
        "dev": len(FAMILIES) * (ROWS_PER_FAMILY - TRAIN_PER_FAMILY),
    }
    if dict(split_counts) != expected_split:
        raise ValueError("fusion split drift")
    for row in rows:
        if row.get("private_identity_content") is not False:
            raise ValueError("fusion curriculum crossed private boundary")
        if row.get("generated_text") is not True:
            raise ValueError("fusion synthetic provenance must be explicit")
        if row.get("data_origin") != "deterministic_public_synthetic_template":
            raise ValueError("fusion data origin drift")
        if row.get("source_authority") != "public_synthetic_training_only":
            raise ValueError("fusion source authority drift")
        target = row["target_view_distribution"]
        available = row["view_available"]
        if len(target) != 3 or len(available) != 3 or len(row["view_reliability"]) != 3:
            raise ValueError("fusion view tensors must have width 3")
        if abs(sum(target) - 1.0) > 1e-6:
            raise ValueError("target view distribution must sum to one")
        if any(mass > 0 and not ok for mass, ok in zip(target, available)):
            raise ValueError("target assigns mass to unavailable view")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = []
    for _name, builder in FAMILIES:
        for index in range(ROWS_PER_FAMILY):
            row = builder(index)
            row["generated_text"] = True
            row["data_origin"] = "deterministic_public_synthetic_template"
            row["source_authority"] = "public_synthetic_training_only"
            row["identity_authority"] = False
            rows.append(row)
    validate(rows)

    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    family_counts = Counter(row["family"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    manifest = {
        "schema": "alice.eipm.n0.v02-cross-context-fusion-curriculum.v0.2",
        "status": "COMPILED_NOT_ACTIVATED",
        "compiled_sha256": sha256_file(output),
        "rows": len(rows),
        "train_rows": split_counts["train"],
        "dev_rows": split_counts["dev"],
        "families": dict(sorted(family_counts.items())),
        "family_count": len(family_counts),
        "view_order": ["semantic", "structured", "evidence"],
        "cross_view_conflict_rows": sum(bool(row["cross_view_conflict"]) for row in rows),
        "missing_view_rows": sum(not all(row["view_available"]) for row in rows),
        "data_origin": "deterministic_public_synthetic_template",
        "source_authority": "public_synthetic_training_only",
        "identity_authority": False,
        "generated_text": True,
        "private_identity_content": False,
        "private_identity_gradient": False,
        "training_authorized": False,
        "hard_parameter_ceiling": None,
        "supersedes_compiler": "build_n0_v02_cross_context_fusion_curriculum.py",
        "supersession_reason": "correct synthetic text provenance before any gradient",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
