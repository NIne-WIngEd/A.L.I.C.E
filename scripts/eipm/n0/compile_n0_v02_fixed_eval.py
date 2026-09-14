#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED = {
    "id",
    "competency",
    "prompt",
    "candidates",
    "preferred_indices",
    "paraphrase_group",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED.difference(row)
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"duplicate benchmark id: {row_id}")
        seen.add(row_id)
        candidates = list(row["candidates"])
        preferred = [int(x) for x in row["preferred_indices"]]
        if len(candidates) < 2 or not preferred:
            raise ValueError(f"invalid candidates/preferred set for {row_id}")
        if min(preferred) < 0 or max(preferred) >= len(candidates):
            raise ValueError(f"preferred index out of range for {row_id}")
        if row.get("training_authorized") is True:
            raise ValueError(f"fixed benchmark row must never authorize training: {row_id}")
        rows.append(row)
    if not rows:
        raise ValueError("fixed benchmark base is empty")
    return rows


def rotate_row(row: dict[str, Any], shift: int) -> dict[str, Any]:
    candidates = list(row["candidates"])
    n = len(candidates)
    shift %= n
    order = list(range(n))[shift:] + list(range(n))[:shift]
    inverse = {old: new for new, old in enumerate(order)}
    compiled = dict(row)
    compiled["id"] = f"{row['id']}.order{shift}"
    compiled["base_id"] = row["id"]
    compiled["candidate_order_variant"] = shift
    compiled["candidates"] = [candidates[index] for index in order]
    compiled["preferred_indices"] = sorted(inverse[int(index)] for index in row["preferred_indices"])
    compiled["eval_only"] = True
    compiled["training_authorized"] = False
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the frozen N0 v0.2 readiness suite.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--order-variants", type=int, default=3)
    parser.add_argument("--minimum-cases-per-competency", type=int, default=2)
    args = parser.parse_args()

    base = Path(args.base)
    output = Path(args.output)
    manifest = Path(args.manifest)
    rows = load_base(base)

    counts = Counter(str(row["competency"]) for row in rows)
    under = {name: count for name, count in counts.items() if count < args.minimum_cases_per_competency}
    if under:
        raise ValueError(f"benchmark competencies below minimum case count: {under}")

    paraphrase_counts = Counter(str(row["paraphrase_group"]) for row in rows)
    weak_groups = sorted(name for name, count in paraphrase_counts.items() if count < 2)
    if weak_groups:
        raise ValueError(f"paraphrase groups must contain at least two cases: {weak_groups[:10]}")

    compiled: list[dict[str, Any]] = []
    for row in rows:
        variants = min(max(args.order_variants, 1), len(row["candidates"]))
        for shift in range(variants):
            compiled.append(rotate_row(row, shift))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in compiled),
        encoding="utf-8",
    )
    receipt = {
        "schema": "alice.eipm.n0.fixed-readiness-receipt.v0.2",
        "base_file": str(base),
        "base_sha256": sha256_file(base),
        "compiled_file": str(output),
        "compiled_sha256": sha256_file(output),
        "base_rows": len(rows),
        "compiled_rows": len(compiled),
        "competencies": dict(sorted(counts.items())),
        "paraphrase_groups": len(paraphrase_counts),
        "candidate_order_variants": args.order_variants,
        "eval_only": True,
        "training_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
