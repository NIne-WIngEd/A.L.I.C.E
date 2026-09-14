#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum import (
    validate_curriculum_manifest,
    validate_curriculum_rows,
)
from alice_personality.n0.v02_objectives import validate_teacher_row_for_v02


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "training/eipm/n0/n0_v02_teacher_bank_v0.4a.json"
DEFAULT_FIXED = ROOT / "evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def normalized_prompt(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the governed public N0 v0.2 teacher bank.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--fixed-eval", default=str(DEFAULT_FIXED))
    parser.add_argument("--output")
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    fixed_path = Path(args.fixed_eval).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    if registry.get("private_identity_data") is not False:
        raise SystemExit("teacher bank registry must declare private_identity_data=false")
    if registry.get("private_identity_gradient_authorized") is not False:
        raise SystemExit("teacher bank registry must not authorize private identity gradients")

    fixed_rows = load_jsonl(fixed_path)
    fixed_prompts = {
        normalized_prompt(str(row.get("prompt", "")))
        for row in fixed_rows
    } | {
        normalized_prompt(str(row.get("prompt_paraphrase", "")))
        for row in fixed_rows
        if row.get("prompt_paraphrase")
    }
    expected_competencies = {str(row["competency"]) for row in fixed_rows}
    if len(expected_competencies) != 43:
        raise SystemExit(f"fixed readiness base must define 43 competencies, got {len(expected_competencies)}")

    all_ids: set[str] = set()
    all_prompts: set[str] = set()
    competency_split: dict[str, Counter[str]] = defaultdict(Counter)
    principle_counts: Counter[str] = Counter()
    preferred_position_counts: Counter[int] = Counter()
    shard_reports: list[dict[str, Any]] = []
    total_rows = 0
    v04_rows = 0
    v04_competency_split: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[str] = []

    for shard in registry.get("shards", []):
        curriculum = (ROOT / str(shard["curriculum"])).resolve()
        manifest = (ROOT / str(shard["manifest"])).resolve()
        principle_required = bool(shard.get("principle_tag_required"))

        row_report = validate_curriculum_rows(curriculum)
        manifest_report = validate_curriculum_manifest(curriculum, manifest)
        rows = load_jsonl(curriculum)
        if len(rows) != int(row_report["row_count"]):
            failures.append(f"row count disagreement for {curriculum}")
        if manifest_report.get("private_identity_data") is not False:
            failures.append(f"private identity flag invalid in {manifest}")
        if manifest_report.get("private_identity_gradient_authorized") not in (False, None):
            failures.append(f"private identity gradient unexpectedly authorized in {manifest}")

        local_ids: set[str] = set()
        local_principles: Counter[str] = Counter()
        for row in rows:
            validate_teacher_row_for_v02(row)
            row_id = str(row["id"])
            if row_id in all_ids:
                failures.append(f"cross-shard duplicate row id: {row_id}")
            all_ids.add(row_id)
            local_ids.add(row_id)

            prompt = normalized_prompt(str(row["prompt"]))
            if prompt in all_prompts:
                failures.append(f"cross-shard exact duplicate prompt: {row_id}")
            all_prompts.add(prompt)
            if prompt in fixed_prompts:
                failures.append(f"teacher prompt copies frozen eval wording: {row_id}")

            competency = str(row["competency"])
            split = str(row["split"])
            if competency not in expected_competencies:
                failures.append(f"unregistered competency {competency} in {row_id}")
            competency_split[competency][split] += 1

            if principle_required:
                v04_rows += 1
                v04_competency_split[competency][split] += 1
                principle = str(row.get("principle_tag", "")).strip()
                if not principle:
                    failures.append(f"v0.4+ row missing principle_tag: {row_id}")
                else:
                    principle_counts[principle] += 1
                    local_principles[principle] += 1
                rationale = str(row.get("rationale", "")).strip()
                if len(rationale) < 48:
                    failures.append(f"v0.4+ rationale too short to teach a governing reason: {row_id}")
                if str(row.get("source")) != "sol_authored_principle_bank_v04a":
                    failures.append(f"unexpected v0.4a source label: {row_id}")

                preferred = [int(index) for index in row["preferred_indices"]]
                if len(preferred) == 1:
                    preferred_position_counts[preferred[0]] += 1

        shard_reports.append(
            {
                "curriculum": str(curriculum.relative_to(ROOT)),
                "rows": len(rows),
                "split_counts": row_report["split_counts"],
                "competencies": len(row_report["competency_counts"]),
                "principle_tag_required": principle_required,
                "distinct_principle_tags": len(local_principles),
            }
        )
        total_rows += len(rows)

    missing_competencies = sorted(expected_competencies.difference(competency_split))
    if missing_competencies:
        failures.append(f"missing competencies: {missing_competencies}")

    for competency in sorted(expected_competencies):
        counts = competency_split[competency]
        if counts["train"] < 1 or counts["dev"] < 1:
            failures.append(
                f"competency lacks train/dev support: {competency} train={counts['train']} dev={counts['dev']}"
            )

    # Wave 0.4a is intentionally a first principle-bank layer: exactly four train
    # and one independently authored dev row for each of the 43 competencies.
    if v04_rows != 215:
        failures.append(f"v0.4a should contain 215 rows, got {v04_rows}")
    for competency in sorted(expected_competencies):
        counts = v04_competency_split[competency]
        if counts["train"] != 4 or counts["dev"] != 1:
            failures.append(
                f"v0.4a distribution mismatch for {competency}: train={counts['train']} dev={counts['dev']}"
            )

    one_off_principles = sorted(tag for tag, count in principle_counts.items() if count < 2)
    if one_off_principles:
        failures.append(f"principle tags must be reusable, one-off tags={one_off_principles}")

    single_preferred_total = sum(preferred_position_counts.values())
    preferred_fraction = {
        str(index): (count / single_preferred_total if single_preferred_total else 0.0)
        for index, count in sorted(preferred_position_counts.items())
    }
    if single_preferred_total:
        dominant = max(preferred_fraction.values())
        if dominant > 0.60:
            failures.append(
                f"preferred answer position is too concentrated: max_fraction={dominant:.4f}"
            )

    full_minimum = int(registry.get("full_multitask_minimum_rows", 1000))
    readiness_target = int(registry.get("initial_readiness_target_rows", 2500))
    report = {
        "schema": "alice.eipm.n0.teacher-bank-audit.v0.4a",
        "status": "PASS" if not failures else "FAIL",
        "registry": str(registry_path.relative_to(ROOT)),
        "registered_rows": total_rows,
        "unique_ids": len(all_ids),
        "competency_count": len(competency_split),
        "v04a_rows": v04_rows,
        "v04a_distinct_principle_tags": len(principle_counts),
        "preferred_position_counts_single_preference_v04a": dict(sorted(preferred_position_counts.items())),
        "preferred_position_fraction_single_preference_v04a": preferred_fraction,
        "full_multitask_minimum_rows": full_minimum,
        "rows_remaining_to_full_multitask_minimum": max(0, full_minimum - total_rows),
        "full_multitask_gate_open": total_rows >= full_minimum and not failures,
        "initial_readiness_target_rows": readiness_target,
        "rows_remaining_to_initial_readiness_target": max(0, readiness_target - total_rows),
        "private_identity_data": False,
        "private_identity_gradient_authorized": False,
        "competency_split_counts": {
            competency: dict(sorted(counts.items()))
            for competency, counts in sorted(competency_split.items())
        },
        "shards": shard_reports,
        "failures": failures,
    }

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if failures:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
