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
DEFAULT_REGISTRY = ROOT / "training/eipm/n0/n0_v02_teacher_bank_v0.4b.json"


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


def normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def display_path(path: Path) -> str:
    """Use repo-relative paths when possible and preserve valid external runtime paths."""
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        return str(resolved.relative_to(ROOT))
    return str(resolved)


def item_fingerprint(prompt: str, candidates: list[Any]) -> str:
    """Fingerprint the semantic item rather than a generic instruction stem.

    Prompts such as "Which pair is closest in meaning?" are legitimate reusable
    task instructions. Leakage/duplication requires the same normalized prompt
    *and* the same candidate content. This preserves the frozen-eval boundary
    without falsely rejecting independent examples that share a generic stem.
    """
    candidate_text = "\x1e".join(normalized_text(str(value)) for value in candidates)
    return normalized_text(prompt) + "\x1f" + candidate_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the governed public N0 v0.2 teacher bank.")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument(
        "--fixed-eval",
        action="append",
        default=[],
        help="Optional eval-only JSONL path; repeatable. If omitted, use registry fixed_eval_suites.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    if registry.get("private_identity_data") is not False:
        raise SystemExit("teacher bank registry must declare private_identity_data=false")
    if registry.get("private_identity_gradient_authorized") is not False:
        raise SystemExit("teacher bank registry must not authorize private identity gradients")

    fixed_specs = list(args.fixed_eval) or [str(value) for value in registry.get("fixed_eval_suites", [])]
    if not fixed_specs:
        raise SystemExit("teacher bank audit requires at least one fixed eval suite")

    fixed_rows: list[dict[str, Any]] = []
    fixed_suite_reports: list[dict[str, Any]] = []
    fixed_items: set[str] = set()
    expected_competencies: set[str] = set()

    for raw_path in fixed_specs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        rows = load_jsonl(path)
        competencies = {str(row["competency"]) for row in rows}
        for row in rows:
            if row.get("eval_only") is not True or row.get("training_authorized") is not False:
                raise SystemExit(f"fixed eval row is not eval-only: {row.get('id')} in {path}")
            candidates = list(row.get("candidates", []))
            fixed_items.add(item_fingerprint(str(row.get("prompt", "")), candidates))
            if row.get("prompt_paraphrase"):
                fixed_items.add(item_fingerprint(str(row["prompt_paraphrase"]), candidates))
        expected_competencies.update(competencies)
        fixed_rows.extend(rows)
        fixed_suite_reports.append(
            {
                "path": display_path(path),
                "rows": len(rows),
                "competencies": len(competencies),
            }
        )

    expected_count = int(registry.get("registered_competencies_expected", len(expected_competencies)))
    if len(expected_competencies) != expected_count:
        raise SystemExit(
            f"fixed readiness suites must define {expected_count} competencies, got {len(expected_competencies)}"
        )

    all_ids: set[str] = set()
    all_items: set[str] = set()
    competency_split: dict[str, Counter[str]] = defaultdict(Counter)
    principle_counts: Counter[str] = Counter()
    preferred_position_counts: Counter[int] = Counter()
    shard_reports: list[dict[str, Any]] = []
    total_rows = 0
    generation_competency_split: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    generation_rows: Counter[str] = Counter()
    failures: list[str] = []

    for shard in registry.get("shards", []):
        curriculum = (ROOT / str(shard["curriculum"])).resolve()
        manifest = (ROOT / str(shard["manifest"])).resolve()
        principle_required = bool(shard.get("principle_tag_required"))
        generation = str(shard.get("generation", "unknown"))

        row_report = validate_curriculum_rows(curriculum)
        manifest_report = validate_curriculum_manifest(curriculum, manifest)
        rows = load_jsonl(curriculum)
        if len(rows) != int(row_report["row_count"]):
            failures.append(f"row count disagreement for {curriculum}")
        if manifest_report.get("private_identity_data") is not False:
            failures.append(f"private identity flag invalid in {manifest}")
        if manifest_report.get("private_identity_gradient_authorized") not in (False, None):
            failures.append(f"private identity gradient unexpectedly authorized in {manifest}")

        local_principles: Counter[str] = Counter()
        for row in rows:
            validate_teacher_row_for_v02(row)
            row_id = str(row["id"])
            if row_id in all_ids:
                failures.append(f"cross-shard duplicate row id: {row_id}")
            all_ids.add(row_id)

            fingerprint = item_fingerprint(str(row["prompt"]), list(row["candidates"]))
            if fingerprint in all_items:
                failures.append(f"cross-shard exact duplicate semantic item: {row_id}")
            all_items.add(fingerprint)
            if fingerprint in fixed_items:
                failures.append(f"teacher semantic item copies frozen eval content: {row_id}")

            competency = str(row["competency"])
            split = str(row["split"])
            if competency not in expected_competencies:
                failures.append(f"unregistered competency {competency} in {row_id}")
            competency_split[competency][split] += 1
            generation_competency_split[generation][competency][split] += 1
            generation_rows[generation] += 1

            if principle_required:
                principle = str(row.get("principle_tag", "")).strip()
                if not principle:
                    failures.append(f"v0.4+ row missing principle_tag: {row_id}")
                else:
                    principle_counts[principle] += 1
                    local_principles[principle] += 1
                rationale = str(row.get("rationale", "")).strip()
                if len(rationale) < 48:
                    failures.append(f"v0.4+ rationale too short to teach a governing reason: {row_id}")

                expected_source = {
                    "principle_bank_v04a": "sol_authored_principle_bank_v04a",
                    "voice_principle_bank_v04b": "sol_authored_voice_principle_bank_v04b",
                    "coverage_wave_v05": "sol_authored_coverage_wave_v05",
                }.get(generation)
                if expected_source and str(row.get("source")) != expected_source:
                    failures.append(
                        f"unexpected source label for {generation}: {row_id} source={row.get('source')!r}"
                    )

                preferred = [int(index) for index in row["preferred_indices"]]
                if len(preferred) == 1:
                    preferred_position_counts[preferred[0]] += 1

        shard_reports.append(
            {
                "curriculum": display_path(curriculum),
                "rows": len(rows),
                "split_counts": row_report["split_counts"],
                "competencies": len(row_report["competency_counts"]),
                "generation": generation,
                "principle_tag_required": principle_required,
                "distinct_principle_tags": len(local_principles),
            }
        )
        total_rows += len(rows)

    registered_expected = int(registry.get("registered_rows_expected", total_rows))
    if total_rows != registered_expected:
        failures.append(f"registered row count expected {registered_expected}, got {total_rows}")

    if "v05_rows" in registry:
        expected_v05_rows = int(registry["v05_rows"])
        if generation_rows["coverage_wave_v05"] != expected_v05_rows:
            failures.append(
                f"v0.5 coverage wave expected {expected_v05_rows} rows, got {generation_rows['coverage_wave_v05']}"
            )

    missing_competencies = sorted(expected_competencies.difference(competency_split))
    if missing_competencies:
        failures.append(f"missing competencies: {missing_competencies}")

    minimum_train = int(registry.get("minimum_train_scenarios_per_competency", 1))
    minimum_dev = int(registry.get("minimum_dev_scenarios_per_competency", 1))
    if minimum_train < 1 or minimum_dev < 1:
        failures.append("per-competency train/dev minimums must be positive")

    for competency in sorted(expected_competencies):
        counts = competency_split[competency]
        if counts["train"] < minimum_train or counts["dev"] < minimum_dev:
            failures.append(
                "competency below coverage floor: "
                f"{competency} train={counts['train']}/{minimum_train} "
                f"dev={counts['dev']}/{minimum_dev}"
            )

    core_competencies = {c for c in expected_competencies if not c.startswith("VOICE-")}
    voice_competencies = {c for c in expected_competencies if c.startswith("VOICE-")}
    expected_core = int(registry.get("core_competencies", len(core_competencies)))
    expected_voice = int(registry.get("voice_competencies", len(voice_competencies)))
    if len(core_competencies) != expected_core:
        failures.append(f"core competency count expected {expected_core}, got {len(core_competencies)}")
    if len(voice_competencies) != expected_voice:
        failures.append(f"voice competency count expected {expected_voice}, got {len(voice_competencies)}")

    if generation_rows["principle_bank_v04a"] != 215:
        failures.append(
            f"v0.4a should contain 215 rows, got {generation_rows['principle_bank_v04a']}"
        )
    for competency in sorted(core_competencies):
        counts = generation_competency_split["principle_bank_v04a"][competency]
        if counts["train"] != 4 or counts["dev"] != 1:
            failures.append(
                f"v0.4a distribution mismatch for {competency}: train={counts['train']} dev={counts['dev']}"
            )

    if generation_rows["voice_principle_bank_v04b"] != 40:
        failures.append(
            f"v0.4b voice wave should contain 40 rows, got {generation_rows['voice_principle_bank_v04b']}"
        )
    for competency in sorted(voice_competencies):
        counts = generation_competency_split["voice_principle_bank_v04b"][competency]
        if counts["train"] != 4 or counts["dev"] != 1:
            failures.append(
                f"v0.4b voice distribution mismatch for {competency}: train={counts['train']} dev={counts['dev']}"
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
    coverage_gate_ok = all(
        competency_split[c]["train"] >= minimum_train
        and competency_split[c]["dev"] >= minimum_dev
        for c in expected_competencies
    )
    report = {
        "schema": "alice.eipm.n0.teacher-bank-audit.v0.5",
        "status": "PASS" if not failures else "FAIL",
        "registry": display_path(registry_path),
        "fixed_eval_suites": fixed_suite_reports,
        "registered_rows": total_rows,
        "unique_ids": len(all_ids),
        "competency_count": len(competency_split),
        "core_competencies": len(core_competencies),
        "voice_competencies": len(voice_competencies),
        "v04a_rows": generation_rows["principle_bank_v04a"],
        "v04b_voice_rows": generation_rows["voice_principle_bank_v04b"],
        "v05_rows": generation_rows["coverage_wave_v05"],
        "distinct_principle_tags_v04plus": len(principle_counts),
        "preferred_position_counts_single_preference_v04plus": dict(sorted(preferred_position_counts.items())),
        "preferred_position_fraction_single_preference_v04plus": preferred_fraction,
        "full_multitask_minimum_rows": full_minimum,
        "rows_remaining_to_full_multitask_minimum": max(0, full_minimum - total_rows),
        "minimum_train_scenarios_per_competency": minimum_train,
        "minimum_dev_scenarios_per_competency": minimum_dev,
        "coverage_gate_ok": coverage_gate_ok,
        "full_multitask_gate_open": total_rows >= full_minimum and coverage_gate_ok and not failures,
        "initial_readiness_target_rows": readiness_target,
        "rows_remaining_to_initial_readiness_target": max(0, readiness_target - total_rows),
        "voice_first_required": bool(registry.get("voice_first_required", False)),
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
