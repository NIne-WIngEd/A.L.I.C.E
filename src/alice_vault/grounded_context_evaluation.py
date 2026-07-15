from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .grounded_context import (
    _atomic_json,
    build_grounded_context,
    load_policy,
)


def evaluate_grounded_context(
    *,
    vault_root: Path,
    benchmark_path: Path,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    device: str = "auto",
    search_fn=None,
    model_loader=None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = [
        case for case in benchmark.get("cases", [])
        if str(case.get("status", "")).casefold() == "approved"
    ]
    if not cases:
        raise ValueError("Benchmark contains no approved cases")

    covered = citation_ok = contradiction_ok = 0
    details = []
    for case in cases:
        built = build_grounded_context(
            vault_root=vault_root,
            query=str(case["question"]),
            pilot_name=pilot_name,
            policy_path=policy_path,
            semantic_policy_path=semantic_policy_path,
            lexical_policy_path=lexical_policy_path,
            device=device,
            save=False,
            search_fn=search_fn,
            model_loader=model_loader,
        )
        package = built["package"]
        expected = {str(x) for x in case.get("expected_source_sha256", [])}
        returned = {str(x["source_content_sha256"]) for x in package["evidence"]}
        has_expected = bool(expected.intersection(returned))
        covered += int(has_expected)

        citations = [x["citation_id"] for x in package["evidence"]]
        good_citations = citations == [
            f"S{i}" for i in range(1, len(citations) + 1)
        ]
        citation_ok += int(good_citations)

        safe_conflicts = all(
            group.get("unresolved") is True and group.get("resolution") is None
            for group in package["contradiction_groups"]
        )
        contradiction_ok += int(safe_conflicts)

        details.append({
            "query_id": case.get("query_id"),
            "expected_source_covered": has_expected,
            "citation_integrity": good_citations,
            "contradiction_safety": safe_conflicts,
        })

    count = len(cases)
    coverage = covered / count
    run_id = str(uuid.uuid4())
    summary = {
        "grounded_context_evaluation_schema_version": 1,
        "run_id": run_id,
        "benchmark_id": benchmark.get("benchmark_id"),
        "pilot_name": pilot_name,
        "approved_case_count": count,
        "expected_source_coverage": round(coverage, 6),
        "minimum_expected_source_coverage": float(
            policy["minimum_expected_source_coverage"]
        ),
        "citation_integrity_rate": round(citation_ok / count, 6),
        "contradiction_safety_rate": round(contradiction_ok / count, 6),
        "passes_coverage_threshold": coverage >= float(
            policy["minimum_expected_source_coverage"]
        ),
        "memory_write_allowed": False,
        "answer_generation_allowed": False,
        "external_action_allowed": False,
    }

    private_root = vault_root / "manifests" / "context" / pilot_name
    exports = vault_root / "manifests" / "exports"
    details_path = private_root / f"context-evaluation-details-{run_id}.json"
    summary_path = exports / f"context-evaluation-summary-{run_id}.json"
    _atomic_json(details_path, {**summary, "cases": details})
    summary["private_details_path"] = str(details_path)
    _atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
