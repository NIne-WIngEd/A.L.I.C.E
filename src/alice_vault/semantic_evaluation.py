from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .retrieval import (
    SearchFilters,
    atomic_json,
    load_chunk_catalog,
    locate_chunk_set,
    search_index,
)
from .semantic_retrieval import (
    hybrid_search,
    load_semantic_policy,
    semantic_search,
)


BENCHMARK_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1


def _read_questions(path: Path) -> list[str]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            output.append(value)
    if not output:
        raise ValueError("Question file contains no questions")
    return output


def create_semantic_benchmark_draft(
    *,
    vault_root: Path,
    questions_path: Path,
    pilot_name: str = "pilot-v1",
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    device: str = "auto",
    model_loader=None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    questions_path = questions_path.expanduser().resolve(
        strict=True
    )
    policy = load_semantic_policy(semantic_policy_path)
    questions = _read_questions(questions_path)
    benchmark_id = str(uuid.uuid4())
    cases: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        result = hybrid_search(
            vault_root=vault_root,
            query=question,
            pilot_name=pilot_name,
            semantic_policy_path=semantic_policy_path,
            lexical_policy_path=lexical_policy_path,
            limit=(
                policy.benchmark.candidate_sources_per_question
            ),
            device=device,
            model_loader=model_loader,
        )
        candidates = []
        for item in result["results"]:
            candidates.append(
                {
                    "rank": item["rank"],
                    "source_content_sha256": item[
                        "source_content_sha256"
                    ],
                    "family": item["family"],
                    "filenames": sorted(
                        {
                            str(provenance["filename"])
                            for provenance in item["provenance"]
                        }
                    ),
                    "rrf_score": item["rrf_score"],
                }
            )
        cases.append(
            {
                "query_id": f"personal-{index:03d}",
                "question": question,
                "status": "pending",
                "expected_source_sha256": [],
                "candidate_sources": candidates,
                "review_notes": "",
            }
        )

    benchmark = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_type": "human_curated_personal_semantic",
        "pilot_name": pilot_name,
        "instructions": {
            "status": "Set each case to approved or excluded.",
            "expected_source_sha256": (
                "For approved cases, copy one or more correct source "
                "hashes from candidate_sources. Search manually if no "
                "candidate is correct."
            ),
            "privacy": (
                "This file is private and must remain inside the vault."
            ),
        },
        "case_count": len(cases),
        "cases": cases,
    }
    private_root = (
        vault_root / "manifests" / "semantic" / pilot_name
    )
    exports = vault_root / "manifests" / "exports"
    private_root.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    benchmark_path = (
        private_root
        / f"semantic-benchmark-draft-{benchmark_id}.json"
    )
    atomic_json(benchmark_path, benchmark)
    summary = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "benchmark_type": benchmark["benchmark_type"],
        "pilot_name": pilot_name,
        "question_count": len(cases),
        "pending_review_count": len(cases),
        "benchmark_path": str(benchmark_path),
    }
    summary_path = (
        exports
        / f"semantic-benchmark-draft-summary-{benchmark_id}.json"
    )
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def validate_semantic_benchmark(
    *,
    vault_root: Path,
    benchmark_path: Path,
    semantic_policy_path: Path | None = None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(
        strict=True
    )
    policy = load_semantic_policy(semantic_policy_path)
    benchmark = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )
    cases = list(benchmark.get("cases", []))
    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=str(benchmark.get("pilot_name", "pilot-v1")),
    )
    _, chunk_records = load_chunk_catalog(chunk_root)
    valid_sources = {
        str(record["source_content_sha256"])
        for record in chunk_records
    }
    errors: list[str] = []
    approved = 0
    excluded = 0
    pending = 0
    query_ids: set[str] = set()

    for index, case in enumerate(cases, start=1):
        query_id = str(case.get("query_id", "")).strip()
        question = str(case.get("question", "")).strip()
        status = str(case.get("status", "")).strip().lower()
        expected = [
            str(value).strip()
            for value in case.get(
                "expected_source_sha256",
                [],
            )
            if str(value).strip()
        ]
        if not query_id or query_id in query_ids:
            errors.append(
                f"Case {index} has a missing or duplicate query_id"
            )
        query_ids.add(query_id)
        if not question:
            errors.append(f"Case {index} has no question")
        if status == "approved":
            approved += 1
            if not expected:
                errors.append(
                    f"Approved case {query_id} has no expected source"
                )
            unknown = sorted(set(expected).difference(valid_sources))
            if unknown:
                errors.append(
                    f"Approved case {query_id} references unknown "
                    f"sources: {unknown}"
                )
        elif status == "excluded":
            excluded += 1
        elif status == "pending":
            pending += 1
        else:
            errors.append(
                f"Case {query_id} has invalid status {status!r}"
            )

    if approved < policy.benchmark.minimum_approved_cases:
        errors.append(
            f"Only {approved} approved cases; minimum is "
            f"{policy.benchmark.minimum_approved_cases}"
        )

    return {
        "benchmark_validation_schema_version": 1,
        "benchmark_id": benchmark.get("benchmark_id"),
        "case_count": len(cases),
        "approved_case_count": approved,
        "excluded_case_count": excluded,
        "pending_case_count": pending,
        "minimum_approved_cases": (
            policy.benchmark.minimum_approved_cases
        ),
        "error_count": len(errors),
        "errors": errors,
        "ready_for_evaluation": not errors,
    }


def _metrics(
    cases: list[dict[str, Any]],
    search: Callable[[str, int], list[str]],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    hits = Counter()
    reciprocal_ranks: list[float] = []
    missed = 0
    max_k = max(k_values)

    for case in cases:
        expected = {
            str(value)
            for value in case["expected_source_sha256"]
        }
        returned = search(str(case["question"]), max_k)
        first_rank = 0
        for rank, source in enumerate(returned, start=1):
            if source in expected:
                first_rank = rank
                break
        for k in k_values:
            if expected.intersection(returned[:k]):
                hits[k] += 1
        reciprocal_ranks.append(
            1.0 / first_rank if first_rank else 0.0
        )
        if not first_rank:
            missed += 1

    count = len(cases)
    return {
        "case_count": count,
        "hit_rate_at_k": {
            str(k): round(hits[k] / count, 6)
            for k in k_values
        },
        "mean_reciprocal_rank_at_10": round(
            sum(reciprocal_ranks) / count,
            6,
        ),
        "missed_cases": missed,
    }


def evaluate_semantic_benchmark(
    *,
    vault_root: Path,
    benchmark_path: Path,
    pilot_name: str = "pilot-v1",
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    device: str = "auto",
    model_loader=None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(strict=True)
    benchmark_path = benchmark_path.expanduser().resolve(
        strict=True
    )
    validation = validate_semantic_benchmark(
        vault_root=vault_root,
        benchmark_path=benchmark_path,
        semantic_policy_path=semantic_policy_path,
    )
    if not validation["ready_for_evaluation"]:
        raise ValueError(
            "Benchmark is not ready:\n- "
            + "\n- ".join(validation["errors"])
        )

    policy = load_semantic_policy(semantic_policy_path)
    benchmark = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )
    cases = [
        case
        for case in benchmark["cases"]
        if str(case["status"]).lower() == "approved"
    ]
    k_values = policy.benchmark.evaluation_k_values

    def lexical_search(question: str, limit: int) -> list[str]:
        result = search_index(
            vault_root=vault_root,
            pilot_name=pilot_name,
            policy_path=lexical_policy_path,
            query=question,
            limit=limit,
            max_chunks_per_source=1,
        )
        return [
            str(item["source_content_sha256"])
            for item in result["results"]
        ]

    def dense_search(question: str, limit: int) -> list[str]:
        result = semantic_search(
            vault_root=vault_root,
            pilot_name=pilot_name,
            policy_path=semantic_policy_path,
            query=question,
            limit=limit,
            device=device,
            model_loader=model_loader,
        )
        return [
            str(item["source_content_sha256"])
            for item in result["results"]
        ]

    def fused_search(question: str, limit: int) -> list[str]:
        result = hybrid_search(
            vault_root=vault_root,
            pilot_name=pilot_name,
            semantic_policy_path=semantic_policy_path,
            lexical_policy_path=lexical_policy_path,
            query=question,
            limit=limit,
            device=device,
            model_loader=model_loader,
        )
        return [
            str(item["source_content_sha256"])
            for item in result["results"]
        ]

    run_id = str(uuid.uuid4())
    summary = {
        "semantic_evaluation_schema_version": (
            EVALUATION_SCHEMA_VERSION
        ),
        "run_id": run_id,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_type": benchmark["benchmark_type"],
        "pilot_name": pilot_name,
        "approved_case_count": len(cases),
        "lexical": _metrics(cases, lexical_search, k_values),
        "semantic": _metrics(cases, dense_search, k_values),
        "hybrid": _metrics(cases, fused_search, k_values),
        "private_text_uploaded": False,
    }
    exports = vault_root / "manifests" / "exports"
    private_root = (
        vault_root / "manifests" / "semantic" / pilot_name
    )
    exports.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)
    details_path = (
        private_root
        / f"semantic-evaluation-details-{run_id}.json"
    )
    atomic_json(
        details_path,
        {
            **summary,
            "benchmark_path": str(benchmark_path),
        },
    )
    summary["private_details_path"] = str(details_path)
    summary_path = (
        exports
        / f"semantic-evaluation-summary-{run_id}.json"
    )
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
