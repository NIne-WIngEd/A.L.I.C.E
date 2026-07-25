"""Final deterministic Memory Core evaluation gates for A.L.I.C.E. P2.9c.

This module evaluates the complete synthetic Phase 2 benchmark against an
already-built authoritative fixture. It does not call a language model, use the
network, invoke tools, mutate memory, or perform external actions.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

from .citation_evaluation import (
    MemoryCitationEvaluationSummary,
    evaluate_memory_answer_submissions,
)
from .cited_answer import (
    MemoryAnswerAuthorization,
    MemoryAnswerSubmission,
    build_memory_answer_submission,
)
from .evaluation_contract import (
    REQUIRED_METRIC_BASELINES,
    MemoryEvaluationBenchmark,
    MemoryEvaluationPolicy,
    canonical_json,
    load_memory_evaluation_benchmark,
    load_memory_evaluation_policy,
)
from .evaluation_fixtures import (
    MemoryEvaluationFixtureSnapshot,
    memory_evaluation_snapshot_id,
)

FINAL_MEMORY_EVALUATION_VERSION = "p2.9c-v1"

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:password|api[_ -]?key|secret|token)\s*[:=]\s*\S+"),
)


class MemoryCoreFinalEvaluationError(RuntimeError):
    """Raised when final Memory Core evaluation cannot be trusted."""


@dataclass(frozen=True)
class MemoryCoreMetricResult:
    metric_id: str
    value: float
    direction: str
    threshold: float
    critical: bool
    passed: bool


@dataclass(frozen=True)
class MemoryCoreFinalEvaluationReport:
    evaluation_version: str
    benchmark_id: str
    benchmark_digest: str
    test_set_version: str
    policy_id: str
    policy_digest: str
    fixture_snapshot_id: str
    case_count: int
    passed_case_count: int
    critical_case_failure_count: int
    metric_results: tuple[MemoryCoreMetricResult, ...]
    citation_summary: MemoryCitationEvaluationSummary
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    private_output_only: bool
    passed: bool
    report_digest: str


def _safe_authorization(authorization: MemoryAnswerAuthorization) -> None:
    if not authorization.allowed:
        raise MemoryCoreFinalEvaluationError(
            "Final Memory Core evaluation requires explicit read authorization."
        )
    if not authorization.actor.strip() or not authorization.purpose.strip():
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation authorization requires actor and purpose."
        )
    if authorization.max_classification not in {
        "PUBLIC",
        "INTERNAL",
        "PRIVATE",
    }:
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation ordinary answer access cannot authorize "
            "HIGHLY_SENSITIVE memory."
        )


def build_deterministic_evaluation_submissions(
    connection: sqlite3.Connection,
    *,
    benchmark: MemoryEvaluationBenchmark,
    authorization: MemoryAnswerAuthorization,
) -> tuple[MemoryAnswerSubmission, ...]:
    """Build one deterministic private answer packet for every benchmark case."""
    _safe_authorization(authorization)
    return tuple(
        build_memory_answer_submission(
            connection,
            case=case,
            memory_ids=case.expected_memory_ids,
            authorization=authorization,
        )
        for case in benchmark.cases
    )


def _suite_rate(
    summary: MemoryCitationEvaluationSummary,
    suite: str,
) -> float:
    cases = tuple(item for item in summary.cases if item.suite == suite)
    if not cases:
        raise MemoryCoreFinalEvaluationError(
            f"Final evaluation benchmark lacks required suite: {suite}"
        )
    return sum(item.passed for item in cases) / len(cases)


def _suite_failure_count(
    summary: MemoryCitationEvaluationSummary,
    suite: str,
) -> int:
    cases = tuple(item for item in summary.cases if item.suite == suite)
    if not cases:
        raise MemoryCoreFinalEvaluationError(
            f"Final evaluation benchmark lacks required suite: {suite}"
        )
    return sum(not item.passed for item in cases)


def _secret_emission_count(
    submissions: tuple[MemoryAnswerSubmission, ...],
) -> int:
    return sum(
        1
        for submission in submissions
        if any(pattern.search(submission.answer_text) for pattern in _SECRET_PATTERNS)
    )


def _metric_value_map(
    *,
    summary: MemoryCitationEvaluationSummary,
    submissions: tuple[MemoryAnswerSubmission, ...],
) -> dict[str, float]:
    candidate_leaks = sum(
        len(item.forbidden_candidate_hits) for item in summary.cases
    )
    return {
        "confirmed_personal_fact_accuracy": _suite_rate(
            summary, "confirmed_fact"
        ),
        "personal_source_attribution_rate": (
            summary.personal_source_attribution_rate
        ),
        "unsupported_personal_claim_rate": (
            summary.unsupported_personal_claim_rate
        ),
        "temporal_classification_accuracy": _suite_rate(summary, "temporal"),
        "conflict_surface_rate": _suite_rate(summary, "conflict"),
        "correction_use_rate": _suite_rate(summary, "correction"),
        "uncertainty_surface_rate": _suite_rate(summary, "uncertainty"),
        "deletion_absence_rate": _suite_rate(summary, "deletion"),
        "permission_violation_count": float(
            _suite_failure_count(summary, "permission")
        ),
        "sensitive_disclosure_count": float(
            _suite_failure_count(summary, "sensitivity")
        ),
        "secret_emission_count": float(_secret_emission_count(submissions)),
        "critical_prompt_injection_success_count": float(
            _suite_failure_count(summary, "prompt_injection")
        ),
        "candidate_leakage_count": float(candidate_leaks),
    }


def _passes(value: float, *, direction: str, threshold: float) -> bool:
    if direction == "minimum":
        return value >= threshold
    if direction == "maximum":
        return value <= threshold
    raise MemoryCoreFinalEvaluationError(
        f"Unsupported metric direction: {direction!r}"
    )


def _metric_results(
    *,
    policy: MemoryEvaluationPolicy,
    values: dict[str, float],
) -> tuple[MemoryCoreMetricResult, ...]:
    policy_gates = {gate.metric_id: gate for gate in policy.metric_gates}
    if set(policy_gates) != set(REQUIRED_METRIC_BASELINES):
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation policy metric set differs from the governing gate set."
        )
    if set(values) != set(REQUIRED_METRIC_BASELINES):
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation did not calculate every governing metric."
        )

    results = []
    for metric_id in REQUIRED_METRIC_BASELINES:
        gate = policy_gates[metric_id]
        value = round(float(values[metric_id]), 6)
        results.append(
            MemoryCoreMetricResult(
                metric_id=metric_id,
                value=value,
                direction=gate.direction,
                threshold=gate.threshold,
                critical=gate.critical,
                passed=_passes(
                    value,
                    direction=gate.direction,
                    threshold=gate.threshold,
                ),
            )
        )
    return tuple(results)


def _report_material(report: MemoryCoreFinalEvaluationReport) -> dict[str, Any]:
    return {
        "evaluation_version": report.evaluation_version,
        "benchmark_id": report.benchmark_id,
        "benchmark_digest": report.benchmark_digest,
        "test_set_version": report.test_set_version,
        "policy_id": report.policy_id,
        "policy_digest": report.policy_digest,
        "fixture_snapshot_id": report.fixture_snapshot_id,
        "case_count": report.case_count,
        "passed_case_count": report.passed_case_count,
        "critical_case_failure_count": report.critical_case_failure_count,
        "metric_results": [
            {
                "metric_id": item.metric_id,
                "value": item.value,
                "direction": item.direction,
                "threshold": item.threshold,
                "critical": item.critical,
                "passed": item.passed,
            }
            for item in report.metric_results
        ],
        "citation_metrics": {
            "case_pass_rate": report.citation_summary.case_pass_rate,
            "claim_count": report.citation_summary.claim_count,
            "supported_claim_count": report.citation_summary.supported_claim_count,
            "cited_claim_count": report.citation_summary.cited_claim_count,
            "unsupported_claim_count": report.citation_summary.unsupported_claim_count,
            "personal_source_attribution_rate": (
                report.citation_summary.personal_source_attribution_rate
            ),
            "expected_source_citation_rate": (
                report.citation_summary.expected_source_citation_rate
            ),
            "claim_citation_coverage": (
                report.citation_summary.claim_citation_coverage
            ),
            "unsupported_personal_claim_rate": (
                report.citation_summary.unsupported_personal_claim_rate
            ),
            "passes_all_p29b_gates": (
                report.citation_summary.passes_all_p29b_gates
            ),
            "cases": [
                {
                    "case_id": item.case_id,
                    "suite": item.suite,
                    "expected_outcome": item.expected_outcome,
                    "actual_outcome": item.actual_outcome,
                    "passed": item.passed,
                    "issues": list(item.issues),
                    "actual_memory_ids": list(item.actual_memory_ids),
                    "actual_source_refs": list(item.actual_source_refs),
                    "actual_knowledge_statuses": list(item.actual_knowledge_statuses),
                    "claim_count": item.claim_count,
                    "supported_claim_count": item.supported_claim_count,
                    "cited_claim_count": item.cited_claim_count,
                    "unsupported_claim_count": item.unsupported_claim_count,
                    "expected_sources_present": item.expected_sources_present,
                    "forbidden_memory_hits": list(item.forbidden_memory_hits),
                    "forbidden_candidate_hits": list(item.forbidden_candidate_hits),
                }
                for item in report.citation_summary.cases
            ],
        },
        "capabilities": {
            "memory_write_allowed": report.memory_write_allowed,
            "external_action_allowed": report.external_action_allowed,
            "tool_calling_allowed": report.tool_calling_allowed,
            "web_access_allowed": report.web_access_allowed,
            "private_output_only": report.private_output_only,
        },
        "passed": report.passed,
    }


def memory_core_final_report_digest(
    report: MemoryCoreFinalEvaluationReport,
) -> str:
    return hashlib.sha256(canonical_json(_report_material(report))).hexdigest()


def verify_memory_core_final_report(
    report: MemoryCoreFinalEvaluationReport,
) -> None:
    """Fail closed if a persisted or transported final report was altered."""
    expected = memory_core_final_report_digest(report)
    if report.report_digest != expected:
        raise MemoryCoreFinalEvaluationError(
            "Final Memory Core evaluation report digest is invalid."
        )
    if report.case_count <= 0 or report.passed_case_count > report.case_count:
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation report case counts are invalid."
        )
    metric_ids = tuple(item.metric_id for item in report.metric_results)
    if len(set(metric_ids)) != len(metric_ids):
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation report contains duplicate metric results."
        )
    if set(metric_ids) != set(REQUIRED_METRIC_BASELINES):
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation report metric coverage is incomplete."
        )
    expected_pass = (
        all(item.passed for item in report.metric_results)
        and report.critical_case_failure_count == 0
        and report.passed_case_count == report.case_count
        and report.citation_summary.passes_all_p29b_gates
        and not report.memory_write_allowed
        and not report.external_action_allowed
        and not report.tool_calling_allowed
        and not report.web_access_allowed
        and report.private_output_only
    )
    if report.passed != expected_pass:
        raise MemoryCoreFinalEvaluationError(
            "Final evaluation report release decision is inconsistent."
        )


def run_memory_core_final_evaluation(
    connection: sqlite3.Connection,
    *,
    fixture: MemoryEvaluationFixtureSnapshot,
    authorization: MemoryAnswerAuthorization,
    benchmark: MemoryEvaluationBenchmark | None = None,
    policy: MemoryEvaluationPolicy | None = None,
    submissions: tuple[MemoryAnswerSubmission, ...] | None = None,
) -> MemoryCoreFinalEvaluationReport:
    """Run all final synthetic Memory Core release gates deterministically."""
    _safe_authorization(authorization)
    resolved_benchmark = benchmark or load_memory_evaluation_benchmark()
    resolved_policy = policy or load_memory_evaluation_policy()

    current_snapshot = memory_evaluation_snapshot_id(connection)
    if fixture.snapshot_id != current_snapshot:
        raise MemoryCoreFinalEvaluationError(
            "Evaluation fixture state changed after snapshot creation."
        )
    if fixture.snapshot_id != resolved_benchmark.fixture_snapshot_id:
        raise MemoryCoreFinalEvaluationError(
            "Evaluation fixture snapshot is not approved by the benchmark."
        )
    if fixture.benchmark_id != resolved_benchmark.benchmark_id:
        raise MemoryCoreFinalEvaluationError(
            "Evaluation fixture and benchmark identifiers disagree."
        )
    if fixture.benchmark_digest != resolved_benchmark.digest:
        raise MemoryCoreFinalEvaluationError(
            "Evaluation fixture and benchmark digests disagree."
        )

    resolved_submissions = submissions or build_deterministic_evaluation_submissions(
        connection,
        benchmark=resolved_benchmark,
        authorization=authorization,
    )
    summary = evaluate_memory_answer_submissions(
        connection,
        benchmark=resolved_benchmark,
        submissions=resolved_submissions,
        authorization=authorization,
        policy=resolved_policy,
    )
    values = _metric_value_map(
        summary=summary,
        submissions=resolved_submissions,
    )
    metrics = _metric_results(policy=resolved_policy, values=values)
    critical_case_ids = {
        case.case_id for case in resolved_benchmark.cases if case.critical
    }
    critical_failures = sum(
        not item.passed
        for item in summary.cases
        if item.case_id in critical_case_ids
    )

    passed = (
        all(item.passed for item in metrics)
        and critical_failures == 0
        and summary.passed_case_count == summary.case_count
        and summary.passes_all_p29b_gates
        and not resolved_policy.memory_write_allowed
        and not resolved_policy.external_action_allowed
        and not resolved_policy.tool_calling_allowed
        and not resolved_policy.web_access_allowed
        and resolved_policy.private_output_only
    )

    report = MemoryCoreFinalEvaluationReport(
        evaluation_version=FINAL_MEMORY_EVALUATION_VERSION,
        benchmark_id=resolved_benchmark.benchmark_id,
        benchmark_digest=resolved_benchmark.digest,
        test_set_version=resolved_benchmark.test_set_version,
        policy_id=resolved_policy.policy_id,
        policy_digest=resolved_policy.digest,
        fixture_snapshot_id=fixture.snapshot_id,
        case_count=summary.case_count,
        passed_case_count=summary.passed_case_count,
        critical_case_failure_count=critical_failures,
        metric_results=metrics,
        citation_summary=summary,
        memory_write_allowed=resolved_policy.memory_write_allowed,
        external_action_allowed=resolved_policy.external_action_allowed,
        tool_calling_allowed=resolved_policy.tool_calling_allowed,
        web_access_allowed=resolved_policy.web_access_allowed,
        private_output_only=resolved_policy.private_output_only,
        passed=passed,
        report_digest="",
    )
    report = replace(report, report_digest=memory_core_final_report_digest(report))
    verify_memory_core_final_report(report)
    return report
