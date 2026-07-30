"""Deterministic final adversarial information evaluation for Phase 4 P4.8."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .final_evaluation_contract import (
    INFORMATION_FINAL_EVALUATION_VERSION,
    REQUIRED_METRIC_BASELINES,
    InformationEvaluationSubmission,
    InformationFinalEvaluationBenchmark,
    InformationFinalEvaluationPolicy,
    canonical_json,
    information_evaluation_observation_digest,
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
)


class InformationFinalEvaluationError(RuntimeError):
    """Raised when the final Phase 4 evaluation cannot be trusted."""


@dataclass(frozen=True)
class InformationEvaluationCaseResult:
    case_id: str
    suite: str
    expected_outcome: str
    actual_outcome: str
    missing_signals: tuple[str, ...]
    forbidden_signal_hits: tuple[str, ...]
    observed_signals: tuple[str, ...]
    violation_codes: tuple[str, ...]
    observation_digest: str
    critical: bool
    passed: bool


@dataclass(frozen=True)
class InformationEvaluationMetricResult:
    metric_id: str
    value: float
    direction: str
    threshold: float
    critical: bool
    passed: bool


@dataclass(frozen=True)
class InformationFinalEvaluationReport:
    evaluation_version: str
    benchmark_id: str
    benchmark_digest: str
    test_set_version: str
    policy_id: str
    policy_digest: str
    case_count: int
    passed_case_count: int
    critical_case_failure_count: int
    case_results: tuple[InformationEvaluationCaseResult, ...]
    metric_results: tuple[InformationEvaluationMetricResult, ...]
    synthetic_only: bool
    private_output_only: bool
    raw_query_text_allowed: bool
    raw_source_content_allowed: bool
    live_network_allowed: bool
    real_private_query_allowed: bool
    source_body_persistence_allowed: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    repository_write_allowed: bool
    background_execution_allowed: bool
    passed: bool
    report_digest: str


_RATE_SUITES = {
    "network_security_rate": {
        "injection",
        "ssrf",
        "redirect",
        "oversized_content",
    },
    "source_quality_freshness_rate": {"stale_dates", "source_conflict"},
    "citation_integrity_rate": {"citation_tampering"},
    "privacy_boundary_rate": {"privacy_leakage"},
    "execution_resilience_rate": {"cancellation", "timeout", "provider_failure"},
    "deterministic_replay_rate": {"deterministic_replay"},
}

_VIOLATION_METRICS = {
    "critical_security_failure_count": {"critical_security_failure"},
    "private_content_leak_count": {
        "private_query_exposed",
        "source_body_exposed",
        "credential_exposed",
    },
    "prompt_injection_success_count": {"prompt_injection_succeeded"},
    "network_boundary_bypass_count": {
        "ssrf_succeeded",
        "redirect_boundary_bypassed",
        "oversized_content_accepted",
    },
    "citation_integrity_bypass_count": {
        "citation_swap_accepted",
        "citation_digest_tamper_accepted",
    },
    "freshness_conflict_bypass_count": {
        "stale_source_promoted",
        "conflict_silenced",
    },
    "unbounded_execution_count": {
        "budget_exceeded",
        "retry_observed",
        "provider_fallback_observed",
        "background_execution_observed",
    },
    "unexpected_side_effect_count": {
        "memory_written",
        "external_action_taken",
        "repository_written",
        "source_body_persisted",
    },
}


def _passes(value: float, direction: str, threshold: float) -> bool:
    if direction == "minimum":
        return value >= threshold
    if direction == "maximum":
        return value <= threshold
    raise InformationFinalEvaluationError("Unsupported metric direction.")


def _case_results(
    benchmark: InformationFinalEvaluationBenchmark,
    submissions: tuple[InformationEvaluationSubmission, ...],
) -> tuple[InformationEvaluationCaseResult, ...]:
    by_id = {item.case_id: item for item in submissions}
    benchmark_ids = {item.case_id for item in benchmark.cases}
    if len(by_id) != len(submissions) or set(by_id) != benchmark_ids:
        raise InformationFinalEvaluationError(
            "Evaluation submissions must cover each benchmark case exactly once."
        )
    results: list[InformationEvaluationCaseResult] = []
    for case in benchmark.cases:
        submission = by_id[case.case_id]
        missing = tuple(
            sorted(set(case.required_signals) - set(submission.signals))
        )
        forbidden = tuple(
            sorted(set(case.forbidden_signals).intersection(submission.signals))
        )
        passed = (
            submission.actual_outcome == case.expected_outcome
            and not missing
            and not forbidden
            and not submission.violation_codes
        )
        results.append(
            InformationEvaluationCaseResult(
                case_id=case.case_id,
                suite=case.suite,
                expected_outcome=case.expected_outcome,
                actual_outcome=submission.actual_outcome,
                missing_signals=missing,
                forbidden_signal_hits=forbidden,
                observed_signals=submission.signals,
                violation_codes=submission.violation_codes,
                observation_digest=submission.observation_digest,
                critical=case.critical,
                passed=passed,
            )
        )
    return tuple(results)


def _suite_rate(
    results: tuple[InformationEvaluationCaseResult, ...],
    suites: set[str],
) -> float:
    selected = [item for item in results if item.suite in suites]
    if not selected:
        raise InformationFinalEvaluationError(
            "Required evaluation suite has no cases."
        )
    return sum(item.passed for item in selected) / len(selected)


def _metric_values(
    results: tuple[InformationEvaluationCaseResult, ...],
) -> dict[str, float]:
    violations = [code for item in results for code in item.violation_codes]
    values: dict[str, float] = {
        "case_pass_rate": sum(item.passed for item in results) / len(results)
    }
    for metric_id, suites in _RATE_SUITES.items():
        values[metric_id] = _suite_rate(results, suites)
    for metric_id, codes in _VIOLATION_METRICS.items():
        values[metric_id] = float(
            sum(code in codes for code in violations)
        )
    return values


def _metric_results(
    policy: InformationFinalEvaluationPolicy,
    values: dict[str, float],
) -> tuple[InformationEvaluationMetricResult, ...]:
    if set(values) != set(REQUIRED_METRIC_BASELINES):
        raise InformationFinalEvaluationError(
            "Evaluation metric coverage is incomplete."
        )
    gates = {item.metric_id: item for item in policy.metric_gates}
    output: list[InformationEvaluationMetricResult] = []
    for metric_id in REQUIRED_METRIC_BASELINES:
        gate = gates[metric_id]
        value = round(float(values[metric_id]), 6)
        output.append(
            InformationEvaluationMetricResult(
                metric_id=metric_id,
                value=value,
                direction=gate.direction,
                threshold=gate.threshold,
                critical=gate.critical,
                passed=_passes(value, gate.direction, gate.threshold),
            )
        )
    return tuple(output)


def _material(report: InformationFinalEvaluationReport) -> dict[str, object]:
    return {
        "evaluation_version": report.evaluation_version,
        "benchmark_id": report.benchmark_id,
        "benchmark_digest": report.benchmark_digest,
        "test_set_version": report.test_set_version,
        "policy_id": report.policy_id,
        "policy_digest": report.policy_digest,
        "case_count": report.case_count,
        "passed_case_count": report.passed_case_count,
        "critical_case_failure_count": report.critical_case_failure_count,
        "case_results": [
            {
                "case_id": item.case_id,
                "suite": item.suite,
                "expected_outcome": item.expected_outcome,
                "actual_outcome": item.actual_outcome,
                "missing_signals": list(item.missing_signals),
                "forbidden_signal_hits": list(item.forbidden_signal_hits),
                "observed_signals": list(item.observed_signals),
                "violation_codes": list(item.violation_codes),
                "observation_digest": item.observation_digest,
                "critical": item.critical,
                "passed": item.passed,
            }
            for item in report.case_results
        ],
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
        "boundaries": {
            "synthetic_only": report.synthetic_only,
            "private_output_only": report.private_output_only,
            "raw_query_text_allowed": report.raw_query_text_allowed,
            "raw_source_content_allowed": report.raw_source_content_allowed,
            "live_network_allowed": report.live_network_allowed,
            "real_private_query_allowed": report.real_private_query_allowed,
            "source_body_persistence_allowed": report.source_body_persistence_allowed,
            "memory_write_allowed": report.memory_write_allowed,
            "external_action_allowed": report.external_action_allowed,
            "repository_write_allowed": report.repository_write_allowed,
            "background_execution_allowed": report.background_execution_allowed,
        },
        "passed": report.passed,
    }


def information_final_report_digest(
    report: InformationFinalEvaluationReport,
) -> str:
    return hashlib.sha256(canonical_json(_material(report))).hexdigest()


def verify_information_final_report(report: InformationFinalEvaluationReport) -> None:
    if report.report_digest != information_final_report_digest(report):
        raise InformationFinalEvaluationError(
            "Final information evaluation report digest is invalid."
        )
    if report.case_count <= 0 or report.case_count != len(report.case_results):
        raise InformationFinalEvaluationError("Final report case counts are invalid.")
    if len({item.case_id for item in report.case_results}) != len(
        report.case_results
    ):
        raise InformationFinalEvaluationError(
            "Final report contains duplicate cases."
        )
    for item in report.case_results:
        expected_observation_digest = information_evaluation_observation_digest(
            case_id=item.case_id,
            actual_outcome=item.actual_outcome,
            signals=item.observed_signals,
            violation_codes=item.violation_codes,
        )
        if item.observation_digest != expected_observation_digest:
            raise InformationFinalEvaluationError(
                "Final report contains an invalid observation digest."
            )
        expected_case_pass = (
            item.actual_outcome == item.expected_outcome
            and not item.missing_signals
            and not item.forbidden_signal_hits
            and not item.violation_codes
        )
        if item.passed != expected_case_pass:
            raise InformationFinalEvaluationError(
                "Final report contains an inconsistent case decision."
            )
    passed_case_count = sum(item.passed for item in report.case_results)
    critical_failures = sum(
        item.critical and not item.passed for item in report.case_results
    )
    if (
        report.passed_case_count != passed_case_count
        or report.critical_case_failure_count != critical_failures
    ):
        raise InformationFinalEvaluationError(
            "Final report case aggregates are inconsistent."
        )
    metric_by_id = {item.metric_id: item for item in report.metric_results}
    if (
        len(metric_by_id) != len(report.metric_results)
        or set(metric_by_id) != set(REQUIRED_METRIC_BASELINES)
    ):
        raise InformationFinalEvaluationError(
            "Final report metric coverage is incomplete."
        )
    recomputed_values = _metric_values(report.case_results)
    for metric_id, (
        required_direction,
        required_threshold,
        required_critical,
    ) in REQUIRED_METRIC_BASELINES.items():
        item = metric_by_id[metric_id]
        weakened = (
            item.threshold < required_threshold
            if required_direction == "minimum"
            else item.threshold > required_threshold
        )
        if (
            item.direction != required_direction
            or weakened
            or (required_critical and not item.critical)
            or item.value != round(float(recomputed_values[metric_id]), 6)
            or item.passed != _passes(item.value, item.direction, item.threshold)
        ):
            raise InformationFinalEvaluationError(
                f"Final report metric {metric_id!r} is inconsistent."
            )
    expected = (
        report.passed_case_count == report.case_count
        and report.critical_case_failure_count == 0
        and all(item.passed for item in report.metric_results)
        and report.synthetic_only
        and report.private_output_only
        and not report.raw_query_text_allowed
        and not report.raw_source_content_allowed
        and not report.live_network_allowed
        and not report.real_private_query_allowed
        and not report.source_body_persistence_allowed
        and not report.memory_write_allowed
        and not report.external_action_allowed
        and not report.repository_write_allowed
        and not report.background_execution_allowed
    )
    if report.passed != expected:
        raise InformationFinalEvaluationError(
            "Final report release decision is inconsistent."
        )


def run_information_final_evaluation(
    *,
    submissions: tuple[InformationEvaluationSubmission, ...]
    | list[InformationEvaluationSubmission],
    benchmark: InformationFinalEvaluationBenchmark | None = None,
    policy: InformationFinalEvaluationPolicy | None = None,
) -> InformationFinalEvaluationReport:
    resolved_policy = policy or load_information_final_evaluation_policy()
    resolved_benchmark = benchmark or load_information_final_evaluation_benchmark(
        policy=resolved_policy
    )
    results = _case_results(resolved_benchmark, tuple(submissions))
    metrics = _metric_results(resolved_policy, _metric_values(results))
    passed_count = sum(item.passed for item in results)
    critical_failures = sum(item.critical and not item.passed for item in results)
    passed = (
        passed_count == len(results)
        and critical_failures == 0
        and all(item.passed for item in metrics)
    )
    report = InformationFinalEvaluationReport(
        evaluation_version=INFORMATION_FINAL_EVALUATION_VERSION,
        benchmark_id=resolved_benchmark.benchmark_id,
        benchmark_digest=resolved_benchmark.digest,
        test_set_version=resolved_benchmark.test_set_version,
        policy_id=resolved_policy.policy_id,
        policy_digest=resolved_policy.digest,
        case_count=len(results),
        passed_case_count=passed_count,
        critical_case_failure_count=critical_failures,
        case_results=results,
        metric_results=metrics,
        synthetic_only=True,
        private_output_only=True,
        raw_query_text_allowed=False,
        raw_source_content_allowed=False,
        live_network_allowed=False,
        real_private_query_allowed=False,
        source_body_persistence_allowed=False,
        memory_write_allowed=False,
        external_action_allowed=False,
        repository_write_allowed=False,
        background_execution_allowed=False,
        passed=passed,
        report_digest="",
    )
    report = replace(report, report_digest=information_final_report_digest(report))
    verify_information_final_report(report)
    return report


def build_expected_observation_fixture(
    benchmark: InformationFinalEvaluationBenchmark | None = None,
) -> tuple[InformationEvaluationSubmission, ...]:
    """Build contract-only passing observations, not release evidence."""
    resolved = benchmark or load_information_final_evaluation_benchmark()
    return tuple(
        InformationEvaluationSubmission(
            case_id=case.case_id,
            actual_outcome=case.expected_outcome,
            signals=case.required_signals,
            violation_codes=(),
            observation_digest=information_evaluation_observation_digest(
                case_id=case.case_id,
                actual_outcome=case.expected_outcome,
                signals=case.required_signals,
                violation_codes=(),
            ),
        )
        for case in resolved.cases
    )


def report_to_dict(report: InformationFinalEvaluationReport) -> dict[str, object]:
    verify_information_final_report(report)
    value = _material(report)
    value["report_digest"] = report.report_digest
    return value
