from __future__ import annotations

from dataclasses import replace

import pytest

from alice_information.final_evaluation import (
    InformationFinalEvaluationError,
    information_final_report_digest,
    run_information_final_evaluation,
    verify_information_final_report,
)

from _information_final_evaluation_helpers import (
    benchmark,
    passing_submissions,
    policy,
    replace_submission,
)


def test_full_synthetic_evaluation_passes() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(),
        benchmark=benchmark(),
        policy=policy(),
    )
    assert report.passed
    assert report.case_count == 24
    assert report.passed_case_count == 24
    assert report.critical_case_failure_count == 0


def test_report_digest_is_deterministic() -> None:
    first = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    second = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    assert first.report_digest == second.report_digest


def test_missing_submission_fails_closed() -> None:
    with pytest.raises(InformationFinalEvaluationError):
        run_information_final_evaluation(
            submissions=passing_submissions()[:-1],
            benchmark=benchmark(),
            policy=policy(),
        )


def test_duplicate_submission_fails_closed() -> None:
    items = passing_submissions()
    with pytest.raises(InformationFinalEvaluationError):
        run_information_final_evaluation(
            submissions=items + (items[0],),
            benchmark=benchmark(),
            policy=policy(),
        )


def test_wrong_outcome_fails_case_and_report() -> None:
    items = replace_submission(
        passing_submissions(), 0, actual_outcome="completed"
    )
    report = run_information_final_evaluation(
        submissions=items, benchmark=benchmark(), policy=policy()
    )
    assert not report.passed
    assert report.passed_case_count == 23
    assert report.critical_case_failure_count == 1


def test_missing_signal_fails_case() -> None:
    items = replace_submission(passing_submissions(), 0, signals=())
    report = run_information_final_evaluation(
        submissions=items, benchmark=benchmark(), policy=policy()
    )
    assert report.case_results[0].missing_signals == (
        "injection_source_blocked",
        "policy_unchanged",
    )


def test_forbidden_signal_fails_case() -> None:
    first = passing_submissions()[0]
    items = replace_submission(
        passing_submissions(),
        0,
        signals=first.signals + ("instruction_executed",),
    )
    report = run_information_final_evaluation(
        submissions=items, benchmark=benchmark(), policy=policy()
    )
    assert report.case_results[0].forbidden_signal_hits == (
        "instruction_executed",
    )


def test_tampered_report_digest_is_rejected() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    with pytest.raises(InformationFinalEvaluationError):
        verify_information_final_report(replace(report, report_digest="0" * 64))


def test_inconsistent_pass_decision_is_rejected() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    tampered = replace(report, passed=False, report_digest="")
    tampered = replace(
        tampered,
        report_digest=information_final_report_digest(tampered),
    )
    with pytest.raises(InformationFinalEvaluationError):
        verify_information_final_report(tampered)


def test_recomputed_metric_tamper_is_rejected() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    metrics = list(report.metric_results)
    metrics[0] = replace(metrics[0], value=0.5, passed=False)
    tampered = replace(
        report,
        metric_results=tuple(metrics),
        passed=False,
        report_digest="",
    )
    tampered = replace(
        tampered,
        report_digest=information_final_report_digest(tampered),
    )
    with pytest.raises(InformationFinalEvaluationError):
        verify_information_final_report(tampered)


def test_report_boundaries_are_content_free_and_offline() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    assert report.synthetic_only and report.private_output_only
    assert not report.raw_query_text_allowed
    assert not report.raw_source_content_allowed
    assert not report.live_network_allowed
    assert not report.real_private_query_allowed
    assert not report.source_body_persistence_allowed
    assert not report.memory_write_allowed
    assert not report.external_action_allowed
    assert not report.repository_write_allowed
    assert not report.background_execution_allowed


def test_report_binds_observation_digest() -> None:
    report = run_information_final_evaluation(
        submissions=passing_submissions(), benchmark=benchmark(), policy=policy()
    )
    assert all(item.observation_digest for item in report.case_results)
    first = report.case_results[0]
    cases = (replace(first, observation_digest="0" * 64),) + report.case_results[1:]
    tampered = replace(report, case_results=cases, report_digest="")
    tampered = replace(
        tampered, report_digest=information_final_report_digest(tampered)
    )
    with pytest.raises(InformationFinalEvaluationError):
        verify_information_final_report(tampered)
