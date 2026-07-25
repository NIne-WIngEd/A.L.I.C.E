"""P2.9c adversarial end-to-end Memory Core evaluation gates."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from alice_memory.cited_answer import MemoryAnswerAuthorization
from alice_memory.evaluation_contract import (
    load_memory_evaluation_benchmark,
    load_memory_evaluation_policy,
)
from alice_memory.evaluation_fixtures import build_memory_evaluation_fixture
from alice_memory.final_evaluation import (
    MemoryCoreFinalEvaluationError,
    build_deterministic_evaluation_submissions,
    memory_core_final_report_digest,
    run_memory_core_final_evaluation,
    verify_memory_core_final_report,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.store import open_memory_store


def _authorization(**changes):
    values = {
        "actor": "p2.9c-evaluator",
        "allowed": True,
        "purpose": "offline synthetic final Memory Core evaluation",
        "max_classification": "PRIVATE",
    }
    values.update(changes)
    return MemoryAnswerAuthorization(**values)


@contextmanager
def _fixture(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    benchmark = load_memory_evaluation_benchmark()
    policy = load_memory_evaluation_policy()
    with open_memory_store(vault, repository_root=repository) as connection:
        snapshot = build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
            benchmark=benchmark,
            key_protector=InMemoryTestKeyProtector(),
        )
        yield connection, snapshot, benchmark, policy


def _perfect(tmp_path: Path):
    manager = _fixture(tmp_path)
    connection, snapshot, benchmark, policy = manager.__enter__()
    report = run_memory_core_final_evaluation(
        connection,
        fixture=snapshot,
        benchmark=benchmark,
        policy=policy,
        authorization=_authorization(),
    )
    return manager, connection, snapshot, benchmark, policy, report


def test_perfect_final_memory_evaluation_passes_all_gates(tmp_path: Path) -> None:
    manager, _connection, _snapshot, _benchmark, _policy, report = _perfect(tmp_path)
    try:
        assert report.passed is True
        assert report.case_count == 13
        assert report.passed_case_count == 13
        assert report.critical_case_failure_count == 0
        assert len(report.metric_results) == 13
        assert all(item.passed for item in report.metric_results)
        assert report.citation_summary.expected_source_citation_rate == 1.0
        assert report.citation_summary.claim_citation_coverage == 1.0
    finally:
        manager.__exit__(None, None, None)


def test_final_report_digest_verifies(tmp_path: Path) -> None:
    manager, _connection, _snapshot, _benchmark, _policy, report = _perfect(tmp_path)
    try:
        verify_memory_core_final_report(report)
        assert report.report_digest == memory_core_final_report_digest(report)
    finally:
        manager.__exit__(None, None, None)


def test_tampered_report_digest_fails_closed(tmp_path: Path) -> None:
    manager, _connection, _snapshot, _benchmark, _policy, report = _perfect(tmp_path)
    try:
        with pytest.raises(MemoryCoreFinalEvaluationError, match="digest"):
            verify_memory_core_final_report(replace(report, report_digest="0" * 64))
    finally:
        manager.__exit__(None, None, None)


def test_fixture_state_change_after_snapshot_fails(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        connection.execute(
            "UPDATE memories SET confidence = 0.5 WHERE memory_id = 'eval-fact-current'"
        )
        with pytest.raises(MemoryCoreFinalEvaluationError, match="changed"):
            run_memory_core_final_evaluation(
                connection,
                fixture=snapshot,
                benchmark=benchmark,
                policy=policy,
                authorization=_authorization(),
            )


def test_unapproved_fixture_snapshot_fails(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        bad = replace(snapshot, snapshot_id="0" * 64)
        with pytest.raises(MemoryCoreFinalEvaluationError, match="changed"):
            run_memory_core_final_evaluation(
                connection,
                fixture=bad,
                benchmark=benchmark,
                policy=policy,
                authorization=_authorization(),
            )


def test_final_evaluation_requires_authorization(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        with pytest.raises(MemoryCoreFinalEvaluationError, match="authorization"):
            run_memory_core_final_evaluation(
                connection,
                fixture=snapshot,
                benchmark=benchmark,
                policy=policy,
                authorization=_authorization(allowed=False),
            )


def test_highly_sensitive_answer_authorization_is_rejected(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        with pytest.raises(MemoryCoreFinalEvaluationError, match="HIGHLY_SENSITIVE"):
            run_memory_core_final_evaluation(
                connection,
                fixture=snapshot,
                benchmark=benchmark,
                policy=policy,
                authorization=_authorization(max_classification="HIGHLY_SENSITIVE"),
            )


def test_permission_case_failure_blocks_release(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        submissions = list(
            build_deterministic_evaluation_submissions(
                connection,
                benchmark=benchmark,
                authorization=_authorization(),
            )
        )
        index = next(i for i, item in enumerate(submissions) if item.case_id == "permission-denial-001")
        submissions[index] = replace(submissions[index], outcome="answerable")
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=policy,
            authorization=_authorization(),
            submissions=tuple(submissions),
        )
        metric = next(item for item in report.metric_results if item.metric_id == "permission_violation_count")
        assert metric.value == 1.0
        assert metric.passed is False
        assert report.critical_case_failure_count == 1
        assert report.passed is False


def test_sensitive_case_failure_blocks_release(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        submissions = list(build_deterministic_evaluation_submissions(connection, benchmark=benchmark, authorization=_authorization()))
        index = next(i for i, item in enumerate(submissions) if item.case_id == "sensitivity-denial-001")
        submissions[index] = replace(submissions[index], outcome="answerable")
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=policy,
            authorization=_authorization(),
            submissions=tuple(submissions),
        )
        metric = next(item for item in report.metric_results if item.metric_id == "sensitive_disclosure_count")
        assert metric.value == 1.0
        assert report.passed is False


def test_candidate_identifier_leak_is_counted(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        submissions = list(build_deterministic_evaluation_submissions(connection, benchmark=benchmark, authorization=_authorization()))
        index = next(i for i, item in enumerate(submissions) if item.case_id == "candidate-boundary-001")
        submissions[index] = replace(
            submissions[index],
            answer_text=submissions[index].answer_text + " eval-candidate-unpromoted",
        )
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=policy,
            authorization=_authorization(),
            submissions=tuple(submissions),
        )
        metric = next(item for item in report.metric_results if item.metric_id == "candidate_leakage_count")
        assert metric.value == 1.0
        assert metric.passed is False
        assert report.passed is False


def test_secret_like_output_is_zero_tolerance(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        submissions = list(build_deterministic_evaluation_submissions(connection, benchmark=benchmark, authorization=_authorization()))
        submissions[0] = replace(
            submissions[0],
            answer_text=submissions[0].answer_text + " api_key=sk-1234567890abcdefghijklmnop",
        )
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=policy,
            authorization=_authorization(),
            submissions=tuple(submissions),
        )
        metric = next(item for item in report.metric_results if item.metric_id == "secret_emission_count")
        assert metric.value == 1.0
        assert metric.passed is False
        assert report.passed is False


def test_prompt_injection_outcome_tamper_is_critical(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        submissions = list(build_deterministic_evaluation_submissions(connection, benchmark=benchmark, authorization=_authorization()))
        index = next(i for i, item in enumerate(submissions) if item.case_id == "prompt-injection-001")
        submissions[index] = replace(submissions[index], outcome="answerable")
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=policy,
            authorization=_authorization(),
            submissions=tuple(submissions),
        )
        metric = next(item for item in report.metric_results if item.metric_id == "critical_prompt_injection_success_count")
        assert metric.value == 1.0
        assert report.critical_case_failure_count == 1
        assert report.passed is False


def test_weakened_policy_capability_boundary_blocks_release(tmp_path: Path) -> None:
    with _fixture(tmp_path) as (connection, snapshot, benchmark, policy):
        weakened = replace(policy, tool_calling_allowed=True)
        report = run_memory_core_final_evaluation(
            connection,
            fixture=snapshot,
            benchmark=benchmark,
            policy=weakened,
            authorization=_authorization(),
        )
        assert report.tool_calling_allowed is True
        assert report.passed is False


def test_report_pass_flag_tamper_is_detected(tmp_path: Path) -> None:
    manager, _connection, _snapshot, _benchmark, _policy, report = _perfect(tmp_path)
    try:
        tampered = replace(report, passed=False, report_digest="")
        tampered = replace(tampered, report_digest=memory_core_final_report_digest(tampered))
        with pytest.raises(MemoryCoreFinalEvaluationError, match="inconsistent"):
            verify_memory_core_final_report(tampered)
    finally:
        manager.__exit__(None, None, None)
