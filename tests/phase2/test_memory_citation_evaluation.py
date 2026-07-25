"""P2.9b authoritative-memory and source-record citation evaluation tests."""

from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path

import pytest

from alice_memory.citation_evaluation import (
    MemoryCitationEvaluationError,
    evaluate_memory_answer_submission,
    evaluate_memory_answer_submissions,
)
from alice_memory.cited_answer import (
    MemoryAnswerAuthorization,
    MemoryAnswerSubmission,
    build_memory_answer_submission,
)
from alice_memory.evaluation_contract import (
    MemoryEvaluationBenchmark,
    MemoryEvaluationCase,
    load_memory_evaluation_benchmark,
)
from alice_memory.evaluation_fixtures import (
    CANDIDATE_IDS,
    MEMORY_IDS,
    build_memory_evaluation_fixture,
)
from alice_memory.store import open_memory_store


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _authorization(**changes) -> MemoryAnswerAuthorization:
    values = {
        "actor": "p29b-evaluator",
        "allowed": True,
        "purpose": "phase2-citation-evaluation",
        "max_classification": "PRIVATE",
    }
    values.update(changes)
    return MemoryAnswerAuthorization(**values)


@contextmanager
def _build_store(tmp_path: Path):
    repository, vault = _setup(tmp_path)
    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        build_memory_evaluation_fixture(
            connection,
            vault,
            repository_root=repository,
        )
        yield connection


def _case(case_id: str) -> MemoryEvaluationCase:
    return next(
        case
        for case in load_memory_evaluation_benchmark().cases
        if case.case_id == case_id
    )


def _submission(
    connection,
    case_id: str,
    memory_ids: tuple[str, ...],
) -> MemoryAnswerSubmission:
    return build_memory_answer_submission(
        connection,
        case=_case(case_id),
        memory_ids=memory_ids,
        authorization=_authorization(),
    )


def _perfect_submissions(connection) -> tuple[MemoryAnswerSubmission, ...]:
    benchmark = load_memory_evaluation_benchmark()
    return tuple(
        build_memory_answer_submission(
            connection,
            case=case,
            memory_ids=case.expected_memory_ids,
            authorization=_authorization(),
        )
        for case in benchmark.cases
    )


def test_valid_confirmed_answer_passes_authoritative_citation_check(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=_submission(
                connection,
                case.case_id,
                case.expected_memory_ids,
            ),
            authorization=_authorization(),
        )

    assert result.passed is True
    assert result.supported_claim_count == 1
    assert result.cited_claim_count == 1
    assert result.unsupported_claim_count == 0
    assert result.expected_sources_present is True


def test_perfect_benchmark_submissions_pass_p29b_gates(
    tmp_path: Path,
) -> None:
    benchmark = load_memory_evaluation_benchmark()
    with _build_store(tmp_path) as connection:
        summary = evaluate_memory_answer_submissions(
            connection,
            benchmark=benchmark,
            submissions=_perfect_submissions(connection),
            authorization=_authorization(),
        )

    assert summary.case_count == 13
    assert summary.passed_case_count == 13
    assert summary.personal_source_attribution_rate == 1.0
    assert summary.expected_source_citation_rate == 1.0
    assert summary.claim_citation_coverage == 1.0
    assert summary.unsupported_personal_claim_rate == 0.0
    assert summary.passes_all_p29b_gates is True
    assert summary.memory_write_allowed is False
    assert summary.external_action_allowed is False
    assert summary.tool_calling_allowed is False
    assert summary.web_access_allowed is False
    assert summary.private_output_only is True


def test_tampered_claim_text_fails_authoritative_support(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad_claim = replace(good.claims[0], text="Fabricated answer.")
        bad = replace(good, claims=(bad_claim,))
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "claim_text_not_authoritative" in result.issues
    assert result.unsupported_claim_count == 1


def test_missing_claim_citations_fails_source_attribution(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad_claim = replace(good.claims[0], citations=())
        bad = replace(good, claims=(bad_claim,))
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "claim_has_no_citation" in result.issues
    assert result.expected_sources_present is False


def test_tampered_source_reference_fails_exact_row_validation(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        citation = replace(
            good.claims[0].citations[0],
            source_ref="fixture:tampered",
        )
        claim = replace(good.claims[0], citations=(citation,))
        bad = replace(good, claims=(claim,))
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "citation_source_row_mismatch" in result.issues


def test_missing_citation_token_in_rendered_answer_fails_coverage(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad = replace(good, answer_text=good.claims[0].text)
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "citation_token_missing_from_answer" in result.issues
    assert result.cited_claim_count == 0


def test_wrong_outcome_fails_case_evaluation(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad = replace(good, outcome="uncertain")
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "outcome_mismatch" in result.issues


def test_wrong_memory_set_fails_even_when_memory_is_authoritative(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        wrong = build_memory_answer_submission(
            connection,
            case=replace(
                case,
                expected_memory_ids=(MEMORY_IDS["private"],),
                expected_source_refs=("fixture:atlas:private-note",),
            ),
            memory_ids=(MEMORY_IDS["private"],),
            authorization=_authorization(),
        )
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=wrong,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "expected_memory_set_mismatch" in result.issues


def test_forbidden_candidate_identifier_in_answer_is_detected(
    tmp_path: Path,
) -> None:
    case = _case("candidate-boundary-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, ())
        bad = replace(
            good,
            answer_text=(
                good.answer_text + " " + CANDIDATE_IDS["unpromoted"]
            ),
        )
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "forbidden_candidate_exposed" in result.issues


def test_duplicate_claims_fail_evaluation(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad = replace(good, claims=(good.claims[0], good.claims[0]))
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "duplicate_claim_id" in result.issues
    assert "duplicate_claim_memory" in result.issues


def test_deleted_memory_claim_is_unsupported(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        bad_claim = replace(
            good.claims[0],
            memory_id=MEMORY_IDS["deleted"],
        )
        bad = replace(good, claims=(bad_claim,))
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=bad,
            authorization=_authorization(),
        )

    assert result.passed is False
    assert "claim_memory_missing" in result.issues


def test_citation_evaluation_requires_authorization(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, case.expected_memory_ids)
        with pytest.raises(Exception, match="denied"):
            evaluate_memory_answer_submission(
                connection,
                case=case,
                submission=good,
                authorization=_authorization(allowed=False),
            )


def test_summary_rejects_duplicate_case_submissions(tmp_path: Path) -> None:
    benchmark = load_memory_evaluation_benchmark()
    with _build_store(tmp_path) as connection:
        submissions = _perfect_submissions(connection)
        duplicate = submissions + (submissions[0],)
        with pytest.raises(
            MemoryCitationEvaluationError,
            match="duplicate case",
        ):
            evaluate_memory_answer_submissions(
                connection,
                benchmark=benchmark,
                submissions=duplicate,
                authorization=_authorization(),
            )


def test_summary_rejects_incomplete_case_coverage(tmp_path: Path) -> None:
    benchmark = load_memory_evaluation_benchmark()
    with _build_store(tmp_path) as connection:
        submissions = _perfect_submissions(connection)[:-1]
        with pytest.raises(
            MemoryCitationEvaluationError,
            match="coverage mismatch",
        ):
            evaluate_memory_answer_submissions(
                connection,
                benchmark=benchmark,
                submissions=submissions,
                authorization=_authorization(),
            )


def test_bad_submission_causes_summary_gate_failure(tmp_path: Path) -> None:
    benchmark = load_memory_evaluation_benchmark()
    with _build_store(tmp_path) as connection:
        submissions = list(_perfect_submissions(connection))
        index = next(
            i
            for i, item in enumerate(submissions)
            if item.case_id == "confirmed-fact-001"
        )
        good = submissions[index]
        claim = replace(good.claims[0], citations=())
        submissions[index] = replace(good, claims=(claim,))
        summary = evaluate_memory_answer_submissions(
            connection,
            benchmark=benchmark,
            submissions=tuple(submissions),
            authorization=_authorization(),
        )

    assert summary.passed_case_count == 12
    assert summary.personal_source_attribution_rate < 1.0
    assert summary.claim_citation_coverage < 1.0
    assert summary.passes_all_p29b_gates is False


def test_nonanswer_cases_pass_only_without_claims(tmp_path: Path) -> None:
    case = _case("unsupported-claim-001")
    with _build_store(tmp_path) as connection:
        good = _submission(connection, case.case_id, ())
        result = evaluate_memory_answer_submission(
            connection,
            case=case,
            submission=good,
            authorization=_authorization(),
        )

    assert result.passed is True
    assert result.claim_count == 0
    assert result.actual_memory_ids == ()
