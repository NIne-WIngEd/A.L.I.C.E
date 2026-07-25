"""P2.9b deterministic source-cited memory-answer tests."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from alice_memory.cited_answer import (
    MemoryAnswerAuthorization,
    MemoryCitedAnswerAuthorizationError,
    MemoryCitedAnswerValidationError,
    build_memory_answer_submission,
)
from alice_memory.evaluation_contract import (
    MemoryEvaluationCase,
    load_memory_evaluation_benchmark,
)
from alice_memory.evaluation_fixtures import (
    CANDIDATE_IDS,
    FIXTURE_CONTENT,
    MEMORY_IDS,
    SOURCE_REFS,
    build_memory_evaluation_fixture,
)
from alice_memory.store import open_memory_store


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _case(case_id: str) -> MemoryEvaluationCase:
    return next(
        case
        for case in load_memory_evaluation_benchmark().cases
        if case.case_id == case_id
    )


def _authorization(**changes) -> MemoryAnswerAuthorization:
    values = {
        "actor": "p29b-test",
        "allowed": True,
        "purpose": "phase2-source-cited-answer-evaluation",
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


def test_confirmed_fact_answer_contains_authoritative_claim_and_source(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("confirmed-fact-001"),
            memory_ids=(MEMORY_IDS["confirmed"],),
            authorization=_authorization(),
        )

    assert answer.outcome == "answerable"
    assert len(answer.claims) == 1
    claim = answer.claims[0]
    assert claim.text == FIXTURE_CONTENT["confirmed"]
    assert claim.memory_id == MEMORY_IDS["confirmed"]
    assert claim.knowledge_status == "verified_fact"
    assert claim.citations[0].source_ref == SOURCE_REFS["confirmed"]
    assert claim.citations[0].token in answer.answer_text


def test_source_attribution_case_cites_exact_source_record(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("source-attribution-001"),
            memory_ids=(MEMORY_IDS["confirmed"],),
            authorization=_authorization(),
        )

    citation = answer.claims[0].citations[0]
    assert citation.memory_source_id
    assert citation.memory_id == MEMORY_IDS["confirmed"]
    assert citation.source_ref == SOURCE_REFS["confirmed"]
    assert citation.support_relation == "supports"


def test_insufficient_evidence_answer_contains_no_claims(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("unsupported-claim-001"),
            memory_ids=(),
            authorization=_authorization(),
        )

    assert answer.claims == ()
    assert answer.outcome == "insufficient_evidence"
    assert "not have enough authoritative memory evidence" in (
        answer.answer_text
    )


def test_denied_answer_contains_no_claims(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("permission-denial-001"),
            memory_ids=(),
            authorization=_authorization(),
        )

    assert answer.claims == ()
    assert answer.outcome == "denied"
    assert answer.answer_text.startswith("Access denied")


def test_temporal_current_answer_uses_current_memory(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("temporal-current-001"),
            memory_ids=(MEMORY_IDS["temporal_current"],),
            authorization=_authorization(),
        )

    assert answer.claims[0].memory_id == MEMORY_IDS["temporal_current"]


def test_temporal_historical_answer_allows_historical_memory(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("temporal-historical-001"),
            memory_ids=(MEMORY_IDS["temporal_old"],),
            authorization=_authorization(),
        )

    assert answer.claims[0].knowledge_status == "historical"


def test_temporal_case_rejects_memory_outside_valid_interval(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="not temporally eligible",
        ):
            build_memory_answer_submission(
                connection,
                case=_case("temporal-current-001"),
                memory_ids=(MEMORY_IDS["temporal_old"],),
                authorization=_authorization(),
            )


def test_correction_case_rejects_corrected_memory(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="Corrected memory",
        ):
            build_memory_answer_submission(
                connection,
                case=_case("correction-001"),
                memory_ids=(MEMORY_IDS["correction_old"],),
                authorization=_authorization(),
            )


def test_conflict_answer_requires_and_preserves_both_records(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("conflict-001"),
            memory_ids=(
                MEMORY_IDS["conflict_a"],
                MEMORY_IDS["conflict_b"],
            ),
            authorization=_authorization(),
        )

    assert answer.outcome == "conflict"
    assert {claim.memory_id for claim in answer.claims} == {
        MEMORY_IDS["conflict_a"],
        MEMORY_IDS["conflict_b"],
    }
    assert answer.answer_text.startswith(
        "Authoritative memory contains a material conflict"
    )


def test_conflict_answer_rejects_single_record(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="at least two",
        ):
            build_memory_answer_submission(
                connection,
                case=_case("conflict-001"),
                memory_ids=(MEMORY_IDS["conflict_a"],),
                authorization=_authorization(),
            )


def test_uncertain_answer_preserves_estimate_status(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("uncertainty-001"),
            memory_ids=(MEMORY_IDS["uncertain"],),
            authorization=_authorization(),
        )

    assert answer.outcome == "uncertain"
    assert answer.claims[0].knowledge_status == "estimate"
    assert answer.answer_text.startswith(
        "The available authoritative memory is uncertain"
    )


def test_answer_requires_explicit_authorization(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(MemoryCitedAnswerAuthorizationError):
            build_memory_answer_submission(
                connection,
                case=_case("confirmed-fact-001"),
                memory_ids=(MEMORY_IDS["confirmed"],),
                authorization=_authorization(allowed=False),
            )


def test_answer_requires_actor_and_purpose(tmp_path: Path) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(MemoryCitedAnswerAuthorizationError):
            build_memory_answer_submission(
                connection,
                case=_case("confirmed-fact-001"),
                memory_ids=(MEMORY_IDS["confirmed"],),
                authorization=_authorization(actor=""),
            )
        with pytest.raises(MemoryCitedAnswerAuthorizationError):
            build_memory_answer_submission(
                connection,
                case=_case("confirmed-fact-001"),
                memory_ids=(MEMORY_IDS["confirmed"],),
                authorization=_authorization(purpose=""),
            )


def test_answer_rejects_highly_sensitive_memory(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="HIGHLY_SENSITIVE",
        ):
            build_memory_answer_submission(
                connection,
                case=case,
                memory_ids=(MEMORY_IDS["sensitive"],),
                authorization=_authorization(),
            )


def test_answer_rejects_deleted_memory(tmp_path: Path) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="does not exist",
        ):
            build_memory_answer_submission(
                connection,
                case=case,
                memory_ids=(MEMORY_IDS["deleted"],),
                authorization=_authorization(),
            )


def test_answer_rejects_unpromoted_candidate_as_authoritative(
    tmp_path: Path,
) -> None:
    case = _case("confirmed-fact-001")
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="does not exist",
        ):
            build_memory_answer_submission(
                connection,
                case=case,
                memory_ids=(CANDIDATE_IDS["unpromoted"],),
                authorization=_authorization(),
            )


def test_answer_rejects_duplicate_and_temporally_invalid_memory_ids(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="duplicate",
        ):
            build_memory_answer_submission(
                connection,
                case=_case("confirmed-fact-001"),
                memory_ids=(
                    MEMORY_IDS["confirmed"],
                    MEMORY_IDS["confirmed"],
                ),
                authorization=_authorization(),
            )
        with pytest.raises(
            MemoryCitedAnswerValidationError,
            match="not temporally eligible",
        ):
            build_memory_answer_submission(
                connection,
                case=_case("temporal-current-001"),
                memory_ids=(MEMORY_IDS["temporal_old"],),
                authorization=_authorization(),
            )


def test_prompt_injection_memory_is_quoted_as_data_with_source(
    tmp_path: Path,
) -> None:
    with _build_store(tmp_path) as connection:
        answer = build_memory_answer_submission(
            connection,
            case=_case("prompt-injection-001"),
            memory_ids=(MEMORY_IDS["injection"],),
            authorization=_authorization(),
        )

    assert answer.claims[0].text == FIXTURE_CONTENT["injection"]
    assert SOURCE_REFS["injection"] in answer.answer_text
    assert len(answer.claims) == 1
