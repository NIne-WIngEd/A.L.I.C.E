"""P2.7b deterministic memory-candidate assessment tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    ASSESSMENT_VERSION,
    MemoryCandidateAssessmentAuthorization,
    MemoryCandidateAssessmentAuthorizationError,
    MemoryCandidateAssessmentStateError,
    assess_memory_candidate,
    load_latest_candidate_assessment,
)
from alice_memory.formation import (
    MemoryCandidateCreateRequest,
    MemoryCandidateWriteAuthorization,
    load_memory_candidate,
    propose_memory_candidate,
)
from alice_memory.lexical_index import authoritative_retrieval_digest
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(vault, repository_root=repository)


def _candidate_authorization() -> MemoryCandidateWriteAuthorization:
    return MemoryCandidateWriteAuthorization(
        actor="test-proposer",
        allowed=True,
        reason="candidate test",
    )


def _assessment_authorization(
    *,
    allowed: bool = True,
    reason: str = "assessment test",
) -> MemoryCandidateAssessmentAuthorization:
    return MemoryCandidateAssessmentAuthorization(
        actor="test-assessor",
        allowed=allowed,
        reason=reason,
    )


def _source(
    *,
    source_type: str = "rayan_direct_statement",
    support_relation: str = "supports",
    source_ref: str = "test:candidate-source",
) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type=source_type,
        source_ref=source_ref,
        support_relation=support_relation,
    )


def _candidate_request(
    *,
    candidate_id: str = "candidate-1",
    content: str = "Rayan is building A.L.I.C.E.",
    memory_key: str | None = "project.alice.status",
    origin: str = "explicit_user",
    knowledge_status: str = "rayan_statement",
    confidence: float = 0.95,
    rayan_confirmed: bool = True,
    verified_at: str | None = None,
    validity_state: str = "current",
    retention_state: str = "durable",
    sources: tuple[MemorySourceSpec, ...] | None = None,
) -> MemoryCandidateCreateRequest:
    model_fields = (
        {
            "policy_version": "formation-v1",
            "model": "qwen3:8b",
            "prompt_version": "candidate-v1",
            "run_id": "run-1",
        }
        if origin == "model_proposed"
        else {}
    )
    return MemoryCandidateCreateRequest(
        candidate_id=candidate_id,
        content=content,
        memory_key=memory_key,
        category="project",
        knowledge_status=knowledge_status,
        confidence=confidence,
        data_classification="PRIVATE",
        recorded_at="2026-07-25T00:00:00Z",
        sources=sources if sources is not None else (_source(),),
        origin=origin,
        verified_at=verified_at,
        rayan_confirmed=rayan_confirmed,
        validity_state=validity_state,
        retention_state=retention_state,
        **model_fields,
    )


def _propose(connection, request: MemoryCandidateCreateRequest):
    return propose_memory_candidate(
        connection,
        request=request,
        authorization=_candidate_authorization(),
        proposed_at="2026-07-25T00:01:00Z",
    )


def _assess(connection, candidate_id: str = "candidate-1"):
    return assess_memory_candidate(
        connection,
        candidate_id=candidate_id,
        authorization=_assessment_authorization(),
        assessed_at="2026-07-25T00:02:00Z",
    )


def _create_authoritative(
    connection,
    *,
    memory_id: str = "memory-1",
    content: str = "Rayan is building A.L.I.C.E.",
    memory_key: str = "project.alice.status",
):
    return create_memory(
        connection,
        request=MemoryCreateRequest(
            memory_id=memory_id,
            content=content,
            memory_key=memory_key,
            category="project",
            knowledge_status="rayan_statement",
            confidence=1.0,
            data_classification="PRIVATE",
            recorded_at="2026-07-25T00:00:00Z",
            sources=(
                _source(source_ref=f"test:memory-source:{memory_id}"),
            ),
            rayan_confirmed=True,
        ),
        authorization=MemoryWriteAuthorization(
            actor="test",
            allowed=True,
            reason="seed authoritative memory",
        ),
        created_at="2026-07-25T00:00:30Z",
    )


def test_confirmed_explicit_user_candidate_becomes_promotion_eligible(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        assessment = _assess(connection)

        assert assessment.outcome == "promotion_eligible"
        assert assessment.reason_codes == (
            "deterministic_eligibility_rules_passed",
        )
        assert assessment.assessment_version == ASSESSMENT_VERSION
        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "validated"
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_assessment_requires_explicit_authorization(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())

        with pytest.raises(MemoryCandidateAssessmentAuthorizationError):
            assess_memory_candidate(
                connection,
                candidate_id="candidate-1",
                authorization=_assessment_authorization(allowed=False),
                assessed_at="2026-07-25T00:02:00Z",
            )

        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "proposed"


def test_model_proposal_always_requires_user_review(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                origin="model_proposed",
                knowledge_status="alice_inference",
                rayan_confirmed=False,
            ),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "review_required"
        assert "model_proposals_require_user_review" in assessment.reason_codes
        assert "knowledge_status_requires_review" in assessment.reason_codes


def test_unconfirmed_explicit_user_candidate_requires_review(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(rayan_confirmed=False),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "review_required"
        assert "explicit_user_candidate_not_confirmed" in assessment.reason_codes


def test_verified_deterministic_import_can_become_eligible(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                origin="deterministic_import",
                knowledge_status="verified_fact",
                confidence=0.95,
                rayan_confirmed=False,
                verified_at="2026-07-25T00:00:00Z",
                sources=(
                    _source(
                        source_type="phase1_source",
                        source_ref="phase1:source:1",
                    ),
                ),
            ),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "promotion_eligible"


def test_low_confidence_candidate_requires_review(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request(confidence=0.60))
        assessment = _assess(connection)

        assert assessment.outcome == "review_required"
        assert "confidence_below_eligibility_threshold" in assessment.reason_codes


def test_exact_authoritative_duplicate_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create_authoritative(connection)
        _propose(connection, _candidate_request())
        assessment = _assess(connection)
        candidate = load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        )

        assert assessment.outcome == "rejected"
        assert assessment.matched_memory_ids == ("memory-1",)
        assert "duplicates_authoritative_memory" in assessment.reason_codes
        assert candidate.candidate_state == "rejected"
        assert candidate.rejection_reason is not None


def test_same_key_with_different_current_memory_requires_review(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create_authoritative(
            connection,
            content="A.L.I.C.E. is in an earlier phase.",
        )
        _propose(connection, _candidate_request())
        assessment = _assess(connection)

        assert assessment.outcome == "review_required"
        assert assessment.matched_memory_ids == ("memory-1",)
        assert "current_memory_exists_for_key" in assessment.reason_codes


def test_only_contradicting_provenance_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                sources=(
                    _source(support_relation="contradicts"),
                ),
            ),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "rejected"
        assert "provenance_only_contradicts" in assessment.reason_codes


def test_mixed_contradictory_provenance_requires_review(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                sources=(
                    _source(source_ref="test:support"),
                    _source(
                        support_relation="contradicts",
                        source_ref="test:contradiction",
                    ),
                ),
            ),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "review_required"
        assert "contradictory_provenance_present" in assessment.reason_codes


def test_archived_candidate_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(retention_state="archived"),
        )
        assessment = _assess(connection)

        assert assessment.outcome == "rejected"
        assert "candidate_starts_archived" in assessment.reason_codes


def test_later_exact_candidate_duplicate_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request(candidate_id="candidate-1"))
        propose_memory_candidate(
            connection,
            request=_candidate_request(candidate_id="candidate-2"),
            authorization=_candidate_authorization(),
            proposed_at="2026-07-25T00:01:01Z",
        )

        assessment = assess_memory_candidate(
            connection,
            candidate_id="candidate-2",
            authorization=_assessment_authorization(),
            assessed_at="2026-07-25T00:02:00Z",
        )

        assert assessment.outcome == "rejected"
        assert assessment.matched_candidate_ids == ("candidate-1",)
        assert "duplicates_earlier_candidate" in assessment.reason_codes


def test_reassessment_is_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        first = _assess(connection)
        second = assess_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=_assessment_authorization(),
            assessed_at="2026-07-25T00:03:00Z",
        )

        assert second == first
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_candidate_events
            WHERE candidate_id = ?
              AND event_type IN ('validated', 'rejected')
            """,
            ("candidate-1",),
        ).fetchone()[0] == 1


def test_assessment_event_is_sanitized(tmp_path: Path) -> None:
    content = "Private candidate plaintext must never enter assessment audit data."
    secret_reason = "private authorization reason"

    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request(content=content))
        assess_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=_assessment_authorization(reason=secret_reason),
            assessed_at="2026-07-25T00:02:00Z",
        )
        row = connection.execute(
            """
            SELECT details_json
            FROM memory_candidate_events
            WHERE candidate_id = ? AND event_type = 'validated'
            """,
            ("candidate-1",),
        ).fetchone()

        assert content not in row["details_json"]
        assert secret_reason not in row["details_json"]


def test_assessed_candidates_remain_outside_authoritative_retrieval(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        before = authoritative_retrieval_digest(connection)
        _propose(connection, _candidate_request())
        _assess(connection)
        after = authoritative_retrieval_digest(connection)

        assert before == after
        assert after[1] == 0


def test_loading_assessment_before_assessment_fails(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())

        with pytest.raises(MemoryCandidateAssessmentStateError):
            load_latest_candidate_assessment(
                connection,
                candidate_id="candidate-1",
            )
