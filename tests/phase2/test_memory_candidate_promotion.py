"""P2.7c authorized memory-candidate promotion tests."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization,
    assess_memory_candidate,
)
from alice_memory.formation import (
    MemoryCandidateCreateRequest,
    MemoryCandidateWriteAuthorization,
    load_memory_candidate,
    propose_memory_candidate,
)
from alice_memory.promotion import (
    MemoryCandidatePromotionAuthorization,
    MemoryCandidatePromotionAuthorizationError,
    MemoryCandidatePromotionStateError,
    MemoryCandidatePromotionValidationError,
    load_candidate_promotion,
    promote_memory_candidate,
)
from alice_memory.service import (
    MemoryContentAccessAuthorization,
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
    load_memory_content,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(vault, repository_root=repository)


def _source(
    *,
    source_type: str = "rayan_direct_statement",
    source_ref: str = "test:promotion-source",
    support_relation: str = "supports",
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
    sources: tuple[MemorySourceSpec, ...] | None = None,
) -> MemoryCandidateCreateRequest:
    model_fields = (
        {
            "policy_version": "formation-v1",
            "model": "qwen3:8b",
            "model_version": "8b",
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
        recorded_at="2026-07-26T00:00:00Z",
        sources=sources if sources is not None else (_source(),),
        origin=origin,
        rayan_confirmed=rayan_confirmed,
        **model_fields,
    )


def _propose(connection, request: MemoryCandidateCreateRequest):
    return propose_memory_candidate(
        connection,
        request=request,
        authorization=MemoryCandidateWriteAuthorization(
            actor="test-proposer",
            allowed=True,
            reason="proposal test",
        ),
        proposed_at="2026-07-26T00:01:00Z",
    )


def _assess(connection, candidate_id: str = "candidate-1"):
    return assess_memory_candidate(
        connection,
        candidate_id=candidate_id,
        authorization=MemoryCandidateAssessmentAuthorization(
            actor="test-assessor",
            allowed=True,
            reason="assessment test",
        ),
        assessed_at="2026-07-26T00:02:00Z",
    )


def _promotion_authorization(
    *,
    candidate_id: str = "candidate-1",
    allowed: bool = True,
    user_confirmed: bool = False,
    authorization_id: str = "promotion-auth-1",
    reason: str = "promotion test",
) -> MemoryCandidatePromotionAuthorization:
    return MemoryCandidatePromotionAuthorization(
        actor="test-promoter",
        allowed=allowed,
        candidate_id=candidate_id,
        authorization_id=authorization_id,
        user_confirmed=user_confirmed,
        reason=reason,
    )


def _promote(
    connection,
    *,
    candidate_id: str = "candidate-1",
    authorization: MemoryCandidatePromotionAuthorization | None = None,
):
    return promote_memory_candidate(
        connection,
        candidate_id=candidate_id,
        authorization=(
            authorization
            if authorization is not None
            else _promotion_authorization(candidate_id=candidate_id)
        ),
        promoted_at="2026-07-26T00:03:00Z",
    )


def _create_authoritative(
    connection,
    *,
    memory_id: str = "memory-existing",
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
            recorded_at="2026-07-26T00:00:00Z",
            sources=(
                _source(
                    source_ref=f"test:authoritative:{memory_id}",
                ),
            ),
            rayan_confirmed=True,
        ),
        authorization=MemoryWriteAuthorization(
            actor="test",
            allowed=True,
            reason="seed memory",
        ),
        created_at="2026-07-26T00:02:30Z",
    )


def test_promotion_creates_one_authoritative_memory_atomically(
    tmp_path: Path,
) -> None:
    content = "Rayan is building A.L.I.C.E."

    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request(content=content))
        assessment = _assess(connection)
        result = _promote(connection)

        assert assessment.outcome == "promotion_eligible"
        assert result.assessment_outcome == "promotion_eligible"
        assert result.candidate.candidate_state == "promoted"
        assert result.candidate.promoted_memory_id == result.memory.memory_id
        assert result.memory.content_sha256 == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_sources"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_derivations"
        ).fetchone()[0] == 1

        assert load_memory_content(
            connection,
            memory_id=result.memory.memory_id,
            authorization=MemoryContentAccessAuthorization(
                actor="test",
                allowed=True,
                reason="verify promoted plaintext",
            ),
        ) == content


def test_promotion_requires_explicit_authorization(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            _promote(
                connection,
                authorization=_promotion_authorization(allowed=False),
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0
        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "validated"


def test_authorization_must_be_bound_to_exact_candidate(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            _promote(
                connection,
                authorization=_promotion_authorization(
                    candidate_id="candidate-other",
                ),
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "authorization_id",
    ("", "contains spaces", "x"),
)
def test_authorization_id_must_be_audit_safe(
    tmp_path: Path,
    authorization_id: str,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            _promote(
                connection,
                authorization=_promotion_authorization(
                    authorization_id=authorization_id,
                ),
            )


def test_unassessed_candidate_cannot_be_promoted(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())

        with pytest.raises(MemoryCandidatePromotionStateError):
            _promote(connection)

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_rejected_candidate_cannot_be_promoted(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create_authoritative(connection)
        _propose(connection, _candidate_request())
        assert _assess(connection).outcome == "rejected"

        with pytest.raises(MemoryCandidatePromotionStateError):
            _promote(connection)

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1



def test_same_key_memory_requires_later_transition_aware_promotion(
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
        assert "current_memory_exists_for_key" in assessment.reason_codes

        with pytest.raises(MemoryCandidatePromotionValidationError):
            _promote(
                connection,
                authorization=_promotion_authorization(
                    user_confirmed=True,
                ),
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "validated"

def test_review_required_candidate_needs_user_confirmation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                confidence=0.60,
                rayan_confirmed=False,
            ),
        )
        assert _assess(connection).outcome == "review_required"

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            _promote(connection)

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_user_confirmation_promotes_reviewed_candidate_and_is_recorded(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                confidence=0.60,
                rayan_confirmed=False,
            ),
        )
        _assess(connection)
        result = _promote(
            connection,
            authorization=_promotion_authorization(
                user_confirmed=True,
            ),
        )

        assert result.assessment_outcome == "review_required"
        assert result.user_confirmed is True
        assert result.memory.rayan_confirmed is True
        assert result.derivation_types == (
            "explicit_user",
            "human_confirmed",
        )


def test_model_proposal_requires_confirmation_and_preserves_derivation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(
                origin="model_proposed",
                knowledge_status="alice_inference",
                rayan_confirmed=False,
            ),
        )
        assert _assess(connection).outcome == "review_required"

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            _promote(connection)

        result = _promote(
            connection,
            authorization=_promotion_authorization(
                user_confirmed=True,
            ),
        )

        rows = connection.execute(
            """
            SELECT
                derivation_type,
                policy_version,
                model,
                model_version,
                prompt_version,
                run_id
            FROM memory_derivations
            WHERE memory_id = ?
            ORDER BY derivation_type
            """,
            (result.memory.memory_id,),
        ).fetchall()

        assert result.memory.rayan_confirmed is True
        assert result.derivation_types == (
            "model_proposed",
            "human_confirmed",
        )
        assert [row["derivation_type"] for row in rows] == [
            "human_confirmed",
            "model_proposed",
        ]
        model_row = next(
            row for row in rows if row["derivation_type"] == "model_proposed"
        )
        assert model_row["policy_version"] == "formation-v1"
        assert model_row["model"] == "qwen3:8b"
        assert model_row["model_version"] == "8b"
        assert model_row["prompt_version"] == "candidate-v1"
        assert model_row["run_id"] == "run-1"


def test_promotion_is_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)
        first = _promote(connection)
        second = promote_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=_promotion_authorization(
                authorization_id="promotion-auth-retry",
            ),
            promoted_at="2026-07-26T00:04:00Z",
        )

        assert second == first
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_candidate_events
            WHERE candidate_id = ? AND event_type = 'promoted'
            """,
            ("candidate-1",),
        ).fetchone()[0] == 1


def test_final_revalidation_blocks_stale_duplicate_assessment(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        assert _assess(connection).outcome == "promotion_eligible"
        _create_authoritative(connection)

        with pytest.raises(MemoryCandidatePromotionValidationError) as exc:
            _promote(connection)

        assert "duplicates_authoritative_memory" in str(exc.value)
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "validated"


def test_promotion_rolls_back_when_derivation_write_fails(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)
        connection.execute(
            """
            CREATE TRIGGER fail_promotion_derivation
            BEFORE INSERT ON memory_derivations
            BEGIN
                SELECT RAISE(ABORT, 'forced derivation failure');
            END
            """
        )

        with pytest.raises(MemoryCandidatePromotionValidationError):
            _promote(connection)

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_sources"
        ).fetchone()[0] == 0
        candidate = load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        )
        assert candidate.candidate_state == "validated"
        assert candidate.promoted_memory_id is None


def test_candidate_digest_mismatch_blocks_promotion(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)
        connection.execute(
            """
            UPDATE memory_candidates
            SET content = 'tampered candidate plaintext'
            WHERE candidate_id = 'candidate-1'
            """
        )

        with pytest.raises(MemoryCandidatePromotionValidationError):
            _promote(connection)

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_promotion_event_is_sanitized(tmp_path: Path) -> None:
    content = "Private plaintext that must not enter promotion audit metadata."
    sensitive_reason = "private free-form reason that must not be logged"

    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request(content=content))
        _assess(connection)
        result = _promote(
            connection,
            authorization=_promotion_authorization(
                reason=sensitive_reason,
            ),
        )
        event = connection.execute(
            """
            SELECT details_json
            FROM memory_candidate_events
            WHERE candidate_id = ? AND event_type = 'promoted'
            """,
            ("candidate-1",),
        ).fetchone()
        details = json.loads(event["details_json"])

        assert content not in event["details_json"]
        assert sensitive_reason not in event["details_json"]
        assert details["promoted_memory_id"] == result.memory.memory_id
        assert details["authorization_id"] == "promotion-auth-1"


def test_load_candidate_promotion_fails_before_promotion(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(connection, _candidate_request())
        _assess(connection)

        with pytest.raises(MemoryCandidatePromotionStateError):
            load_candidate_promotion(
                connection,
                candidate_id="candidate-1",
            )
