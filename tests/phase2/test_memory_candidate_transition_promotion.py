"""P2.7d transition-aware memory-candidate promotion tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization,
    load_latest_candidate_assessment,
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
    promote_memory_candidate,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
    load_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.transition_promotion import (
    MemoryCandidateTransitionAuthorization,
    MemoryCandidateTransitionAuthorizationError,
    MemoryCandidateTransitionStateError,
    MemoryCandidateTransitionValidationError,
    load_candidate_transition_promotion,
    promote_memory_candidate_with_transition,
)


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(vault, repository_root=repository)


def _source(
    *,
    source_ref: str = "test:transition-candidate",
    support_relation: str = "supports",
) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="rayan_direct_statement",
        source_ref=source_ref,
        support_relation=support_relation,
    )


def _authoritative_request(
    *,
    memory_id: str = "memory-old",
    content: str = "Rayan lives in the old location.",
    memory_key: str = "profile.location",
    classification: str = "PRIVATE",
    valid_from: str | None = "2026-01-01T00:00:00Z",
    valid_to: str | None = None,
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key=memory_key,
        category="profile",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification=classification,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at="2026-07-27T00:00:00Z",
        sources=(
            _source(source_ref=f"test:authoritative:{memory_id}"),
        ),
        rayan_confirmed=True,
    )


def _create_authoritative(connection, **kwargs):
    return create_memory(
        connection,
        request=_authoritative_request(**kwargs),
        authorization=MemoryWriteAuthorization(
            actor="test-seeder",
            allowed=True,
            reason="seed transition target",
        ),
        created_at="2026-07-27T00:00:30Z",
    )


def _candidate_request(
    *,
    candidate_id: str = "candidate-1",
    content: str = "Rayan lives in the new location.",
    memory_key: str = "profile.location",
    classification: str = "PRIVATE",
    valid_from: str | None = "2026-07-27T00:00:00Z",
    support_relation: str = "supports",
    origin: str = "explicit_user",
    rayan_confirmed: bool = True,
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
        category="profile",
        knowledge_status="rayan_statement",
        confidence=0.95,
        data_classification=classification,
        valid_from=valid_from,
        recorded_at="2026-07-27T00:01:00Z",
        sources=(
            _source(
                source_ref=f"test:candidate:{candidate_id}",
                support_relation=support_relation,
            ),
        ),
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
            reason="transition proposal",
        ),
        proposed_at="2026-07-27T00:01:30Z",
    )


def _assess(connection, candidate_id: str = "candidate-1"):
    return assess_memory_candidate(
        connection,
        candidate_id=candidate_id,
        authorization=MemoryCandidateAssessmentAuthorization(
            actor="test-assessor",
            allowed=True,
            reason="transition assessment",
        ),
        assessed_at="2026-07-27T00:02:00Z",
    )


def _authorization(
    transition_type: str,
    *,
    candidate_id: str = "candidate-1",
    target_memory_id: str = "memory-old",
    allowed: bool = True,
    user_confirmed: bool = True,
    authorization_id: str = "transition-auth-1",
) -> MemoryCandidateTransitionAuthorization:
    return MemoryCandidateTransitionAuthorization(
        actor="test-reviewer",
        allowed=allowed,
        candidate_id=candidate_id,
        target_memory_id=target_memory_id,
        transition_type=transition_type,
        authorization_id=authorization_id,
        user_confirmed=user_confirmed,
        reason="explicit transition decision",
    )


def _transition(
    connection,
    transition_type: str,
    *,
    candidate_id: str = "candidate-1",
    target_memory_id: str = "memory-old",
    authorization: MemoryCandidateTransitionAuthorization | None = None,
):
    return promote_memory_candidate_with_transition(
        connection,
        candidate_id=candidate_id,
        authorization=(
            authorization
            if authorization is not None
            else _authorization(
                transition_type,
                candidate_id=candidate_id,
                target_memory_id=target_memory_id,
            )
        ),
        promoted_at="2026-07-27T00:03:00Z",
    )


def _seed_review_required(connection, **candidate_kwargs):
    _create_authoritative(connection)
    _propose(connection, _candidate_request(**candidate_kwargs))
    assessment = _assess(connection)
    assert assessment.outcome == "review_required"
    assert "current_memory_exists_for_key" in assessment.reason_codes


def test_correction_promotes_candidate_and_preserves_old_record(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        result = _transition(connection, "correction")

        assert result.transition_type == "correction"
        assert result.transition_action == "corrected"
        assert result.candidate.candidate_state == "promoted"
        assert result.memory.memory_id == result.candidate.promoted_memory_id
        assert result.memory.validity_state == "current"
        assert result.memory.rayan_confirmed is True
        assert result.target.memory_id == "memory-old"
        assert result.target.validity_state == "historical"
        assert result.target.knowledge_status == "superseded"
        assert result.relation is not None
        assert result.relation.relation_type == "corrects"
        assert result.relation.from_memory_id == result.memory.memory_id
        assert result.relation.to_memory_id == "memory-old"
        assert result.derivation_types == ("explicit_user", "human_confirmed")

        source_count = connection.execute(
            "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?",
            (result.memory.memory_id,),
        ).fetchone()[0]
        assert source_count == 1


def test_supersession_closes_target_validity_interval(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(
            connection,
            valid_from="2026-08-01T00:00:00Z",
        )

        result = _transition(connection, "supersession")

        assert result.transition_action == "superseded"
        assert result.target.validity_state == "historical"
        assert result.target.knowledge_status == "historical"
        assert result.target.valid_to == "2026-08-01T00:00:00Z"
        assert result.memory.valid_from == "2026-08-01T00:00:00Z"
        assert result.relation is not None
        assert result.relation.relation_type == "supersedes"


def test_conflict_preserves_both_and_marks_both_disputed(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        result = _transition(connection, "conflict")

        assert result.transition_action == "conflicted"
        assert result.memory.validity_state == "disputed"
        assert result.memory.knowledge_status == "disputed"
        assert result.target.validity_state == "disputed"
        assert result.target.knowledge_status == "disputed"
        assert result.relation is not None
        assert result.relation.relation_type == "conflicts_with"


def test_exact_duplicate_is_resolved_as_rejected_noop(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        content = "Rayan lives in the same location."
        _create_authoritative(connection, content=content)
        _propose(connection, _candidate_request(content=content))
        assessment = _assess(connection)
        assert assessment.outcome == "rejected"

        result = _transition(
            connection,
            "duplicate",
            authorization=_authorization(
                "duplicate",
                user_confirmed=False,
            ),
        )

        assert result.is_noop is True
        assert result.candidate.candidate_state == "rejected"
        assert result.candidate.promoted_memory_id is None
        assert result.memory.memory_id == "memory-old"
        assert result.relation is None
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1


def test_stale_validated_candidate_can_fail_closed_as_duplicate_noop(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        content = "Rayan lives in the same location."
        _propose(connection, _candidate_request(content=content))
        assessment = _assess(connection)
        assert assessment.outcome == "promotion_eligible"
        _create_authoritative(connection, content=content)

        result = _transition(
            connection,
            "duplicate",
            authorization=_authorization(
                "duplicate",
                user_confirmed=False,
            ),
        )

        assert result.candidate.candidate_state == "rejected"
        assert "duplicates_authoritative_memory" in result.reason_codes
        latest = load_latest_candidate_assessment(
            connection,
            candidate_id="candidate-1",
        )
        assert latest.outcome == "rejected"
        assert "duplicates_authoritative_memory" in latest.reason_codes


def test_transition_requires_explicit_authorization(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "correction",
                authorization=_authorization(
                    "correction",
                    allowed=False,
                ),
            )


def test_authorization_must_be_bound_to_exact_candidate(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "correction",
                authorization=_authorization(
                    "correction",
                    candidate_id="different-candidate",
                ),
            )


def test_nonduplicate_transition_requires_user_confirmation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "correction",
                authorization=_authorization(
                    "correction",
                    user_confirmed=False,
                ),
            )


def test_invalid_transition_type_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "rewrite",
                authorization=_authorization("rewrite"),
            )


def test_unsafe_authorization_id_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "correction",
                authorization=_authorization(
                    "correction",
                    authorization_id="unsafe authorization",
                ),
            )


def test_unassessed_candidate_cannot_use_transition_path(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create_authoritative(connection)
        _propose(connection, _candidate_request())

        with pytest.raises(MemoryCandidateTransitionStateError):
            _transition(connection, "correction")


def test_target_must_match_deterministic_assessment(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)
        _create_authoritative(
            connection,
            memory_id="memory-unrelated",
            content="Unrelated value",
            memory_key="profile.unrelated",
        )

        with pytest.raises(MemoryCandidateTransitionValidationError):
            _transition(
                connection,
                "correction",
                target_memory_id="memory-unrelated",
            )


def test_exact_duplicate_cannot_be_labeled_correction(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        content = "Rayan lives in the same location."
        _create_authoritative(connection, content=content)
        _propose(connection, _candidate_request(content=content))
        _assess(connection)

        with pytest.raises(MemoryCandidateTransitionValidationError):
            _transition(connection, "correction")


def test_correction_cannot_downgrade_target_classification(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection, classification="INTERNAL")

        with pytest.raises(MemoryCandidateTransitionValidationError):
            _transition(connection, "correction")

        target = load_memory(connection, memory_id="memory-old")
        assert target.validity_state == "current"
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        ).candidate_state == "validated"


def test_supersession_requires_candidate_valid_from(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection, valid_from=None)

        with pytest.raises(MemoryCandidateTransitionValidationError):
            _transition(connection, "supersession")

        assert load_memory(connection, memory_id="memory-old").valid_to is None
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1


def test_transition_promotion_is_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)

        first = _transition(connection, "correction")
        second = _transition(connection, "correction")

        assert second.memory.memory_id == first.memory.memory_id
        assert second.relation == first.relation
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 2
        assert connection.execute(
            """
            SELECT COUNT(*) FROM memory_candidate_events
            WHERE candidate_id = 'candidate-1' AND event_type = 'promoted'
            """
        ).fetchone()[0] == 1


def test_duplicate_resolution_is_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        content = "Rayan lives in the same location."
        _create_authoritative(connection, content=content)
        _propose(connection, _candidate_request(content=content))
        _assess(connection)
        authorization = _authorization(
            "duplicate",
            user_confirmed=False,
        )

        first = _transition(
            connection,
            "duplicate",
            authorization=authorization,
        )
        second = _transition(
            connection,
            "duplicate",
            authorization=authorization,
        )

        assert second.memory.memory_id == first.memory.memory_id
        assert connection.execute(
            """
            SELECT COUNT(*) FROM memory_candidate_events
            WHERE candidate_id = 'candidate-1' AND event_type = 'inspected'
            """
        ).fetchone()[0] == 1


def test_transition_rolls_back_when_derivation_insert_fails(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(connection)
        connection.execute(
            """
            CREATE TRIGGER fail_transition_derivation
            BEFORE INSERT ON memory_derivations
            BEGIN
                SELECT RAISE(ABORT, 'forced derivation failure');
            END
            """
        )

        with pytest.raises(MemoryCandidateTransitionValidationError):
            _transition(connection, "correction")

        target = load_memory(connection, memory_id="memory-old")
        assert target.validity_state == "current"
        assert target.knowledge_status == "rayan_statement"
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        candidate = load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        )
        assert candidate.candidate_state == "validated"
        assert candidate.promoted_memory_id is None


def test_model_candidate_still_requires_human_confirmation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _seed_review_required(
            connection,
            origin="model_proposed",
            rayan_confirmed=False,
        )

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            _transition(
                connection,
                "conflict",
                authorization=_authorization(
                    "conflict",
                    user_confirmed=False,
                ),
            )


def test_transition_event_is_sanitized(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        secret_marker = "PRIVATE-CANDIDATE-CONTENT-DO-NOT-AUDIT"
        _seed_review_required(connection, content=secret_marker)

        result = _transition(connection, "correction")
        row = connection.execute(
            """
            SELECT details_json FROM memory_candidate_events
            WHERE candidate_id = 'candidate-1' AND event_type = 'promoted'
            """
        ).fetchone()
        details_json = str(row["details_json"])
        details = json.loads(details_json)

        assert secret_marker not in details_json
        assert details["promotion_mode"] == "transition_aware"
        assert details["transition_type"] == "correction"
        assert details["target_memory_id"] == "memory-old"
        assert details["promoted_memory_id"] == result.memory.memory_id


def test_ordinary_promotion_is_not_misreported_as_transition_promotion(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _propose(
            connection,
            _candidate_request(memory_key="project.new-key"),
        )
        assessment = _assess(connection)
        assert assessment.outcome == "promotion_eligible"
        promote_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=MemoryCandidatePromotionAuthorization(
                actor="test-promoter",
                allowed=True,
                candidate_id="candidate-1",
                authorization_id="ordinary-auth-1",
            ),
            promoted_at="2026-07-27T00:03:00Z",
        )

        with pytest.raises(MemoryCandidateTransitionStateError):
            load_candidate_transition_promotion(
                connection,
                candidate_id="candidate-1",
            )
