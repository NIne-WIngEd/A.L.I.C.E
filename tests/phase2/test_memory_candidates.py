"""P2.7a non-authoritative memory-candidate staging tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alice_memory.formation import (
    MemoryCandidateAlreadyExistsError,
    MemoryCandidateAuthorizationError,
    MemoryCandidateCreateRequest,
    MemoryCandidateValidationError,
    MemoryCandidateWriteAuthorization,
    load_memory_candidate,
    load_memory_candidate_content,
    propose_memory_candidate,
)
from alice_memory.service import MemoryContentAccessAuthorization
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


def _source() -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="rayan_direct_statement",
        source_ref="test-suite:candidate-statement",
        support_relation="supports",
    )


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(
        vault,
        repository_root=repository,
    )


def _authorization(
    *,
    allowed: bool = True,
    reason: str = "candidate test",
) -> MemoryCandidateWriteAuthorization:
    return MemoryCandidateWriteAuthorization(
        actor="test",
        allowed=allowed,
        reason=reason,
    )


def _content_authorization() -> MemoryContentAccessAuthorization:
    return MemoryContentAccessAuthorization(
        actor="test",
        allowed=True,
        reason="inspect candidate plaintext",
    )


def _request(
    *,
    candidate_id: str = "candidate-1",
    content: str = "Rayan is building A.L.I.C.E.",
    classification: str = "PRIVATE",
    origin: str = "explicit_user",
    rayan_confirmed: bool = True,
    policy_version: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    run_id: str | None = None,
    recorded_at: str = "2026-07-24T00:00:00Z",
    valid_from: str | None = None,
    sources: tuple[MemorySourceSpec, ...] | None = None,
) -> MemoryCandidateCreateRequest:
    return MemoryCandidateCreateRequest(
        candidate_id=candidate_id,
        content=content,
        memory_key="project.alice.status",
        category="project",
        knowledge_status="rayan_statement",
        confidence=0.95,
        data_classification=classification,
        recorded_at=recorded_at,
        sources=sources if sources is not None else (_source(),),
        origin=origin,
        valid_from=valid_from,
        rayan_confirmed=rayan_confirmed,
        policy_version=policy_version,
        model=model,
        prompt_version=prompt_version,
        run_id=run_id,
    )


def test_authorized_proposal_is_staged_but_not_authoritative(
    tmp_path: Path,
) -> None:
    content = "Rayan is building A.L.I.C.E."

    with _open(tmp_path) as connection:
        candidate = propose_memory_candidate(
            connection,
            request=_request(content=content),
            authorization=_authorization(),
            proposed_at="2026-07-24T00:01:00Z",
        )

        assert candidate.candidate_id == "candidate-1"
        assert candidate.candidate_state == "proposed"
        assert candidate.origin == "explicit_user"
        assert candidate.proposed_by == "test"
        assert not hasattr(candidate, "content")
        assert candidate.content_sha256 == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_sources"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_proposal_requires_explicit_authorization(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryCandidateAuthorizationError):
            propose_memory_candidate(
                connection,
                request=_request(),
                authorization=_authorization(allowed=False),
                proposed_at="2026-07-24T00:01:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0


def test_candidate_event_is_sanitized(
    tmp_path: Path,
) -> None:
    content = "Private candidate text that must not enter audit metadata."
    sensitive_reason = "private reason that must not be persisted"

    with _open(tmp_path) as connection:
        propose_memory_candidate(
            connection,
            request=_request(content=content),
            authorization=_authorization(reason=sensitive_reason),
            proposed_at="2026-07-24T00:01:00Z",
        )

        row = connection.execute(
            """
            SELECT event_type, actor, details_json
            FROM memory_candidate_events
            WHERE candidate_id = ?
            """,
            ("candidate-1",),
        ).fetchone()

        assert row["event_type"] == "proposed"
        assert row["actor"] == "test"
        assert content not in row["details_json"]
        assert sensitive_reason not in row["details_json"]


def test_candidate_plaintext_requires_explicit_content_authorization(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        propose_memory_candidate(
            connection,
            request=_request(),
            authorization=_authorization(),
            proposed_at="2026-07-24T00:01:00Z",
        )

        with pytest.raises(MemoryCandidateAuthorizationError):
            load_memory_candidate_content(
                connection,
                candidate_id="candidate-1",
                authorization=None,
            )

        assert load_memory_candidate_content(
            connection,
            candidate_id="candidate-1",
            authorization=_content_authorization(),
        ) == "Rayan is building A.L.I.C.E."


def test_public_candidate_load_is_metadata_only(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        propose_memory_candidate(
            connection,
            request=_request(),
            authorization=_authorization(),
            proposed_at="2026-07-24T00:01:00Z",
        )

        candidate = load_memory_candidate(
            connection,
            candidate_id="candidate-1",
        )

        assert candidate.candidate_state == "proposed"
        assert not hasattr(candidate, "content")


@pytest.mark.parametrize("classification", ["HIGHLY_SENSITIVE", "SECRETS"])
def test_unsafe_candidate_classifications_fail_closed(
    tmp_path: Path,
    classification: str,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryCandidateValidationError):
            propose_memory_candidate(
                connection,
                request=_request(classification=classification),
                authorization=_authorization(),
                proposed_at="2026-07-24T00:01:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0


def test_model_proposal_requires_derivation_metadata(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryCandidateValidationError):
            propose_memory_candidate(
                connection,
                request=_request(
                    origin="model_proposed",
                    rayan_confirmed=False,
                ),
                authorization=_authorization(),
                proposed_at="2026-07-24T00:01:00Z",
            )


def test_model_proposal_cannot_claim_user_confirmation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryCandidateValidationError):
            propose_memory_candidate(
                connection,
                request=_request(
                    origin="model_proposed",
                    rayan_confirmed=True,
                    policy_version="formation-v1",
                    model="qwen3:8b",
                    prompt_version="candidate-v1",
                    run_id="run-1",
                ),
                authorization=_authorization(),
                proposed_at="2026-07-24T00:01:00Z",
            )


def test_model_proposal_with_complete_metadata_is_staged(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        candidate = propose_memory_candidate(
            connection,
            request=_request(
                origin="model_proposed",
                rayan_confirmed=False,
                policy_version="formation-v1",
                model="qwen3:8b",
                prompt_version="candidate-v1",
                run_id="run-1",
            ),
            authorization=_authorization(),
            proposed_at="2026-07-24T00:01:00Z",
        )

        assert candidate.origin == "model_proposed"
        assert candidate.rayan_confirmed is False
        assert candidate.model == "qwen3:8b"
        assert candidate.run_id == "run-1"


def test_candidate_timestamps_are_normalized_to_utc(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        candidate = propose_memory_candidate(
            connection,
            request=_request(
                recorded_at="2026-07-24T14:00:00+14:00",
                valid_from="2026-07-23T19:00:00-05:00",
            ),
            authorization=_authorization(),
            proposed_at="2026-07-24T01:00:00+01:00",
        )

        assert candidate.recorded_at == "2026-07-24T00:00:00Z"
        assert candidate.valid_from == "2026-07-24T00:00:00Z"
        assert candidate.created_at == "2026-07-24T00:00:00Z"


def test_duplicate_candidate_id_is_rejected(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        propose_memory_candidate(
            connection,
            request=_request(),
            authorization=_authorization(),
            proposed_at="2026-07-24T00:01:00Z",
        )

        with pytest.raises(MemoryCandidateAlreadyExistsError):
            propose_memory_candidate(
                connection,
                request=_request(),
                authorization=_authorization(),
                proposed_at="2026-07-24T00:02:00Z",
            )


def test_invalid_provenance_leaves_no_partial_candidate(
    tmp_path: Path,
) -> None:
    duplicate_sources = (_source(), _source())

    with _open(tmp_path) as connection:
        with pytest.raises(MemoryCandidateValidationError):
            propose_memory_candidate(
                connection,
                request=_request(sources=duplicate_sources),
                authorization=_authorization(),
                proposed_at="2026-07-24T00:01:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidates"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_sources"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_candidate_events"
        ).fetchone()[0] == 0
