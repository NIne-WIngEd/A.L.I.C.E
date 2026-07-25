"""P2.7e adversarial security gates for memory formation and promotion."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.candidate_assessment import (
    MemoryCandidateAssessmentAuthorization,
    assess_memory_candidate,
)
from alice_memory.formation import (
    MemoryCandidateCreateRequest,
    MemoryCandidateWriteAuthorization,
    propose_memory_candidate,
)
from alice_memory.hybrid_retrieval import (
    hybrid_search_memories,
    search_memories_semantic,
)
from alice_memory.lexical_index import (
    authoritative_retrieval_digest,
    build_memory_lexical_index,
    memory_lexical_index_path,
    verify_memory_lexical_index,
)
from alice_memory.promotion import (
    MemoryCandidatePromotionAuthorization,
    MemoryCandidatePromotionAuthorizationError,
    MemoryCandidatePromotionStateError,
    load_candidate_promotion,
    promote_memory_candidate,
)
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
    StaleMemoryLexicalIndexError,
)
from alice_memory.semantic_index import (
    StaleMemorySemanticIndexError,
    build_memory_semantic_index,
    verify_memory_semantic_index,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.transition_promotion import (
    MemoryCandidateTransitionAuthorization,
    MemoryCandidateTransitionAuthorizationError,
    MemoryCandidateTransitionStateError,
    load_candidate_transition_promotion,
    promote_memory_candidate_with_transition,
)


class FakeEncoder:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            values = [0.0] * self.dimension
            lowered = text.casefold()
            if "candidateonly" in lowered:
                values[0] = 1.0
            elif "baselineonly" in lowered:
                values[1] = 1.0
            else:
                values[2] = 1.0
            rows.append(values)
        return rows


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _source(
    source_ref: str,
    *,
    source_type: str = "rayan_direct_statement",
    support_relation: str = "supports",
) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type=source_type,
        source_ref=source_ref,
        support_relation=support_relation,
    )


def _create_memory(
    connection,
    *,
    memory_id: str,
    content: str,
    memory_key: str,
    classification: str = "PRIVATE",
    valid_from: str | None = "2026-01-01T00:00:00Z",
):
    return create_memory(
        connection,
        request=MemoryCreateRequest(
            memory_id=memory_id,
            content=content,
            memory_key=memory_key,
            category="profile",
            knowledge_status="rayan_statement",
            confidence=1.0,
            data_classification=classification,
            valid_from=valid_from,
            recorded_at="2026-07-28T00:00:00Z",
            sources=(
                _source(f"security:memory:{memory_id}"),
            ),
            rayan_confirmed=True,
        ),
        authorization=MemoryWriteAuthorization(
            actor="security-seeder",
            allowed=True,
            reason="seed authoritative security fixture",
        ),
        created_at="2026-07-28T00:00:30Z",
    )


def _candidate_request(
    *,
    candidate_id: str = "candidate-1",
    content: str = "candidateonly new memory",
    memory_key: str = "profile.security",
    origin: str = "explicit_user",
    valid_from: str | None = "2026-07-28T00:00:00Z",
) -> MemoryCandidateCreateRequest:
    model_fields = (
        {
            "policy_version": "formation-v1",
            "model": "qwen3:8b",
            "model_version": "8b",
            "prompt_version": "candidate-v1",
            "run_id": f"run:{candidate_id}",
        }
        if origin == "model_proposed"
        else {}
    )
    return MemoryCandidateCreateRequest(
        candidate_id=candidate_id,
        content=content,
        memory_key=memory_key,
        category="profile",
        knowledge_status=(
            "alice_inference" if origin == "model_proposed" else "rayan_statement"
        ),
        confidence=0.95,
        data_classification="PRIVATE",
        valid_from=valid_from,
        recorded_at="2026-07-28T00:01:00Z",
        sources=(
            _source(
                f"security:candidate:{candidate_id}",
                source_type=(
                    "alice_inference"
                    if origin == "model_proposed"
                    else "rayan_direct_statement"
                ),
                support_relation=(
                    "derived_from" if origin == "model_proposed" else "supports"
                ),
            ),
        ),
        origin=origin,
        rayan_confirmed=origin == "explicit_user",
        **model_fields,
    )


def _propose(
    connection,
    request: MemoryCandidateCreateRequest,
    *,
    actor: str = "security-proposer",
    reason: str = "authorized security proposal",
):
    return propose_memory_candidate(
        connection,
        request=request,
        authorization=MemoryCandidateWriteAuthorization(
            actor=actor,
            allowed=True,
            reason=reason,
        ),
        proposed_at="2026-07-28T00:01:30Z",
    )


def _assess(connection, candidate_id: str = "candidate-1"):
    return assess_memory_candidate(
        connection,
        candidate_id=candidate_id,
        authorization=MemoryCandidateAssessmentAuthorization(
            actor="security-assessor",
            allowed=True,
            reason="authorized deterministic assessment",
        ),
        assessed_at="2026-07-28T00:02:00Z",
    )


def _promotion_authorization(
    candidate_id: str = "candidate-1",
    *,
    actor: str = "security-promoter",
    user_confirmed: bool = False,
    reason: str = "authorized ordinary promotion",
):
    return MemoryCandidatePromotionAuthorization(
        actor=actor,
        allowed=True,
        candidate_id=candidate_id,
        authorization_id=f"promotion:{candidate_id}",
        user_confirmed=user_confirmed,
        reason=reason,
    )


def _transition_authorization(
    transition_type: str,
    target_memory_id: str,
    *,
    candidate_id: str = "candidate-1",
    actor: str = "security-reviewer",
    user_confirmed: bool = True,
):
    return MemoryCandidateTransitionAuthorization(
        actor=actor,
        allowed=True,
        candidate_id=candidate_id,
        target_memory_id=target_memory_id,
        transition_type=transition_type,
        authorization_id=f"transition:{transition_type}:{candidate_id}",
        user_confirmed=user_confirmed,
        reason="authorized transition security decision",
    )


def _read_authorization():
    return MemoryRetrievalAuthorization(
        actor="security-reader",
        allowed=True,
        purpose="p2.7e retrieval isolation gate",
        max_classification="PRIVATE",
    )


def _build_indexes(connection, vault, repository, model):
    lexical = build_memory_lexical_index(
        connection,
        vault,
        repository_root=repository,
        built_at="2026-07-28T00:03:00Z",
    )
    semantic = build_memory_semantic_index(
        connection,
        vault,
        model=model,
        repository_root=repository,
        built_at="2026-07-28T00:03:00Z",
    )
    return lexical, semantic


def test_candidate_lifecycle_does_not_change_authoritative_digest(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_memory(
            connection,
            memory_id="baseline",
            content="baselineonly authoritative memory",
            memory_key="profile.baseline",
        )
        before = authoritative_retrieval_digest(connection)

        _propose(connection, _candidate_request())
        after_proposal = authoritative_retrieval_digest(connection)
        _assess(connection)
        after_assessment = authoritative_retrieval_digest(connection)

        assert before == after_proposal == after_assessment
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1


def test_candidates_never_enter_derived_indexes_before_promotion(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(policy.model.embedding_dimension)

    with open_memory_store(vault, repository_root=repository) as connection:
        _create_memory(
            connection,
            memory_id="baseline",
            content="baselineonly authoritative memory",
            memory_key="profile.baseline",
        )
        _propose(connection, _candidate_request())
        _assess(connection)
        lexical, semantic = _build_indexes(
            connection,
            vault,
            repository,
            model,
        )

        assert lexical.record_count == semantic.record_count == 1
        assert search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            repository_root=repository,
        ).results == ()

        semantic_results = search_memories_semantic(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            model=model,
            repository_root=repository,
        ).results
        hybrid_results = hybrid_search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            model=model,
            repository_root=repository,
        ).results

        assert {item.memory_id for item in semantic_results} <= {"baseline"}
        assert {item.memory_id for item in hybrid_results} <= {"baseline"}


def test_promotion_invalidates_old_indexes_and_rebuild_surfaces_memory(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(policy.model.embedding_dimension)

    with open_memory_store(vault, repository_root=repository) as connection:
        _create_memory(
            connection,
            memory_id="baseline",
            content="baselineonly authoritative memory",
            memory_key="profile.baseline",
        )
        _propose(connection, _candidate_request())
        _assess(connection)
        _build_indexes(connection, vault, repository, model)

        promoted = promote_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=_promotion_authorization(),
            promoted_at="2026-07-28T00:04:00Z",
        )

        with pytest.raises(StaleMemoryLexicalIndexError):
            verify_memory_lexical_index(
                connection,
                memory_lexical_index_path(
                    vault,
                    repository_root=repository,
                ),
            )
        with pytest.raises(StaleMemorySemanticIndexError):
            verify_memory_semantic_index(
                connection,
                vault,
                repository_root=repository,
            )

        _build_indexes(connection, vault, repository, model)
        lexical = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            repository_root=repository,
        )
        semantic = search_memories_semantic(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            model=model,
            repository_root=repository,
        )
        hybrid = hybrid_search_memories(
            connection,
            vault,
            request=MemorySearchRequest(query="candidateonly"),
            authorization=_read_authorization(),
            model=model,
            repository_root=repository,
        )

        assert lexical.results[0].memory_id == promoted.memory.memory_id
        assert semantic.results[0].memory_id == promoted.memory.memory_id
        assert hybrid.results[0].memory_id == promoted.memory.memory_id


def test_model_proposer_cannot_self_authorize_ordinary_promotion(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _propose(
            connection,
            _candidate_request(origin="model_proposed"),
            actor="model:qwen3",
        )
        _assess(connection)

        with pytest.raises(MemoryCandidatePromotionAuthorizationError):
            promote_memory_candidate(
                connection,
                candidate_id="candidate-1",
                authorization=_promotion_authorization(
                    actor="model:qwen3",
                    user_confirmed=True,
                ),
                promoted_at="2026-07-28T00:04:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_model_proposer_cannot_self_authorize_transition(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_memory(
            connection,
            memory_id="target",
            content="old model-controlled value",
            memory_key="profile.security",
        )
        _propose(
            connection,
            _candidate_request(origin="model_proposed"),
            actor="model:qwen3",
        )
        _assess(connection)

        with pytest.raises(MemoryCandidateTransitionAuthorizationError):
            promote_memory_candidate_with_transition(
                connection,
                candidate_id="candidate-1",
                authorization=_transition_authorization(
                    "correction",
                    "target",
                    actor="model:qwen3",
                    user_confirmed=True,
                ),
                promoted_at="2026-07-28T00:04:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1


def test_duplicate_resolution_cannot_be_rebound_to_second_target(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    content = "candidateonly exact duplicate"
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_memory(
            connection,
            memory_id="duplicate-a",
            content=content,
            memory_key="profile.security",
        )
        _create_memory(
            connection,
            memory_id="duplicate-b",
            content=content,
            memory_key="profile.security",
        )
        _propose(
            connection,
            _candidate_request(content=content),
        )
        _assess(connection)

        first = promote_memory_candidate_with_transition(
            connection,
            candidate_id="candidate-1",
            authorization=_transition_authorization(
                "duplicate",
                "duplicate-a",
                user_confirmed=False,
            ),
            promoted_at="2026-07-28T00:04:00Z",
        )
        with pytest.raises(MemoryCandidateTransitionStateError):
            promote_memory_candidate_with_transition(
                connection,
                candidate_id="candidate-1",
                authorization=_transition_authorization(
                    "duplicate",
                    "duplicate-b",
                    user_confirmed=False,
                ),
                promoted_at="2026-07-28T00:05:00Z",
            )

        loaded = load_candidate_transition_promotion(
            connection,
            candidate_id="candidate-1",
        )
        assert first == loaded
        assert loaded.target.memory_id == "duplicate-a"
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_candidate_events
            WHERE candidate_id = 'candidate-1'
              AND event_type = 'inspected'
            """
        ).fetchone()[0] == 1


def _transition_promote_correction(connection):
    _create_memory(
        connection,
        memory_id="target",
        content="old correction value",
        memory_key="profile.security",
    )
    _propose(connection, _candidate_request())
    _assess(connection)
    return promote_memory_candidate_with_transition(
        connection,
        candidate_id="candidate-1",
        authorization=_transition_authorization("correction", "target"),
        promoted_at="2026-07-28T00:04:00Z",
    )


def test_ordinary_loader_rejects_transition_promoted_candidate(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _transition_promote_correction(connection)

        with pytest.raises(MemoryCandidatePromotionStateError):
            load_candidate_promotion(
                connection,
                candidate_id="candidate-1",
            )


def test_ordinary_promotion_cannot_replay_after_transition_promotion(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _transition_promote_correction(connection)
        before = connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]

        with pytest.raises(MemoryCandidatePromotionStateError):
            promote_memory_candidate(
                connection,
                candidate_id="candidate-1",
                authorization=_promotion_authorization(user_confirmed=True),
                promoted_at="2026-07-28T00:05:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == before


def _ordinary_promotion(connection):
    _propose(connection, _candidate_request())
    _assess(connection)
    return promote_memory_candidate(
        connection,
        candidate_id="candidate-1",
        authorization=_promotion_authorization(),
        promoted_at="2026-07-28T00:04:00Z",
    )


def test_ordinary_loader_fails_closed_on_tampered_memory_link(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _ordinary_promotion(connection)
        row = connection.execute(
            """
            SELECT candidate_event_id, details_json
            FROM memory_candidate_events
            WHERE candidate_id = 'candidate-1'
              AND event_type = 'promoted'
            """
        ).fetchone()
        details = json.loads(row["details_json"])
        details["promoted_memory_id"] = "tampered-memory-id"
        connection.execute(
            """
            UPDATE memory_candidate_events
            SET details_json = ?
            WHERE candidate_event_id = ?
            """,
            (json.dumps(details), row["candidate_event_id"]),
        )

        with pytest.raises(MemoryCandidatePromotionStateError):
            load_candidate_promotion(
                connection,
                candidate_id="candidate-1",
            )


def test_ordinary_loader_fails_closed_when_derivation_is_removed(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _ordinary_promotion(connection)
        connection.execute(
            "DELETE FROM memory_derivations WHERE memory_id = ?",
            (result.memory.memory_id,),
        )

        with pytest.raises(MemoryCandidatePromotionStateError):
            load_candidate_promotion(
                connection,
                candidate_id="candidate-1",
            )


def test_transition_loader_fails_closed_when_relation_is_tampered(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        result = _transition_promote_correction(connection)
        assert result.relation is not None
        connection.execute(
            """
            UPDATE memory_relations
            SET relation_type = 'supports'
            WHERE relation_id = ?
            """,
            (result.relation.relation_id,),
        )

        with pytest.raises(MemoryCandidateTransitionStateError):
            load_candidate_transition_promotion(
                connection,
                candidate_id="candidate-1",
            )


def test_candidate_audit_stream_never_contains_plaintext_or_reasons(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    content = "candidateonly PLAINTEXT-MARKER-27E"
    proposal_reason = "PROPOSAL-PRIVATE-REASON-27E"
    promotion_reason = "PROMOTION-PRIVATE-REASON-27E"

    with open_memory_store(vault, repository_root=repository) as connection:
        _propose(
            connection,
            _candidate_request(content=content),
            reason=proposal_reason,
        )
        _assess(connection)
        promote_memory_candidate(
            connection,
            candidate_id="candidate-1",
            authorization=_promotion_authorization(reason=promotion_reason),
            promoted_at="2026-07-28T00:04:00Z",
        )

        audit_values = [
            str(row[0] or "")
            for row in connection.execute(
                """
                SELECT details_json FROM memory_candidate_events
                UNION ALL
                SELECT details_json FROM memory_events
                """
            ).fetchall()
        ]
        audit_blob = "\n".join(audit_values)

        assert content not in audit_blob
        assert proposal_reason not in audit_blob
        assert promotion_reason not in audit_blob


@pytest.mark.parametrize("classification", ["HIGHLY_SENSITIVE", "SECRETS"])
def test_direct_sql_cannot_reclassify_candidate_into_unsafe_storage(
    tmp_path: Path,
    classification: str,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        _propose(connection, _candidate_request())

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE memory_candidates
                SET data_classification = ?
                WHERE candidate_id = 'candidate-1'
                """,
                (classification,),
            )

        assert connection.execute(
            """
            SELECT data_classification
            FROM memory_candidates
            WHERE candidate_id = 'candidate-1'
            """
        ).fetchone()[0] == "PRIVATE"


def test_forged_promoted_state_without_audit_event_fails_closed(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        memory = _create_memory(
            connection,
            memory_id="unrelated",
            content="unrelated authoritative content",
            memory_key="profile.unrelated",
        )
        _propose(connection, _candidate_request())
        connection.execute(
            """
            UPDATE memory_candidates
            SET candidate_state = 'promoted',
                promoted_memory_id = ?
            WHERE candidate_id = 'candidate-1'
            """,
            (memory.memory_id,),
        )

        with pytest.raises(MemoryCandidatePromotionStateError):
            load_candidate_promotion(
                connection,
                candidate_id="candidate-1",
            )
        with pytest.raises(MemoryCandidateTransitionStateError):
            load_candidate_transition_promotion(
                connection,
                candidate_id="candidate-1",
            )
