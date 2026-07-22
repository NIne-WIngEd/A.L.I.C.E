"""P2.5 metadata-safe lexical memory retrieval tests."""

from __future__ import annotations

from pathlib import Path

from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    archive_memory,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    correct_memory,
    mark_memory_conflict,
    supersede_memory,
)


def _test_source() -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref="test-suite:synthetic-memory",
        support_relation="supports",
    )


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _write_auth() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="retrieval test",
    )


def _read_auth(
    *,
    max_classification: str = "PRIVATE",
) -> MemoryRetrievalAuthorization:
    return MemoryRetrievalAuthorization(
        actor="test",
        allowed=True,
        purpose="unit test retrieval",
        max_classification=max_classification,
    )


def _request(
    memory_id: str,
    content: str,
    *,
    memory_key: str = "project.phase",
    classification: str = "PRIVATE",
    valid_from: str | None = None,
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key=memory_key,
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification=classification,
        valid_from=valid_from,
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def _build(
    connection,
    vault,
    repository,
) -> None:
    build_memory_lexical_index(
        connection,
        vault,
        repository_root=repository,
        built_at="2026-07-21T01:00:00Z",
    )


def test_search_returns_metadata_only(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "memory-1",
                "alpha private retrieval phrase",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        _build(connection, vault, repository)

        response = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alpha retrieval",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert [item.memory_id for item in response.results] == [
            "memory-1"
        ]
        assert not hasattr(response.results[0], "content")
        assert not hasattr(response.results[0], "snippet")


def test_archived_memory_is_hidden_by_default(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "memory-1",
                "alpha archived retrieval phrase",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        archive_memory(
            connection,
            memory_id="memory-1",
            authorization=_write_auth(),
            archived_at="2026-07-21T00:01:00Z",
        )
        _build(connection, vault, repository)

        hidden = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alpha archived",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )
        visible = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alpha archived",
                include_archived=True,
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert hidden.results == ()
        assert [item.memory_id for item in visible.results] == [
            "memory-1"
        ]


def test_classification_ceiling_filters_results(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "public",
                "alpha shared phrase",
                classification="PUBLIC",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "private",
                "alpha shared phrase",
                classification="PRIVATE",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:01Z",
        )
        _build(connection, vault, repository)

        response = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alpha shared",
            ),
            authorization=_read_auth(
                max_classification="PUBLIC",
            ),
            repository_root=repository,
        )

        assert [item.memory_id for item in response.results] == [
            "public"
        ]


def test_corrected_false_memory_is_not_returned_as_truth(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "wrong",
                "wrongtoken project phase",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        correct_memory(
            connection,
            memory_id="wrong",
            replacement=_request(
                "correct",
                "correcttoken project phase",
            ),
            authorization=_write_auth(),
            corrected_at="2026-07-21T00:01:00Z",
        )
        _build(connection, vault, repository)

        response = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="wrongtoken project",
                include_historical=True,
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert response.results == ()


def test_past_temporal_search_returns_historical_superseded_memory(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "old",
                "phaseword old state",
                valid_from="2026-01-01T00:00:00Z",
            ),
            authorization=_write_auth(),
            created_at="2026-01-01T00:00:00Z",
        )
        supersede_memory(
            connection,
            memory_id="old",
            replacement=_request(
                "new",
                "phaseword new state",
                valid_from="2026-06-01T00:00:00Z",
            ),
            authorization=_write_auth(),
            superseded_at="2026-06-01T00:00:00Z",
        )
        _build(connection, vault, repository)

        past = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="phaseword state",
                at="2026-05-01T00:00:00Z",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )
        current = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="phaseword state",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert [item.memory_id for item in past.results] == [
            "old"
        ]
        assert [item.memory_id for item in current.results] == [
            "new"
        ]


def test_conflict_expansion_surfaces_unmatched_conflicting_memory(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "first",
                "alphaunique candidate",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "second",
                "betadifferent candidate",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:01Z",
        )
        mark_memory_conflict(
            connection,
            first_memory_id="first",
            second_memory_id="second",
            authorization=_write_auth(),
            disputed_at="2026-07-21T00:01:00Z",
        )
        _build(connection, vault, repository)

        response = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alphaunique",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )

        assert {
            item.memory_id
            for item in response.results
        } == {
            "first",
            "second",
        }

        by_id = {
            item.memory_id: item
            for item in response.results
        }
        assert by_id["first"].conflict_memory_ids == (
            "second",
        )
        assert by_id["second"].conflict_memory_ids == (
            "first",
        )
