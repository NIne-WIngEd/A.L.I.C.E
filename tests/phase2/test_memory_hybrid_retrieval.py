"""P2.5b semantic and hybrid retrieval tests."""

from __future__ import annotations

from pathlib import Path

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.hybrid_retrieval import (
    hybrid_search_memories,
    search_memories_semantic,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
)
from alice_memory.semantic_index import (
    build_memory_semantic_index,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    correct_memory,
    mark_memory_conflict,
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
            if (
                "car" in lowered
                or "vehicle" in lowered
                or "automobile" in lowered
            ):
                values[0] = 1.0
            elif (
                "wrongtoken" in lowered
                or "correcttoken" in lowered
            ):
                values[1] = 1.0
            elif "alphaunique" in lowered:
                values[2] = 1.0
            else:
                values[3] = 1.0
            rows.append(values)
        return rows


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _write_auth():
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="hybrid test",
    )


def _read_auth():
    return MemoryRetrievalAuthorization(
        actor="test",
        allowed=True,
        purpose="hybrid retrieval test",
        max_classification="PRIVATE",
    )


def _request(
    memory_id: str,
    content: str,
    *,
    key: str = "hybrid.test",
):
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key=key,
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def _build_all(
    connection,
    vault,
    repository,
    model,
):
    build_memory_lexical_index(
        connection,
        vault,
        repository_root=repository,
        built_at="2026-07-21T00:10:00Z",
    )
    build_memory_semantic_index(
        connection,
        vault,
        model=model,
        repository_root=repository,
        built_at="2026-07-21T00:10:00Z",
    )


def test_semantic_retrieval_finds_paraphrase_lexical_misses(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(
        policy.model.embedding_dimension
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "car-memory",
                "I own a car",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        _build_all(
            connection,
            vault,
            repository,
            model,
        )

        lexical = search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="vehicle",
            ),
            authorization=_read_auth(),
            repository_root=repository,
        )
        semantic = search_memories_semantic(
            connection,
            vault,
            request=MemorySearchRequest(
                query="vehicle",
            ),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )

        assert lexical.results == ()
        assert [
            item.memory_id
            for item in semantic.results
        ] == ["car-memory"]
        assert semantic.results[0].matched_by == "semantic"
        assert not hasattr(
            semantic.results[0],
            "content",
        )


def test_hybrid_retrieval_uses_rrf_and_returns_metadata_only(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(
        policy.model.embedding_dimension
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "shared",
                "vehicle car",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "semantic-only",
                "automobile",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:01Z",
        )
        _build_all(
            connection,
            vault,
            repository,
            model,
        )

        response = hybrid_search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="vehicle",
            ),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )

        assert response.results[0].memory_id == "shared"
        assert response.results[0].matched_by == "hybrid"
        assert not hasattr(
            response.results[0],
            "content",
        )


def test_semantic_retrieval_excludes_corrected_false_memory(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(
        policy.model.embedding_dimension
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        create_memory(
            connection,
            request=_request(
                "wrong",
                "wrongtoken state",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        correct_memory(
            connection,
            memory_id="wrong",
            replacement=_request(
                "correct",
                "correcttoken state",
            ),
            authorization=_write_auth(),
            corrected_at="2026-07-21T00:01:00Z",
        )
        _build_all(
            connection,
            vault,
            repository,
            model,
        )

        response = search_memories_semantic(
            connection,
            vault,
            request=MemorySearchRequest(
                query="wrongtoken",
                include_historical=True,
            ),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )

        assert all(
            item.memory_id != "wrong"
            for item in response.results
        )


def test_hybrid_conflict_expansion_surfaces_both_records(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(
        policy.model.embedding_dimension
    )

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
                "other candidate",
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
        _build_all(
            connection,
            vault,
            repository,
            model,
        )

        response = hybrid_search_memories(
            connection,
            vault,
            request=MemorySearchRequest(
                query="alphaunique",
            ),
            authorization=_read_auth(),
            model=model,
            repository_root=repository,
        )

        assert {
            item.memory_id
            for item in response.results
        } == {
            "first",
            "second",
        }
