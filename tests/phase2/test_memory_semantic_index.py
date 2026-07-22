"""P2.5b semantic memory-index tests with a deterministic fake encoder."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.semantic_index import (
    MemorySemanticIndexError,
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
            if "car" in lowered or "vehicle" in lowered:
                values[0] = 1.0
            elif "alpha" in lowered:
                values[1] = 1.0
            else:
                values[2] = 1.0
            rows.append(values)
        return rows


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


def _auth():
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="semantic test",
    )


def _request(memory_id: str, content: str):
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key="semantic.test",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def test_semantic_index_builds_and_verifies(
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
                "memory-1",
                "car memory",
            ),
            authorization=_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        built = build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )
        verified, path = verify_memory_semantic_index(
            connection,
            vault,
            repository_root=repository,
        )

        assert built.record_count == 1
        assert verified.index_id == built.index_id
        assert path.exists()


def test_semantic_index_fails_closed_when_memory_changes(
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
                "memory-1",
                "car memory",
            ),
            authorization=_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "memory-2",
                "alpha memory",
            ),
            authorization=_auth(),
            created_at="2026-07-21T00:02:00Z",
        )

        with pytest.raises(
            StaleMemorySemanticIndexError
        ):
            verify_memory_semantic_index(
                connection,
                vault,
                repository_root=repository,
            )


def test_semantic_index_rejects_model_dimension_mismatch(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    policy = load_semantic_policy()
    model = FakeEncoder(
        policy.model.embedding_dimension + 1
    )

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with pytest.raises(
            MemorySemanticIndexError
        ):
            build_memory_semantic_index(
                connection,
                vault,
                model=model,
                repository_root=repository,
                built_at="2026-07-21T00:01:00Z",
            )
