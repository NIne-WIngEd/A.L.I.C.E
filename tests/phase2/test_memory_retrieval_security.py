"""P2.5 retrieval authorization and stale-index security tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.retrieval import search_memories
from alice_memory.retrieval_models import (
    MemoryRetrievalAuthorization,
    MemoryRetrievalAuthorizationError,
    MemorySearchRequest,
    StaleMemoryLexicalIndexError,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


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


def _request(
    memory_id: str,
    content: str,
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key="security.test",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def _write_auth() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="security test",
    )


def test_retrieval_is_default_deny(
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
                "alpha security memory",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        with pytest.raises(MemoryRetrievalAuthorizationError):
            search_memories(
                connection,
                vault,
                request=MemorySearchRequest(
                    query="alpha",
                ),
                authorization=MemoryRetrievalAuthorization(
                    actor="test",
                    allowed=False,
                    purpose="denied test",
                ),
                repository_root=repository,
            )


def test_retrieval_requires_non_empty_purpose(
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
                "alpha security memory",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        with pytest.raises(MemoryRetrievalAuthorizationError):
            search_memories(
                connection,
                vault,
                request=MemorySearchRequest(
                    query="alpha",
                ),
                authorization=MemoryRetrievalAuthorization(
                    actor="test",
                    allowed=True,
                    purpose="   ",
                ),
                repository_root=repository,
            )


def test_highly_sensitive_classification_ceiling_is_rejected_until_p2_6(
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
                "alpha security memory",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        with pytest.raises(MemoryRetrievalAuthorizationError):
            search_memories(
                connection,
                vault,
                request=MemorySearchRequest(
                    query="alpha",
                ),
                authorization=MemoryRetrievalAuthorization(
                    actor="test",
                    allowed=True,
                    purpose="unit test",
                    max_classification="HIGHLY_SENSITIVE",
                ),
                repository_root=repository,
            )


def test_stale_index_never_serves_after_authoritative_change(
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
                "alpha security memory",
            ),
            authorization=_write_auth(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        connection.execute(
            """
            UPDATE memories
            SET deletion_state = 'pending_deletion',
                updated_at = '2026-07-21T00:02:00Z'
            WHERE memory_id = 'memory-1'
            """
        )

        with pytest.raises(StaleMemoryLexicalIndexError):
            search_memories(
                connection,
                vault,
                request=MemorySearchRequest(
                    query="alpha",
                ),
                authorization=MemoryRetrievalAuthorization(
                    actor="test",
                    allowed=True,
                    purpose="unit test",
                ),
                repository_root=repository,
            )
