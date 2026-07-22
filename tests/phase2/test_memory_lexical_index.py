"""P2.5 private lexical-index build, verification, and rebuild tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.lexical_index import (
    build_memory_lexical_index,
    memory_lexical_index_path,
    verify_memory_lexical_index,
)
from alice_memory.retrieval_models import StaleMemoryLexicalIndexError
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


def _authorization() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="lexical index test",
    )


def _request(
    memory_id: str,
    content: str,
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        sources=(_test_source(),),
        memory_id=memory_id,
        content=content,
        memory_key="project.alpha",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def test_lexical_index_is_created_under_private_vault(
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
                "alpha retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        manifest = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        path = memory_lexical_index_path(
            vault,
            repository_root=repository,
        )

        assert path.exists()
        assert vault.resolve() in path.parents
        assert repository.resolve() not in path.parents
        assert manifest.record_count == 1


def test_lexical_index_fails_closed_when_authoritative_memory_changes(
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
                "alpha retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        create_memory(
            connection,
            request=_request(
                "memory-2",
                "beta retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:02:00Z",
        )

        path = memory_lexical_index_path(
            vault,
            repository_root=repository,
        )

        with pytest.raises(StaleMemoryLexicalIndexError):
            verify_memory_lexical_index(
                connection,
                path,
            )


def test_rebuild_restores_index_verification(
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
                "alpha retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        create_memory(
            connection,
            request=_request(
                "memory-2",
                "beta retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:02:00Z",
        )

        rebuilt = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:03:00Z",
        )

        verified = verify_memory_lexical_index(
            connection,
            memory_lexical_index_path(
                vault,
                repository_root=repository,
            ),
        )

        assert rebuilt.record_count == 2
        assert verified.authoritative_digest == rebuilt.authoritative_digest


def test_pending_deletion_memory_is_excluded_from_rebuilt_index(
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
                "alpha retrieval memory",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        connection.execute(
            """
            UPDATE memories
            SET deletion_state = 'pending_deletion',
                updated_at = '2026-07-21T00:01:00Z'
            WHERE memory_id = 'memory-1'
            """
        )

        manifest = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:02:00Z",
        )

        assert manifest.record_count == 0
