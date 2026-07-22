"""Cross-cutting provenance invariants for authoritative memory and retrieval."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alice_memory.lexical_index import (
    build_memory_lexical_index,
    memory_lexical_index_path,
    search_memory_lexical_candidates,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryValidationError,
    MemoryWriteAuthorization,
    create_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    InvalidMemoryTransitionError,
    correct_memory,
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
        reason="provenance invariant test",
    )


def _source(
    *,
    source_ref: str = "interaction:test-1",
) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="rayan_direct_statement",
        source_ref=source_ref,
        support_relation="supports",
    )


def _request(
    memory_id: str,
    content: str,
    *,
    sources: tuple[MemorySourceSpec, ...],
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        category="project",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
        sources=sources,
    )


def test_authoritative_memory_creation_requires_provenance(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    "memory-1",
                    "Unprovenanced memory",
                    sources=(),
                ),
                authorization=_authorization(),
                created_at="2026-07-21T00:00:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events"
        ).fetchone()[0] == 0


def test_source_insert_failure_rolls_back_memory_and_event(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_memory_sources
            BEFORE INSERT ON memory_sources
            BEGIN
                SELECT RAISE(ABORT, 'synthetic source failure');
            END
            """
        )

        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    "memory-1",
                    "Atomic memory",
                    sources=(_source(),),
                ),
                authorization=_authorization(),
                created_at="2026-07-21T00:00:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_sources"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events"
        ).fetchone()[0] == 0


def test_correction_replacement_requires_its_own_provenance(
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
                "Old statement",
                sources=(_source(source_ref="interaction:old"),),
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(InvalidMemoryTransitionError):
            correct_memory(
                connection,
                memory_id="old",
                replacement=_request(
                    "new",
                    "New statement",
                    sources=(),
                ),
                authorization=_authorization(),
                corrected_at="2026-07-21T00:01:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_relations
            WHERE relation_type = 'corrects'
            """
        ).fetchone()[0] == 0


def test_retrieval_index_excludes_legacy_unprovenanced_memory(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)

    with open_memory_store(
        vault,
        repository_root=repository,
    ) as connection:
        content = "legacy orphan token"
        connection.execute(
            """
            INSERT INTO memories (
                memory_id,
                schema_version,
                content,
                content_sha256,
                memory_key,
                category,
                knowledge_status,
                confidence,
                data_classification,
                valid_from,
                valid_to,
                time_precision,
                recorded_at,
                verified_at,
                rayan_confirmed,
                validity_state,
                retention_state,
                deletion_state,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-orphan",
                1,
                content,
                hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "legacy.test",
                "project",
                "verified_fact",
                1.0,
                "PRIVATE",
                None,
                None,
                None,
                "2026-07-21T00:00:00Z",
                None,
                1,
                "current",
                "durable",
                "active",
                "2026-07-21T00:00:00Z",
                "2026-07-21T00:00:00Z",
            ),
        )

        manifest = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-21T00:01:00Z",
        )

        assert manifest.record_count == 0
        assert search_memory_lexical_candidates(
            memory_lexical_index_path(
                vault,
                repository_root=repository,
            ),
            query="legacy orphan",
            limit=10,
        ) == []
