"""P2.3 authorization-aware memory inspection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.inspection import (
    MemoryContentAccessAuthorization,
    MemoryInspectionAuthorizationError,
    inspect_memory,
    list_memory_summaries,
)
from alice_memory.service import (
    MemoryCreateRequest,
    MemoryWriteAuthorization,
    archive_memory,
    create_memory,
)
from alice_memory.store import open_memory_store


def _authorization() -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason="inspection test",
    )


def _content_authorization(
    *,
    allowed: bool = True,
    actor: str = "test",
) -> MemoryContentAccessAuthorization:
    return MemoryContentAccessAuthorization(
        actor=actor,
        allowed=allowed,
        reason="explicit plaintext inspection test",
    )


def _request(
    memory_id: str,
    *,
    category: str = "project",
    content: str = "Private memory content",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        category=category,
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
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


def test_inspection_excludes_plaintext_by_default(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("memory-1"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        inspected = inspect_memory(
            connection,
            memory_id="memory-1",
        )

        assert inspected.content is None
        assert inspected.content_sha256
        assert inspected.category == "project"
        assert inspected.events[0].event_type == "created"


def test_plaintext_inspection_requires_explicit_authorization(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("memory-1"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryInspectionAuthorizationError):
            inspect_memory(
                connection,
                memory_id="memory-1",
                include_content=True,
            )


def test_denied_plaintext_authorization_is_rejected(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("memory-1"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryInspectionAuthorizationError):
            inspect_memory(
                connection,
                memory_id="memory-1",
                include_content=True,
                content_authorization=_content_authorization(
                    allowed=False,
                ),
            )


def test_plaintext_authorization_requires_actor(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("memory-1"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryInspectionAuthorizationError):
            inspect_memory(
                connection,
                memory_id="memory-1",
                include_content=True,
                content_authorization=_content_authorization(
                    actor="   ",
                ),
            )


def test_inspection_can_include_private_plaintext_when_authorized(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "memory-1",
                content="Explicitly requested content",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        inspected = inspect_memory(
            connection,
            memory_id="memory-1",
            include_content=True,
            content_authorization=_content_authorization(),
        )

        assert inspected.content == "Explicitly requested content"


def test_memory_summary_never_contains_plaintext(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "memory-1",
                content="Do not expose this in summaries",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )

        summaries = list_memory_summaries(connection)

        assert len(summaries) == 1
        assert not hasattr(summaries[0], "content")


def test_archived_memories_are_hidden_by_default(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request("memory-1"),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        archive_memory(
            connection,
            memory_id="memory-1",
            authorization=_authorization(),
            archived_at="2026-07-21T00:00:01Z",
        )

        assert list_memory_summaries(connection) == ()

        with_archived = list_memory_summaries(
            connection,
            include_archived=True,
        )

        assert len(with_archived) == 1
        assert with_archived[0].retention_state == "archived"


def test_memory_summaries_can_filter_category(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(
                "memory-project",
                category="project",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:00Z",
        )
        create_memory(
            connection,
            request=_request(
                "memory-goal",
                category="goal",
            ),
            authorization=_authorization(),
            created_at="2026-07-21T00:00:01Z",
        )

        summaries = list_memory_summaries(
            connection,
            category="goal",
        )

        assert [item.memory_id for item in summaries] == [
            "memory-goal"
        ]
