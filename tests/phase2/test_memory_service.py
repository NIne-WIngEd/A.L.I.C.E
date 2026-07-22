"""P2.3 deterministic memory lifecycle service tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from alice_memory.service import (
    MemoryAlreadyExistsError,
    MemoryContentAccessAuthorization,
    MemoryContentAuthorizationError,
    MemoryCreateRequest,
    MemoryValidationError,
    MemoryWriteAuthorization,
    MemoryWriteAuthorizationError,
    archive_memory,
    create_memory,
    load_memory,
    load_memory_content,
)
from alice_memory.store import open_memory_store


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(
        vault,
        repository_root=repository,
    )


def _request(
    *,
    memory_id: str = "memory-1",
    content: str = "A verified test memory.",
    classification: str = "PRIVATE",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        category="project",
        knowledge_status="verified_fact",
        confidence=0.95,
        data_classification=classification,
        recorded_at="2026-07-21T00:00:00Z",
        verified_at="2026-07-21T00:00:00Z",
        rayan_confirmed=True,
    )


def _authorized(
    *,
    reason: str = "unit test",
) -> MemoryWriteAuthorization:
    return MemoryWriteAuthorization(
        actor="test",
        allowed=True,
        reason=reason,
    )


def _content_authorized() -> MemoryContentAccessAuthorization:
    return MemoryContentAccessAuthorization(
        actor="test",
        allowed=True,
        reason="explicit content test",
    )


def test_unauthorized_memory_creation_is_rejected(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryWriteAuthorizationError):
            create_memory(
                connection,
                request=_request(),
                authorization=MemoryWriteAuthorization(
                    actor="test",
                    allowed=False,
                ),
                created_at="2026-07-21T00:00:00Z",
            )


def test_authorized_memory_creation_returns_metadata_only_and_persists_event(
    tmp_path: Path,
) -> None:
    content = "A verified test memory."

    with _open(tmp_path) as connection:
        memory = create_memory(
            connection,
            request=_request(content=content),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        assert memory.memory_id == "memory-1"
        assert not hasattr(memory, "content")
        assert memory.content_sha256 == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        event = connection.execute(
            """
            SELECT event_type, actor, details_json
            FROM memory_events
            WHERE memory_id = ?
            """,
            ("memory-1",),
        ).fetchone()

        assert event["event_type"] == "created"
        assert event["actor"] == "test"
        assert content not in event["details_json"]


def test_free_form_authorization_reason_is_not_persisted_in_event(
    tmp_path: Path,
) -> None:
    sensitive_reason = "contains private context that must not be logged"

    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(
                reason=sensitive_reason,
            ),
            created_at="2026-07-21T00:00:00Z",
        )

        details = connection.execute(
            """
            SELECT details_json
            FROM memory_events
            WHERE memory_id = ?
            """,
            ("memory-1",),
        ).fetchone()["details_json"]

        assert sensitive_reason not in details


def test_public_load_memory_is_metadata_only(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        memory = load_memory(
            connection,
            memory_id="memory-1",
        )

        assert not hasattr(memory, "content")
        assert memory.content_sha256


def test_plaintext_load_requires_explicit_authorization(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryContentAuthorizationError):
            load_memory_content(
                connection,
                memory_id="memory-1",
                authorization=None,
            )

        assert load_memory_content(
            connection,
            memory_id="memory-1",
            authorization=_content_authorized(),
        ) == "A verified test memory."


def test_highly_sensitive_memory_creation_is_blocked_until_p2_6(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    classification="HIGHLY_SENSITIVE",
                ),
                authorization=_authorized(),
                created_at="2026-07-21T00:00:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_secrets_are_rejected_before_database_insert(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(
                    classification="SECRETS",
                ),
                authorization=_authorized(),
                created_at="2026-07-21T00:00:00Z",
            )

        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_empty_content_is_rejected(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(content="   "),
                authorization=_authorized(),
                created_at="2026-07-21T00:00:00Z",
            )


def test_duplicate_memory_id_is_rejected(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryAlreadyExistsError):
            create_memory(
                connection,
                request=_request(),
                authorization=_authorized(),
                created_at="2026-07-21T00:00:01Z",
            )


def test_archive_memory_requires_authorization(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        with pytest.raises(MemoryWriteAuthorizationError):
            archive_memory(
                connection,
                memory_id="memory-1",
                authorization=MemoryWriteAuthorization(
                    actor="test",
                    allowed=False,
                ),
                archived_at="2026-07-21T00:00:01Z",
            )


def test_archive_memory_returns_metadata_only_and_creates_event(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        archived = archive_memory(
            connection,
            memory_id="memory-1",
            authorization=_authorized(),
            archived_at="2026-07-21T00:00:01Z",
        )

        assert archived.retention_state == "archived"
        assert not hasattr(archived, "content")

        events = connection.execute(
            """
            SELECT event_type
            FROM memory_events
            WHERE memory_id = ?
            ORDER BY created_at
            """,
            ("memory-1",),
        ).fetchall()

        assert [row["event_type"] for row in events] == [
            "created",
            "archived",
        ]


def test_archive_memory_is_idempotent(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        create_memory(
            connection,
            request=_request(),
            authorization=_authorized(),
            created_at="2026-07-21T00:00:00Z",
        )

        archive_memory(
            connection,
            memory_id="memory-1",
            authorization=_authorized(),
            archived_at="2026-07-21T00:00:01Z",
        )
        second = archive_memory(
            connection,
            memory_id="memory-1",
            authorization=_authorized(),
            archived_at="2026-07-21T00:00:02Z",
        )

        assert second.updated_at == "2026-07-21T00:00:01Z"

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_events
            WHERE memory_id = ?
              AND event_type = 'archived'
            """,
            ("memory-1",),
        ).fetchone()[0]

        assert count == 1
