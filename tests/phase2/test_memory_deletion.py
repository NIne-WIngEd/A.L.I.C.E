"""P2.8a ordinary-memory deletion lifecycle and tombstone tests."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from alice_memory.deletion import (
    MAX_STRONG_CONFIRMATION_SECONDS,
    ORDINARY_MEMORY_DELETION_SCOPE,
    MemoryDeletionAuthorization,
    MemoryDeletionAuthorizationError,
    MemoryDeletionCancellationAuthorization,
    MemoryDeletionRequestAuthorization,
    MemoryDeletionStateError,
    MemoryDeletionValidationError,
    MemoryTombstoneNotFoundError,
    _tombstone_id,
    cancel_memory_deletion,
    delete_memory,
    load_memory_deletion,
    load_memory_tombstone,
    request_memory_deletion,
)
from alice_memory.inspection import list_memory_summaries
from alice_memory.lexical_index import authoritative_retrieval_digest
from alice_memory.service import (
    MemoryAlreadyExistsError,
    MemoryContentAccessAuthorization,
    MemoryContentAuthorizationError,
    MemoryCreateRequest,
    MemoryNotFoundError,
    MemoryWriteAuthorization,
    create_memory,
    load_memory,
    load_memory_content,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return open_memory_store(vault, repository_root=repository)


def _source(ref: str = "test-suite:deletion") -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref=ref,
        support_relation="supports",
    )


def _request(
    *,
    memory_id: str = "memory-delete-1",
    content: str = "Delete this ordinary memory safely.",
    classification: str = "PRIVATE",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        category="project",
        knowledge_status="verified_fact",
        confidence=0.96,
        data_classification=classification,
        recorded_at="2026-07-25T10:00:00Z",
        verified_at="2026-07-25T10:00:00Z",
        rayan_confirmed=True,
        sources=(_source(f"test-suite:{memory_id}"),),
    )


def _create(
    connection,
    *,
    memory_id: str = "memory-delete-1",
    content: str | None = None,
):
    return create_memory(
        connection,
        request=_request(
            memory_id=memory_id,
            content=content or f"Ordinary memory {memory_id}.",
        ),
        authorization=MemoryWriteAuthorization(
            actor="test-writer",
            allowed=True,
            reason="synthetic fixture",
        ),
        created_at="2026-07-25T10:00:00Z",
    )


def _request_auth(
    *,
    memory_id: str = "memory-delete-1",
    allowed: bool = True,
    actor: str = "rayan",
    authorization_id: str = "delete-request-001",
    scope: str = ORDINARY_MEMORY_DELETION_SCOPE,
    reason: str | None = "private reason must not be persisted",
) -> MemoryDeletionRequestAuthorization:
    return MemoryDeletionRequestAuthorization(
        actor=actor,
        allowed=allowed,
        memory_id=memory_id,
        deletion_scope=scope,
        authorization_id=authorization_id,
        reason=reason,
    )


def _cancel_auth(
    *,
    memory_id: str = "memory-delete-1",
    authorization_id: str = "delete-cancel-001",
) -> MemoryDeletionCancellationAuthorization:
    return MemoryDeletionCancellationAuthorization(
        actor="rayan",
        allowed=True,
        memory_id=memory_id,
        deletion_scope=ORDINARY_MEMORY_DELETION_SCOPE,
        authorization_id=authorization_id,
        reason="private cancellation reason",
    )


def _delete_auth(
    *,
    memory_id: str = "memory-delete-1",
    allowed: bool = True,
    strongly_confirmed: bool = True,
    actor: str = "rayan",
    authorization_id: str = "delete-final-001",
    scope: str = ORDINARY_MEMORY_DELETION_SCOPE,
    issued_at: str = "2026-07-25T10:01:00Z",
    expires_at: str = "2026-07-25T10:03:00Z",
    reason: str | None = "sensitive free-form deletion reason",
) -> MemoryDeletionAuthorization:
    return MemoryDeletionAuthorization(
        actor=actor,
        allowed=allowed,
        memory_id=memory_id,
        deletion_scope=scope,
        authorization_id=authorization_id,
        strongly_confirmed=strongly_confirmed,
        issued_at=issued_at,
        expires_at=expires_at,
        reason=reason,
    )


def _content_auth() -> MemoryContentAccessAuthorization:
    return MemoryContentAccessAuthorization(
        actor="test-reader",
        allowed=True,
        reason="test read",
    )


def _request_deletion(connection, *, memory_id: str = "memory-delete-1"):
    return request_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_request_auth(memory_id=memory_id),
        requested_at="2026-07-25T10:00:30Z",
    )


def _complete_deletion(connection, *, memory_id: str = "memory-delete-1"):
    _request_deletion(connection, memory_id=memory_id)
    return delete_memory(
        connection,
        memory_id=memory_id,
        authorization=_delete_auth(memory_id=memory_id),
        deleted_at="2026-07-25T10:02:00Z",
    )


def test_request_requires_explicit_target_bound_authorization(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)

        with pytest.raises(MemoryDeletionAuthorizationError):
            request_memory_deletion(
                connection,
                memory_id="memory-delete-1",
                authorization=_request_auth(allowed=False),
                requested_at="2026-07-25T10:00:30Z",
            )

        with pytest.raises(MemoryDeletionAuthorizationError):
            request_memory_deletion(
                connection,
                memory_id="memory-delete-1",
                authorization=_request_auth(memory_id="other-memory"),
                requested_at="2026-07-25T10:00:30Z",
            )

        assert (
            load_memory(
                connection,
                memory_id="memory-delete-1",
            ).deletion_state
            == "active"
        )


def test_request_rejects_unsupported_scope_and_unsafe_audit_id(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)

        with pytest.raises(MemoryDeletionAuthorizationError):
            request_memory_deletion(
                connection,
                memory_id="memory-delete-1",
                authorization=_request_auth(scope="memory-only"),
                requested_at="2026-07-25T10:00:30Z",
            )

        with pytest.raises(MemoryDeletionAuthorizationError):
            request_memory_deletion(
                connection,
                memory_id="memory-delete-1",
                authorization=_request_auth(authorization_id="contains spaces"),
                requested_at="2026-07-25T10:00:30Z",
            )


def test_request_sets_pending_state_and_writes_sanitized_event(tmp_path: Path) -> None:
    content = "Private plaintext that must not enter deletion audit metadata."
    private_reason = "This reason is also private and must not be logged."

    with _open(tmp_path) as connection:
        _create(connection, content=content)
        result = request_memory_deletion(
            connection,
            memory_id="memory-delete-1",
            authorization=_request_auth(reason=private_reason),
            requested_at="2026-07-25T10:00:30Z",
        )

        assert result.memory.deletion_state == "pending_deletion"
        assert result.deletion_scope == ORDINARY_MEMORY_DELETION_SCOPE
        row = connection.execute(
            "SELECT event_type, actor, details_json "
            "FROM memory_events WHERE event_id = ?",
            (result.request_event_id,),
        ).fetchone()
        assert row["event_type"] == "deletion_requested"
        assert row["actor"] == "rayan"
        assert content not in row["details_json"]
        assert private_reason not in row["details_json"]
        details = json.loads(row["details_json"])
        assert details["operation"] == "deletion_requested"
        assert details["new_deletion_state"] == "pending_deletion"


def test_pending_deletion_is_hidden_from_summaries_digest_and_plaintext(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        before_digest, before_count = authoritative_retrieval_digest(connection)
        assert before_count == 1
        assert len(list_memory_summaries(connection)) == 1

        _request_deletion(connection)

        after_digest, after_count = authoritative_retrieval_digest(connection)
        assert after_count == 0
        assert after_digest != before_digest
        assert list_memory_summaries(connection) == ()
        with pytest.raises(MemoryContentAuthorizationError):
            load_memory_content(
                connection,
                memory_id="memory-delete-1",
                authorization=_content_auth(),
            )


def test_repeated_request_is_idempotent(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        first = _request_deletion(connection)
        second = request_memory_deletion(
            connection,
            memory_id="memory-delete-1",
            authorization=_request_auth(authorization_id="delete-request-002"),
            requested_at="2026-07-25T10:00:45Z",
        )

        assert second.request_event_id == first.request_event_id
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'deletion_requested'"
        ).fetchone()[0] == 1


def test_cancellation_restores_active_state_and_plaintext_access(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create(connection, content="Restored after cancellation.")
        request_result = _request_deletion(connection)
        cancelled = cancel_memory_deletion(
            connection,
            memory_id="memory-delete-1",
            authorization=_cancel_auth(),
            cancelled_at="2026-07-25T10:00:45Z",
        )

        assert cancelled.request_event_id == request_result.request_event_id
        assert cancelled.memory.deletion_state == "active"
        assert load_memory_content(
            connection,
            memory_id="memory-delete-1",
            authorization=_content_auth(),
        ) == "Restored after cancellation."
        row = connection.execute(
            "SELECT event_type, details_json FROM memory_events WHERE event_id = ?",
            (cancelled.cancellation_event_id,),
        ).fetchone()
        assert row["event_type"] == "reclassified"
        assert json.loads(row["details_json"])["operation"] == "deletion_cancelled"
        assert "private cancellation reason" not in row["details_json"]


def test_cancellation_requires_pending_state(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        with pytest.raises(MemoryDeletionStateError):
            cancel_memory_deletion(
                connection,
                memory_id="memory-delete-1",
                authorization=_cancel_auth(),
                cancelled_at="2026-07-25T10:00:45Z",
            )


def test_cancelled_request_cannot_be_finalized_without_new_request(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _request_deletion(connection)
        cancel_memory_deletion(
            connection,
            memory_id="memory-delete-1",
            authorization=_cancel_auth(),
            cancelled_at="2026-07-25T10:00:45Z",
        )
        # Simulate corrupted/direct state mutation. The lifecycle audit still
        # fails closed because the latest request was cancelled.
        connection.execute(
            "UPDATE memories SET deletion_state = 'pending_deletion' "
            "WHERE memory_id = ?",
            ("memory-delete-1",),
        )

        with pytest.raises(MemoryDeletionStateError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T10:02:00Z",
            )


def test_final_deletion_requires_pending_request_and_strong_confirmation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        with pytest.raises(MemoryDeletionStateError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T10:02:00Z",
            )

        _request_deletion(connection)
        with pytest.raises(MemoryDeletionAuthorizationError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(strongly_confirmed=False),
                deleted_at="2026-07-25T10:02:00Z",
            )

        assert (
            load_memory(
                connection,
                memory_id="memory-delete-1",
            ).deletion_state
            == "pending_deletion"
        )


def test_strong_confirmation_window_is_short_lived(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _request_deletion(connection)

        with pytest.raises(MemoryDeletionAuthorizationError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(expires_at="2026-07-25T10:03:01Z"),
                deleted_at="2026-07-25T10:02:00Z",
            )

        assert MAX_STRONG_CONFIRMATION_SECONDS == 120


def test_expired_or_not_yet_valid_confirmation_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _request_deletion(connection)

        with pytest.raises(MemoryDeletionAuthorizationError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T10:03:01Z",
            )

        with pytest.raises(MemoryDeletionAuthorizationError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T10:00:59Z",
            )


def test_completed_deletion_removes_memory_and_cascaded_dependents(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _create(connection, memory_id="memory-other")
        connection.execute(
            """
            INSERT INTO memory_relations (
                relation_id, from_memory_id, to_memory_id, relation_type, created_at
            ) VALUES (?, ?, ?, 'supports', ?)
            """,
            (
                "relation-delete",
                "memory-delete-1",
                "memory-other",
                "2026-07-25T10:00:10Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_derivations (
                derivation_id, memory_id, derivation_type, created_at
            ) VALUES (?, ?, 'explicit_user', ?)
            """,
            ("derivation-delete", "memory-delete-1", "2026-07-25T10:00:10Z"),
        )
        connection.execute(
            """
            INSERT INTO memory_entities (
                memory_entity_id, memory_id, entity_type, entity_value, created_at
            ) VALUES (?, ?, 'project', 'deletion-test', ?)
            """,
            ("entity-delete", "memory-delete-1", "2026-07-25T10:00:10Z"),
        )

        result = _complete_deletion(connection)

        with pytest.raises(MemoryNotFoundError):
            load_memory(connection, memory_id="memory-delete-1")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?",
            ("memory-delete-1",),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_relations "
            "WHERE from_memory_id = ? OR to_memory_id = ?",
            ("memory-delete-1", "memory-delete-1"),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_derivations WHERE memory_id = ?",
            ("memory-delete-1",),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_id = ?",
            ("memory-delete-1",),
        ).fetchone()[0] == 0
        assert result.tombstone.deleted_memory_id == "memory-delete-1"
        assert load_memory(connection, memory_id="memory-other")


def test_tombstone_and_deleted_event_are_sanitized_and_linked(tmp_path: Path) -> None:
    content = "Extremely private ordinary-memory plaintext."
    private_reason = "Detailed private deletion explanation."

    with _open(tmp_path) as connection:
        _create(connection, content=content)
        _request_deletion(connection)
        result = delete_memory(
            connection,
            memory_id="memory-delete-1",
            authorization=_delete_auth(reason=private_reason),
            deleted_at="2026-07-25T10:02:00Z",
        )

        tombstone = result.tombstone
        assert tombstone.content_sha256 == hashlib.sha256(content.encode()).hexdigest()
        assert tombstone.event_id is not None
        row = connection.execute(
            "SELECT memory_id, event_type, details_json "
            "FROM memory_events WHERE event_id = ?",
            (tombstone.event_id,),
        ).fetchone()
        assert row["memory_id"] is None
        assert row["event_type"] == "deleted"
        assert content not in row["details_json"]
        assert private_reason not in row["details_json"]
        assert content not in json.dumps(tombstone.__dict__)


def test_deletion_is_idempotent_after_tombstone_exists(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        first = _complete_deletion(connection)
        second = delete_memory(
            connection,
            memory_id="memory-delete-1",
            authorization=_delete_auth(authorization_id="delete-final-002"),
            deleted_at="2026-07-25T10:02:30Z",
        )

        assert second.tombstone == first.tombstone
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tombstones WHERE deleted_memory_id = ?",
            ("memory-delete-1",),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'deleted'"
        ).fetchone()[0] == 1


def test_deleted_memory_identifier_cannot_be_reused(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _complete_deletion(connection)

        with pytest.raises(MemoryAlreadyExistsError):
            _create(connection)


def test_request_after_completed_deletion_fails_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _complete_deletion(connection)

        with pytest.raises(MemoryDeletionStateError):
            _request_deletion(connection)


def test_missing_tombstone_raises_specific_error(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        with pytest.raises(MemoryTombstoneNotFoundError):
            load_memory_tombstone(connection, memory_id="missing")


def test_tampered_deleted_event_fails_closed(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        result = _complete_deletion(connection)
        connection.execute(
            "UPDATE memory_events SET details_json = ? WHERE event_id = ?",
            ('{"operation":"wrong"}', result.tombstone.event_id),
        )

        with pytest.raises(MemoryDeletionStateError):
            load_memory_deletion(connection, memory_id="memory-delete-1")


def test_tombstone_collision_rolls_back_entire_deletion(tmp_path: Path) -> None:
    with _open(tmp_path) as connection:
        _create(connection)
        _request_deletion(connection)
        connection.execute(
            """
            INSERT INTO memory_tombstones (
                tombstone_id, deleted_memory_id, content_sha256,
                deleted_at, deletion_scope, event_id
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                _tombstone_id("memory-delete-1"),
                "different-memory",
                "0" * 64,
                "2026-07-25T10:01:00Z",
                ORDINARY_MEMORY_DELETION_SCOPE,
            ),
        )

        with pytest.raises(MemoryDeletionValidationError):
            delete_memory(
                connection,
                memory_id="memory-delete-1",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T10:02:00Z",
            )

        assert (
            load_memory(
                connection,
                memory_id="memory-delete-1",
            ).deletion_state
            == "pending_deletion"
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'deleted'"
        ).fetchone()[0] == 0


def test_highly_sensitive_memory_is_rejected_by_ordinary_deletion_path(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as connection:
        # Direct fixture insertion is limited to this validation test. The row
        # uses the encrypted sentinel shape, but the protected deletion path is
        # intentionally outside P2.8a.
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, schema_version, content, content_sha256, memory_key,
                category, knowledge_status, confidence, data_classification,
                valid_from, valid_to, time_precision, recorded_at, verified_at,
                rayan_confirmed, validity_state, retention_state, deletion_state,
                created_at, updated_at
            ) VALUES (?, 3, ?, ?, NULL, 'profile', 'rayan_statement', 1.0,
                      'HIGHLY_SENSITIVE', NULL, NULL, NULL, ?, NULL, 1,
                      'current', 'durable', 'active', ?, ?)
            """,
            (
                "sensitive-memory",
                "[ALICE:HIGHLY_SENSITIVE:ENCRYPTED]",
                hashlib.sha256(b"sensitive").hexdigest(),
                "2026-07-25T10:00:00Z",
                "2026-07-25T10:00:00Z",
                "2026-07-25T10:00:00Z",
            ),
        )

        with pytest.raises(MemoryDeletionValidationError):
            request_memory_deletion(
                connection,
                memory_id="sensitive-memory",
                authorization=_request_auth(memory_id="sensitive-memory"),
                requested_at="2026-07-25T10:00:30Z",
            )

        assert (
            load_memory(
                connection,
                memory_id="sensitive-memory",
            ).deletion_state
            == "active"
        )
