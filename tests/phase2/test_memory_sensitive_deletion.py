"""P2.8c protected HIGHLY_SENSITIVE deletion tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from alice_vault.semantic_retrieval import load_semantic_policy
from alice_memory.deletion import (
    MemoryTombstoneNotFoundError,
    load_memory_tombstone,
)
from alice_memory.deletion_indexes import (
    delete_sensitive_memory_with_index_purge,
    rebuild_indexes_after_memory_deletion,
)
from alice_memory.lexical_index import (
    build_memory_lexical_index,
    memory_lexical_index_path,
)
from alice_memory.semantic_index import (
    build_memory_semantic_index,
    memory_semantic_index_root,
)
from alice_memory.sensitive_access import (
    SensitiveMemoryAccessAuthorization,
    SensitiveMemoryAccessValidationError,
    load_sensitive_memory_content,
)
from alice_memory.sensitive_crypto import InMemoryTestKeyProtector
from alice_memory.sensitive_deletion import (
    SENSITIVE_MEMORY_DELETION_SCOPE,
    SensitiveMemoryDeletionAuthorization,
    SensitiveMemoryDeletionAuthorizationError,
    SensitiveMemoryDeletionCancellationAuthorization,
    SensitiveMemoryDeletionRequestAuthorization,
    SensitiveMemoryDeletionStateError,
    SensitiveMemoryDeletionValidationError,
    cancel_sensitive_memory_deletion,
    delete_sensitive_memory,
    load_sensitive_memory_deletion,
    request_sensitive_memory_deletion,
)
from alice_memory.sensitive_storage import (
    SensitiveMemoryStorageError,
    SensitiveMemoryWriteAuthorization,
    create_sensitive_memory,
    load_sensitive_payload_record,
)
from alice_memory.service import (
    MemoryAlreadyExistsError,
    MemoryCreateRequest,
    MemoryNotFoundError,
    MemoryWriteAuthorization,
    create_memory,
    load_memory,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


class FakeEncoder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            values = [0.0] * self.dimension
            lowered = str(text).casefold()
            if "keeptoken" in lowered:
                values[0] = 1.0
            else:
                values[1] = 1.0
            rows.append(values)
        return rows


def _setup(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault


def _source(memory_id: str) -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="approved_manual_entry",
        source_ref=f"test-suite:{memory_id}",
        support_relation="supports",
    )


def _sensitive_request(
    memory_id: str = "sensitive-delete",
    *,
    content: str = "private deletion fixture plaintext",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key=f"sensitive.{memory_id}",
        category="episodic",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at="2026-07-25T13:00:00Z",
        rayan_confirmed=True,
        sources=(_source(memory_id),),
    )


def _ordinary_request(memory_id: str = "ordinary") -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content="keeptoken ordinary memory",
        memory_key=f"ordinary.{memory_id}",
        category="project",
        knowledge_status="verified_fact",
        confidence=1.0,
        data_classification="PRIVATE",
        recorded_at="2026-07-25T13:00:00Z",
        verified_at="2026-07-25T13:00:00Z",
        rayan_confirmed=True,
        sources=(_source(memory_id),),
    )


def _create_sensitive(connection, vault, repository, protector, memory_id="sensitive-delete"):
    return create_sensitive_memory(
        connection,
        vault,
        request=_sensitive_request(memory_id),
        authorization=SensitiveMemoryWriteAuthorization(
            actor="rayan",
            allowed=True,
            purpose="memory.user_requested_storage",
            authorization_id=f"create-{memory_id}",
            directly_requested=True,
        ),
        created_at="2026-07-25T13:00:00Z",
        repository_root=repository,
        key_protector=protector,
    )


def _request_auth(memory_id="sensitive-delete", **changes):
    values = dict(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_sensitive_deletion",
        authorization_id=f"request-{memory_id}",
        memory_id=memory_id,
        deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
        directly_requested=True,
    )
    values.update(changes)
    return SensitiveMemoryDeletionRequestAuthorization(**values)


def _cancel_auth(memory_id="sensitive-delete", **changes):
    values = dict(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_sensitive_deletion",
        authorization_id=f"cancel-{memory_id}",
        memory_id=memory_id,
        deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
        directly_requested=True,
    )
    values.update(changes)
    return SensitiveMemoryDeletionCancellationAuthorization(**values)


def _delete_auth(memory_id="sensitive-delete", **changes):
    values = dict(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_sensitive_deletion",
        authorization_id=f"delete-{memory_id}",
        memory_id=memory_id,
        deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
        directly_requested=True,
        strongly_confirmed=True,
        issued_at="2026-07-25T13:01:00Z",
        expires_at="2026-07-25T13:03:00Z",
    )
    values.update(changes)
    return SensitiveMemoryDeletionAuthorization(**values)


def _read_auth(memory_id="sensitive-delete"):
    return SensitiveMemoryAccessAuthorization(
        actor="rayan",
        allowed=True,
        purpose="memory.local_sensitive_access",
        authorization_id=f"read-{memory_id}",
        allowed_operations=("read_plaintext",),
        expires_at="2026-07-25T13:10:00Z",
        memory_ids=(memory_id,),
    )


def _request_delete(connection, memory_id="sensitive-delete"):
    return request_sensitive_memory_deletion(
        connection,
        memory_id=memory_id,
        authorization=_request_auth(memory_id),
        requested_at="2026-07-25T13:00:30Z",
    )


def _model():
    policy = load_semantic_policy()
    return FakeEncoder(policy.model.embedding_dimension)


def test_request_marks_pending_and_blocks_sensitive_plaintext(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        result = _request_delete(connection)

        assert result.memory.deletion_state == "pending_deletion"
        assert load_sensitive_payload_record(
            connection, memory_id="sensitive-delete"
        ).ciphertext
        with pytest.raises(SensitiveMemoryAccessValidationError):
            load_sensitive_memory_content(
                connection,
                vault,
                memory_id="sensitive-delete",
                authorization=_read_auth(),
                accessed_at="2026-07-25T13:00:40Z",
                repository_root=repository,
                key_protector=protector,
            )


def test_request_is_idempotent_and_does_not_duplicate_audit(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        first = _request_delete(connection)
        second = _request_delete(connection)

        assert second.request_event_id == first.request_event_id
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'deletion_requested'"
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sensitive_memory_access_events
            WHERE operation = 'request_deletion' AND decision = 'allowed'
            """
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"allowed": False},
        {"directly_requested": False},
        {"memory_id": "other-memory"},
        {"deletion_scope": "ordinary_memory_and_dependents"},
    ],
)
def test_request_requires_exact_direct_authorization(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        with pytest.raises(SensitiveMemoryDeletionAuthorizationError):
            request_sensitive_memory_deletion(
                connection,
                memory_id="sensitive-delete",
                authorization=_request_auth(**changes),
                requested_at="2026-07-25T13:00:30Z",
            )
        assert load_memory(
            connection, memory_id="sensitive-delete"
        ).deletion_state == "active"


def test_sensitive_path_rejects_ordinary_memory(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    with open_memory_store(vault, repository_root=repository) as connection:
        create_memory(
            connection,
            request=_ordinary_request(),
            authorization=MemoryWriteAuthorization(actor="test", allowed=True),
            created_at="2026-07-25T13:00:00Z",
        )
        with pytest.raises(SensitiveMemoryDeletionValidationError):
            request_sensitive_memory_deletion(
                connection,
                memory_id="ordinary",
                authorization=_request_auth("ordinary"),
                requested_at="2026-07-25T13:00:30Z",
            )


def test_cancellation_restores_active_payload_access(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        request = _request_delete(connection)
        result = cancel_sensitive_memory_deletion(
            connection,
            memory_id="sensitive-delete",
            authorization=_cancel_auth(),
            cancelled_at="2026-07-25T13:00:45Z",
        )

        assert result.request_event_id == request.request_event_id
        assert result.memory.deletion_state == "active"
        assert load_sensitive_memory_content(
            connection,
            vault,
            memory_id="sensitive-delete",
            authorization=_read_auth(),
            accessed_at="2026-07-25T13:00:50Z",
            repository_root=repository,
            key_protector=protector,
        ) == "private deletion fixture plaintext"


def test_final_deletion_requires_open_request(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        with pytest.raises(SensitiveMemoryDeletionStateError):
            delete_sensitive_memory(
                connection,
                memory_id="sensitive-delete",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T13:02:00Z",
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"strongly_confirmed": False},
        {"directly_requested": False},
        {"memory_id": "other-memory"},
        {"deletion_scope": "ordinary_memory_and_dependents"},
        {"expires_at": "2026-07-25T13:03:01Z"},
        {"expires_at": "2026-07-25T13:01:00Z"},
    ],
)
def test_final_deletion_requires_exact_short_lived_strong_confirmation(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        _request_delete(connection)
        with pytest.raises(SensitiveMemoryDeletionAuthorizationError):
            delete_sensitive_memory(
                connection,
                memory_id="sensitive-delete",
                authorization=_delete_auth(**changes),
                deleted_at="2026-07-25T13:02:00Z",
            )
        assert load_memory(
            connection, memory_id="sensitive-delete"
        ).deletion_state == "pending_deletion"
        assert load_sensitive_payload_record(
            connection, memory_id="sensitive-delete"
        ).ciphertext


def test_final_deletion_removes_payload_and_preserves_sanitized_evidence(
    tmp_path: Path,
) -> None:
    plaintext = "private deletion fixture plaintext"
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        connection.execute(
            """
            INSERT INTO memory_entities (
                memory_entity_id, memory_id, entity_type, entity_value,
                normalized_value, created_at
            ) VALUES ('entity-1', 'sensitive-delete', 'topic', 'private',
                      'private', '2026-07-25T13:00:10Z')
            """
        )
        _request_delete(connection)
        result = delete_sensitive_memory(
            connection,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
        )

        with pytest.raises(MemoryNotFoundError):
            load_memory(connection, memory_id="sensitive-delete")
        with pytest.raises(SensitiveMemoryStorageError):
            load_sensitive_payload_record(connection, memory_id="sensitive-delete")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_sources WHERE memory_id = 'sensitive-delete'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_id = 'sensitive-delete'"
        ).fetchone()[0] == 0
        assert result.tombstone.deleted_memory_id == "sensitive-delete"
        assert result.tombstone.deletion_scope == SENSITIVE_MEMORY_DELETION_SCOPE

        events = connection.execute(
            "SELECT details_json FROM memory_events ORDER BY created_at"
        ).fetchall()
        access = connection.execute(
            """
            SELECT memory_id, purpose, authorization_id, operation, decision
            FROM sensitive_memory_access_events ORDER BY created_at
            """
        ).fetchall()
        serialized = json.dumps(
            [str(row["details_json"]) for row in events]
            + [dict(row) for row in access],
            sort_keys=True,
        )
        assert plaintext not in serialized
        assert all(row["memory_id"] is None for row in access)
        assert any(row["operation"] == "delete" for row in access)


def test_completed_sensitive_deletion_is_idempotent(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        _request_delete(connection)
        first = delete_sensitive_memory(
            connection,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
        )
        second = delete_sensitive_memory(
            connection,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
        )
        assert second == first
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_tombstones"
        ).fetchone()[0] == 1
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sensitive_memory_access_events
            WHERE operation = 'delete' AND decision = 'allowed'
            """
        ).fetchone()[0] == 1


def test_deleted_sensitive_identifier_cannot_be_reused(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        _request_delete(connection)
        delete_sensitive_memory(
            connection,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
        )
        with pytest.raises(MemoryAlreadyExistsError):
            _create_sensitive(connection, vault, repository, protector)


def test_tombstone_failure_rolls_back_memory_and_encrypted_payload(
    tmp_path: Path,
) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        _request_delete(connection)
        connection.execute(
            """
            CREATE TRIGGER reject_sensitive_tombstone
            BEFORE INSERT ON memory_tombstones
            BEGIN
                SELECT RAISE(ABORT, 'reject sensitive tombstone');
            END
            """
        )
        with pytest.raises(SensitiveMemoryDeletionValidationError):
            delete_sensitive_memory(
                connection,
                memory_id="sensitive-delete",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T13:02:00Z",
            )

        assert load_memory(
            connection, memory_id="sensitive-delete"
        ).deletion_state == "pending_deletion"
        assert load_sensitive_payload_record(
            connection, memory_id="sensitive-delete"
        ).ciphertext
        with pytest.raises(MemoryTombstoneNotFoundError):
            load_memory_tombstone(connection, memory_id="sensitive-delete")
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_events WHERE event_type = 'deleted'"
        ).fetchone()[0] == 0


def test_tampered_request_or_completion_audit_fails_closed(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    with open_memory_store(vault, repository_root=repository) as connection:
        _create_sensitive(connection, vault, repository, protector)
        request = _request_delete(connection)
        connection.execute(
            "UPDATE memory_events SET details_json = '{}' WHERE event_id = ?",
            (request.request_event_id,),
        )
        with pytest.raises(SensitiveMemoryDeletionStateError):
            delete_sensitive_memory(
                connection,
                memory_id="sensitive-delete",
                authorization=_delete_auth(),
                deleted_at="2026-07-25T13:02:00Z",
            )

    repository2 = tmp_path / "repo2"
    vault2 = tmp_path / "vault2"
    repository2.mkdir()
    vault2.mkdir()
    with open_memory_store(vault2, repository_root=repository2) as connection:
        _create_sensitive(connection, vault2, repository2, protector)
        _request_delete(connection)
        result = delete_sensitive_memory(
            connection,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
        )
        connection.execute(
            "UPDATE memory_events SET details_json = '{}' WHERE event_id = ?",
            (result.tombstone.event_id,),
        )
        with pytest.raises(SensitiveMemoryDeletionStateError):
            load_sensitive_memory_deletion(
                connection,
                memory_id="sensitive-delete",
            )


def test_sensitive_deletion_purges_and_rebuilds_all_indexes(tmp_path: Path) -> None:
    repository, vault = _setup(tmp_path)
    protector = InMemoryTestKeyProtector()
    model = _model()
    with open_memory_store(vault, repository_root=repository) as connection:
        create_memory(
            connection,
            request=_ordinary_request("keep"),
            authorization=MemoryWriteAuthorization(actor="test", allowed=True),
            created_at="2026-07-25T13:00:00Z",
        )
        _create_sensitive(connection, vault, repository, protector)
        build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-25T13:00:10Z",
        )
        build_memory_semantic_index(
            connection,
            vault,
            model=model,
            repository_root=repository,
            built_at="2026-07-25T13:00:10Z",
        )
        _request_delete(connection)
        result = delete_sensitive_memory_with_index_purge(
            connection,
            vault,
            memory_id="sensitive-delete",
            authorization=_delete_auth(),
            deleted_at="2026-07-25T13:02:00Z",
            repository_root=repository,
        )
        assert result.purge.lexical_root.exists() is False
        assert result.purge.semantic_root.exists() is False

        rebuilt = rebuild_indexes_after_memory_deletion(
            connection,
            vault,
            memory_id="sensitive-delete",
            model=model,
            built_at="2026-07-25T13:03:00Z",
            repository_root=repository,
        )
        assert rebuilt.verification.lexical_manifest.record_count == 1
        assert rebuilt.verification.semantic_manifest.record_count == 1
        assert memory_lexical_index_path(
            vault, repository_root=repository
        ).exists()
        assert memory_semantic_index_root(
            vault, repository_root=repository
        ).exists()
