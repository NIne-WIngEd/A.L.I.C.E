"""P2.6a encrypted HIGHLY_SENSITIVE persistence tests."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from alice_memory.inspection import (
    MemoryInspectionAuthorizationError,
    inspect_memory,
    list_memory_summaries,
)
from alice_memory.lexical_index import build_memory_lexical_index
from alice_memory.sensitive_crypto import (
    InMemoryTestKeyProtector,
    SensitiveMasterKeyStore,
    decrypt_sensitive_payload,
)
from alice_memory.sensitive_storage import (
    SENSITIVE_CONTENT_SENTINEL,
    SensitiveMemoryAuthorizationError,
    SensitiveMemoryWriteAuthorization,
    create_sensitive_memory,
    load_sensitive_payload_record,
)
from alice_memory.service import (
    MemoryContentAccessAuthorization,
    MemoryContentAuthorizationError,
    MemoryCreateRequest,
    MemoryValidationError,
    MemoryWriteAuthorization,
    archive_memory,
    create_memory,
    load_memory_content,
)
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store
from alice_memory.temporal import (
    list_current_memories_for_key,
    list_memory_history,
    resolve_memory_at,
)


def _source() -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="rayan_direct_statement",
        source_ref="test-suite:direct-sensitive-statement",
        support_relation="supports",
    )


def _request(
    *,
    memory_id: str = "sensitive-1",
    content: str = "A deeply private life event.",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key="life.private-event",
        category="episodic",
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at="2026-07-22T00:00:00Z",
        sources=(_source(),),
        rayan_confirmed=True,
    )


def _authorization(
    *,
    allowed: bool = True,
    directly_requested: bool = True,
) -> SensitiveMemoryWriteAuthorization:
    return SensitiveMemoryWriteAuthorization(
        actor="rayan",
        allowed=allowed,
        purpose="memory.user_requested_storage",
        authorization_id="auth-sensitive-create-1",
        directly_requested=directly_requested,
    )


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return (
        repository,
        vault,
        open_memory_store(vault, repository_root=repository),
    )


def test_sensitive_creation_encrypts_payload_and_never_stores_plaintext_in_memories(
    tmp_path: Path,
) -> None:
    content = "A deeply private life event."
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()

    with manager as connection:
        memory = create_sensitive_memory(
            connection,
            vault,
            request=_request(content=content),
            authorization=_authorization(),
            created_at="2026-07-22T00:00:00Z",
            repository_root=repository,
            key_protector=protector,
        )

        row = connection.execute(
            "SELECT content, content_sha256 FROM memories WHERE memory_id = ?",
            (memory.memory_id,),
        ).fetchone()
        payload = load_sensitive_payload_record(
            connection,
            memory_id=memory.memory_id,
        )
        source_count = connection.execute(
            "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?",
            (memory.memory_id,),
        ).fetchone()[0]
        access_event = connection.execute(
            """
            SELECT actor, purpose, authorization_id, operation, decision
            FROM sensitive_memory_access_events
            WHERE memory_id = ?
            """,
            (memory.memory_id,),
        ).fetchone()

        assert row["content"] == SENSITIVE_CONTENT_SENTINEL
        assert content not in row["content"]
        assert row["content_sha256"] == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        assert content.encode("utf-8") not in payload.ciphertext
        assert source_count == 1
        assert access_event["actor"] == "rayan"
        assert access_event["operation"] == "create"
        assert access_event["decision"] == "allowed"
        assert content not in access_event["purpose"]

        master_key = SensitiveMasterKeyStore(
            vault,
            protector=protector,
            repository_root=repository,
        ).load_or_create_key()
        assert decrypt_sensitive_payload(
            master_key=master_key,
            memory_id=memory.memory_id,
            content_sha256=memory.content_sha256,
            payload=payload.encrypted_payload(),
        ) == content


def test_sensitive_creation_requires_direct_purpose_bound_authorization(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    with manager as connection:
        with pytest.raises(SensitiveMemoryAuthorizationError):
            create_sensitive_memory(
                connection,
                vault,
                request=_request(),
                authorization=_authorization(directly_requested=False),
                created_at="2026-07-22T00:00:00Z",
                repository_root=repository,
                key_protector=InMemoryTestKeyProtector(),
            )
        assert connection.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0] == 0


def test_ordinary_memory_creation_path_still_rejects_highly_sensitive(
    tmp_path: Path,
) -> None:
    _repository, _vault, manager = _open(tmp_path)
    with manager as connection:
        with pytest.raises(MemoryValidationError):
            create_memory(
                connection,
                request=_request(),
                authorization=MemoryWriteAuthorization(
                    actor="test",
                    allowed=True,
                ),
                created_at="2026-07-22T00:00:00Z",
            )


def test_ordinary_plaintext_inspection_and_temporal_paths_fail_closed(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    with manager as connection:
        create_sensitive_memory(
            connection,
            vault,
            request=_request(),
            authorization=_authorization(),
            created_at="2026-07-22T00:00:00Z",
            repository_root=repository,
            key_protector=InMemoryTestKeyProtector(),
        )

        with pytest.raises(MemoryContentAuthorizationError):
            load_memory_content(
                connection,
                memory_id="sensitive-1",
                authorization=MemoryContentAccessAuthorization(
                    actor="test",
                    allowed=True,
                ),
            )
        with pytest.raises(MemoryInspectionAuthorizationError):
            inspect_memory(connection, memory_id="sensitive-1")
        assert list_memory_summaries(connection) == ()
        assert list_current_memories_for_key(
            connection,
            memory_key="life.private-event",
        ) == ()
        assert list_memory_history(
            connection,
            memory_key="life.private-event",
        ) == ()
        assert resolve_memory_at(
            connection,
            memory_key="life.private-event",
            at="2026-07-22T00:00:00Z",
        ).memories == ()
        with pytest.raises(MemoryValidationError):
            archive_memory(
                connection,
                memory_id="sensitive-1",
                authorization=MemoryWriteAuthorization(
                    actor="test",
                    allowed=True,
                ),
                archived_at="2026-07-22T00:01:00Z",
            )


def test_sensitive_memory_is_excluded_from_ordinary_lexical_index(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    with manager as connection:
        create_sensitive_memory(
            connection,
            vault,
            request=_request(content="uniquesensitivetoken only here"),
            authorization=_authorization(),
            created_at="2026-07-22T00:00:00Z",
            repository_root=repository,
            key_protector=InMemoryTestKeyProtector(),
        )
        manifest = build_memory_lexical_index(
            connection,
            vault,
            repository_root=repository,
            built_at="2026-07-22T00:01:00Z",
        )
        assert manifest.record_count == 0


def test_sensitive_payload_insert_failure_rolls_back_memory_provenance_and_events(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    with manager as connection:
        connection.execute(
            """
            CREATE TRIGGER force_sensitive_payload_failure
            BEFORE INSERT ON memory_sensitive_payloads
            BEGIN
                SELECT RAISE(ABORT, 'forced sensitive payload failure');
            END
            """
        )

        with pytest.raises(sqlite3.IntegrityError):
            create_sensitive_memory(
                connection,
                vault,
                request=_request(),
                authorization=_authorization(),
                created_at="2026-07-22T00:00:00Z",
                repository_root=repository,
                key_protector=InMemoryTestKeyProtector(),
            )

        for table in (
            "memories",
            "memory_sources",
            "memory_events",
            "memory_sensitive_payloads",
            "sensitive_memory_access_events",
        ):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
