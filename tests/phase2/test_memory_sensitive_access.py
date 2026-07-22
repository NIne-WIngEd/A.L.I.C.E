"""P2.6b purpose-bound HIGHLY_SENSITIVE local access tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alice_memory.sensitive_access import (
    SensitiveMemoryAccessAuthorization,
    SensitiveMemoryAccessAuthorizationError,
    SensitiveMemoryAccessValidationError,
    SensitiveMemoryMetadataSearchRequest,
    inspect_sensitive_memory_metadata,
    inspect_sensitive_memory_provenance,
    load_sensitive_memory_content,
    search_sensitive_memory_metadata,
)
from alice_memory.sensitive_crypto import (
    InMemoryTestKeyProtector,
    SensitiveKeyProtectionError,
    SensitiveMasterKeyStore,
    SensitivePayloadIntegrityError,
)
from alice_memory.sensitive_storage import (
    SensitiveMemoryWriteAuthorization,
    create_sensitive_memory,
)
from alice_memory.service import MemoryCreateRequest
from alice_memory.sources import MemorySourceSpec
from alice_memory.store import open_memory_store


ACCESSED_AT = "2026-07-22T12:00:00Z"
EXPIRES_AT = "2026-07-22T12:05:00Z"


def _source(ref: str = "test-suite:sensitive-access") -> MemorySourceSpec:
    return MemorySourceSpec(
        source_type="rayan_direct_statement",
        source_ref=ref,
        support_relation="supports",
    )


def _request(
    memory_id: str,
    *,
    content: str,
    memory_key: str,
    category: str = "episodic",
    recorded_at: str = "2026-07-22T00:00:00Z",
) -> MemoryCreateRequest:
    return MemoryCreateRequest(
        memory_id=memory_id,
        content=content,
        memory_key=memory_key,
        category=category,
        knowledge_status="rayan_statement",
        confidence=1.0,
        data_classification="HIGHLY_SENSITIVE",
        recorded_at=recorded_at,
        sources=(_source(f"test-suite:{memory_id}"),),
        rayan_confirmed=True,
    )


def _write_auth(memory_id: str) -> SensitiveMemoryWriteAuthorization:
    return SensitiveMemoryWriteAuthorization(
        actor="rayan",
        allowed=True,
        purpose="memory.user_requested_storage",
        authorization_id=f"auth-create-{memory_id}",
        directly_requested=True,
    )


def _access_auth(
    *,
    operations: tuple[str, ...],
    memory_ids: tuple[str, ...] = (),
    allowed: bool = True,
    expires_at: str = EXPIRES_AT,
    authorization_id: str = "auth-sensitive-access-1",
) -> SensitiveMemoryAccessAuthorization:
    return SensitiveMemoryAccessAuthorization(
        actor="rayan",
        allowed=allowed,
        purpose="memory.local_sensitive_access",
        authorization_id=authorization_id,
        allowed_operations=operations,
        expires_at=expires_at,
        memory_ids=memory_ids,
    )


def _open(tmp_path: Path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    return repository, vault, open_memory_store(
        vault,
        repository_root=repository,
    )


def _create(
    connection,
    repository: Path,
    vault: Path,
    protector: InMemoryTestKeyProtector,
    *,
    memory_id: str,
    content: str,
    memory_key: str,
    category: str = "episodic",
    recorded_at: str = "2026-07-22T00:00:00Z",
):
    return create_sensitive_memory(
        connection,
        vault,
        request=_request(
            memory_id,
            content=content,
            memory_key=memory_key,
            category=category,
            recorded_at=recorded_at,
        ),
        authorization=_write_auth(memory_id),
        created_at=recorded_at,
        repository_root=repository,
        key_protector=protector,
    )


def _access_events(connection, *, operation: str):
    return connection.execute(
        """
        SELECT memory_id, actor, purpose, authorization_id, operation, decision
        FROM sensitive_memory_access_events
        WHERE operation = ?
        ORDER BY created_at, access_event_id
        """,
        (operation,),
    ).fetchall()


def test_metadata_search_is_selector_based_and_returns_no_plaintext(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-a",
            content="alpha private narrative",
            memory_key="life.alpha",
        )
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-b",
            content="beta private narrative",
            memory_key="life.beta",
        )

        response = search_sensitive_memory_metadata(
            connection,
            request=SensitiveMemoryMetadataSearchRequest(category="episodic"),
            authorization=_access_auth(operations=("search_metadata",)),
            accessed_at=ACCESSED_AT,
        )

        assert {item.memory_id for item in response.results} == {
            "sensitive-a",
            "sensitive-b",
        }
        assert all(item.data_classification == "HIGHLY_SENSITIVE" for item in response.results)
        assert all(not hasattr(item, "content") for item in response.results)
        assert all(not hasattr(item, "snippet") for item in response.results)
        events = _access_events(connection, operation="search_metadata")
        assert len(events) == 1
        assert events[0]["decision"] == "allowed"
        assert events[0]["memory_id"] is None


def test_metadata_search_requires_a_deterministic_non_content_selector(
    tmp_path: Path,
) -> None:
    _repository, _vault, manager = _open(tmp_path)
    with manager as connection:
        with pytest.raises(SensitiveMemoryAccessValidationError):
            search_sensitive_memory_metadata(
                connection,
                request=SensitiveMemoryMetadataSearchRequest(),
                authorization=_access_auth(operations=("search_metadata",)),
                accessed_at=ACCESSED_AT,
            )
        events = _access_events(connection, operation="search_metadata")
        assert len(events) == 1
        assert events[0]["decision"] == "denied"



def test_metadata_search_rejects_invalid_classification_metadata_selector(
    tmp_path: Path,
) -> None:
    _repository, _vault, manager = _open(tmp_path)
    with manager as connection:
        with pytest.raises(SensitiveMemoryAccessValidationError):
            search_sensitive_memory_metadata(
                connection,
                request=SensitiveMemoryMetadataSearchRequest(
                    category="not-a-category"
                ),
                authorization=_access_auth(operations=("search_metadata",)),
                accessed_at=ACCESSED_AT,
            )
        events = _access_events(connection, operation="search_metadata")
        assert len(events) == 1
        assert events[0]["decision"] == "denied"

def test_denied_and_expired_search_authorizations_are_audited(
    tmp_path: Path,
) -> None:
    _repository, _vault, manager = _open(tmp_path)
    with manager as connection:
        request = SensitiveMemoryMetadataSearchRequest(category="episodic")
        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            search_sensitive_memory_metadata(
                connection,
                request=request,
                authorization=_access_auth(
                    operations=("search_metadata",),
                    allowed=False,
                    authorization_id="auth-denied-search",
                ),
                accessed_at=ACCESSED_AT,
            )
        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            search_sensitive_memory_metadata(
                connection,
                request=request,
                authorization=_access_auth(
                    operations=("search_metadata",),
                    expires_at="2026-07-22T11:59:59Z",
                    authorization_id="auth-expired-search",
                ),
                accessed_at=ACCESSED_AT,
            )

        events = _access_events(connection, operation="search_metadata")
        assert [row["decision"] for row in events] == ["denied", "denied"]


def test_exact_metadata_inspection_requires_memory_scope(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="private content",
            memory_key="life.private",
        )

        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            inspect_sensitive_memory_metadata(
                connection,
                memory_id="sensitive-1",
                authorization=_access_auth(
                    operations=("inspect_metadata",),
                    memory_ids=("different-memory",),
                ),
                accessed_at=ACCESSED_AT,
            )

        metadata = inspect_sensitive_memory_metadata(
            connection,
            memory_id="sensitive-1",
            authorization=_access_auth(
                operations=("inspect_metadata",),
                memory_ids=("sensitive-1",),
                authorization_id="auth-inspect-metadata",
            ),
            accessed_at=ACCESSED_AT,
        )
        assert metadata.memory_id == "sensitive-1"
        assert metadata.memory_key == "life.private"
        assert not hasattr(metadata, "content")

        events = _access_events(connection, operation="inspect_metadata")
        assert sorted(row["decision"] for row in events) == ["allowed", "denied"]


def test_provenance_inspection_is_exact_scoped_and_private_safe(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="private content",
            memory_key="life.private",
        )
        sources = inspect_sensitive_memory_provenance(
            connection,
            memory_id="sensitive-1",
            authorization=_access_auth(
                operations=("inspect_provenance",),
                memory_ids=("sensitive-1",),
            ),
            accessed_at=ACCESSED_AT,
        )
        assert len(sources) == 1
        assert sources[0].memory_id == "sensitive-1"
        assert sources[0].source_type == "rayan_direct_statement"
        assert not hasattr(sources[0], "content")
        events = _access_events(connection, operation="inspect_provenance")
        assert len(events) == 1
        assert events[0]["decision"] == "allowed"


def test_plaintext_read_requires_exact_scope_and_allowed_operation(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="private content",
            memory_key="life.private",
        )

        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            load_sensitive_memory_content(
                connection,
                vault,
                memory_id="sensitive-1",
                authorization=_access_auth(
                    operations=("inspect_metadata",),
                    memory_ids=("sensitive-1",),
                ),
                accessed_at=ACCESSED_AT,
                repository_root=repository,
                key_protector=protector,
            )
        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            load_sensitive_memory_content(
                connection,
                vault,
                memory_id="sensitive-1",
                authorization=_access_auth(
                    operations=("read_plaintext",),
                    memory_ids=("different-memory",),
                    authorization_id="auth-wrong-scope",
                ),
                accessed_at=ACCESSED_AT,
                repository_root=repository,
                key_protector=protector,
            )

        events = _access_events(connection, operation="read_plaintext")
        assert [row["decision"] for row in events] == ["denied", "denied"]


def test_plaintext_read_decrypts_after_authorization_and_audits_without_content(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    plaintext = "A deeply private life event that must never appear in audit logs."
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content=plaintext,
            memory_key="life.private",
        )
        loaded = load_sensitive_memory_content(
            connection,
            vault,
            memory_id="sensitive-1",
            authorization=_access_auth(
                operations=("read_plaintext",),
                memory_ids=("sensitive-1",),
            ),
            accessed_at=ACCESSED_AT,
            repository_root=repository,
            key_protector=protector,
        )
        assert loaded == plaintext

        events = _access_events(connection, operation="read_plaintext")
        assert len(events) == 1
        assert events[0]["decision"] == "allowed"
        serialized = "|".join(str(value) for value in events[0])
        assert plaintext not in serialized
        assert events[0]["purpose"] == "memory.local_sensitive_access"


def test_plaintext_read_fails_closed_when_master_key_is_missing(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="private content",
            memory_key="life.private",
        )
        key_store = SensitiveMasterKeyStore(
            vault,
            protector=protector,
            repository_root=repository,
        )
        key_store.path.unlink()

        with pytest.raises(SensitiveKeyProtectionError):
            load_sensitive_memory_content(
                connection,
                vault,
                memory_id="sensitive-1",
                authorization=_access_auth(
                    operations=("read_plaintext",),
                    memory_ids=("sensitive-1",),
                ),
                accessed_at=ACCESSED_AT,
                repository_root=repository,
                key_protector=protector,
            )
        assert not key_store.path.exists()
        events = _access_events(connection, operation="read_plaintext")
        assert events[-1]["decision"] == "denied"


def test_tampered_ciphertext_is_denied_and_audited(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="private content",
            memory_key="life.private",
        )
        ciphertext = connection.execute(
            "SELECT ciphertext FROM memory_sensitive_payloads WHERE memory_id = ?",
            ("sensitive-1",),
        ).fetchone()[0]
        tampered = bytearray(ciphertext)
        tampered[0] ^= 0x01
        connection.execute(
            "UPDATE memory_sensitive_payloads SET ciphertext = ? WHERE memory_id = ?",
            (bytes(tampered), "sensitive-1"),
        )

        with pytest.raises(SensitivePayloadIntegrityError):
            load_sensitive_memory_content(
                connection,
                vault,
                memory_id="sensitive-1",
                authorization=_access_auth(
                    operations=("read_plaintext",),
                    memory_ids=("sensitive-1",),
                ),
                accessed_at=ACCESSED_AT,
                repository_root=repository,
                key_protector=protector,
            )
        events = _access_events(connection, operation="read_plaintext")
        assert events[-1]["decision"] == "denied"


def test_metadata_search_cannot_match_sensitive_plaintext_token(
    tmp_path: Path,
) -> None:
    repository, vault, manager = _open(tmp_path)
    protector = InMemoryTestKeyProtector()
    with manager as connection:
        _create(
            connection,
            repository,
            vault,
            protector,
            memory_id="sensitive-1",
            content="uniquesecretplaintexttoken",
            memory_key="life.private",
        )
        response = search_sensitive_memory_metadata(
            connection,
            request=SensitiveMemoryMetadataSearchRequest(
                memory_key="uniquesecretplaintexttoken"
            ),
            authorization=_access_auth(operations=("search_metadata",)),
            accessed_at=ACCESSED_AT,
        )
        assert response.results == ()


def test_nonexistent_exact_target_denial_does_not_break_audit_foreign_key(
    tmp_path: Path,
) -> None:
    _repository, _vault, manager = _open(tmp_path)
    with manager as connection:
        with pytest.raises(SensitiveMemoryAccessAuthorizationError):
            inspect_sensitive_memory_metadata(
                connection,
                memory_id="missing-memory",
                authorization=_access_auth(
                    operations=("inspect_metadata",),
                    memory_ids=("different-memory",),
                ),
                accessed_at=ACCESSED_AT,
            )
        events = _access_events(connection, operation="inspect_metadata")
        assert len(events) == 1
        assert events[0]["memory_id"] is None
        assert events[0]["decision"] == "denied"
