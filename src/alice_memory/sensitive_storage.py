"""Encrypted HIGHLY_SENSITIVE memory creation foundation for P2.6a.

This module provides protected creation. Purpose-bound metadata discovery and
plaintext retrieval are implemented separately in ``sensitive_access``. Ordinary
P2.3-P2.5 APIs continue to fail closed for HIGHLY_SENSITIVE content.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from .sensitive_crypto import (
    EncryptedSensitivePayload,
    SensitiveKeyProtector,
    SensitiveMasterKeyStore,
    WindowsDPAPIKeyProtector,
    encrypt_sensitive_payload,
)
from .service import (
    MemoryCreateRequest,
    MemoryRecord,
    MemoryValidationError,
    _insert_memory_in_transaction,
    _normalize_create_request,
    _normalize_timestamp,
    load_memory,
)
from .store import transaction

SENSITIVE_CONTENT_SENTINEL = "[ALICE:HIGHLY_SENSITIVE:ENCRYPTED]"


class SensitiveMemoryStorageError(RuntimeError):
    """Base error for encrypted sensitive-memory persistence."""


class SensitiveMemoryAuthorizationError(SensitiveMemoryStorageError):
    """Raised when protected creation lacks purpose-bound authorization."""

_SAFE_AUDIT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")


def _require_safe_audit_identifier(value: str, *, field_name: str) -> None:
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(value):
        raise SensitiveMemoryAuthorizationError(
            f"{field_name} must be a 3-128 character audit-safe identifier "
            "containing only letters, numbers, underscore, dot, colon, or hyphen."
        )


@dataclass(frozen=True)
class SensitiveMemoryWriteAuthorization:
    """Narrow authorization for one directly requested sensitive-memory write."""

    actor: str
    allowed: bool
    purpose: str
    authorization_id: str
    directly_requested: bool


@dataclass(frozen=True)
class SensitivePayloadRecord:
    memory_id: str
    ciphertext: bytes
    nonce: bytes
    algorithm: str
    key_id: str
    aad_version: int
    created_at: str
    updated_at: str

    def encrypted_payload(self) -> EncryptedSensitivePayload:
        return EncryptedSensitivePayload(
            ciphertext=self.ciphertext,
            nonce=self.nonce,
            algorithm=self.algorithm,
            key_id=self.key_id,
            aad_version=self.aad_version,
        )


def _require_sensitive_write_authorization(
    authorization: SensitiveMemoryWriteAuthorization,
) -> None:
    if not authorization.allowed:
        raise SensitiveMemoryAuthorizationError(
            "Sensitive-memory creation denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise SensitiveMemoryAuthorizationError(
            "Sensitive-memory creation requires a non-empty actor."
        )
    _require_safe_audit_identifier(
        authorization.purpose,
        field_name="Sensitive-memory purpose",
    )
    _require_safe_audit_identifier(
        authorization.authorization_id,
        field_name="Sensitive-memory authorization_id",
    )
    if not authorization.directly_requested:
        raise SensitiveMemoryAuthorizationError(
            "P2.6a supports only directly requested HIGHLY_SENSITIVE memory "
            "creation; autonomous sensitive-memory mutation remains disabled."
        )


def load_sensitive_payload_record(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> SensitivePayloadRecord:
    """Load encrypted payload bytes without decrypting them."""
    row = connection.execute(
        """
        SELECT
            memory_id,
            ciphertext,
            nonce,
            algorithm,
            key_id,
            aad_version,
            created_at,
            updated_at
        FROM memory_sensitive_payloads
        WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        raise SensitiveMemoryStorageError(
            f"Encrypted sensitive payload not found: {memory_id}"
        )
    return SensitivePayloadRecord(
        memory_id=str(row["memory_id"]),
        ciphertext=bytes(row["ciphertext"]),
        nonce=bytes(row["nonce"]),
        algorithm=str(row["algorithm"]),
        key_id=str(row["key_id"]),
        aad_version=int(row["aad_version"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_sensitive_memory(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    request: MemoryCreateRequest,
    authorization: SensitiveMemoryWriteAuthorization,
    created_at: str,
    repository_root: str | Path | None = None,
    key_protector: SensitiveKeyProtector | None = None,
) -> MemoryRecord:
    """Atomically persist metadata, provenance, and an encrypted payload.

    The ordinary memories.content column receives only a fixed sentinel. The
    SHA-256 remains the digest of the real plaintext so provenance/integrity
    semantics are preserved without making the content searchable.
    """
    _require_sensitive_write_authorization(authorization)

    if request.data_classification != "HIGHLY_SENSITIVE":
        raise MemoryValidationError(
            "create_sensitive_memory requires HIGHLY_SENSITIVE classification."
        )

    if request.memory_id is None:
        request = replace(request, memory_id=str(uuid.uuid4()))

    request = _normalize_create_request(
        request,
        allow_highly_sensitive=True,
    )
    created_at = _normalize_timestamp(
        created_at,
        field_name="created_at",
    )
    memory_id = request.memory_id
    assert memory_id is not None

    protector = key_protector or WindowsDPAPIKeyProtector()
    key_store = SensitiveMasterKeyStore(
        vault_root,
        protector=protector,
        repository_root=repository_root,
    )
    master_key = key_store.load_or_create_key()
    content_sha256 = hashlib.sha256(
        request.content.encode("utf-8")
    ).hexdigest()
    encrypted = encrypt_sensitive_payload(
        master_key=master_key,
        memory_id=memory_id,
        content=request.content,
        content_sha256=content_sha256,
    )

    with transaction(connection):
        persisted_id = _insert_memory_in_transaction(
            connection,
            request=request,
            actor=authorization.actor,
            created_at=created_at,
            stored_content=SENSITIVE_CONTENT_SENTINEL,
            content_sha256=content_sha256,
        )
        if persisted_id != memory_id:
            raise SensitiveMemoryStorageError(
                "Sensitive memory identifier changed during persistence."
            )

        connection.execute(
            """
            INSERT INTO memory_sensitive_payloads (
                memory_id,
                ciphertext,
                nonce,
                algorithm,
                key_id,
                aad_version,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                encrypted.ciphertext,
                encrypted.nonce,
                encrypted.algorithm,
                encrypted.key_id,
                encrypted.aad_version,
                created_at,
                created_at,
            ),
        )

        connection.execute(
            """
            INSERT INTO sensitive_memory_access_events (
                access_event_id,
                memory_id,
                actor,
                purpose,
                authorization_id,
                operation,
                decision,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                memory_id,
                authorization.actor,
                authorization.purpose,
                authorization.authorization_id,
                "create",
                "allowed",
                created_at,
            ),
        )

    return load_memory(connection, memory_id=memory_id)
