"""Purpose-bound local access to encrypted HIGHLY_SENSITIVE memory.

P2.6b deliberately keeps sensitive content out of ordinary lexical, semantic,
and hybrid retrieval. Discovery is metadata-only and requires deterministic
selectors. Plaintext decryption requires an exact memory scope and a live,
purpose-bound authorization. All normal allowed/denied access decisions are
recorded without plaintext or free-form purpose text.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from .provenance import AttachedMemorySource, list_memory_sources
from .schema import KNOWLEDGE_STATUSES, MEMORY_CATEGORIES
from .sensitive_crypto import (
    SensitiveCryptoError,
    SensitiveKeyProtector,
    SensitiveMasterKeyStore,
    WindowsDPAPIKeyProtector,
    decrypt_sensitive_payload,
)
from .sensitive_storage import (
    SensitiveMemoryStorageError,
    load_sensitive_payload_record,
)
from .service import (
    MemoryNotFoundError,
    MemoryRecord,
    MemoryValidationError,
    _normalize_timestamp,
    load_memory,
)

SENSITIVE_ACCESS_OPERATIONS = (
    "search_metadata",
    "inspect_metadata",
    "inspect_provenance",
    "read_plaintext",
)

_SAFE_AUDIT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")


class SensitiveMemoryAccessError(RuntimeError):
    """Base error for purpose-bound sensitive-memory access."""


class SensitiveMemoryAccessAuthorizationError(SensitiveMemoryAccessError):
    """Raised when a sensitive-memory access decision is denied."""


class SensitiveMemoryAccessValidationError(SensitiveMemoryAccessError):
    """Raised when a sensitive-memory request is malformed or inapplicable."""


@dataclass(frozen=True)
class SensitiveMemoryAccessAuthorization:
    """Narrow local authorization for HIGHLY_SENSITIVE memory access.

    ``memory_ids`` is mandatory for exact-resource operations
    (inspect_metadata, inspect_provenance, read_plaintext). Metadata search may
    be broader, but the request itself must include at least one deterministic
    non-content selector.
    """

    actor: str
    allowed: bool
    purpose: str
    authorization_id: str
    allowed_operations: tuple[str, ...]
    expires_at: str
    memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitiveMemoryMetadataSearchRequest:
    """Deterministic metadata-only sensitive-memory discovery request."""

    memory_id: str | None = None
    memory_key: str | None = None
    category: str | None = None
    knowledge_status: str | None = None
    recorded_from: str | None = None
    recorded_to: str | None = None
    include_historical: bool = False
    include_archived: bool = False
    limit: int = 25


@dataclass(frozen=True)
class SensitiveMemoryMetadata:
    """Metadata-only representation; plaintext and snippets are absent."""

    memory_id: str
    content_sha256: str
    memory_key: str | None
    category: str
    knowledge_status: str
    confidence: float
    data_classification: str
    valid_from: str | None
    valid_to: str | None
    time_precision: str | None
    recorded_at: str
    verified_at: str | None
    rayan_confirmed: bool
    validity_state: str
    retention_state: str
    deletion_state: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SensitiveMemoryMetadataSearchResponse:
    results: tuple[SensitiveMemoryMetadata, ...]


def _safe_identifier(value: str, *, field_name: str) -> str:
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(value):
        raise SensitiveMemoryAccessAuthorizationError(
            f"{field_name} must be a 3-128 character audit-safe identifier "
            "containing only letters, numbers, underscore, dot, colon, or hyphen."
        )
    return value


def _canonical_timestamp(value: str, *, field_name: str) -> str:
    try:
        return _normalize_timestamp(value, field_name=field_name)
    except MemoryValidationError as exc:
        raise SensitiveMemoryAccessValidationError(str(exc)) from exc


def _record_access_event(
    connection: sqlite3.Connection,
    *,
    memory_id: str | None,
    authorization: SensitiveMemoryAccessAuthorization,
    operation: str,
    decision: str,
    created_at: str,
) -> None:
    """Persist a sanitized access decision. No plaintext or free-form reason."""
    audit_memory_id = memory_id
    if memory_id is not None:
        exists = connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if exists is None:
            audit_memory_id = None

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
            audit_memory_id,
            authorization.actor,
            authorization.purpose,
            authorization.authorization_id,
            operation,
            decision,
            created_at,
        ),
    )


def _validate_authorization_identifiers(
    authorization: SensitiveMemoryAccessAuthorization,
) -> None:
    _safe_identifier(authorization.actor, field_name="Sensitive-memory actor")
    _safe_identifier(authorization.purpose, field_name="Sensitive-memory purpose")
    _safe_identifier(
        authorization.authorization_id,
        field_name="Sensitive-memory authorization_id",
    )

    if not authorization.allowed_operations:
        raise SensitiveMemoryAccessAuthorizationError(
            "Sensitive-memory authorization requires at least one allowed operation."
        )

    unknown = sorted(
        set(authorization.allowed_operations) - set(SENSITIVE_ACCESS_OPERATIONS)
    )
    if unknown:
        raise SensitiveMemoryAccessAuthorizationError(
            "Sensitive-memory authorization contains unsupported operations: "
            + ", ".join(unknown)
        )

    if len(set(authorization.allowed_operations)) != len(
        authorization.allowed_operations
    ):
        raise SensitiveMemoryAccessAuthorizationError(
            "Sensitive-memory allowed_operations must not contain duplicates."
        )

    if len(set(authorization.memory_ids)) != len(authorization.memory_ids):
        raise SensitiveMemoryAccessAuthorizationError(
            "Sensitive-memory memory_ids scope must not contain duplicates."
        )
    if any(not memory_id.strip() for memory_id in authorization.memory_ids):
        raise SensitiveMemoryAccessAuthorizationError(
            "Sensitive-memory memory_ids scope cannot contain empty identifiers."
        )


def _require_access(
    connection: sqlite3.Connection,
    *,
    authorization: SensitiveMemoryAccessAuthorization,
    operation: str,
    accessed_at: str,
    memory_id: str | None = None,
    require_exact_scope: bool = False,
) -> str:
    """Authorize one operation and audit ordinary denials before raising."""
    if operation not in SENSITIVE_ACCESS_OPERATIONS:
        raise SensitiveMemoryAccessValidationError(
            f"Unsupported sensitive-memory access operation: {operation!r}"
        )

    _validate_authorization_identifiers(authorization)
    canonical_accessed_at = _canonical_timestamp(
        accessed_at,
        field_name="accessed_at",
    )

    denial: str | None = None
    if not authorization.allowed:
        denial = "Sensitive-memory access denied by explicit authorization."
    elif operation not in authorization.allowed_operations:
        denial = (
            "Sensitive-memory authorization does not allow operation "
            f"{operation!r}."
        )
    else:
        expires_at = _canonical_timestamp(
            authorization.expires_at,
            field_name="authorization.expires_at",
        )
        if canonical_accessed_at > expires_at:
            denial = "Sensitive-memory authorization has expired."

    if denial is None and require_exact_scope:
        if memory_id is None or memory_id not in authorization.memory_ids:
            denial = (
                "Sensitive-memory exact-resource access requires the target "
                "memory_id in the authorization scope."
            )

    if denial is not None:
        _record_access_event(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation=operation,
            decision="denied",
            created_at=canonical_accessed_at,
        )
        raise SensitiveMemoryAccessAuthorizationError(denial)

    return canonical_accessed_at


def _metadata(record: MemoryRecord) -> SensitiveMemoryMetadata:
    return SensitiveMemoryMetadata(
        memory_id=record.memory_id,
        content_sha256=record.content_sha256,
        memory_key=record.memory_key,
        category=record.category,
        knowledge_status=record.knowledge_status,
        confidence=record.confidence,
        data_classification=record.data_classification,
        valid_from=record.valid_from,
        valid_to=record.valid_to,
        time_precision=record.time_precision,
        recorded_at=record.recorded_at,
        verified_at=record.verified_at,
        rayan_confirmed=record.rayan_confirmed,
        validity_state=record.validity_state,
        retention_state=record.retention_state,
        deletion_state=record.deletion_state,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _require_sensitive_record(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryRecord:
    try:
        record = load_memory(connection, memory_id=memory_id)
    except MemoryNotFoundError as exc:
        raise SensitiveMemoryAccessValidationError(
            f"Sensitive memory not found: {memory_id}"
        ) from exc

    if record.data_classification != "HIGHLY_SENSITIVE":
        raise SensitiveMemoryAccessValidationError(
            f"Memory is not HIGHLY_SENSITIVE: {memory_id}"
        )
    if record.deletion_state != "active":
        raise SensitiveMemoryAccessValidationError(
            "Sensitive memory is not active and cannot be accessed."
        )

    payload_exists = connection.execute(
        "SELECT 1 FROM memory_sensitive_payloads WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if payload_exists is None:
        raise SensitiveMemoryAccessValidationError(
            f"Encrypted sensitive payload is missing: {memory_id}"
        )
    return record


def _validate_search_request(
    request: SensitiveMemoryMetadataSearchRequest,
) -> tuple[str | None, str | None]:
    if request.limit < 1 or request.limit > 100:
        raise SensitiveMemoryAccessValidationError(
            "Sensitive metadata search limit must be between 1 and 100."
        )

    for field_name, value in (
        ("memory_id", request.memory_id),
        ("memory_key", request.memory_key),
    ):
        if value is not None and not value.strip():
            raise SensitiveMemoryAccessValidationError(
                f"request.{field_name} cannot be empty when provided."
            )

    if (
        request.category is not None
        and request.category not in MEMORY_CATEGORIES
    ):
        raise SensitiveMemoryAccessValidationError(
            f"Unsupported sensitive-memory category: {request.category!r}"
        )
    if (
        request.knowledge_status is not None
        and request.knowledge_status not in KNOWLEDGE_STATUSES
    ):
        raise SensitiveMemoryAccessValidationError(
            "Unsupported sensitive-memory knowledge_status: "
            f"{request.knowledge_status!r}"
        )

    selectors = (
        request.memory_id,
        request.memory_key,
        request.category,
        request.knowledge_status,
        request.recorded_from,
        request.recorded_to,
    )
    if not any(value is not None for value in selectors):
        raise SensitiveMemoryAccessValidationError(
            "Sensitive metadata search requires at least one deterministic "
            "non-content selector."
        )

    recorded_from = (
        None
        if request.recorded_from is None
        else _canonical_timestamp(
            request.recorded_from,
            field_name="request.recorded_from",
        )
    )
    recorded_to = (
        None
        if request.recorded_to is None
        else _canonical_timestamp(
            request.recorded_to,
            field_name="request.recorded_to",
        )
    )
    if (
        recorded_from is not None
        and recorded_to is not None
        and recorded_to < recorded_from
    ):
        raise SensitiveMemoryAccessValidationError(
            "request.recorded_to cannot be earlier than request.recorded_from."
        )
    return recorded_from, recorded_to


def search_sensitive_memory_metadata(
    connection: sqlite3.Connection,
    *,
    request: SensitiveMemoryMetadataSearchRequest,
    authorization: SensitiveMemoryAccessAuthorization,
    accessed_at: str,
) -> SensitiveMemoryMetadataSearchResponse:
    """Search HIGHLY_SENSITIVE memory by metadata only; never by content."""
    canonical_accessed_at = _require_access(
        connection,
        authorization=authorization,
        operation="search_metadata",
        accessed_at=accessed_at,
    )

    try:
        recorded_from, recorded_to = _validate_search_request(request)
    except SensitiveMemoryAccessValidationError:
        _record_access_event(
            connection,
            memory_id=None,
            authorization=authorization,
            operation="search_metadata",
            decision="denied",
            created_at=canonical_accessed_at,
        )
        raise

    clauses = [
        "data_classification = 'HIGHLY_SENSITIVE'",
        "deletion_state = 'active'",
        "EXISTS (SELECT 1 FROM memory_sensitive_payloads AS payload "
        "WHERE payload.memory_id = memories.memory_id)",
    ]
    parameters: list[object] = []

    if not request.include_archived:
        clauses.append("retention_state <> 'archived'")
    if not request.include_historical:
        clauses.append("validity_state <> 'historical'")
    if request.memory_id is not None:
        clauses.append("memory_id = ?")
        parameters.append(request.memory_id)
    if request.memory_key is not None:
        clauses.append("memory_key = ?")
        parameters.append(request.memory_key)
    if request.category is not None:
        clauses.append("category = ?")
        parameters.append(request.category)
    if request.knowledge_status is not None:
        clauses.append("knowledge_status = ?")
        parameters.append(request.knowledge_status)
    if recorded_from is not None:
        clauses.append("recorded_at >= ?")
        parameters.append(recorded_from)
    if recorded_to is not None:
        clauses.append("recorded_at <= ?")
        parameters.append(recorded_to)

    parameters.append(request.limit)
    rows = connection.execute(
        f"""
        SELECT memory_id
        FROM memories
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at DESC, memory_id
        LIMIT ?
        """,
        parameters,
    ).fetchall()

    results = tuple(
        _metadata(load_memory(connection, memory_id=str(row["memory_id"])))
        for row in rows
    )
    _record_access_event(
        connection,
        memory_id=None,
        authorization=authorization,
        operation="search_metadata",
        decision="allowed",
        created_at=canonical_accessed_at,
    )
    return SensitiveMemoryMetadataSearchResponse(results=results)


def inspect_sensitive_memory_metadata(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: SensitiveMemoryAccessAuthorization,
    accessed_at: str,
) -> SensitiveMemoryMetadata:
    """Inspect one exact sensitive memory as metadata only."""
    canonical_accessed_at = _require_access(
        connection,
        authorization=authorization,
        operation="inspect_metadata",
        accessed_at=accessed_at,
        memory_id=memory_id,
        require_exact_scope=True,
    )
    try:
        record = _require_sensitive_record(connection, memory_id=memory_id)
    except SensitiveMemoryAccessValidationError:
        _record_access_event(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="inspect_metadata",
            decision="denied",
            created_at=canonical_accessed_at,
        )
        raise

    _record_access_event(
        connection,
        memory_id=memory_id,
        authorization=authorization,
        operation="inspect_metadata",
        decision="allowed",
        created_at=canonical_accessed_at,
    )
    return _metadata(record)


def inspect_sensitive_memory_provenance(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: SensitiveMemoryAccessAuthorization,
    accessed_at: str,
) -> tuple[AttachedMemorySource, ...]:
    """Inspect private-safe provenance for one exact sensitive memory."""
    canonical_accessed_at = _require_access(
        connection,
        authorization=authorization,
        operation="inspect_provenance",
        accessed_at=accessed_at,
        memory_id=memory_id,
        require_exact_scope=True,
    )
    try:
        _require_sensitive_record(connection, memory_id=memory_id)
    except SensitiveMemoryAccessValidationError:
        _record_access_event(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="inspect_provenance",
            decision="denied",
            created_at=canonical_accessed_at,
        )
        raise

    sources = list_memory_sources(connection, memory_id=memory_id)
    _record_access_event(
        connection,
        memory_id=memory_id,
        authorization=authorization,
        operation="inspect_provenance",
        decision="allowed",
        created_at=canonical_accessed_at,
    )
    return sources


def load_sensitive_memory_content(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    memory_id: str,
    authorization: SensitiveMemoryAccessAuthorization,
    accessed_at: str,
    repository_root: str | Path | None = None,
    key_protector: SensitiveKeyProtector | None = None,
) -> str:
    """Decrypt one exact HIGHLY_SENSITIVE memory after authorization."""
    canonical_accessed_at = _require_access(
        connection,
        authorization=authorization,
        operation="read_plaintext",
        accessed_at=accessed_at,
        memory_id=memory_id,
        require_exact_scope=True,
    )

    try:
        record = _require_sensitive_record(connection, memory_id=memory_id)
        payload = load_sensitive_payload_record(connection, memory_id=memory_id)
        protector = key_protector or WindowsDPAPIKeyProtector()
        master_key = SensitiveMasterKeyStore(
            vault_root,
            protector=protector,
            repository_root=repository_root,
        ).load_key()
        plaintext = decrypt_sensitive_payload(
            master_key=master_key,
            memory_id=memory_id,
            content_sha256=record.content_sha256,
            payload=payload.encrypted_payload(),
        )
        if hashlib.sha256(plaintext.encode("utf-8")).hexdigest() != record.content_sha256:
            raise SensitiveCryptoError(
                "Sensitive plaintext digest does not match authoritative metadata."
            )
    except (
        SensitiveMemoryAccessValidationError,
        SensitiveMemoryStorageError,
        SensitiveCryptoError,
    ):
        _record_access_event(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="read_plaintext",
            decision="denied",
            created_at=canonical_accessed_at,
        )
        raise

    _record_access_event(
        connection,
        memory_id=memory_id,
        authorization=authorization,
        operation="read_plaintext",
        decision="allowed",
        created_at=canonical_accessed_at,
    )
    return plaintext
