"""Deterministic Phase 2 memory lifecycle and authorized read service.

P2.3 provides explicit, permission-gated creation and archival operations plus
metadata-safe reads. Plaintext reads require explicit authorization.

Until P2.6 introduces the dedicated protected storage and purpose-based access
layer, HIGHLY_SENSITIVE memory writes and reads fail closed. SECRETS are never
ordinary memories.

P2.3 intentionally does not perform automatic memory formation, model calls,
correction/supersession logic, deletion, vector indexing, or external actions.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass

from .schema import (
    DATA_CLASSIFICATIONS,
    KNOWLEDGE_STATUSES,
    MEMORY_CATEGORIES,
    MEMORY_STORABLE_CLASSIFICATIONS,
    RETENTION_STATES,
    SCHEMA_VERSION,
    VALIDITY_STATES,
)
from .store import transaction


class MemoryServiceError(RuntimeError):
    """Base error for deterministic Memory Core lifecycle operations."""


class MemoryWriteAuthorizationError(MemoryServiceError):
    """Raised when a mutation is attempted without explicit authorization."""


class MemoryContentAuthorizationError(MemoryServiceError):
    """Raised when plaintext memory access is not explicitly authorized."""


class MemoryValidationError(MemoryServiceError):
    """Raised when a requested memory does not satisfy service invariants."""


class MemoryAlreadyExistsError(MemoryServiceError):
    """Raised when a requested memory ID already exists."""


class MemoryNotFoundError(MemoryServiceError):
    """Raised when a requested memory cannot be found."""


@dataclass(frozen=True)
class MemoryWriteAuthorization:
    """Explicit deterministic authorization context for one memory mutation."""

    actor: str
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MemoryContentAccessAuthorization:
    """Explicit authorization for plaintext memory access."""

    actor: str
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MemoryCreateRequest:
    """Explicit input for creating one durable memory record."""

    content: str
    category: str
    knowledge_status: str
    confidence: float
    data_classification: str
    recorded_at: str
    validity_state: str = "current"
    retention_state: str = "durable"
    memory_id: str | None = None
    memory_key: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    time_precision: str | None = None
    verified_at: str | None = None
    rayan_confirmed: bool = False


@dataclass(frozen=True)
class MemoryRecord:
    """Metadata-safe authoritative memory record."""

    memory_id: str
    schema_version: int
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
class _StoredMemoryRecord:
    """Internal database representation that includes plaintext content."""

    metadata: MemoryRecord
    content: str


def _require_write_authorization(
    authorization: MemoryWriteAuthorization,
) -> None:
    if not authorization.allowed:
        raise MemoryWriteAuthorizationError(
            "Memory mutation denied by explicit write authorization."
        )
    if not authorization.actor.strip():
        raise MemoryWriteAuthorizationError(
            "Authorized memory mutations require a non-empty actor."
        )


def _require_content_authorization(
    authorization: MemoryContentAccessAuthorization | None,
) -> None:
    if authorization is None or not authorization.allowed:
        raise MemoryContentAuthorizationError(
            "Plaintext memory access requires explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryContentAuthorizationError(
            "Authorized plaintext access requires a non-empty actor."
        )


def _validate_create_request(request: MemoryCreateRequest) -> None:
    if not request.content.strip():
        raise MemoryValidationError("Memory content cannot be empty.")

    if request.category not in MEMORY_CATEGORIES:
        raise MemoryValidationError(
            f"Unsupported memory category: {request.category!r}"
        )

    if request.knowledge_status not in KNOWLEDGE_STATUSES:
        raise MemoryValidationError(
            f"Unsupported knowledge status: {request.knowledge_status!r}"
        )

    if not 0.0 <= request.confidence <= 1.0:
        raise MemoryValidationError(
            "Memory confidence must be between 0.0 and 1.0."
        )

    if request.data_classification not in DATA_CLASSIFICATIONS:
        raise MemoryValidationError(
            "Unsupported data classification: "
            f"{request.data_classification!r}"
        )

    if request.data_classification not in MEMORY_STORABLE_CLASSIFICATIONS:
        raise MemoryValidationError(
            "SECRETS are prohibited from ordinary A.L.I.C.E. memory storage."
        )

    if request.data_classification == "HIGHLY_SENSITIVE":
        raise MemoryValidationError(
            "HIGHLY_SENSITIVE memory storage is disabled until the dedicated "
            "protected storage and purpose-based access layer is enabled."
        )

    if request.validity_state not in VALIDITY_STATES:
        raise MemoryValidationError(
            f"Unsupported validity state: {request.validity_state!r}"
        )

    if request.retention_state not in RETENTION_STATES:
        raise MemoryValidationError(
            f"Unsupported retention state: {request.retention_state!r}"
        )

    if (
        request.valid_from is not None
        and request.valid_to is not None
        and request.valid_to < request.valid_from
    ):
        raise MemoryValidationError(
            "valid_to cannot be earlier than valid_from."
        )


def _row_to_stored_memory(row: sqlite3.Row) -> _StoredMemoryRecord:
    metadata = MemoryRecord(
        memory_id=str(row["memory_id"]),
        schema_version=int(row["schema_version"]),
        content_sha256=str(row["content_sha256"]),
        memory_key=(
            None if row["memory_key"] is None else str(row["memory_key"])
        ),
        category=str(row["category"]),
        knowledge_status=str(row["knowledge_status"]),
        confidence=float(row["confidence"]),
        data_classification=str(row["data_classification"]),
        valid_from=(
            None if row["valid_from"] is None else str(row["valid_from"])
        ),
        valid_to=(
            None if row["valid_to"] is None else str(row["valid_to"])
        ),
        time_precision=(
            None
            if row["time_precision"] is None
            else str(row["time_precision"])
        ),
        recorded_at=str(row["recorded_at"]),
        verified_at=(
            None if row["verified_at"] is None else str(row["verified_at"])
        ),
        rayan_confirmed=bool(row["rayan_confirmed"]),
        validity_state=str(row["validity_state"]),
        retention_state=str(row["retention_state"]),
        deletion_state=str(row["deletion_state"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
    return _StoredMemoryRecord(
        metadata=metadata,
        content=str(row["content"]),
    )


def _load_stored_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> _StoredMemoryRecord:
    row = connection.execute(
        """
        SELECT *
        FROM memories
        WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()

    if row is None:
        raise MemoryNotFoundError(f"Memory not found: {memory_id}")

    return _row_to_stored_memory(row)


def load_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryRecord:
    """Load one authoritative memory as metadata only."""
    return _load_stored_memory(
        connection,
        memory_id=memory_id,
    ).metadata


def load_memory_content(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: MemoryContentAccessAuthorization | None,
) -> str:
    """Load plaintext only with explicit authorization."""
    _require_content_authorization(authorization)
    stored = _load_stored_memory(
        connection,
        memory_id=memory_id,
    )

    if stored.metadata.data_classification == "HIGHLY_SENSITIVE":
        raise MemoryContentAuthorizationError(
            "HIGHLY_SENSITIVE plaintext access is disabled until the "
            "dedicated protected storage and purpose-based access layer "
            "is enabled."
        )

    return stored.content


def create_memory(
    connection: sqlite3.Connection,
    *,
    request: MemoryCreateRequest,
    authorization: MemoryWriteAuthorization,
    created_at: str,
) -> MemoryRecord:
    """Create one explicit durable memory and a sanitized lifecycle event."""
    _require_write_authorization(authorization)
    _validate_create_request(request)

    memory_id = request.memory_id or str(uuid.uuid4())
    content_sha256 = hashlib.sha256(
        request.content.encode("utf-8")
    ).hexdigest()

    event_id = str(uuid.uuid4())
    event_details = json.dumps(
        {
            "category": request.category,
            "knowledge_status": request.knowledge_status,
            "data_classification": request.data_classification,
            "rayan_confirmed": request.rayan_confirmed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    try:
        with transaction(connection):
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
                    memory_id,
                    SCHEMA_VERSION,
                    request.content,
                    content_sha256,
                    request.memory_key,
                    request.category,
                    request.knowledge_status,
                    request.confidence,
                    request.data_classification,
                    request.valid_from,
                    request.valid_to,
                    request.time_precision,
                    request.recorded_at,
                    request.verified_at,
                    int(request.rayan_confirmed),
                    request.validity_state,
                    request.retention_state,
                    "active",
                    created_at,
                    created_at,
                ),
            )

            connection.execute(
                """
                INSERT INTO memory_events (
                    event_id,
                    memory_id,
                    event_type,
                    actor,
                    details_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    memory_id,
                    "created",
                    authorization.actor,
                    event_details,
                    created_at,
                ),
            )
    except sqlite3.IntegrityError as exc:
        existing = connection.execute(
            "SELECT 1 FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if existing is not None:
            raise MemoryAlreadyExistsError(
                f"Memory already exists: {memory_id}"
            ) from exc
        raise MemoryValidationError(
            f"Memory creation failed database validation: {exc}"
        ) from exc

    return load_memory(
        connection,
        memory_id=memory_id,
    )


def archive_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: MemoryWriteAuthorization,
    archived_at: str,
) -> MemoryRecord:
    """Archive a memory without deleting its history or provenance.

    Re-archiving an already archived memory is idempotent and creates no
    duplicate lifecycle event.
    """
    _require_write_authorization(authorization)
    existing = load_memory(
        connection,
        memory_id=memory_id,
    )

    if existing.deletion_state != "active":
        raise MemoryValidationError(
            "Only active, non-deleted memories can be archived."
        )

    if existing.retention_state == "archived":
        return existing

    event_id = str(uuid.uuid4())
    event_details = json.dumps(
        {
            "previous_retention_state": existing.retention_state,
            "new_retention_state": "archived",
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with transaction(connection):
        connection.execute(
            """
            UPDATE memories
            SET retention_state = ?,
                updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "archived",
                archived_at,
                memory_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO memory_events (
                event_id,
                memory_id,
                event_type,
                actor,
                details_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                memory_id,
                "archived",
                authorization.actor,
                event_details,
                archived_at,
            ),
        )

    return load_memory(
        connection,
        memory_id=memory_id,
    )
