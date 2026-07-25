"""Authorized ordinary-memory deletion lifecycle for A.L.I.C.E. P2.8a.

Deletion is a destructive P4 operation. Ordinary memories therefore move
through an explicit request state before irreversible removal. Final deletion
requires strong, short-lived authorization bound to the exact memory and
scope. The memory row and dependent authoritative records are removed in one
transaction while a sanitized tombstone and audit event remain.

HIGHLY_SENSITIVE deletion is intentionally excluded from this module. It
requires the dedicated protected-payload deletion path added separately.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .service import (
    MemoryNotFoundError,
    MemoryRecord,
    MemoryValidationError,
    _normalize_timestamp,
    load_memory,
)
from .store import transaction

ORDINARY_MEMORY_DELETION_SCOPE = "ordinary_memory_and_dependents"
MAX_STRONG_CONFIRMATION_SECONDS = 120


class MemoryDeletionError(RuntimeError):
    """Base error for ordinary-memory deletion operations."""


class MemoryDeletionAuthorizationError(MemoryDeletionError):
    """Raised when deletion authorization is absent, stale, or mis-scoped."""


class MemoryDeletionStateError(MemoryDeletionError):
    """Raised when a memory cannot enter the requested deletion transition."""


class MemoryDeletionValidationError(MemoryDeletionError):
    """Raised when deterministic deletion validation fails."""


class MemoryTombstoneNotFoundError(MemoryDeletionError):
    """Raised when no tombstone exists for a deleted memory identifier."""


@dataclass(frozen=True)
class MemoryDeletionRequestAuthorization:
    """Explicit authorization bound to one reversible deletion request."""

    actor: str
    allowed: bool
    memory_id: str
    deletion_scope: str
    authorization_id: str
    reason: str | None = None


@dataclass(frozen=True)
class MemoryDeletionCancellationAuthorization:
    """Explicit authorization bound to cancellation of one pending request."""

    actor: str
    allowed: bool
    memory_id: str
    deletion_scope: str
    authorization_id: str
    reason: str | None = None


@dataclass(frozen=True)
class MemoryDeletionAuthorization:
    """Strong, short-lived authorization for irreversible deletion."""

    actor: str
    allowed: bool
    memory_id: str
    deletion_scope: str
    authorization_id: str
    strongly_confirmed: bool
    issued_at: str
    expires_at: str
    reason: str | None = None


@dataclass(frozen=True)
class MemoryDeletionRequestResult:
    """Metadata-safe result of entering pending-deletion state."""

    memory: MemoryRecord
    request_event_id: str
    authorization_id: str
    deletion_scope: str
    requested_by: str
    requested_at: str


@dataclass(frozen=True)
class MemoryDeletionCancellationResult:
    """Metadata-safe result of restoring a pending memory to active state."""

    memory: MemoryRecord
    cancellation_event_id: str
    request_event_id: str
    authorization_id: str
    deletion_scope: str
    cancelled_by: str
    cancelled_at: str


@dataclass(frozen=True)
class MemoryTombstoneRecord:
    """Sanitized durable evidence that one memory identifier was deleted."""

    tombstone_id: str
    deleted_memory_id: str
    content_sha256: str
    deleted_at: str
    deletion_scope: str
    event_id: str | None


@dataclass(frozen=True)
class MemoryDeletionResult:
    """Metadata-safe result of completed irreversible deletion."""

    tombstone: MemoryTombstoneRecord
    authorization_id: str
    request_event_id: str
    deleted_by: str
    deleted_at: str


@dataclass(frozen=True)
class _DeletionLifecycleEvent:
    event_id: str
    event_type: str
    operation: str
    actor: str
    authorization_id: str
    deletion_scope: str
    request_event_id: str | None
    target_memory_id: str
    created_at: str


_TOMBSTONE_NAMESPACE = uuid.UUID("89dba647-e550-46d5-8781-e3c3574fa74f")
_SAFE_AUDIT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")


def _tombstone_id(memory_id: str) -> str:
    return str(uuid.uuid5(_TOMBSTONE_NAMESPACE, memory_id))


def _require_safe_identifier(value: str, *, field_name: str) -> None:
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(value):
        raise MemoryDeletionAuthorizationError(
            f"{field_name} must be a 3-128 character audit-safe identifier "
            "containing only letters, numbers, underscore, dot, colon, or "
            "hyphen."
        )


_DeletionAuthorizationLike = (
    MemoryDeletionRequestAuthorization
    | MemoryDeletionCancellationAuthorization
    | MemoryDeletionAuthorization
)


def _require_common_authorization(
    authorization: _DeletionAuthorizationLike,
    *,
    memory_id: str,
) -> None:
    if not authorization.allowed:
        raise MemoryDeletionAuthorizationError(
            "Memory deletion operation denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryDeletionAuthorizationError(
            "Authorized memory deletion operations require a non-empty actor."
        )
    if authorization.memory_id != memory_id:
        raise MemoryDeletionAuthorizationError(
            "Memory deletion authorization is not bound to the requested "
            "memory."
        )
    if authorization.deletion_scope != ORDINARY_MEMORY_DELETION_SCOPE:
        raise MemoryDeletionAuthorizationError(
            "Ordinary-memory deletion authorization has an unsupported or "
            "mis-scoped deletion_scope."
        )
    _require_safe_identifier(
        authorization.authorization_id,
        field_name="Deletion authorization_id",
    )


def _parse_utc(value: str, *, field_name: str) -> tuple[str, datetime]:
    try:
        normalized = _normalize_timestamp(value, field_name=field_name)
    except MemoryValidationError as exc:
        raise MemoryDeletionAuthorizationError(str(exc)) from exc
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return normalized, parsed.astimezone(timezone.utc)


def _require_final_authorization(
    authorization: MemoryDeletionAuthorization,
    *,
    memory_id: str,
    deleted_at: str,
) -> str:
    _require_common_authorization(authorization, memory_id=memory_id)
    if not authorization.strongly_confirmed:
        raise MemoryDeletionAuthorizationError(
            "Irreversible memory deletion requires strong confirmation."
        )

    issued_at, issued_time = _parse_utc(
        authorization.issued_at,
        field_name="authorization.issued_at",
    )
    expires_at, expires_time = _parse_utc(
        authorization.expires_at,
        field_name="authorization.expires_at",
    )
    normalized_deleted_at, deleted_time = _parse_utc(
        deleted_at,
        field_name="deleted_at",
    )

    lifetime = (expires_time - issued_time).total_seconds()
    if lifetime <= 0:
        raise MemoryDeletionAuthorizationError(
            "Deletion authorization must expire after it is issued."
        )
    if lifetime > MAX_STRONG_CONFIRMATION_SECONDS:
        raise MemoryDeletionAuthorizationError(
            "Strong deletion authorization cannot remain valid for more than "
            f"{MAX_STRONG_CONFIRMATION_SECONDS} seconds."
        )
    if deleted_time < issued_time or deleted_time > expires_time:
        raise MemoryDeletionAuthorizationError(
            "Strong deletion authorization is not valid at deleted_at."
        )

    # Canonical parsing above is intentional even though only deleted_at is
    # persisted. It prevents alternate timestamp representations from
    # bypassing the two-minute confirmation window.
    assert issued_at and expires_at
    return normalized_deleted_at


def _require_ordinary_memory(memory: MemoryRecord) -> None:
    if memory.data_classification == "HIGHLY_SENSITIVE":
        raise MemoryDeletionValidationError(
            "HIGHLY_SENSITIVE deletion requires the dedicated protected "
            "payload deletion path."
        )


def _row_to_tombstone(row: sqlite3.Row) -> MemoryTombstoneRecord:
    return MemoryTombstoneRecord(
        tombstone_id=str(row["tombstone_id"]),
        deleted_memory_id=str(row["deleted_memory_id"]),
        content_sha256=str(row["content_sha256"]),
        deleted_at=str(row["deleted_at"]),
        deletion_scope=str(row["deletion_scope"]),
        event_id=None if row["event_id"] is None else str(row["event_id"]),
    )


def load_memory_tombstone(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryTombstoneRecord:
    """Load sanitized deletion evidence for one former memory identifier."""
    row = connection.execute(
        """
        SELECT
            tombstone_id,
            deleted_memory_id,
            content_sha256,
            deleted_at,
            deletion_scope,
            event_id
        FROM memory_tombstones
        WHERE deleted_memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        raise MemoryTombstoneNotFoundError(
            f"Memory tombstone not found: {memory_id}"
        )
    return _row_to_tombstone(row)


def _parse_lifecycle_event(row: sqlite3.Row) -> _DeletionLifecycleEvent | None:
    try:
        details = json.loads(str(row["details_json"]))
    except (TypeError, ValueError):
        return None
    if not isinstance(details, dict):
        return None

    operation = str(details.get("operation", ""))
    if operation not in {"deletion_requested", "deletion_cancelled"}:
        return None

    return _DeletionLifecycleEvent(
        event_id=str(row["event_id"]),
        event_type=str(row["event_type"]),
        operation=operation,
        actor=str(row["actor"]),
        authorization_id=str(details.get("authorization_id", "")),
        deletion_scope=str(details.get("deletion_scope", "")),
        request_event_id=(
            None
            if details.get("request_event_id") in (None, "")
            else str(details.get("request_event_id"))
        ),
        target_memory_id=str(details.get("target_memory_id", "")),
        created_at=str(row["created_at"]),
    )


def _latest_deletion_lifecycle_event(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> _DeletionLifecycleEvent | None:
    rows = connection.execute(
        """
        SELECT event_id, event_type, actor, details_json, created_at
        FROM memory_events
        WHERE memory_id = ?
          AND event_type IN ('deletion_requested', 'reclassified')
        ORDER BY created_at DESC, event_id DESC
        """,
        (memory_id,),
    ).fetchall()
    for row in rows:
        parsed = _parse_lifecycle_event(row)
        if parsed is not None:
            return parsed
    return None


def _open_deletion_request(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> _DeletionLifecycleEvent:
    event = _latest_deletion_lifecycle_event(
        connection,
        memory_id=memory_id,
    )
    if event is None or event.operation != "deletion_requested":
        raise MemoryDeletionStateError(
            "Memory has no open deletion request."
        )
    if event.event_type != "deletion_requested":
        raise MemoryDeletionStateError(
            "Deletion request audit event has an unexpected event type."
        )
    if event.deletion_scope != ORDINARY_MEMORY_DELETION_SCOPE:
        raise MemoryDeletionStateError(
            "Deletion request audit event has an unsupported scope."
        )
    if event.target_memory_id != memory_id:
        raise MemoryDeletionStateError(
            "Deletion request audit event is not bound to the memory."
        )
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(event.authorization_id):
        raise MemoryDeletionStateError(
            "Deletion request audit event has an invalid authorization_id."
        )
    return event


def _request_result(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryDeletionRequestResult:
    memory = load_memory(connection, memory_id=memory_id)
    event = _open_deletion_request(connection, memory_id=memory_id)
    if memory.deletion_state != "pending_deletion":
        raise MemoryDeletionStateError(
            "Open deletion request does not match memory deletion state."
        )
    return MemoryDeletionRequestResult(
        memory=memory,
        request_event_id=event.event_id,
        authorization_id=event.authorization_id,
        deletion_scope=event.deletion_scope,
        requested_by=event.actor,
        requested_at=event.created_at,
    )


def request_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: MemoryDeletionRequestAuthorization,
    requested_at: str,
) -> MemoryDeletionRequestResult:
    """Move one ordinary memory into reversible pending-deletion state."""
    _require_common_authorization(authorization, memory_id=memory_id)
    try:
        requested_at = _normalize_timestamp(
            requested_at,
            field_name="requested_at",
        )
    except MemoryValidationError as exc:
        raise MemoryDeletionValidationError(str(exc)) from exc

    try:
        existing = load_memory(connection, memory_id=memory_id)
    except MemoryNotFoundError as exc:
        try:
            load_memory_tombstone(connection, memory_id=memory_id)
        except MemoryTombstoneNotFoundError:
            raise exc
        raise MemoryDeletionStateError(
            "Deleted memories cannot receive a new deletion request."
        ) from exc

    _require_ordinary_memory(existing)
    if existing.deletion_state == "pending_deletion":
        return _request_result(connection, memory_id=memory_id)
    if existing.deletion_state != "active":
        raise MemoryDeletionStateError(
            "Only active ordinary memories can request deletion."
        )

    event_id = str(uuid.uuid4())
    details_json = json.dumps(
        {
            "authorization_id": authorization.authorization_id,
            "deletion_scope": authorization.deletion_scope,
            "new_deletion_state": "pending_deletion",
            "operation": "deletion_requested",
            "previous_deletion_state": "active",
            "target_memory_id": memory_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE memories
            SET deletion_state = 'pending_deletion',
                updated_at = ?
            WHERE memory_id = ?
              AND deletion_state = 'active'
            """,
            (requested_at, memory_id),
        )
        if cursor.rowcount != 1:
            raise MemoryDeletionStateError(
                "Memory state changed before deletion request could commit."
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
            VALUES (?, ?, 'deletion_requested', ?, ?, ?)
            """,
            (
                event_id,
                memory_id,
                authorization.actor,
                details_json,
                requested_at,
            ),
        )

    return _request_result(connection, memory_id=memory_id)


def cancel_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: MemoryDeletionCancellationAuthorization,
    cancelled_at: str,
) -> MemoryDeletionCancellationResult:
    """Restore one pending ordinary memory to active state before deletion."""
    _require_common_authorization(authorization, memory_id=memory_id)
    try:
        cancelled_at = _normalize_timestamp(
            cancelled_at,
            field_name="cancelled_at",
        )
    except MemoryValidationError as exc:
        raise MemoryDeletionValidationError(str(exc)) from exc

    memory = load_memory(connection, memory_id=memory_id)
    _require_ordinary_memory(memory)
    if memory.deletion_state != "pending_deletion":
        raise MemoryDeletionStateError(
            "Only pending-deletion memories can cancel deletion."
        )

    request_event = _open_deletion_request(
        connection,
        memory_id=memory_id,
    )
    if request_event.deletion_scope != authorization.deletion_scope:
        raise MemoryDeletionAuthorizationError(
            "Cancellation scope does not match the open deletion request."
        )

    cancellation_event_id = str(uuid.uuid4())
    details_json = json.dumps(
        {
            "authorization_id": authorization.authorization_id,
            "deletion_scope": authorization.deletion_scope,
            "new_deletion_state": "active",
            "operation": "deletion_cancelled",
            "previous_deletion_state": "pending_deletion",
            "request_event_id": request_event.event_id,
            "target_memory_id": memory_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE memories
            SET deletion_state = 'active',
                updated_at = ?
            WHERE memory_id = ?
              AND deletion_state = 'pending_deletion'
            """,
            (cancelled_at, memory_id),
        )
        if cursor.rowcount != 1:
            raise MemoryDeletionStateError(
                "Memory state changed before deletion cancellation could "
                "commit."
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
            VALUES (?, ?, 'reclassified', ?, ?, ?)
            """,
            (
                cancellation_event_id,
                memory_id,
                authorization.actor,
                details_json,
                cancelled_at,
            ),
        )

    return MemoryDeletionCancellationResult(
        memory=load_memory(connection, memory_id=memory_id),
        cancellation_event_id=cancellation_event_id,
        request_event_id=request_event.event_id,
        authorization_id=authorization.authorization_id,
        deletion_scope=authorization.deletion_scope,
        cancelled_by=authorization.actor,
        cancelled_at=cancelled_at,
    )


def _dependent_counts(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> dict[str, int]:
    queries = {
        "sources": "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?",
        "relations": (
            "SELECT COUNT(*) FROM memory_relations "
            "WHERE from_memory_id = ? OR to_memory_id = ?"
        ),
        "derivations": (
            "SELECT COUNT(*) FROM memory_derivations WHERE memory_id = ?"
        ),
        "entities": "SELECT COUNT(*) FROM memory_entities WHERE memory_id = ?",
        "candidate_links": (
            "SELECT COUNT(*) FROM memory_candidates "
            "WHERE promoted_memory_id = ?"
        ),
    }
    counts: dict[str, int] = {}
    for name, query in queries.items():
        parameters = (memory_id, memory_id) if name == "relations" else (memory_id,)
        counts[name] = int(connection.execute(query, parameters).fetchone()[0])
    return counts


def _verify_authoritative_dependents_removed(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> None:
    remaining = {
        "sources": connection.execute(
            "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0],
        "relations": connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_relations
            WHERE from_memory_id = ? OR to_memory_id = ?
            """,
            (memory_id, memory_id),
        ).fetchone()[0],
        "derivations": connection.execute(
            "SELECT COUNT(*) FROM memory_derivations WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0],
        "entities": connection.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0],
        "sensitive_payloads": connection.execute(
            "SELECT COUNT(*) FROM memory_sensitive_payloads WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0],
        "candidate_links": connection.execute(
            """
            SELECT COUNT(*)
            FROM memory_candidates
            WHERE promoted_memory_id = ?
            """,
            (memory_id,),
        ).fetchone()[0],
    }
    nonzero = {name: int(count) for name, count in remaining.items() if count}
    if nonzero:
        raise MemoryDeletionValidationError(
            "Dependent authoritative records remained after memory deletion: "
            + ", ".join(f"{name}={count}" for name, count in sorted(nonzero.items()))
        )


def _load_deleted_event(
    connection: sqlite3.Connection,
    *,
    event_id: str,
) -> tuple[sqlite3.Row, dict[str, object]]:
    row = connection.execute(
        """
        SELECT event_id, memory_id, event_type, actor, details_json, created_at
        FROM memory_events
        WHERE event_id = ?
        """,
        (event_id,),
    ).fetchone()
    if row is None:
        raise MemoryDeletionStateError(
            "Memory tombstone references a missing deletion event."
        )
    if str(row["event_type"]) != "deleted":
        raise MemoryDeletionStateError(
            "Memory tombstone references a non-deletion event."
        )
    if row["memory_id"] is not None:
        raise MemoryDeletionStateError(
            "Completed deletion event still references an active memory row."
        )
    try:
        details = json.loads(str(row["details_json"]))
    except (TypeError, ValueError) as exc:
        raise MemoryDeletionStateError(
            "Deletion event contains invalid JSON."
        ) from exc
    if not isinstance(details, dict):
        raise MemoryDeletionStateError(
            "Deletion event details must be a JSON object."
        )
    return row, details


def load_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryDeletionResult:
    """Load and integrity-check one completed deletion result."""
    tombstone = load_memory_tombstone(connection, memory_id=memory_id)
    if tombstone.event_id is None:
        raise MemoryDeletionStateError(
            "Memory tombstone is missing its deletion event link."
        )
    event, details = _load_deleted_event(
        connection,
        event_id=tombstone.event_id,
    )

    if str(details.get("operation", "")) != "memory_deleted":
        raise MemoryDeletionStateError(
            "Deletion event contains an unsupported operation."
        )
    if str(details.get("deleted_memory_id", "")) != memory_id:
        raise MemoryDeletionStateError(
            "Deletion event does not match the tombstoned memory identifier."
        )
    if str(details.get("deletion_scope", "")) != tombstone.deletion_scope:
        raise MemoryDeletionStateError(
            "Deletion event scope does not match the tombstone."
        )
    if str(event["created_at"]) != tombstone.deleted_at:
        raise MemoryDeletionStateError(
            "Deletion event timestamp does not match the tombstone."
        )

    authorization_id = str(details.get("authorization_id", ""))
    request_event_id = str(details.get("request_event_id", ""))
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(authorization_id):
        raise MemoryDeletionStateError(
            "Deletion event has an invalid authorization_id."
        )
    if not request_event_id:
        raise MemoryDeletionStateError(
            "Deletion event is missing request_event_id."
        )

    request_row = connection.execute(
        """
        SELECT event_type, details_json
        FROM memory_events
        WHERE event_id = ?
        """,
        (request_event_id,),
    ).fetchone()
    if request_row is None or str(request_row["event_type"]) != "deletion_requested":
        raise MemoryDeletionStateError(
            "Deletion event references an invalid request event."
        )
    try:
        request_details = json.loads(str(request_row["details_json"]))
    except (TypeError, ValueError) as exc:
        raise MemoryDeletionStateError(
            "Deletion request event contains invalid JSON."
        ) from exc
    if (
        not isinstance(request_details, dict)
        or request_details.get("operation") != "deletion_requested"
        or request_details.get("deletion_scope") != tombstone.deletion_scope
        or request_details.get("target_memory_id") != memory_id
    ):
        raise MemoryDeletionStateError(
            "Deletion request event does not match the completed deletion."
        )

    return MemoryDeletionResult(
        tombstone=tombstone,
        authorization_id=authorization_id,
        request_event_id=request_event_id,
        deleted_by=str(event["actor"]),
        deleted_at=str(event["created_at"]),
    )


def delete_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: MemoryDeletionAuthorization,
    deleted_at: str,
) -> MemoryDeletionResult:
    """Irreversibly delete one pending ordinary memory and retain a tombstone."""
    normalized_deleted_at = _require_final_authorization(
        authorization,
        memory_id=memory_id,
        deleted_at=deleted_at,
    )

    try:
        existing_result = load_memory_deletion(
            connection,
            memory_id=memory_id,
        )
    except MemoryTombstoneNotFoundError:
        existing_result = None

    if existing_result is not None:
        if existing_result.tombstone.deletion_scope != authorization.deletion_scope:
            raise MemoryDeletionAuthorizationError(
                "Existing deletion scope does not match authorization."
            )
        return existing_result

    memory = load_memory(connection, memory_id=memory_id)
    _require_ordinary_memory(memory)
    if memory.deletion_state != "pending_deletion":
        raise MemoryDeletionStateError(
            "Memory must have an open pending-deletion request before final "
            "deletion."
        )

    request_event = _open_deletion_request(
        connection,
        memory_id=memory_id,
    )
    if request_event.deletion_scope != authorization.deletion_scope:
        raise MemoryDeletionAuthorizationError(
            "Final deletion scope does not match the open deletion request."
        )

    deleted_event_id = str(uuid.uuid4())
    tombstone_id = _tombstone_id(memory_id)

    try:
        with transaction(connection):
            current = load_memory(connection, memory_id=memory_id)
            _require_ordinary_memory(current)
            if current.deletion_state != "pending_deletion":
                raise MemoryDeletionStateError(
                    "Memory state changed before deletion could commit."
                )
            current_request = _open_deletion_request(
                connection,
                memory_id=memory_id,
            )
            if current_request.event_id != request_event.event_id:
                raise MemoryDeletionStateError(
                    "Open deletion request changed before deletion could "
                    "commit."
                )

            dependent_counts = _dependent_counts(
                connection,
                memory_id=memory_id,
            )
            details_json = json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "deleted_memory_id": memory_id,
                    "deletion_scope": authorization.deletion_scope,
                    "dependent_counts": dependent_counts,
                    "operation": "memory_deleted",
                    "request_event_id": current_request.event_id,
                    "strong_confirmation": True,
                },
                sort_keys=True,
                separators=(",", ":"),
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
                VALUES (?, ?, 'deleted', ?, ?, ?)
                """,
                (
                    deleted_event_id,
                    memory_id,
                    authorization.actor,
                    details_json,
                    normalized_deleted_at,
                ),
            )

            cursor = connection.execute(
                """
                DELETE FROM memories
                WHERE memory_id = ?
                  AND deletion_state = 'pending_deletion'
                """,
                (memory_id,),
            )
            if cursor.rowcount != 1:
                raise MemoryDeletionStateError(
                    "Memory state changed before deletion could commit."
                )

            _verify_authoritative_dependents_removed(
                connection,
                memory_id=memory_id,
            )

            connection.execute(
                """
                INSERT INTO memory_tombstones (
                    tombstone_id,
                    deleted_memory_id,
                    content_sha256,
                    deleted_at,
                    deletion_scope,
                    event_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tombstone_id,
                    memory_id,
                    current.content_sha256,
                    normalized_deleted_at,
                    authorization.deletion_scope,
                    deleted_event_id,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise MemoryDeletionValidationError(
            f"Memory deletion failed database validation: {exc}"
        ) from exc

    return load_memory_deletion(connection, memory_id=memory_id)
