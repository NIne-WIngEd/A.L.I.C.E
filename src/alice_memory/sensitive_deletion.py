"""Protected HIGHLY_SENSITIVE memory deletion for A.L.I.C.E. P2.8c.

Sensitive deletion is a destructive P4 operation. It uses a dedicated path
that is bound to the exact memory, purpose, deletion scope, and short-lived
strong confirmation. Request and cancellation remain reversible. Final
completion atomically removes the authoritative row, encrypted payload, and
dependent records while preserving only sanitized audit evidence and a
content-digest tombstone.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .deletion import (
    MAX_STRONG_CONFIRMATION_SECONDS,
    MemoryDeletionCancellationResult,
    MemoryDeletionRequestResult,
    MemoryDeletionResult,
    MemoryDeletionStateError,
    MemoryTombstoneNotFoundError,
    _dependent_counts,
    _tombstone_id,
    _verify_authoritative_dependents_removed,
    load_memory_deletion,
    load_memory_tombstone,
)
from .deletion_integrity import (
    MemoryDeletionIntegrityError,
    collect_promoted_candidate_lineage,
    promoted_candidate_lineage_audit,
    purge_promoted_candidate_lineage,
    require_no_unmanaged_active_derivative_tables,
    validate_deletion_lifecycle_details,
    verify_promoted_candidate_lineage_removed,
)
from .sensitive_storage import (
    SENSITIVE_CONTENT_SENTINEL,
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
from .store import transaction

SENSITIVE_MEMORY_DELETION_SCOPE = (
    "highly_sensitive_memory_encrypted_payload_and_dependents"
)

_SAFE_AUDIT_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")


class SensitiveMemoryDeletionError(RuntimeError):
    """Base error for protected sensitive-memory deletion."""


class SensitiveMemoryDeletionAuthorizationError(SensitiveMemoryDeletionError):
    """Raised when authorization is denied, stale, weak, or mis-scoped."""


class SensitiveMemoryDeletionStateError(SensitiveMemoryDeletionError):
    """Raised when sensitive deletion lifecycle state is inconsistent."""


class SensitiveMemoryDeletionValidationError(SensitiveMemoryDeletionError):
    """Raised when a target is not a valid encrypted sensitive memory."""


@dataclass(frozen=True)
class SensitiveMemoryDeletionRequestAuthorization:
    """Explicit direct authorization for one reversible deletion request."""

    actor: str
    allowed: bool
    purpose: str
    authorization_id: str
    memory_id: str
    deletion_scope: str
    directly_requested: bool


@dataclass(frozen=True)
class SensitiveMemoryDeletionCancellationAuthorization:
    """Explicit direct authorization for cancelling one pending request."""

    actor: str
    allowed: bool
    purpose: str
    authorization_id: str
    memory_id: str
    deletion_scope: str
    directly_requested: bool


@dataclass(frozen=True)
class SensitiveMemoryDeletionAuthorization:
    """Strong, exact, short-lived authorization for irreversible deletion."""

    actor: str
    allowed: bool
    purpose: str
    authorization_id: str
    memory_id: str
    deletion_scope: str
    directly_requested: bool
    strongly_confirmed: bool
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class _SensitiveDeletionLifecycleEvent:
    event_id: str
    event_type: str
    operation: str
    actor: str
    authorization_id: str
    deletion_scope: str
    request_event_id: str | None
    target_memory_id: str
    created_at: str


_SensitiveAuthorization = (
    SensitiveMemoryDeletionRequestAuthorization
    | SensitiveMemoryDeletionCancellationAuthorization
    | SensitiveMemoryDeletionAuthorization
)


def _safe_identifier(value: str, *, field_name: str) -> str:
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(value):
        raise SensitiveMemoryDeletionAuthorizationError(
            f"{field_name} must be a 3-128 character audit-safe identifier "
            "containing only letters, numbers, underscore, dot, colon, or "
            "hyphen."
        )
    return value


def _canonical_timestamp(value: str, *, field_name: str) -> str:
    try:
        return _normalize_timestamp(value, field_name=field_name)
    except MemoryValidationError as exc:
        raise SensitiveMemoryDeletionValidationError(str(exc)) from exc


def _parse_utc(value: str, *, field_name: str) -> tuple[str, datetime]:
    canonical = _canonical_timestamp(value, field_name=field_name)
    parsed = datetime.fromisoformat(canonical.replace("Z", "+00:00"))
    return canonical, parsed.astimezone(timezone.utc)


def _record_sensitive_deletion_decision(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: _SensitiveAuthorization,
    operation: str,
    decision: str,
    created_at: str,
) -> None:
    audit_memory_id: str | None = memory_id
    if connection.execute(
        "SELECT 1 FROM memories WHERE memory_id = ?",
        (memory_id,),
    ).fetchone() is None:
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


def _require_common_authorization(
    connection: sqlite3.Connection,
    *,
    authorization: _SensitiveAuthorization,
    memory_id: str,
    operation: str,
    occurred_at: str,
) -> str:
    canonical = _canonical_timestamp(occurred_at, field_name=f"{operation}_at")
    _safe_identifier(authorization.actor, field_name="Sensitive deletion actor")
    _safe_identifier(authorization.purpose, field_name="Sensitive deletion purpose")
    _safe_identifier(
        authorization.authorization_id,
        field_name="Sensitive deletion authorization_id",
    )

    denial: str | None = None
    if not authorization.allowed:
        denial = "Sensitive-memory deletion operation denied by authorization."
    elif authorization.memory_id != memory_id:
        denial = "Sensitive deletion authorization is not bound to the target memory."
    elif authorization.deletion_scope != SENSITIVE_MEMORY_DELETION_SCOPE:
        denial = "Sensitive deletion authorization has an unsupported scope."
    elif not authorization.directly_requested:
        denial = (
            "HIGHLY_SENSITIVE deletion changes require a direct user request; "
            "autonomous mutation is disabled."
        )

    if denial is not None:
        _record_sensitive_deletion_decision(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation=operation,
            decision="denied",
            created_at=canonical,
        )
        raise SensitiveMemoryDeletionAuthorizationError(denial)
    return canonical


def _require_final_authorization(
    connection: sqlite3.Connection,
    *,
    authorization: SensitiveMemoryDeletionAuthorization,
    memory_id: str,
    deleted_at: str,
) -> str:
    canonical_deleted_at = _require_common_authorization(
        connection,
        authorization=authorization,
        memory_id=memory_id,
        operation="delete",
        occurred_at=deleted_at,
    )

    denial: str | None = None
    if not authorization.strongly_confirmed:
        denial = "Irreversible sensitive-memory deletion requires strong confirmation."
    else:
        _issued_at, issued_time = _parse_utc(
            authorization.issued_at,
            field_name="authorization.issued_at",
        )
        _expires_at, expires_time = _parse_utc(
            authorization.expires_at,
            field_name="authorization.expires_at",
        )
        _deleted_at, deleted_time = _parse_utc(
            canonical_deleted_at,
            field_name="deleted_at",
        )
        lifetime = (expires_time - issued_time).total_seconds()
        if lifetime <= 0:
            denial = "Sensitive deletion authorization must expire after issue."
        elif lifetime > MAX_STRONG_CONFIRMATION_SECONDS:
            denial = (
                "Strong sensitive deletion authorization cannot remain valid "
                f"for more than {MAX_STRONG_CONFIRMATION_SECONDS} seconds."
            )
        elif deleted_time < issued_time or deleted_time > expires_time:
            denial = "Strong sensitive deletion authorization is not currently valid."

    if denial is not None:
        _record_sensitive_deletion_decision(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="delete",
            decision="denied",
            created_at=canonical_deleted_at,
        )
        raise SensitiveMemoryDeletionAuthorizationError(denial)
    return canonical_deleted_at


def _require_sensitive_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryRecord:
    try:
        memory = load_memory(connection, memory_id=memory_id)
    except MemoryNotFoundError as exc:
        raise SensitiveMemoryDeletionValidationError(
            f"Sensitive memory not found: {memory_id}"
        ) from exc

    if memory.data_classification != "HIGHLY_SENSITIVE":
        raise SensitiveMemoryDeletionValidationError(
            "The protected deletion path accepts only HIGHLY_SENSITIVE memory."
        )
    row = connection.execute(
        "SELECT content FROM memories WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if row is None or str(row["content"]) != SENSITIVE_CONTENT_SENTINEL:
        raise SensitiveMemoryDeletionValidationError(
            "Sensitive memory does not contain the encrypted-content sentinel."
        )
    try:
        load_sensitive_payload_record(connection, memory_id=memory_id)
    except SensitiveMemoryStorageError as exc:
        raise SensitiveMemoryDeletionValidationError(
            "Encrypted sensitive payload is missing or invalid."
        ) from exc
    return memory


def _parse_lifecycle_event(
    row: sqlite3.Row,
) -> _SensitiveDeletionLifecycleEvent | None:
    try:
        details = json.loads(str(row["details_json"]))
    except (TypeError, ValueError):
        return None
    if not isinstance(details, dict):
        return None
    if details.get("memory_kind") != "highly_sensitive":
        if str(row["event_type"]) == "deletion_requested":
            raise SensitiveMemoryDeletionStateError(
                "Sensitive deletion request lacks protected-memory proof."
            )
        return None
    operation = str(details.get("operation", ""))
    if operation not in {"deletion_requested", "deletion_cancelled"}:
        if str(row["event_type"]) == "deletion_requested":
            raise SensitiveMemoryDeletionStateError(
                "Sensitive deletion request has invalid structure."
            )
        return None
    try:
        validate_deletion_lifecycle_details(
            details,
            operation=operation,
            target_memory_id=str(details.get("target_memory_id", "")),
            deletion_scope=SENSITIVE_MEMORY_DELETION_SCOPE,
            memory_kind="highly_sensitive",
        )
    except MemoryDeletionIntegrityError as exc:
        raise SensitiveMemoryDeletionStateError(str(exc)) from exc
    return _SensitiveDeletionLifecycleEvent(
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


def _latest_lifecycle_event(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> _SensitiveDeletionLifecycleEvent | None:
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
        event = _parse_lifecycle_event(row)
        if event is not None:
            return event
    return None


def _open_request(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> _SensitiveDeletionLifecycleEvent:
    event = _latest_lifecycle_event(connection, memory_id=memory_id)
    if event is None or event.operation != "deletion_requested":
        raise SensitiveMemoryDeletionStateError(
            "Sensitive memory has no open deletion request."
        )
    if event.event_type != "deletion_requested":
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request has an invalid event type."
        )
    if event.deletion_scope != SENSITIVE_MEMORY_DELETION_SCOPE:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request has an unsupported scope."
        )
    if event.target_memory_id != memory_id:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request is not bound to the memory."
        )
    if not _SAFE_AUDIT_IDENTIFIER.fullmatch(event.authorization_id):
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request has an invalid authorization_id."
        )
    return event


def _request_result(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryDeletionRequestResult:
    memory = _require_sensitive_memory(connection, memory_id=memory_id)
    event = _open_request(connection, memory_id=memory_id)
    if memory.deletion_state != "pending_deletion":
        raise SensitiveMemoryDeletionStateError(
            "Open sensitive deletion request does not match memory state."
        )
    return MemoryDeletionRequestResult(
        memory=memory,
        request_event_id=event.event_id,
        authorization_id=event.authorization_id,
        deletion_scope=event.deletion_scope,
        requested_by=event.actor,
        requested_at=event.created_at,
    )


def request_sensitive_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: SensitiveMemoryDeletionRequestAuthorization,
    requested_at: str,
) -> MemoryDeletionRequestResult:
    """Move one encrypted sensitive memory into reversible pending deletion."""
    requested_at = _require_common_authorization(
        connection,
        authorization=authorization,
        memory_id=memory_id,
        operation="request_deletion",
        occurred_at=requested_at,
    )

    try:
        memory = _require_sensitive_memory(connection, memory_id=memory_id)
    except SensitiveMemoryDeletionValidationError as exc:
        try:
            tombstone = load_memory_tombstone(connection, memory_id=memory_id)
        except MemoryTombstoneNotFoundError:
            raise exc
        if tombstone.deletion_scope == SENSITIVE_MEMORY_DELETION_SCOPE:
            raise SensitiveMemoryDeletionStateError(
                "Deleted sensitive memories cannot receive a new deletion request."
            ) from exc
        raise exc

    if memory.deletion_state == "pending_deletion":
        return _request_result(connection, memory_id=memory_id)
    if memory.deletion_state != "active":
        raise SensitiveMemoryDeletionStateError(
            "Only active sensitive memories can request deletion."
        )

    event_id = str(uuid.uuid4())
    details_json = json.dumps(
        {
            "authorization_id": authorization.authorization_id,
            "deletion_scope": authorization.deletion_scope,
            "memory_kind": "highly_sensitive",
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
            SET deletion_state = 'pending_deletion', updated_at = ?
            WHERE memory_id = ? AND deletion_state = 'active'
            """,
            (requested_at, memory_id),
        )
        if cursor.rowcount != 1:
            raise SensitiveMemoryDeletionStateError(
                "Sensitive memory state changed before request commit."
            )
        connection.execute(
            """
            INSERT INTO memory_events (
                event_id, memory_id, event_type, actor, details_json, created_at
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
        _record_sensitive_deletion_decision(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="request_deletion",
            decision="allowed",
            created_at=requested_at,
        )

    return _request_result(connection, memory_id=memory_id)


def cancel_sensitive_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: SensitiveMemoryDeletionCancellationAuthorization,
    cancelled_at: str,
) -> MemoryDeletionCancellationResult:
    """Restore one pending sensitive memory to active state."""
    cancelled_at = _require_common_authorization(
        connection,
        authorization=authorization,
        memory_id=memory_id,
        operation="cancel_deletion",
        occurred_at=cancelled_at,
    )
    memory = _require_sensitive_memory(connection, memory_id=memory_id)
    if memory.deletion_state != "pending_deletion":
        raise SensitiveMemoryDeletionStateError(
            "Only pending sensitive memories can cancel deletion."
        )
    request = _open_request(connection, memory_id=memory_id)

    event_id = str(uuid.uuid4())
    details_json = json.dumps(
        {
            "authorization_id": authorization.authorization_id,
            "deletion_scope": authorization.deletion_scope,
            "memory_kind": "highly_sensitive",
            "new_deletion_state": "active",
            "operation": "deletion_cancelled",
            "previous_deletion_state": "pending_deletion",
            "request_event_id": request.event_id,
            "target_memory_id": memory_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE memories
            SET deletion_state = 'active', updated_at = ?
            WHERE memory_id = ? AND deletion_state = 'pending_deletion'
            """,
            (cancelled_at, memory_id),
        )
        if cursor.rowcount != 1:
            raise SensitiveMemoryDeletionStateError(
                "Sensitive memory state changed before cancellation commit."
            )
        connection.execute(
            """
            INSERT INTO memory_events (
                event_id, memory_id, event_type, actor, details_json, created_at
            )
            VALUES (?, ?, 'reclassified', ?, ?, ?)
            """,
            (
                event_id,
                memory_id,
                authorization.actor,
                details_json,
                cancelled_at,
            ),
        )
        _record_sensitive_deletion_decision(
            connection,
            memory_id=memory_id,
            authorization=authorization,
            operation="cancel_deletion",
            decision="allowed",
            created_at=cancelled_at,
        )

    return MemoryDeletionCancellationResult(
        memory=load_memory(connection, memory_id=memory_id),
        cancellation_event_id=event_id,
        request_event_id=request.event_id,
        authorization_id=authorization.authorization_id,
        deletion_scope=authorization.deletion_scope,
        cancelled_by=authorization.actor,
        cancelled_at=cancelled_at,
    )


def load_sensitive_memory_deletion(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> MemoryDeletionResult:
    """Load and integrity-check a completed protected sensitive deletion."""
    try:
        result = load_memory_deletion(connection, memory_id=memory_id)
    except MemoryTombstoneNotFoundError:
        raise
    except MemoryDeletionStateError as exc:
        raise SensitiveMemoryDeletionStateError(str(exc)) from exc

    if result.tombstone.deletion_scope != SENSITIVE_MEMORY_DELETION_SCOPE:
        raise SensitiveMemoryDeletionStateError(
            "Tombstone does not represent a protected sensitive deletion."
        )
    event = connection.execute(
        "SELECT actor, details_json, created_at FROM memory_events WHERE event_id = ?",
        (result.tombstone.event_id,),
    ).fetchone()
    if event is None:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion event is missing."
        )
    try:
        details = json.loads(str(event["details_json"]))
    except (TypeError, ValueError) as exc:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion event contains invalid JSON."
        ) from exc
    if (
        not isinstance(details, dict)
        or details.get("memory_kind") != "highly_sensitive"
        or details.get("encrypted_payload_deleted") is not True
        or details.get("operation") != "memory_deleted"
    ):
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion event lacks protected-payload proof."
        )

    request_row = connection.execute(
        "SELECT details_json FROM memory_events WHERE event_id = ?",
        (result.request_event_id,),
    ).fetchone()
    if request_row is None:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request event is missing."
        )
    try:
        request_details = json.loads(str(request_row["details_json"]))
    except (TypeError, ValueError) as exc:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request contains invalid JSON."
        ) from exc
    if (
        not isinstance(request_details, dict)
        or request_details.get("memory_kind") != "highly_sensitive"
        or request_details.get("operation") != "deletion_requested"
    ):
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion request lacks protected-memory proof."
        )

    if connection.execute(
        "SELECT 1 FROM memories WHERE memory_id = ?",
        (memory_id,),
    ).fetchone() is not None:
        raise SensitiveMemoryDeletionStateError(
            "Deleted sensitive memory still exists in the authoritative table."
        )
    if connection.execute(
        "SELECT 1 FROM memory_sensitive_payloads WHERE memory_id = ?",
        (memory_id,),
    ).fetchone() is not None:
        raise SensitiveMemoryDeletionStateError(
            "Encrypted sensitive payload remained after deletion."
        )

    audit_rows = connection.execute(
        """
        SELECT actor, purpose, authorization_id, operation, decision, created_at
        FROM sensitive_memory_access_events
        WHERE authorization_id = ?
          AND operation = 'delete'
          AND decision = 'allowed'
          AND created_at = ?
        """,
        (result.authorization_id, result.deleted_at),
    ).fetchall()
    if len(audit_rows) != 1 or str(audit_rows[0]["actor"]) != result.deleted_by:
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion audit decision is missing or inconsistent."
        )
    if audit_rows[0]["purpose"] is None or not _SAFE_AUDIT_IDENTIFIER.fullmatch(
        str(audit_rows[0]["purpose"])
    ):
        raise SensitiveMemoryDeletionStateError(
            "Sensitive deletion audit purpose is invalid."
        )
    return result


def delete_sensitive_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    authorization: SensitiveMemoryDeletionAuthorization,
    deleted_at: str,
) -> MemoryDeletionResult:
    """Atomically remove encrypted sensitive memory and retain a tombstone."""
    deleted_at = _require_final_authorization(
        connection,
        authorization=authorization,
        memory_id=memory_id,
        deleted_at=deleted_at,
    )

    try:
        existing = load_sensitive_memory_deletion(
            connection,
            memory_id=memory_id,
        )
    except MemoryTombstoneNotFoundError:
        existing = None
    if existing is not None:
        return existing

    memory = _require_sensitive_memory(connection, memory_id=memory_id)
    if memory.deletion_state != "pending_deletion":
        raise SensitiveMemoryDeletionStateError(
            "Sensitive memory requires an open pending-deletion request."
        )
    request = _open_request(connection, memory_id=memory_id)
    payload = load_sensitive_payload_record(connection, memory_id=memory_id)
    deleted_event_id = str(uuid.uuid4())

    try:
        with transaction(connection):
            current = _require_sensitive_memory(connection, memory_id=memory_id)
            if current.deletion_state != "pending_deletion":
                raise SensitiveMemoryDeletionStateError(
                    "Sensitive memory state changed before deletion commit."
                )
            current_request = _open_request(connection, memory_id=memory_id)
            if current_request.event_id != request.event_id:
                raise SensitiveMemoryDeletionStateError(
                    "Open sensitive deletion request changed before commit."
                )
            current_payload = load_sensitive_payload_record(
                connection,
                memory_id=memory_id,
            )
            if (
                current_payload.key_id != payload.key_id
                or current_payload.ciphertext != payload.ciphertext
                or current_payload.nonce != payload.nonce
            ):
                raise SensitiveMemoryDeletionStateError(
                    "Encrypted payload changed before deletion could commit."
                )

            try:
                require_no_unmanaged_active_derivative_tables(connection)
                candidate_lineage = collect_promoted_candidate_lineage(
                    connection,
                    memory_id=memory_id,
                    content_sha256=current.content_sha256,
                )
            except MemoryDeletionIntegrityError as exc:
                raise SensitiveMemoryDeletionValidationError(str(exc)) from exc

            dependent_counts = _dependent_counts(
                connection,
                memory_id=memory_id,
            )
            details_json = json.dumps(
                {
                    "authorization_id": authorization.authorization_id,
                    "content_sha256": current.content_sha256,
                    "deleted_memory_id": memory_id,
                    "deletion_scope": authorization.deletion_scope,
                    "dependent_counts": dependent_counts,
                    "encrypted_payload_deleted": True,
                    "memory_kind": "highly_sensitive",
                    "operation": "memory_deleted",
                    "promoted_candidate_lineage": (
                        promoted_candidate_lineage_audit(candidate_lineage)
                    ),
                    "request_event_id": current_request.event_id,
                    "strong_confirmation": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            connection.execute(
                """
                INSERT INTO memory_events (
                    event_id, memory_id, event_type, actor, details_json, created_at
                )
                VALUES (?, ?, 'deleted', ?, ?, ?)
                """,
                (
                    deleted_event_id,
                    memory_id,
                    authorization.actor,
                    details_json,
                    deleted_at,
                ),
            )
            _record_sensitive_deletion_decision(
                connection,
                memory_id=memory_id,
                authorization=authorization,
                operation="delete",
                decision="allowed",
                created_at=deleted_at,
            )

            try:
                purge_promoted_candidate_lineage(
                    connection,
                    lineage=candidate_lineage,
                )
            except MemoryDeletionIntegrityError as exc:
                raise SensitiveMemoryDeletionValidationError(str(exc)) from exc

            cursor = connection.execute(
                """
                DELETE FROM memories
                WHERE memory_id = ? AND deletion_state = 'pending_deletion'
                """,
                (memory_id,),
            )
            if cursor.rowcount != 1:
                raise SensitiveMemoryDeletionStateError(
                    "Sensitive memory state changed before deletion commit."
                )
            _verify_authoritative_dependents_removed(
                connection,
                memory_id=memory_id,
            )
            try:
                verify_promoted_candidate_lineage_removed(
                    connection,
                    lineage=candidate_lineage,
                )
            except MemoryDeletionIntegrityError as exc:
                raise SensitiveMemoryDeletionValidationError(str(exc)) from exc
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
                    _tombstone_id(memory_id),
                    memory_id,
                    current.content_sha256,
                    deleted_at,
                    authorization.deletion_scope,
                    deleted_event_id,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise SensitiveMemoryDeletionValidationError(
            f"Sensitive memory deletion failed database validation: {exc}"
        ) from exc

    return load_sensitive_memory_deletion(connection, memory_id=memory_id)
