"""Deterministic temporal, correction, conflict, and supersession logic.

P2.4 keeps authoritative history append-preserving:
- corrections create a replacement and mark the corrected record non-current;
- supersessions create a successor and retain the prior record as historical;
- conflicts retain both records and mark both disputed;
- valid-time resolution uses half-open intervals [valid_from, valid_to).

All transition writes are atomic and permission-gated. Public results are
metadata-only; plaintext stays behind the P2.3 authorization boundary.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from .service import (
    MemoryAlreadyExistsError,
    MemoryCreateRequest,
    MemoryRecord,
    MemoryValidationError,
    MemoryWriteAuthorization,
    _insert_memory_in_transaction,
    _normalize_create_request,
    _require_write_authorization,
    load_memory,
)
from .store import transaction


class MemoryTransitionError(RuntimeError):
    """Base error for deterministic memory-state transitions."""


class InvalidMemoryTransitionError(MemoryTransitionError):
    """Raised when a requested transition violates lifecycle semantics."""


@dataclass(frozen=True)
class MemoryRelation:
    relation_id: str
    from_memory_id: str
    to_memory_id: str
    relation_type: str
    created_at: str


@dataclass(frozen=True)
class MemoryTransitionResult:
    previous: MemoryRecord
    replacement: MemoryRecord
    relation: MemoryRelation


@dataclass(frozen=True)
class ConflictResult:
    first: MemoryRecord
    second: MemoryRecord
    relation: MemoryRelation


@dataclass(frozen=True)
class TemporalResolution:
    memory_key: str
    at: str
    memories: tuple[MemoryRecord, ...]
    conflict_pairs: tuple[tuple[str, str], ...]

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflict_pairs)


_RELATION_NAMESPACE = uuid.UUID(
    "48fb5dbe-43b2-4b84-80dd-bf288e229f58"
)

_CLASSIFICATION_RANK = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "PRIVATE": 2,
    "HIGHLY_SENSITIVE": 3,
}


def _parse_timestamp(
    value: str,
    *,
    field_name: str,
) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise InvalidMemoryTransitionError(
            f"{field_name} must be an ISO-8601 timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise InvalidMemoryTransitionError(
            f"{field_name} must include a timezone offset."
        )

    return parsed.astimezone(timezone.utc)


def _optional_timestamp(
    value: str | None,
    *,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    return _parse_timestamp(
        value,
        field_name=field_name,
    )


def _canonical_timestamp(
    value: str,
    *,
    field_name: str,
) -> str:
    return (
        _parse_timestamp(
            value,
            field_name=field_name,
        )
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relation_id(
    *,
    from_memory_id: str,
    to_memory_id: str,
    relation_type: str,
) -> str:
    canonical = "|".join(
        (
            from_memory_id,
            to_memory_id,
            relation_type,
        )
    )
    return str(
        uuid.uuid5(
            _RELATION_NAMESPACE,
            canonical,
        )
    )


def _load_relation(
    connection: sqlite3.Connection,
    *,
    relation_id: str,
) -> MemoryRelation:
    row = connection.execute(
        """
        SELECT
            relation_id,
            from_memory_id,
            to_memory_id,
            relation_type,
            created_at
        FROM memory_relations
        WHERE relation_id = ?
        """,
        (relation_id,),
    ).fetchone()

    if row is None:
        raise MemoryTransitionError(
            f"Memory relation was not persisted: {relation_id}"
        )

    return MemoryRelation(
        relation_id=str(row["relation_id"]),
        from_memory_id=str(row["from_memory_id"]),
        to_memory_id=str(row["to_memory_id"]),
        relation_type=str(row["relation_type"]),
        created_at=str(row["created_at"]),
    )


def _require_transitionable(memory: MemoryRecord) -> None:
    if memory.deletion_state != "active":
        raise InvalidMemoryTransitionError(
            "Deleted or deletion-pending memories cannot transition."
        )
    if memory.retention_state == "archived":
        raise InvalidMemoryTransitionError(
            "Archived memories cannot transition."
        )
    if memory.validity_state not in {"current", "disputed"}:
        raise InvalidMemoryTransitionError(
            "Only current or disputed memories can transition."
        )


def _normalize_replacement_request(
    previous: MemoryRecord,
    request: MemoryCreateRequest,
) -> MemoryCreateRequest:
    memory_key = request.memory_key

    if previous.memory_key is not None:
        if memory_key is None:
            memory_key = previous.memory_key
        elif memory_key != previous.memory_key:
            raise InvalidMemoryTransitionError(
                "Replacement memory_key must match the previous memory_key."
            )

    previous_rank = _CLASSIFICATION_RANK[
        previous.data_classification
    ]
    replacement_rank = _CLASSIFICATION_RANK[
        request.data_classification
    ]
    if replacement_rank < previous_rank:
        raise InvalidMemoryTransitionError(
            "Correction or supersession cannot downgrade data classification."
        )

    normalized = replace(
        request,
        memory_key=memory_key,
        validity_state="current",
    )

    try:
        normalized = _normalize_create_request(normalized)
    except MemoryValidationError as exc:
        raise InvalidMemoryTransitionError(
            f"Invalid correction or supersession replacement: {exc}"
        ) from exc

    if normalized.retention_state == "archived":
        raise InvalidMemoryTransitionError(
            "A correction or supersession replacement cannot start archived."
        )

    return normalized


def _insert_transition_event(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    event_type: str,
    actor: str,
    created_at: str,
    details: dict[str, object],
) -> None:
    """Write a sanitized transition event inside the caller's transaction."""
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
            str(uuid.uuid4()),
            memory_id,
            event_type,
            actor,
            json.dumps(
                details,
                sort_keys=True,
                separators=(",", ":"),
            ),
            created_at,
        ),
    )


def correct_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    replacement: MemoryCreateRequest,
    authorization: MemoryWriteAuthorization,
    corrected_at: str,
) -> MemoryTransitionResult:
    """Correct an inaccurate memory without erasing the corrected record."""
    _require_write_authorization(authorization)
    corrected_at = _canonical_timestamp(
        corrected_at,
        field_name="corrected_at",
    )

    previous = load_memory(
        connection,
        memory_id=memory_id,
    )
    _require_transitionable(previous)

    normalized = _normalize_replacement_request(
        previous,
        replacement,
    )

    with transaction(connection):
        replacement_id = _insert_memory_in_transaction(
            connection,
            request=normalized,
            actor=authorization.actor,
            created_at=corrected_at,
        )

        connection.execute(
            """
            UPDATE memories
            SET knowledge_status = ?,
                validity_state = ?,
                updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "superseded",
                "historical",
                corrected_at,
                memory_id,
            ),
        )

        relation_id = _relation_id(
            from_memory_id=replacement_id,
            to_memory_id=memory_id,
            relation_type="corrects",
        )
        connection.execute(
            """
            INSERT INTO memory_relations (
                relation_id,
                from_memory_id,
                to_memory_id,
                relation_type,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                relation_id,
                replacement_id,
                memory_id,
                "corrects",
                corrected_at,
            ),
        )

        _insert_transition_event(
            connection,
            memory_id=memory_id,
            event_type="corrected",
            actor=authorization.actor,
            created_at=corrected_at,
            details={
                "replacement_memory_id": replacement_id,
                "relation_id": relation_id,
            },
        )

    return MemoryTransitionResult(
        previous=load_memory(
            connection,
            memory_id=memory_id,
        ),
        replacement=load_memory(
            connection,
            memory_id=replacement_id,
        ),
        relation=_load_relation(
            connection,
            relation_id=relation_id,
        ),
    )


def supersede_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    replacement: MemoryCreateRequest,
    authorization: MemoryWriteAuthorization,
    superseded_at: str,
) -> MemoryTransitionResult:
    """Replace a once-valid memory with a new valid-time successor."""
    _require_write_authorization(authorization)
    superseded_at = _canonical_timestamp(
        superseded_at,
        field_name="superseded_at",
    )

    previous = load_memory(
        connection,
        memory_id=memory_id,
    )
    _require_transitionable(previous)

    normalized = _normalize_replacement_request(
        previous,
        replacement,
    )
    replacement_start = _optional_timestamp(
        normalized.valid_from,
        field_name="replacement.valid_from",
    )
    if replacement_start is None:
        raise InvalidMemoryTransitionError(
            "Supersession requires replacement.valid_from so the previous "
            "validity interval can be closed deterministically."
        )

    previous_start = _optional_timestamp(
        previous.valid_from,
        field_name="previous.valid_from",
    )
    if (
        previous_start is not None
        and replacement_start < previous_start
    ):
        raise InvalidMemoryTransitionError(
            "Replacement valid_from cannot precede the previous valid_from."
        )

    previous_end = _optional_timestamp(
        previous.valid_to,
        field_name="previous.valid_to",
    )
    closed_valid_to = normalized.valid_from
    if (
        previous_end is not None
        and previous_end < replacement_start
    ):
        closed_valid_to = previous.valid_to

    with transaction(connection):
        replacement_id = _insert_memory_in_transaction(
            connection,
            request=normalized,
            actor=authorization.actor,
            created_at=superseded_at,
        )

        connection.execute(
            """
            UPDATE memories
            SET knowledge_status = ?,
                validity_state = ?,
                valid_to = ?,
                updated_at = ?
            WHERE memory_id = ?
            """,
            (
                "historical",
                "historical",
                closed_valid_to,
                superseded_at,
                memory_id,
            ),
        )

        relation_id = _relation_id(
            from_memory_id=replacement_id,
            to_memory_id=memory_id,
            relation_type="supersedes",
        )
        connection.execute(
            """
            INSERT INTO memory_relations (
                relation_id,
                from_memory_id,
                to_memory_id,
                relation_type,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                relation_id,
                replacement_id,
                memory_id,
                "supersedes",
                superseded_at,
            ),
        )

        _insert_transition_event(
            connection,
            memory_id=memory_id,
            event_type="superseded",
            actor=authorization.actor,
            created_at=superseded_at,
            details={
                "replacement_memory_id": replacement_id,
                "relation_id": relation_id,
                "closed_valid_to": closed_valid_to,
            },
        )

    return MemoryTransitionResult(
        previous=load_memory(
            connection,
            memory_id=memory_id,
        ),
        replacement=load_memory(
            connection,
            memory_id=replacement_id,
        ),
        relation=_load_relation(
            connection,
            relation_id=relation_id,
        ),
    )


def mark_memory_conflict(
    connection: sqlite3.Connection,
    *,
    first_memory_id: str,
    second_memory_id: str,
    authorization: MemoryWriteAuthorization,
    disputed_at: str,
) -> ConflictResult:
    """Preserve two contradictory memories and mark both as disputed."""
    _require_write_authorization(authorization)
    disputed_at = _canonical_timestamp(
        disputed_at,
        field_name="disputed_at",
    )

    if first_memory_id == second_memory_id:
        raise InvalidMemoryTransitionError(
            "A memory cannot conflict with itself."
        )

    first = load_memory(
        connection,
        memory_id=first_memory_id,
    )
    second = load_memory(
        connection,
        memory_id=second_memory_id,
    )
    _require_transitionable(first)
    _require_transitionable(second)

    if (
        first.memory_key is not None
        and second.memory_key is not None
        and first.memory_key != second.memory_key
    ):
        raise InvalidMemoryTransitionError(
            "Conflicting memories with explicit memory_key values must share "
            "the same key."
        )

    from_memory_id, to_memory_id = sorted(
        (
            first_memory_id,
            second_memory_id,
        )
    )
    relation_id = _relation_id(
        from_memory_id=from_memory_id,
        to_memory_id=to_memory_id,
        relation_type="conflicts_with",
    )

    existing_relation = connection.execute(
        """
        SELECT 1
        FROM memory_relations
        WHERE relation_id = ?
        """,
        (relation_id,),
    ).fetchone()
    if existing_relation is not None:
        return ConflictResult(
            first=first,
            second=second,
            relation=_load_relation(
                connection,
                relation_id=relation_id,
            ),
        )

    with transaction(connection):
        connection.execute(
            """
            UPDATE memories
            SET knowledge_status = ?,
                validity_state = ?,
                updated_at = ?
            WHERE memory_id IN (?, ?)
            """,
            (
                "disputed",
                "disputed",
                disputed_at,
                first_memory_id,
                second_memory_id,
            ),
        )

        connection.execute(
            """
            INSERT INTO memory_relations (
                relation_id,
                from_memory_id,
                to_memory_id,
                relation_type,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                relation_id,
                from_memory_id,
                to_memory_id,
                "conflicts_with",
                disputed_at,
            ),
        )

        _insert_transition_event(
            connection,
            memory_id=first_memory_id,
            event_type="conflict_marked",
            actor=authorization.actor,
            created_at=disputed_at,
            details={
                "other_memory_id": second_memory_id,
                "relation_id": relation_id,
            },
        )
        _insert_transition_event(
            connection,
            memory_id=second_memory_id,
            event_type="conflict_marked",
            actor=authorization.actor,
            created_at=disputed_at,
            details={
                "other_memory_id": first_memory_id,
                "relation_id": relation_id,
            },
        )

    return ConflictResult(
        first=load_memory(
            connection,
            memory_id=first_memory_id,
        ),
        second=load_memory(
            connection,
            memory_id=second_memory_id,
        ),
        relation=_load_relation(
            connection,
            relation_id=relation_id,
        ),
    )


def list_current_memories_for_key(
    connection: sqlite3.Connection,
    *,
    memory_key: str,
) -> tuple[MemoryRecord, ...]:
    """Return current or disputed active memories for one logical key."""
    rows = connection.execute(
        """
        SELECT memory_id
        FROM memories
        WHERE memory_key = ?
          AND deletion_state = 'active'
          AND retention_state <> 'archived'
          AND validity_state IN ('current', 'disputed')
        ORDER BY recorded_at, memory_id
        """,
        (memory_key,),
    ).fetchall()

    return tuple(
        load_memory(
            connection,
            memory_id=str(row["memory_id"]),
        )
        for row in rows
    )


def list_memory_history(
    connection: sqlite3.Connection,
    *,
    memory_key: str,
) -> tuple[MemoryRecord, ...]:
    """Return all non-deleted records for one logical key."""
    rows = connection.execute(
        """
        SELECT memory_id
        FROM memories
        WHERE memory_key = ?
          AND deletion_state = 'active'
        """,
        (memory_key,),
    ).fetchall()

    memories = [
        load_memory(
            connection,
            memory_id=str(row["memory_id"]),
        )
        for row in rows
    ]

    def sort_key(memory: MemoryRecord) -> tuple[bool, datetime, datetime, str]:
        valid_from = _optional_timestamp(
            memory.valid_from,
            field_name="memory.valid_from",
        )
        recorded_at = _parse_timestamp(
            memory.recorded_at,
            field_name="memory.recorded_at",
        )
        return (
            valid_from is None,
            valid_from or datetime.max.replace(tzinfo=timezone.utc),
            recorded_at,
            memory.memory_id,
        )

    return tuple(sorted(memories, key=sort_key))


def resolve_memory_at(
    connection: sqlite3.Connection,
    *,
    memory_key: str,
    at: str,
    include_archived: bool = False,
) -> TemporalResolution:
    """Resolve metadata-only memories valid at one point in time.

    Valid-time intervals are half-open: valid_from <= at < valid_to.
    Corrected records are never treated as valid-time truth after a correction;
    they remain available through list_memory_history().
    """
    at_time = _parse_timestamp(
        at,
        field_name="at",
    )

    rows = connection.execute(
        """
        SELECT memory_id
        FROM memories
        WHERE memory_key = ?
          AND deletion_state = 'active'
        """,
        (memory_key,),
    ).fetchall()

    corrected_targets = {
        str(row["to_memory_id"])
        for row in connection.execute(
            """
            SELECT to_memory_id
            FROM memory_relations
            WHERE relation_type = 'corrects'
            """
        ).fetchall()
    }

    candidates: list[MemoryRecord] = []
    for row in rows:
        memory = load_memory(
            connection,
            memory_id=str(row["memory_id"]),
        )

        if (
            not include_archived
            and memory.retention_state == "archived"
        ):
            continue

        if memory.memory_id in corrected_targets:
            continue

        valid_from = _optional_timestamp(
            memory.valid_from,
            field_name="memory.valid_from",
        )
        valid_to = _optional_timestamp(
            memory.valid_to,
            field_name="memory.valid_to",
        )

        if valid_from is not None and at_time < valid_from:
            continue
        if valid_to is not None and at_time >= valid_to:
            continue

        candidates.append(memory)

    candidates.sort(
        key=lambda memory: (
            _parse_timestamp(
                memory.recorded_at,
                field_name="memory.recorded_at",
            ),
            memory.memory_id,
        )
    )
    memories = tuple(candidates)
    candidate_ids = {
        memory.memory_id
        for memory in memories
    }

    relation_rows = connection.execute(
        """
        SELECT from_memory_id, to_memory_id
        FROM memory_relations
        WHERE relation_type = 'conflicts_with'
        ORDER BY from_memory_id, to_memory_id
        """
    ).fetchall()

    conflict_pairs = tuple(
        (
            str(row["from_memory_id"]),
            str(row["to_memory_id"]),
        )
        for row in relation_rows
        if (
            str(row["from_memory_id"]) in candidate_ids
            and str(row["to_memory_id"]) in candidate_ids
        )
    )

    return TemporalResolution(
        memory_key=memory_key,
        at=at,
        memories=memories,
        conflict_pairs=conflict_pairs,
    )
