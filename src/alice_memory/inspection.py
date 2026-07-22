"""Inspectable, authorization-aware views of Phase 2 memory state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .provenance import AttachedMemorySource, list_memory_sources
from .service import (
    MemoryContentAccessAuthorization,
    MemoryContentAuthorizationError,
    load_memory,
    load_memory_content,
)

MemoryInspectionAuthorizationError = MemoryContentAuthorizationError


@dataclass(frozen=True)
class MemoryRelationView:
    relation_id: str
    from_memory_id: str
    to_memory_id: str
    relation_type: str
    created_at: str


@dataclass(frozen=True)
class MemoryEventView:
    event_id: str
    event_type: str
    actor: str
    details: dict[str, object] | None
    created_at: str


@dataclass(frozen=True)
class MemoryInspection:
    memory_id: str
    content: str | None
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
    sources: tuple[AttachedMemorySource, ...]
    relations: tuple[MemoryRelationView, ...]
    events: tuple[MemoryEventView, ...]


@dataclass(frozen=True)
class MemorySummary:
    memory_id: str
    content_sha256: str
    memory_key: str | None
    category: str
    knowledge_status: str
    confidence: float
    data_classification: str
    validity_state: str
    retention_state: str
    deletion_state: str
    recorded_at: str
    updated_at: str


def _relations_for_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[MemoryRelationView, ...]:
    rows = connection.execute(
        """
        SELECT
            relation_id,
            from_memory_id,
            to_memory_id,
            relation_type,
            created_at
        FROM memory_relations
        WHERE from_memory_id = ?
           OR to_memory_id = ?
        ORDER BY created_at, relation_id
        """,
        (
            memory_id,
            memory_id,
        ),
    ).fetchall()

    return tuple(
        MemoryRelationView(
            relation_id=str(row["relation_id"]),
            from_memory_id=str(row["from_memory_id"]),
            to_memory_id=str(row["to_memory_id"]),
            relation_type=str(row["relation_type"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    )


def _events_for_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[MemoryEventView, ...]:
    rows = connection.execute(
        """
        SELECT
            event_id,
            event_type,
            actor,
            details_json,
            created_at
        FROM memory_events
        WHERE memory_id = ?
        ORDER BY created_at, event_id
        """,
        (memory_id,),
    ).fetchall()

    events: list[MemoryEventView] = []
    for row in rows:
        details_json = row["details_json"]
        details = (
            None
            if details_json is None
            else json.loads(str(details_json))
        )
        events.append(
            MemoryEventView(
                event_id=str(row["event_id"]),
                event_type=str(row["event_type"]),
                actor=str(row["actor"]),
                details=details,
                created_at=str(row["created_at"]),
            )
        )

    return tuple(events)


def inspect_memory(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    include_content: bool = False,
    content_authorization: MemoryContentAccessAuthorization | None = None,
) -> MemoryInspection:
    """Inspect one memory; plaintext content is excluded by default."""
    record = load_memory(
        connection,
        memory_id=memory_id,
    )

    if record.data_classification == "HIGHLY_SENSITIVE":
        raise MemoryInspectionAuthorizationError(
            "HIGHLY_SENSITIVE inspection requires the dedicated "
            "purpose-bound sensitive-memory access path."
        )

    content = None
    if include_content:
        content = load_memory_content(
            connection,
            memory_id=memory_id,
            authorization=content_authorization,
        )

    return MemoryInspection(
        memory_id=record.memory_id,
        content=content,
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
        sources=list_memory_sources(
            connection,
            memory_id=memory_id,
        ),
        relations=_relations_for_memory(
            connection,
            memory_id=memory_id,
        ),
        events=_events_for_memory(
            connection,
            memory_id=memory_id,
        ),
    )


def list_memory_summaries(
    connection: sqlite3.Connection,
    *,
    include_archived: bool = False,
    category: str | None = None,
) -> tuple[MemorySummary, ...]:
    """List metadata-only memory summaries, excluding deleted states."""
    clauses = [
        "deletion_state = 'active'",
        "data_classification <> 'HIGHLY_SENSITIVE'",
    ]
    parameters: list[object] = []

    if not include_archived:
        clauses.append("retention_state <> 'archived'")

    if category is not None:
        clauses.append("category = ?")
        parameters.append(category)

    where_clause = " AND ".join(clauses)

    rows = connection.execute(
        f"""
        SELECT
            memory_id,
            content_sha256,
            memory_key,
            category,
            knowledge_status,
            confidence,
            data_classification,
            validity_state,
            retention_state,
            deletion_state,
            recorded_at,
            updated_at
        FROM memories
        WHERE {where_clause}
        ORDER BY updated_at DESC, memory_id
        """,
        parameters,
    ).fetchall()

    return tuple(
        MemorySummary(
            memory_id=str(row["memory_id"]),
            content_sha256=str(row["content_sha256"]),
            memory_key=(
                None
                if row["memory_key"] is None
                else str(row["memory_key"])
            ),
            category=str(row["category"]),
            knowledge_status=str(row["knowledge_status"]),
            confidence=float(row["confidence"]),
            data_classification=str(row["data_classification"]),
            validity_state=str(row["validity_state"]),
            retention_state=str(row["retention_state"]),
            deletion_state=str(row["deletion_state"]),
            recorded_at=str(row["recorded_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    )
