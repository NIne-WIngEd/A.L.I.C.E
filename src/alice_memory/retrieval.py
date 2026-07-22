"""Authorization-aware metadata-safe Phase 2 memory retrieval."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .lexical_index import (
    memory_lexical_index_path,
    search_memory_lexical_candidates,
    verify_memory_lexical_index,
)
from .retrieval_models import (
    MemoryRetrievalAuthorization,
    MemoryRetrievalAuthorizationError,
    MemoryRetrievalValidationError,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchResult,
)
from .service import MemoryRecord, load_memory


_CLASSIFICATION_RANK = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "PRIVATE": 2,
}


def _require_retrieval_authorization(
    authorization: MemoryRetrievalAuthorization,
) -> None:
    if not authorization.allowed:
        raise MemoryRetrievalAuthorizationError(
            "Memory retrieval denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryRetrievalAuthorizationError(
            "Authorized retrieval requires a non-empty actor."
        )
    if not authorization.purpose.strip():
        raise MemoryRetrievalAuthorizationError(
            "Authorized retrieval requires a non-empty purpose."
        )
    if authorization.max_classification not in _CLASSIFICATION_RANK:
        raise MemoryRetrievalAuthorizationError(
            "P2.5 retrieval supports a maximum classification of PRIVATE. "
            "HIGHLY_SENSITIVE retrieval remains disabled until P2.6."
        )


def _validate_request(
    request: MemorySearchRequest,
) -> None:
    if not request.query.strip():
        raise MemoryRetrievalValidationError(
            "Memory search query cannot be empty."
        )
    if request.limit < 1 or request.limit > 100:
        raise MemoryRetrievalValidationError(
            "Memory search limit must be between 1 and 100."
        )


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
        raise MemoryRetrievalValidationError(
            f"{field_name} must be an ISO-8601 timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise MemoryRetrievalValidationError(
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


def _corrected_targets(
    connection: sqlite3.Connection,
) -> set[str]:
    return {
        str(row["to_memory_id"])
        for row in connection.execute(
            """
            SELECT to_memory_id
            FROM memory_relations
            WHERE relation_type = 'corrects'
            """
        ).fetchall()
    }


def _classification_allowed(
    record: MemoryRecord,
    authorization: MemoryRetrievalAuthorization,
) -> bool:
    rank = _CLASSIFICATION_RANK.get(
        record.data_classification
    )
    if rank is None:
        return False
    return rank <= _CLASSIFICATION_RANK[
        authorization.max_classification
    ]


def _temporally_eligible(
    record: MemoryRecord,
    *,
    request: MemorySearchRequest,
    corrected_targets: set[str],
) -> bool:
    if record.memory_id in corrected_targets:
        return False

    if (
        not request.include_archived
        and record.retention_state == "archived"
    ):
        return False

    if request.memory_key is not None:
        if record.memory_key != request.memory_key:
            return False

    if request.category is not None:
        if record.category != request.category:
            return False

    if request.at is None:
        if (
            not request.include_historical
            and record.validity_state == "historical"
        ):
            return False
        return True

    at = _parse_timestamp(
        request.at,
        field_name="request.at",
    )
    valid_from = _optional_timestamp(
        record.valid_from,
        field_name="memory.valid_from",
    )
    valid_to = _optional_timestamp(
        record.valid_to,
        field_name="memory.valid_to",
    )

    if valid_from is not None and at < valid_from:
        return False
    if valid_to is not None and at >= valid_to:
        return False
    return True


def _conflict_ids(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT
            CASE
                WHEN from_memory_id = ? THEN to_memory_id
                ELSE from_memory_id
            END AS other_memory_id
        FROM memory_relations
        WHERE relation_type = 'conflicts_with'
          AND (
              from_memory_id = ?
              OR to_memory_id = ?
          )
        ORDER BY other_memory_id
        """,
        (
            memory_id,
            memory_id,
            memory_id,
        ),
    ).fetchall()

    return tuple(
        str(row["other_memory_id"])
        for row in rows
    )


def _result_from_record(
    record: MemoryRecord,
    *,
    score: float,
    conflict_memory_ids: tuple[str, ...],
) -> MemorySearchResult:
    return MemorySearchResult(
        memory_id=record.memory_id,
        score=score,
        memory_key=record.memory_key,
        category=record.category,
        knowledge_status=record.knowledge_status,
        confidence=record.confidence,
        data_classification=record.data_classification,
        valid_from=record.valid_from,
        valid_to=record.valid_to,
        recorded_at=record.recorded_at,
        validity_state=record.validity_state,
        retention_state=record.retention_state,
        conflict_memory_ids=conflict_memory_ids,
    )


def search_memories(
    connection: sqlite3.Connection,
    vault_root: str | Path,
    *,
    request: MemorySearchRequest,
    authorization: MemoryRetrievalAuthorization,
    repository_root: str | Path | None = None,
) -> MemorySearchResponse:
    """Search memory lexically and return metadata-safe authorized results.

    The derived index is verified against the authoritative Memory Core before
    use. A stale index fails closed instead of serving outdated or deleted data.
    """
    _require_retrieval_authorization(
        authorization
    )
    _validate_request(
        request
    )

    index_path = memory_lexical_index_path(
        vault_root,
        repository_root=repository_root,
    )
    manifest = verify_memory_lexical_index(
        connection,
        index_path,
    )

    candidate_limit = min(
        max(
            request.limit * 8,
            request.limit,
        ),
        800,
    )
    candidates = search_memory_lexical_candidates(
        index_path,
        query=request.query,
        limit=candidate_limit,
    )

    corrected = _corrected_targets(
        connection
    )
    ranked: list[tuple[MemoryRecord, float]] = []
    seen: set[str] = set()

    for memory_id, score in candidates:
        record = load_memory(
            connection,
            memory_id=memory_id,
        )
        if not _classification_allowed(
            record,
            authorization,
        ):
            continue
        if not _temporally_eligible(
            record,
            request=request,
            corrected_targets=corrected,
        ):
            continue

        ranked.append(
            (
                record,
                score,
            )
        )
        seen.add(
            memory_id
        )

        if len(ranked) >= request.limit:
            break

    if request.expand_conflicts:
        expanded = list(ranked)
        for record, score in tuple(ranked):
            for other_id in _conflict_ids(
                connection,
                memory_id=record.memory_id,
            ):
                if other_id in seen:
                    continue

                other = load_memory(
                    connection,
                    memory_id=other_id,
                )
                if not _classification_allowed(
                    other,
                    authorization,
                ):
                    continue
                if not _temporally_eligible(
                    other,
                    request=request,
                    corrected_targets=corrected,
                ):
                    continue

                expanded.append(
                    (
                        other,
                        score,
                    )
                )
                seen.add(
                    other_id
                )
        ranked = expanded

    results = tuple(
        _result_from_record(
            record,
            score=score,
            conflict_memory_ids=tuple(
                other_id
                for other_id in _conflict_ids(
                    connection,
                    memory_id=record.memory_id,
                )
                if other_id in seen
            ),
        )
        for record, score in ranked
    )

    return MemorySearchResponse(
        query=request.query,
        results=results,
        index_id=manifest.index_id,
        authoritative_digest=manifest.authoritative_digest,
    )
