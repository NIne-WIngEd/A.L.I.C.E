"""Non-authoritative memory-candidate staging for A.L.I.C.E. P2.7a.

A candidate is a proposal, not something A.L.I.C.E. knows. Candidate rows stay
outside the authoritative ``memories`` table. They therefore cannot enter the
existing lexical, semantic, or hybrid memory indexes.

P2.7a only stages ordinary candidates. HIGHLY_SENSITIVE proposals require a
future protected candidate path. SECRETS remain prohibited.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, replace

from .schema import CANDIDATE_ORIGINS, SCHEMA_VERSION
from .service import (
    MemoryContentAccessAuthorization,
    MemoryContentAuthorizationError,
    MemoryCreateRequest,
    MemoryValidationError,
    _normalize_create_request,
    _normalize_timestamp,
    _require_content_authorization,
)
from .sources import (
    MemorySourceSpec,
    MemorySourceValidationError,
    validate_memory_sources,
)
from .store import transaction


class MemoryCandidateError(RuntimeError):
    """Base error for candidate formation and staging."""


class MemoryCandidateAuthorizationError(MemoryCandidateError):
    """Raised when candidate mutation lacks explicit authorization."""


class MemoryCandidateValidationError(MemoryCandidateError):
    """Raised when a candidate violates deterministic staging rules."""


class MemoryCandidateAlreadyExistsError(MemoryCandidateError):
    """Raised when a candidate identifier is already present."""


class MemoryCandidateNotFoundError(MemoryCandidateError):
    """Raised when a candidate cannot be found."""


@dataclass(frozen=True)
class MemoryCandidateWriteAuthorization:
    """Explicit deterministic authorization for one candidate proposal."""

    actor: str
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MemoryCandidateCreateRequest:
    """Proposed memory fields plus formation provenance.

    The request mirrors the fields required for a future authoritative memory,
    but persistence here creates only a non-authoritative candidate.
    """

    content: str
    category: str
    knowledge_status: str
    confidence: float
    data_classification: str
    recorded_at: str
    sources: tuple[MemorySourceSpec, ...]
    origin: str
    validity_state: str = "current"
    retention_state: str = "durable"
    candidate_id: str | None = None
    memory_key: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    time_precision: str | None = None
    verified_at: str | None = None
    rayan_confirmed: bool = False
    policy_version: str | None = None
    model: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class MemoryCandidateRecord:
    """Metadata-safe representation of one staged candidate."""

    candidate_id: str
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
    candidate_state: str
    origin: str
    proposed_by: str
    policy_version: str | None
    model: str | None
    model_version: str | None
    prompt_version: str | None
    run_id: str | None
    promoted_memory_id: str | None
    rejection_reason: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class _StoredMemoryCandidate:
    metadata: MemoryCandidateRecord
    content: str


_CANDIDATE_SOURCE_NAMESPACE = uuid.UUID(
    "264723fd-d4d3-4818-ae60-f1bf00169b5e"
)


def _require_candidate_write_authorization(
    authorization: MemoryCandidateWriteAuthorization,
) -> None:
    if not authorization.allowed:
        raise MemoryCandidateAuthorizationError(
            "Memory-candidate proposal denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryCandidateAuthorizationError(
            "Authorized candidate proposals require a non-empty actor."
        )


def _require_non_empty_optional(
    value: str | None,
    *,
    field_name: str,
) -> None:
    if value is None or not value.strip():
        raise MemoryCandidateValidationError(
            f"{field_name} is required for model-proposed candidates."
        )


def _normalize_candidate_request(
    request: MemoryCandidateCreateRequest,
) -> MemoryCandidateCreateRequest:
    if request.origin not in CANDIDATE_ORIGINS:
        raise MemoryCandidateValidationError(
            f"Unsupported candidate origin: {request.origin!r}"
        )

    if request.origin == "model_proposed":
        if request.rayan_confirmed:
            raise MemoryCandidateValidationError(
                "A model-proposed candidate cannot claim user confirmation."
            )
        _require_non_empty_optional(
            request.policy_version,
            field_name="policy_version",
        )
        _require_non_empty_optional(
            request.model,
            field_name="model",
        )
        _require_non_empty_optional(
            request.prompt_version,
            field_name="prompt_version",
        )
        _require_non_empty_optional(
            request.run_id,
            field_name="run_id",
        )

    memory_request = MemoryCreateRequest(
        content=request.content,
        category=request.category,
        knowledge_status=request.knowledge_status,
        confidence=request.confidence,
        data_classification=request.data_classification,
        recorded_at=request.recorded_at,
        sources=request.sources,
        validity_state=request.validity_state,
        retention_state=request.retention_state,
        memory_key=request.memory_key,
        valid_from=request.valid_from,
        valid_to=request.valid_to,
        time_precision=request.time_precision,
        verified_at=request.verified_at,
        rayan_confirmed=request.rayan_confirmed,
    )

    try:
        normalized = _normalize_create_request(memory_request)
    except MemoryValidationError as exc:
        raise MemoryCandidateValidationError(
            f"Invalid memory candidate: {exc}"
        ) from exc

    return replace(
        request,
        valid_from=normalized.valid_from,
        valid_to=normalized.valid_to,
        recorded_at=normalized.recorded_at,
        verified_at=normalized.verified_at,
    )


def _candidate_source_id(
    *,
    candidate_id: str,
    source: MemorySourceSpec,
) -> str:
    canonical = "|".join(
        (
            candidate_id,
            source.source_type,
            source.source_ref,
            source.chunk_id or "",
            source.file_id or "",
            source.support_relation,
        )
    )
    return str(uuid.uuid5(_CANDIDATE_SOURCE_NAMESPACE, canonical))


def _insert_candidate_sources(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    sources: tuple[MemorySourceSpec, ...],
    created_at: str,
) -> None:
    try:
        validate_memory_sources(sources)
    except MemorySourceValidationError as exc:
        raise MemoryCandidateValidationError(
            f"Invalid candidate provenance: {exc}"
        ) from exc

    for source in sources:
        connection.execute(
            """
            INSERT INTO memory_candidate_sources (
                candidate_source_id,
                candidate_id,
                source_type,
                source_ref,
                source_content_sha256,
                source_text_sha256,
                chunk_id,
                file_id,
                source_date,
                support_relation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _candidate_source_id(
                    candidate_id=candidate_id,
                    source=source,
                ),
                candidate_id,
                source.source_type,
                source.source_ref,
                source.source_content_sha256,
                source.source_text_sha256,
                source.chunk_id,
                source.file_id,
                source.source_date,
                source.support_relation,
                created_at,
            ),
        )


def _row_to_stored_candidate(
    row: sqlite3.Row,
) -> _StoredMemoryCandidate:
    metadata = MemoryCandidateRecord(
        candidate_id=str(row["candidate_id"]),
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
        candidate_state=str(row["candidate_state"]),
        origin=str(row["origin"]),
        proposed_by=str(row["proposed_by"]),
        policy_version=(
            None
            if row["policy_version"] is None
            else str(row["policy_version"])
        ),
        model=None if row["model"] is None else str(row["model"]),
        model_version=(
            None
            if row["model_version"] is None
            else str(row["model_version"])
        ),
        prompt_version=(
            None
            if row["prompt_version"] is None
            else str(row["prompt_version"])
        ),
        run_id=None if row["run_id"] is None else str(row["run_id"]),
        promoted_memory_id=(
            None
            if row["promoted_memory_id"] is None
            else str(row["promoted_memory_id"])
        ),
        rejection_reason=(
            None
            if row["rejection_reason"] is None
            else str(row["rejection_reason"])
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
    return _StoredMemoryCandidate(
        metadata=metadata,
        content=str(row["content"]),
    )


def _load_stored_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> _StoredMemoryCandidate:
    row = connection.execute(
        """
        SELECT *
        FROM memory_candidates
        WHERE candidate_id = ?
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise MemoryCandidateNotFoundError(
            f"Memory candidate not found: {candidate_id}"
        )
    return _row_to_stored_candidate(row)


def load_memory_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> MemoryCandidateRecord:
    """Load candidate metadata without exposing proposed plaintext."""
    return _load_stored_candidate(
        connection,
        candidate_id=candidate_id,
    ).metadata


def load_memory_candidate_content(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    authorization: MemoryContentAccessAuthorization | None,
) -> str:
    """Load proposed plaintext only through explicit content authorization."""
    try:
        _require_content_authorization(authorization)
    except MemoryContentAuthorizationError as exc:
        raise MemoryCandidateAuthorizationError(
            "Candidate plaintext access requires explicit authorization."
        ) from exc

    return _load_stored_candidate(
        connection,
        candidate_id=candidate_id,
    ).content


def propose_memory_candidate(
    connection: sqlite3.Connection,
    *,
    request: MemoryCandidateCreateRequest,
    authorization: MemoryCandidateWriteAuthorization,
    proposed_at: str,
) -> MemoryCandidateRecord:
    """Atomically stage one non-authoritative candidate with provenance."""
    _require_candidate_write_authorization(authorization)
    request = _normalize_candidate_request(request)
    proposed_at = _normalize_timestamp(
        proposed_at,
        field_name="proposed_at",
    )
    candidate_id = request.candidate_id or str(uuid.uuid4())

    if connection.execute(
        "SELECT 1 FROM memory_candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone() is not None:
        raise MemoryCandidateAlreadyExistsError(
            f"Memory candidate already exists: {candidate_id}"
        )

    content_sha256 = hashlib.sha256(
        request.content.encode("utf-8")
    ).hexdigest()
    event_details = json.dumps(
        {
            "category": request.category,
            "knowledge_status": request.knowledge_status,
            "data_classification": request.data_classification,
            "origin": request.origin,
            "rayan_confirmed": request.rayan_confirmed,
            "source_count": len(request.sources),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    try:
        with transaction(connection):
            connection.execute(
                """
                INSERT INTO memory_candidates (
                    candidate_id,
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
                    candidate_state,
                    origin,
                    proposed_by,
                    policy_version,
                    model,
                    model_version,
                    prompt_version,
                    run_id,
                    promoted_memory_id,
                    rejection_reason,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    candidate_id,
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
                    "proposed",
                    request.origin,
                    authorization.actor,
                    request.policy_version,
                    request.model,
                    request.model_version,
                    request.prompt_version,
                    request.run_id,
                    None,
                    None,
                    proposed_at,
                    proposed_at,
                ),
            )

            _insert_candidate_sources(
                connection,
                candidate_id=candidate_id,
                sources=request.sources,
                created_at=proposed_at,
            )

            connection.execute(
                """
                INSERT INTO memory_candidate_events (
                    candidate_event_id,
                    candidate_id,
                    event_type,
                    actor,
                    details_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    candidate_id,
                    "proposed",
                    authorization.actor,
                    event_details,
                    proposed_at,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise MemoryCandidateValidationError(
            f"Memory-candidate creation failed database validation: {exc}"
        ) from exc

    return load_memory_candidate(
        connection,
        candidate_id=candidate_id,
    )
