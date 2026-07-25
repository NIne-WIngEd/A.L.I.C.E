"""Authorized promotion of assessed memory candidates for A.L.I.C.E. P2.7c.

Promotion is the only bridge from the non-authoritative candidate layer into
authoritative Phase 2 memory. It is explicit, candidate-bound, atomic, and
revalidates deterministic assessment rules while holding the write lock.

A language model may propose a candidate, but a model-origin candidate always
requires explicit user confirmation before promotion.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass

from .candidate_assessment import (
    ASSESSMENT_OUTCOMES,
    MemoryCandidateAssessment,
    MemoryCandidateAssessmentStateError,
    _assessment_decision,
    load_latest_candidate_assessment,
)
from .formation import (
    MemoryCandidateNotFoundError,
    MemoryCandidateRecord,
    load_memory_candidate,
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
from .sources import MemorySourceSpec
from .store import transaction


class MemoryCandidatePromotionError(RuntimeError):
    """Base error for candidate promotion."""


class MemoryCandidatePromotionAuthorizationError(
    MemoryCandidatePromotionError
):
    """Raised when promotion lacks valid candidate-bound authorization."""


class MemoryCandidatePromotionStateError(MemoryCandidatePromotionError):
    """Raised when a candidate cannot enter the promotion transition."""


class MemoryCandidatePromotionValidationError(
    MemoryCandidatePromotionError
):
    """Raised when final deterministic promotion checks fail."""


@dataclass(frozen=True)
class MemoryCandidatePromotionAuthorization:
    """Explicit authorization bound to one candidate promotion attempt."""

    actor: str
    allowed: bool
    candidate_id: str
    authorization_id: str
    user_confirmed: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class MemoryCandidatePromotionResult:
    """Metadata-safe record of one completed candidate promotion."""

    candidate: MemoryCandidateRecord
    memory: MemoryRecord
    assessment_outcome: str
    reason_codes: tuple[str, ...]
    authorization_id: str
    user_confirmed: bool
    derivation_types: tuple[str, ...]
    promoted_by: str
    promoted_at: str


_PROMOTED_MEMORY_NAMESPACE = uuid.UUID(
    "103b2607-f1ac-41aa-9216-ed5a4b521393"
)
_DERIVATION_NAMESPACE = uuid.UUID(
    "07ca462b-3485-457b-a180-93cc80c0e446"
)
_SAFE_AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9_.:-]{3,128}$")


def _require_promotion_authorization(
    authorization: MemoryCandidatePromotionAuthorization,
    *,
    candidate_id: str,
) -> None:
    if not authorization.allowed:
        raise MemoryCandidatePromotionAuthorizationError(
            "Memory-candidate promotion denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryCandidatePromotionAuthorizationError(
            "Authorized candidate promotion requires a non-empty actor."
        )
    if authorization.candidate_id != candidate_id:
        raise MemoryCandidatePromotionAuthorizationError(
            "Promotion authorization is not bound to the requested candidate."
        )
    if not _SAFE_AUTHORIZATION_ID.fullmatch(
        authorization.authorization_id
    ):
        raise MemoryCandidatePromotionAuthorizationError(
            "Promotion authorization_id must be a 3-128 character audit-safe "
            "identifier containing only letters, numbers, underscore, dot, "
            "colon, or hyphen."
        )


def _require_model_proposal_confirmation_actor(
    candidate: MemoryCandidateRecord,
    authorization: MemoryCandidatePromotionAuthorization,
) -> None:
    """Prevent the model-side proposer from authorizing its own promotion."""
    if candidate.origin != "model_proposed":
        return
    if not authorization.user_confirmed:
        raise MemoryCandidatePromotionAuthorizationError(
            "Model-proposed candidates require explicit user confirmation."
        )
    if authorization.actor.strip() == candidate.proposed_by.strip():
        raise MemoryCandidatePromotionAuthorizationError(
            "The actor that proposed a model candidate cannot authorize its "
            "promotion, even when a confirmation flag is supplied."
        )


def _candidate_content(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> str:
    row = connection.execute(
        "SELECT content FROM memory_candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise MemoryCandidateNotFoundError(
            f"Memory candidate not found: {candidate_id}"
        )
    return str(row["content"])


def _candidate_sources(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> tuple[MemorySourceSpec, ...]:
    rows = connection.execute(
        """
        SELECT
            source_type,
            source_ref,
            source_content_sha256,
            source_text_sha256,
            chunk_id,
            file_id,
            source_date,
            support_relation
        FROM memory_candidate_sources
        WHERE candidate_id = ?
        ORDER BY candidate_source_id
        """,
        (candidate_id,),
    ).fetchall()
    return tuple(
        MemorySourceSpec(
            source_type=str(row["source_type"]),
            source_ref=str(row["source_ref"]),
            source_content_sha256=(
                None
                if row["source_content_sha256"] is None
                else str(row["source_content_sha256"])
            ),
            source_text_sha256=(
                None
                if row["source_text_sha256"] is None
                else str(row["source_text_sha256"])
            ),
            chunk_id=(
                None if row["chunk_id"] is None else str(row["chunk_id"])
            ),
            file_id=(
                None if row["file_id"] is None else str(row["file_id"])
            ),
            source_date=(
                None
                if row["source_date"] is None
                else str(row["source_date"])
            ),
            support_relation=str(row["support_relation"]),
        )
        for row in rows
    )


def _promoted_memory_id(candidate_id: str) -> str:
    return str(uuid.uuid5(_PROMOTED_MEMORY_NAMESPACE, candidate_id))


def _derivation_id(
    *,
    memory_id: str,
    candidate_id: str,
    derivation_type: str,
) -> str:
    return str(
        uuid.uuid5(
            _DERIVATION_NAMESPACE,
            "|".join(
                (
                    memory_id,
                    candidate_id,
                    derivation_type,
                )
            ),
        )
    )


def _promotion_request(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
    authorization: MemoryCandidatePromotionAuthorization,
) -> MemoryCreateRequest:
    content = _candidate_content(
        connection,
        candidate_id=candidate.candidate_id,
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != candidate.content_sha256:
        raise MemoryCandidatePromotionValidationError(
            "Candidate plaintext does not match its stored SHA-256 digest."
        )

    request = MemoryCreateRequest(
        memory_id=_promoted_memory_id(candidate.candidate_id),
        content=content,
        memory_key=candidate.memory_key,
        category=candidate.category,
        knowledge_status=candidate.knowledge_status,
        confidence=candidate.confidence,
        data_classification=candidate.data_classification,
        recorded_at=candidate.recorded_at,
        sources=_candidate_sources(
            connection,
            candidate_id=candidate.candidate_id,
        ),
        validity_state=candidate.validity_state,
        retention_state=candidate.retention_state,
        valid_from=candidate.valid_from,
        valid_to=candidate.valid_to,
        time_precision=candidate.time_precision,
        verified_at=candidate.verified_at,
        rayan_confirmed=(
            candidate.rayan_confirmed or authorization.user_confirmed
        ),
    )
    try:
        return _normalize_create_request(request)
    except MemoryValidationError as exc:
        raise MemoryCandidatePromotionValidationError(
            f"Candidate cannot become authoritative memory: {exc}"
        ) from exc


def _insert_derivations(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
    memory_id: str,
    authorization: MemoryCandidatePromotionAuthorization,
    promoted_at: str,
) -> tuple[str, ...]:
    derivation_types = [candidate.origin]
    if authorization.user_confirmed:
        derivation_types.append("human_confirmed")

    for derivation_type in derivation_types:
        connection.execute(
            """
            INSERT INTO memory_derivations (
                derivation_id,
                memory_id,
                derivation_type,
                policy_version,
                model,
                model_version,
                prompt_version,
                run_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _derivation_id(
                    memory_id=memory_id,
                    candidate_id=candidate.candidate_id,
                    derivation_type=derivation_type,
                ),
                memory_id,
                derivation_type,
                candidate.policy_version,
                candidate.model,
                candidate.model_version,
                candidate.prompt_version,
                candidate.run_id,
                promoted_at,
            ),
        )

    return tuple(derivation_types)


def _load_promoted_event(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> tuple[sqlite3.Row, dict[str, object]]:
    rows = connection.execute(
        """
        SELECT actor, details_json, created_at
        FROM memory_candidate_events
        WHERE candidate_id = ?
          AND event_type = 'promoted'
        ORDER BY created_at DESC, candidate_event_id DESC
        """,
        (candidate_id,),
    ).fetchall()

    for row in rows:
        try:
            details = json.loads(str(row["details_json"]))
        except (TypeError, ValueError) as exc:
            raise MemoryCandidatePromotionStateError(
                "Candidate promotion event contains invalid JSON."
            ) from exc
        if not isinstance(details, dict):
            raise MemoryCandidatePromotionStateError(
                "Candidate promotion event must contain a JSON object."
            )

        mode = details.get("promotion_mode")
        if mode in (None, "ordinary"):
            return row, details
        if mode == "transition_aware":
            continue
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion event contains an unsupported promotion mode."
        )

    raise MemoryCandidatePromotionStateError(
        "Promoted candidate is missing its ordinary promotion audit event."
    )


def _persisted_derivation_types(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT derivation_type
        FROM memory_derivations
        WHERE memory_id = ?
        ORDER BY derivation_type
        """,
        (memory_id,),
    ).fetchall()
    return tuple(str(row["derivation_type"]) for row in rows)


def load_candidate_promotion(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> MemoryCandidatePromotionResult:
    """Load and integrity-check one completed ordinary promotion."""
    candidate = load_memory_candidate(
        connection,
        candidate_id=candidate_id,
    )
    if candidate.candidate_state != "promoted":
        raise MemoryCandidatePromotionStateError(
            "Memory candidate has not been promoted."
        )
    if candidate.promoted_memory_id is None:
        raise MemoryCandidatePromotionStateError(
            "Promoted candidate is missing promoted_memory_id."
        )

    event, details = _load_promoted_event(
        connection,
        candidate_id=candidate_id,
    )

    outcome = str(details.get("assessment_outcome", ""))
    if outcome not in ASSESSMENT_OUTCOMES:
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion event contains an unsupported assessment "
            "outcome."
        )

    event_memory_id = str(details.get("promoted_memory_id", ""))
    if event_memory_id != candidate.promoted_memory_id:
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion event does not match promoted_memory_id."
        )

    authorization_id = str(details.get("authorization_id", ""))
    if not _SAFE_AUTHORIZATION_ID.fullmatch(authorization_id):
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion event contains an invalid authorization_id."
        )

    derivation_types = tuple(
        str(value) for value in details.get("derivation_types", ())
    )
    if candidate.origin not in derivation_types:
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion event is missing the origin derivation."
        )

    user_confirmed = bool(details.get("user_confirmed", False))
    if user_confirmed != ("human_confirmed" in derivation_types):
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion confirmation metadata is inconsistent."
        )
    if candidate.origin == "model_proposed" and not user_confirmed:
        raise MemoryCandidatePromotionStateError(
            "Model-proposed promotion is missing human confirmation."
        )

    persisted_derivations = set(
        _persisted_derivation_types(
            connection,
            memory_id=candidate.promoted_memory_id,
        )
    )
    if not set(derivation_types).issubset(persisted_derivations):
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion derivation records are missing or inconsistent."
        )

    memory = load_memory(
        connection,
        memory_id=candidate.promoted_memory_id,
    )
    if memory.content_sha256 != candidate.content_sha256:
        raise MemoryCandidatePromotionStateError(
            "Promoted memory content digest does not match its candidate."
        )
    if user_confirmed and not memory.rayan_confirmed:
        raise MemoryCandidatePromotionStateError(
            "User-confirmed promotion did not persist confirmation metadata."
        )

    return MemoryCandidatePromotionResult(
        candidate=candidate,
        memory=memory,
        assessment_outcome=outcome,
        reason_codes=tuple(
            str(value) for value in details.get("reason_codes", ())
        ),
        authorization_id=authorization_id,
        user_confirmed=user_confirmed,
        derivation_types=derivation_types,
        promoted_by=str(event["actor"]),
        promoted_at=str(event["created_at"]),
    )


def _require_promotable_decision(
    *,
    outcome: str,
    reason_codes: tuple[str, ...],
    authorization: MemoryCandidatePromotionAuthorization,
) -> None:
    transition_required = {
        "current_memory_exists_for_key",
    }.intersection(reason_codes)
    if transition_required:
        raise MemoryCandidatePromotionValidationError(
            "Candidate requires a transition-aware promotion path before it "
            "can become authoritative: "
            + ";".join(sorted(transition_required))
        )

    if outcome == "rejected":
        raise MemoryCandidatePromotionValidationError(
            "Final deterministic assessment rejected candidate promotion: "
            + ";".join(reason_codes)
        )
    if outcome == "review_required" and not authorization.user_confirmed:
        raise MemoryCandidatePromotionAuthorizationError(
            "Candidate requires explicit user confirmation before promotion."
        )
    if outcome not in {"review_required", "promotion_eligible"}:
        raise MemoryCandidatePromotionValidationError(
            f"Unsupported final assessment outcome: {outcome!r}"
        )


def promote_memory_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    authorization: MemoryCandidatePromotionAuthorization,
    promoted_at: str,
) -> MemoryCandidatePromotionResult:
    """Atomically promote one assessed candidate into authoritative memory.

    Deterministic assessment rules are recomputed while the write transaction
    is held. This prevents a stale assessment from bypassing a newly created
    duplicate or conflict condition.
    """
    _require_promotion_authorization(
        authorization,
        candidate_id=candidate_id,
    )
    promoted_at = _normalize_timestamp(
        promoted_at,
        field_name="promoted_at",
    )

    initial = load_memory_candidate(
        connection,
        candidate_id=candidate_id,
    )
    _require_model_proposal_confirmation_actor(
        initial,
        authorization,
    )
    if initial.candidate_state == "promoted":
        return load_candidate_promotion(
            connection,
            candidate_id=candidate_id,
        )
    if initial.candidate_state == "proposed":
        raise MemoryCandidatePromotionStateError(
            "Candidate must be assessed before promotion."
        )
    if initial.candidate_state == "rejected":
        raise MemoryCandidatePromotionStateError(
            "Rejected candidates cannot be promoted."
        )
    if initial.candidate_state != "validated":
        raise MemoryCandidatePromotionStateError(
            f"Unsupported candidate state for promotion: "
            f"{initial.candidate_state!r}"
        )

    try:
        load_latest_candidate_assessment(
            connection,
            candidate_id=candidate_id,
        )
    except MemoryCandidateAssessmentStateError as exc:
        raise MemoryCandidatePromotionStateError(
            "Validated candidate is missing its persisted assessment."
        ) from exc

    memory_id: str | None = None

    try:
        with transaction(connection):
            candidate = load_memory_candidate(
                connection,
                candidate_id=candidate_id,
            )
            _require_model_proposal_confirmation_actor(
                candidate,
                authorization,
            )
            if candidate.candidate_state == "promoted":
                memory_id = candidate.promoted_memory_id
            else:
                if candidate.candidate_state != "validated":
                    raise MemoryCandidatePromotionStateError(
                        "Candidate state changed before promotion could commit."
                    )

                (
                    outcome,
                    reason_codes,
                    _matched_memory_ids,
                    _matched_candidate_ids,
                ) = _assessment_decision(
                    connection,
                    candidate=candidate,
                )
                _require_promotable_decision(
                    outcome=outcome,
                    reason_codes=reason_codes,
                    authorization=authorization,
                )

                request = _promotion_request(
                    connection,
                    candidate=candidate,
                    authorization=authorization,
                )
                memory_id = _insert_memory_in_transaction(
                    connection,
                    request=request,
                    actor=authorization.actor,
                    created_at=promoted_at,
                )

                derivation_types = _insert_derivations(
                    connection,
                    candidate=candidate,
                    memory_id=memory_id,
                    authorization=authorization,
                    promoted_at=promoted_at,
                )

                cursor = connection.execute(
                    """
                    UPDATE memory_candidates
                    SET candidate_state = 'promoted',
                        promoted_memory_id = ?,
                        rejection_reason = NULL,
                        updated_at = ?
                    WHERE candidate_id = ?
                      AND candidate_state = 'validated'
                      AND promoted_memory_id IS NULL
                    """,
                    (
                        memory_id,
                        promoted_at,
                        candidate_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MemoryCandidatePromotionStateError(
                        "Candidate state changed before promotion could commit."
                    )

                details_json = json.dumps(
                    {
                        "assessment_outcome": outcome,
                        "authorization_id": authorization.authorization_id,
                        "derivation_types": list(derivation_types),
                        "promoted_memory_id": memory_id,
                        "promotion_mode": "ordinary",
                        "reason_codes": list(reason_codes),
                        "user_confirmed": authorization.user_confirmed,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
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
                    VALUES (?, ?, 'promoted', ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        candidate_id,
                        authorization.actor,
                        details_json,
                        promoted_at,
                    ),
                )
    except (sqlite3.IntegrityError, MemoryValidationError) as exc:
        raise MemoryCandidatePromotionValidationError(
            f"Candidate promotion failed database validation: {exc}"
        ) from exc

    if memory_id is None:
        raise MemoryCandidatePromotionStateError(
            "Candidate promotion completed without a memory identifier."
        )

    return load_candidate_promotion(
        connection,
        candidate_id=candidate_id,
    )
