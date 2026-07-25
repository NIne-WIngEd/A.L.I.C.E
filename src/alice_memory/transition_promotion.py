"""Transition-aware candidate promotion for A.L.I.C.E. P2.7d.

Ordinary P2.7c promotion refuses candidates when a current authoritative memory
already occupies the same logical key. This module handles that boundary with
an explicit, candidate-bound, target-bound transition decision.

A caller must choose one deterministic transition:
- ``duplicate`` records a no-op resolution and keeps the candidate rejected;
- ``correction`` creates a replacement and preserves the corrected record;
- ``supersession`` creates a valid-time successor and closes prior validity;
- ``conflict`` preserves both memories and marks both disputed.

The candidate or language model cannot select or authorize its own transition.
All authoritative changes, provenance copying, derivation records, relations,
candidate state, and audit events commit or roll back as one transaction.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

from .candidate_assessment import (
    ASSESSMENT_VERSION,
    MemoryCandidateAssessmentStateError,
    _assessment_decision,
    load_latest_candidate_assessment,
)
from .formation import MemoryCandidateRecord, load_memory_candidate
from .promotion import (
    MemoryCandidatePromotionAuthorization,
    MemoryCandidatePromotionAuthorizationError,
    _insert_derivations,
    _promoted_memory_id,
    _promotion_request,
    _require_promotion_authorization,
)
from .service import (
    MemoryAlreadyExistsError,
    MemoryCreateRequest,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryValidationError,
    _insert_memory_in_transaction,
    _normalize_timestamp,
    load_memory,
)
from .store import transaction
from .temporal import (
    InvalidMemoryTransitionError,
    MemoryRelation,
    _insert_transition_event,
    _load_relation,
    _normalize_replacement_request,
    _optional_timestamp,
    _relation_id,
    _require_transitionable,
)

TRANSITION_TYPES = (
    "duplicate",
    "correction",
    "supersession",
    "conflict",
)


class MemoryCandidateTransitionPromotionError(RuntimeError):
    """Base error for transition-aware candidate promotion."""


class MemoryCandidateTransitionAuthorizationError(
    MemoryCandidateTransitionPromotionError
):
    """Raised when transition authorization is missing or improperly bound."""


class MemoryCandidateTransitionStateError(
    MemoryCandidateTransitionPromotionError
):
    """Raised when candidate state cannot enter the selected transition."""


class MemoryCandidateTransitionValidationError(
    MemoryCandidateTransitionPromotionError
):
    """Raised when deterministic transition checks reject the operation."""


@dataclass(frozen=True)
class MemoryCandidateTransitionAuthorization:
    """Explicit authorization bound to candidate, target, and transition type."""

    actor: str
    allowed: bool
    candidate_id: str
    target_memory_id: str
    transition_type: str
    authorization_id: str
    user_confirmed: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class MemoryCandidateTransitionResult:
    """Metadata-safe result of a completed transition-aware resolution."""

    candidate: MemoryCandidateRecord
    memory: MemoryRecord
    target: MemoryRecord
    transition_type: str
    transition_action: str
    relation: MemoryRelation | None
    assessment_outcome: str
    reason_codes: tuple[str, ...]
    authorization_id: str
    user_confirmed: bool
    derivation_types: tuple[str, ...]
    resolved_by: str
    resolved_at: str

    @property
    def is_noop(self) -> bool:
        return self.transition_type == "duplicate"


_TRANSITION_ACTIONS = {
    "duplicate": "duplicate_noop",
    "correction": "corrected",
    "supersession": "superseded",
    "conflict": "conflicted",
}


def _base_promotion_authorization(
    authorization: MemoryCandidateTransitionAuthorization,
) -> MemoryCandidatePromotionAuthorization:
    return MemoryCandidatePromotionAuthorization(
        actor=authorization.actor,
        allowed=authorization.allowed,
        candidate_id=authorization.candidate_id,
        authorization_id=authorization.authorization_id,
        user_confirmed=authorization.user_confirmed,
        reason=authorization.reason,
    )


def _require_transition_authorization(
    authorization: MemoryCandidateTransitionAuthorization,
    *,
    candidate_id: str,
) -> None:
    try:
        _require_promotion_authorization(
            _base_promotion_authorization(authorization),
            candidate_id=candidate_id,
        )
    except MemoryCandidatePromotionAuthorizationError as exc:
        raise MemoryCandidateTransitionAuthorizationError(str(exc)) from exc

    if authorization.transition_type not in TRANSITION_TYPES:
        raise MemoryCandidateTransitionAuthorizationError(
            "Unsupported candidate transition type: "
            f"{authorization.transition_type!r}"
        )
    if not authorization.target_memory_id.strip():
        raise MemoryCandidateTransitionAuthorizationError(
            "Transition authorization requires a non-empty target_memory_id."
        )
    if (
        authorization.transition_type != "duplicate"
        and not authorization.user_confirmed
    ):
        raise MemoryCandidateTransitionAuthorizationError(
            "Correction, supersession, and conflict promotion require explicit "
            "user confirmation."
        )


def _load_resolution_event(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> tuple[sqlite3.Row, dict[str, object]]:
    rows = connection.execute(
        """
        SELECT event_type, actor, details_json, created_at
        FROM memory_candidate_events
        WHERE candidate_id = ?
          AND event_type IN ('promoted', 'inspected')
        ORDER BY created_at DESC, candidate_event_id DESC
        """,
        (candidate_id,),
    ).fetchall()

    for row in rows:
        try:
            details = json.loads(str(row["details_json"]))
        except (TypeError, ValueError):
            continue
        if not isinstance(details, dict):
            continue
        if (
            row["event_type"] == "promoted"
            and details.get("promotion_mode") == "transition_aware"
        ):
            return row, details
        if (
            row["event_type"] == "inspected"
            and details.get("operation") == "duplicate_noop"
        ):
            return row, details

    raise MemoryCandidateTransitionStateError(
        "Candidate has no transition-aware promotion or duplicate resolution."
    )


def load_candidate_transition_promotion(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> MemoryCandidateTransitionResult:
    """Load a completed transition-aware result without exposing plaintext."""
    candidate = load_memory_candidate(
        connection,
        candidate_id=candidate_id,
    )
    event, details = _load_resolution_event(
        connection,
        candidate_id=candidate_id,
    )

    transition_type = str(details.get("transition_type", ""))
    if transition_type not in TRANSITION_TYPES:
        raise MemoryCandidateTransitionStateError(
            "Transition audit event contains an unsupported transition type."
        )
    target_memory_id = str(details.get("target_memory_id", ""))
    if not target_memory_id:
        raise MemoryCandidateTransitionStateError(
            "Transition audit event is missing target_memory_id."
        )

    target = load_memory(
        connection,
        memory_id=target_memory_id,
    )
    relation_id = details.get("relation_id")
    relation = (
        None
        if relation_id in (None, "")
        else _load_relation(
            connection,
            relation_id=str(relation_id),
        )
    )

    if transition_type == "duplicate":
        if candidate.candidate_state != "rejected":
            raise MemoryCandidateTransitionStateError(
                "Duplicate-resolved candidate must remain rejected."
            )
        memory = target
    else:
        if candidate.candidate_state != "promoted":
            raise MemoryCandidateTransitionStateError(
                "Transition-promoted candidate is not marked promoted."
            )
        if candidate.promoted_memory_id is None:
            raise MemoryCandidateTransitionStateError(
                "Transition-promoted candidate is missing promoted_memory_id."
            )
        memory = load_memory(
            connection,
            memory_id=candidate.promoted_memory_id,
        )

    return MemoryCandidateTransitionResult(
        candidate=candidate,
        memory=memory,
        target=target,
        transition_type=transition_type,
        transition_action=str(details.get("transition_action", "")),
        relation=relation,
        assessment_outcome=str(details.get("assessment_outcome", "")),
        reason_codes=tuple(
            str(value) for value in details.get("reason_codes", ())
        ),
        authorization_id=str(details.get("authorization_id", "")),
        user_confirmed=bool(details.get("user_confirmed", False)),
        derivation_types=tuple(
            str(value) for value in details.get("derivation_types", ())
        ),
        resolved_by=str(event["actor"]),
        resolved_at=str(event["created_at"]),
    )


def _require_assessed_candidate(candidate: MemoryCandidateRecord) -> None:
    if candidate.candidate_state == "proposed":
        raise MemoryCandidateTransitionStateError(
            "Candidate must be assessed before transition-aware promotion."
        )
    if candidate.candidate_state == "promoted":
        return
    if candidate.candidate_state not in {"validated", "rejected"}:
        raise MemoryCandidateTransitionStateError(
            "Unsupported candidate state for transition-aware promotion: "
            f"{candidate.candidate_state!r}"
        )


def _validate_transition_target(
    *,
    candidate: MemoryCandidateRecord,
    target: MemoryRecord,
    transition_type: str,
    outcome: str,
    reason_codes: tuple[str, ...],
    matched_memory_ids: tuple[str, ...],
) -> None:
    if target.memory_id not in matched_memory_ids:
        raise MemoryCandidateTransitionValidationError(
            "Authorized target is not one of the deterministic assessment "
            "matches for this candidate."
        )

    is_exact_duplicate = target.content_sha256 == candidate.content_sha256
    if transition_type == "duplicate":
        if outcome != "rejected" or "duplicates_authoritative_memory" not in (
            reason_codes
        ):
            raise MemoryCandidateTransitionValidationError(
                "Duplicate no-op requires an exact authoritative duplicate."
            )
        if not is_exact_duplicate:
            raise MemoryCandidateTransitionValidationError(
                "Duplicate target content digest does not match the candidate."
            )
        return

    if is_exact_duplicate:
        raise MemoryCandidateTransitionValidationError(
            "Exact duplicates must use the duplicate no-op resolution."
        )
    if outcome != "review_required":
        raise MemoryCandidateTransitionValidationError(
            "Authoritative transitions require a review-required assessment."
        )
    if "current_memory_exists_for_key" not in reason_codes:
        raise MemoryCandidateTransitionValidationError(
            "Selected transition requires a current authoritative memory for "
            "the candidate key."
        )


def _write_duplicate_resolution(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
    target: MemoryRecord,
    outcome: str,
    reason_codes: tuple[str, ...],
    matched_memory_ids: tuple[str, ...],
    authorization: MemoryCandidateTransitionAuthorization,
    resolved_at: str,
) -> None:
    existing_rows = connection.execute(
        """
        SELECT details_json
        FROM memory_candidate_events
        WHERE candidate_id = ?
          AND event_type = 'inspected'
        ORDER BY created_at DESC, candidate_event_id DESC
        """,
        (candidate.candidate_id,),
    ).fetchall()
    for row in existing_rows:
        try:
            details = json.loads(str(row["details_json"]))
        except (TypeError, ValueError):
            continue
        if (
            isinstance(details, dict)
            and details.get("operation") == "duplicate_noop"
            and details.get("target_memory_id") == target.memory_id
        ):
            return

    previous_state = candidate.candidate_state
    rejection_reason = ";".join(reason_codes)
    cursor = connection.execute(
        """
        UPDATE memory_candidates
        SET candidate_state = 'rejected',
            rejection_reason = ?,
            updated_at = ?
        WHERE candidate_id = ?
          AND candidate_state IN ('validated', 'rejected')
          AND promoted_memory_id IS NULL
        """,
        (
            rejection_reason,
            resolved_at,
            candidate.candidate_id,
        ),
    )
    if cursor.rowcount != 1:
        raise MemoryCandidateTransitionStateError(
            "Candidate state changed before duplicate resolution could commit."
        )

    if previous_state == "validated":
        assessment_details = json.dumps(
            {
                "assessment_version": ASSESSMENT_VERSION,
                "matched_candidate_ids": [],
                "matched_memory_ids": list(matched_memory_ids),
                "outcome": "rejected",
                "reason_codes": list(reason_codes),
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
            VALUES (?, ?, 'rejected', ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                candidate.candidate_id,
                authorization.actor,
                assessment_details,
                resolved_at,
            ),
        )

    details_json = json.dumps(
        {
            "assessment_outcome": outcome,
            "authorization_id": authorization.authorization_id,
            "derivation_types": [],
            "operation": "duplicate_noop",
            "reason_codes": list(reason_codes),
            "target_memory_id": target.memory_id,
            "transition_action": _TRANSITION_ACTIONS["duplicate"],
            "transition_type": "duplicate",
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
        VALUES (?, ?, 'inspected', ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            candidate.candidate_id,
            authorization.actor,
            details_json,
            resolved_at,
        ),
    )


def _apply_correction(
    connection: sqlite3.Connection,
    *,
    target: MemoryRecord,
    request: MemoryCreateRequest,
    actor: str,
    transitioned_at: str,
) -> tuple[str, MemoryRelation]:
    _require_transitionable(target)
    normalized = _normalize_replacement_request(target, request)
    memory_id = _insert_memory_in_transaction(
        connection,
        request=normalized,
        actor=actor,
        created_at=transitioned_at,
    )

    connection.execute(
        """
        UPDATE memories
        SET knowledge_status = 'superseded',
            validity_state = 'historical',
            updated_at = ?
        WHERE memory_id = ?
        """,
        (transitioned_at, target.memory_id),
    )
    relation_id = _relation_id(
        from_memory_id=memory_id,
        to_memory_id=target.memory_id,
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
        VALUES (?, ?, ?, 'corrects', ?)
        """,
        (
            relation_id,
            memory_id,
            target.memory_id,
            transitioned_at,
        ),
    )
    _insert_transition_event(
        connection,
        memory_id=target.memory_id,
        event_type="corrected",
        actor=actor,
        created_at=transitioned_at,
        details={
            "replacement_memory_id": memory_id,
            "relation_id": relation_id,
        },
    )
    return memory_id, _load_relation(connection, relation_id=relation_id)


def _apply_supersession(
    connection: sqlite3.Connection,
    *,
    target: MemoryRecord,
    request: MemoryCreateRequest,
    actor: str,
    transitioned_at: str,
) -> tuple[str, MemoryRelation]:
    _require_transitionable(target)
    normalized = _normalize_replacement_request(target, request)
    replacement_start = _optional_timestamp(
        normalized.valid_from,
        field_name="replacement.valid_from",
    )
    if replacement_start is None:
        raise InvalidMemoryTransitionError(
            "Supersession requires candidate.valid_from."
        )

    previous_start = _optional_timestamp(
        target.valid_from,
        field_name="previous.valid_from",
    )
    if previous_start is not None and replacement_start < previous_start:
        raise InvalidMemoryTransitionError(
            "Candidate valid_from cannot precede target valid_from."
        )

    previous_end = _optional_timestamp(
        target.valid_to,
        field_name="previous.valid_to",
    )
    closed_valid_to = normalized.valid_from
    if previous_end is not None and previous_end < replacement_start:
        closed_valid_to = target.valid_to

    memory_id = _insert_memory_in_transaction(
        connection,
        request=normalized,
        actor=actor,
        created_at=transitioned_at,
    )
    connection.execute(
        """
        UPDATE memories
        SET knowledge_status = 'historical',
            validity_state = 'historical',
            valid_to = ?,
            updated_at = ?
        WHERE memory_id = ?
        """,
        (
            closed_valid_to,
            transitioned_at,
            target.memory_id,
        ),
    )
    relation_id = _relation_id(
        from_memory_id=memory_id,
        to_memory_id=target.memory_id,
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
        VALUES (?, ?, ?, 'supersedes', ?)
        """,
        (
            relation_id,
            memory_id,
            target.memory_id,
            transitioned_at,
        ),
    )
    _insert_transition_event(
        connection,
        memory_id=target.memory_id,
        event_type="superseded",
        actor=actor,
        created_at=transitioned_at,
        details={
            "closed_valid_to": closed_valid_to,
            "relation_id": relation_id,
            "replacement_memory_id": memory_id,
        },
    )
    return memory_id, _load_relation(connection, relation_id=relation_id)


def _apply_conflict(
    connection: sqlite3.Connection,
    *,
    target: MemoryRecord,
    request: MemoryCreateRequest,
    actor: str,
    transitioned_at: str,
) -> tuple[str, MemoryRelation]:
    _require_transitionable(target)
    memory_id = _insert_memory_in_transaction(
        connection,
        request=request,
        actor=actor,
        created_at=transitioned_at,
    )

    connection.execute(
        """
        UPDATE memories
        SET knowledge_status = 'disputed',
            validity_state = 'disputed',
            updated_at = ?
        WHERE memory_id IN (?, ?)
        """,
        (
            transitioned_at,
            target.memory_id,
            memory_id,
        ),
    )

    from_memory_id, to_memory_id = sorted((target.memory_id, memory_id))
    relation_id = _relation_id(
        from_memory_id=from_memory_id,
        to_memory_id=to_memory_id,
        relation_type="conflicts_with",
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
        VALUES (?, ?, ?, 'conflicts_with', ?)
        """,
        (
            relation_id,
            from_memory_id,
            to_memory_id,
            transitioned_at,
        ),
    )
    _insert_transition_event(
        connection,
        memory_id=target.memory_id,
        event_type="conflict_marked",
        actor=actor,
        created_at=transitioned_at,
        details={
            "other_memory_id": memory_id,
            "relation_id": relation_id,
        },
    )
    _insert_transition_event(
        connection,
        memory_id=memory_id,
        event_type="conflict_marked",
        actor=actor,
        created_at=transitioned_at,
        details={
            "other_memory_id": target.memory_id,
            "relation_id": relation_id,
        },
    )
    return memory_id, _load_relation(connection, relation_id=relation_id)


def _mark_candidate_promoted(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
    memory_id: str,
    target_memory_id: str,
    transition_type: str,
    relation: MemoryRelation,
    outcome: str,
    reason_codes: tuple[str, ...],
    matched_memory_ids: tuple[str, ...],
    authorization: MemoryCandidateTransitionAuthorization,
    derivation_types: tuple[str, ...],
    promoted_at: str,
) -> None:
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
            candidate.candidate_id,
        ),
    )
    if cursor.rowcount != 1:
        raise MemoryCandidateTransitionStateError(
            "Candidate state changed before transition promotion could commit."
        )

    details_json = json.dumps(
        {
            "assessment_outcome": outcome,
            "authorization_id": authorization.authorization_id,
            "derivation_types": list(derivation_types),
            "promoted_memory_id": memory_id,
            "promotion_mode": "transition_aware",
            "reason_codes": list(reason_codes),
            "relation_id": relation.relation_id,
            "target_memory_id": target_memory_id,
            "transition_action": _TRANSITION_ACTIONS[transition_type],
            "transition_type": transition_type,
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
            candidate.candidate_id,
            authorization.actor,
            details_json,
            promoted_at,
        ),
    )


def promote_memory_candidate_with_transition(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    authorization: MemoryCandidateTransitionAuthorization,
    promoted_at: str,
) -> MemoryCandidateTransitionResult:
    """Resolve a candidate through an explicit target-bound transition."""
    _require_transition_authorization(
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
    _require_assessed_candidate(initial)

    if initial.candidate_state == "promoted":
        result = load_candidate_transition_promotion(
            connection,
            candidate_id=candidate_id,
        )
        if (
            result.transition_type != authorization.transition_type
            or result.target.memory_id != authorization.target_memory_id
        ):
            raise MemoryCandidateTransitionStateError(
                "Candidate was already resolved through a different transition."
            )
        return result

    try:
        load_latest_candidate_assessment(
            connection,
            candidate_id=candidate_id,
        )
    except MemoryCandidateAssessmentStateError as exc:
        raise MemoryCandidateTransitionStateError(
            "Assessed candidate is missing its persisted assessment."
        ) from exc

    try:
        with transaction(connection):
            candidate = load_memory_candidate(
                connection,
                candidate_id=candidate_id,
            )
            _require_assessed_candidate(candidate)

            if candidate.candidate_state == "promoted":
                pass
            else:
                target = load_memory(
                    connection,
                    memory_id=authorization.target_memory_id,
                )
                (
                    outcome,
                    reason_codes,
                    matched_memory_ids,
                    _matched_candidate_ids,
                ) = _assessment_decision(
                    connection,
                    candidate=candidate,
                )
                _validate_transition_target(
                    candidate=candidate,
                    target=target,
                    transition_type=authorization.transition_type,
                    outcome=outcome,
                    reason_codes=reason_codes,
                    matched_memory_ids=matched_memory_ids,
                )

                if authorization.transition_type == "duplicate":
                    _write_duplicate_resolution(
                        connection,
                        candidate=candidate,
                        target=target,
                        outcome=outcome,
                        reason_codes=reason_codes,
                        matched_memory_ids=matched_memory_ids,
                        authorization=authorization,
                        resolved_at=promoted_at,
                    )
                else:
                    if candidate.candidate_state != "validated":
                        raise MemoryCandidateTransitionStateError(
                            "Only validated candidates can modify authoritative "
                            "memory through a transition."
                        )

                    base_authorization = _base_promotion_authorization(
                        authorization
                    )
                    request = _promotion_request(
                        connection,
                        candidate=candidate,
                        authorization=base_authorization,
                    )

                    if authorization.transition_type == "correction":
                        memory_id, relation = _apply_correction(
                            connection,
                            target=target,
                            request=request,
                            actor=authorization.actor,
                            transitioned_at=promoted_at,
                        )
                    elif authorization.transition_type == "supersession":
                        memory_id, relation = _apply_supersession(
                            connection,
                            target=target,
                            request=request,
                            actor=authorization.actor,
                            transitioned_at=promoted_at,
                        )
                    else:
                        memory_id, relation = _apply_conflict(
                            connection,
                            target=target,
                            request=request,
                            actor=authorization.actor,
                            transitioned_at=promoted_at,
                        )

                    expected_memory_id = _promoted_memory_id(candidate_id)
                    if memory_id != expected_memory_id:
                        raise MemoryCandidateTransitionStateError(
                            "Transition promotion produced an unexpected memory "
                            "identifier."
                        )
                    derivation_types = _insert_derivations(
                        connection,
                        candidate=candidate,
                        memory_id=memory_id,
                        authorization=base_authorization,
                        promoted_at=promoted_at,
                    )
                    _mark_candidate_promoted(
                        connection,
                        candidate=candidate,
                        memory_id=memory_id,
                        target_memory_id=target.memory_id,
                        transition_type=authorization.transition_type,
                        relation=relation,
                        outcome=outcome,
                        reason_codes=reason_codes,
                        matched_memory_ids=matched_memory_ids,
                        authorization=authorization,
                        derivation_types=derivation_types,
                        promoted_at=promoted_at,
                    )
    except (
        sqlite3.IntegrityError,
        MemoryAlreadyExistsError,
        MemoryNotFoundError,
        MemoryValidationError,
        InvalidMemoryTransitionError,
    ) as exc:
        raise MemoryCandidateTransitionValidationError(
            f"Transition-aware candidate promotion failed: {exc}"
        ) from exc

    result = load_candidate_transition_promotion(
        connection,
        candidate_id=candidate_id,
    )
    if (
        result.transition_type != authorization.transition_type
        or result.target.memory_id != authorization.target_memory_id
    ):
        raise MemoryCandidateTransitionStateError(
            "Candidate transition result does not match authorization binding."
        )
    return result
