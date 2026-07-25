"""Deterministic assessment of staged memory candidates for A.L.I.C.E. P2.7b.

Assessment never creates authoritative memory. It classifies a staged proposal
as rejected, requiring user review, or eligible for a later promotion step.
The language model cannot mark its own proposal promotion-eligible because all
model-origin candidates require review.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass

from .formation import (
    MemoryCandidateNotFoundError,
    MemoryCandidateRecord,
    load_memory_candidate,
)
from .service import _normalize_timestamp
from .store import transaction

ASSESSMENT_VERSION = "p2.7b-v1"

ASSESSMENT_OUTCOMES = (
    "rejected",
    "review_required",
    "promotion_eligible",
)


class MemoryCandidateAssessmentError(RuntimeError):
    """Base error for deterministic candidate assessment."""


class MemoryCandidateAssessmentAuthorizationError(
    MemoryCandidateAssessmentError
):
    """Raised when assessment lacks explicit authorization."""


class MemoryCandidateAssessmentStateError(MemoryCandidateAssessmentError):
    """Raised when a candidate cannot enter the assessment transition."""


@dataclass(frozen=True)
class MemoryCandidateAssessmentAuthorization:
    """Explicit authorization for one deterministic candidate assessment."""

    actor: str
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True)
class MemoryCandidateAssessment:
    """Metadata-safe result of one persisted deterministic assessment."""

    candidate_id: str
    outcome: str
    reason_codes: tuple[str, ...]
    matched_memory_ids: tuple[str, ...]
    matched_candidate_ids: tuple[str, ...]
    assessed_by: str
    assessed_at: str
    assessment_version: str


def _require_assessment_authorization(
    authorization: MemoryCandidateAssessmentAuthorization,
) -> None:
    if not authorization.allowed:
        raise MemoryCandidateAssessmentAuthorizationError(
            "Memory-candidate assessment denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryCandidateAssessmentAuthorizationError(
            "Authorized candidate assessment requires a non-empty actor."
        )


def _candidate_plaintext(
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
) -> tuple[sqlite3.Row, ...]:
    rows = connection.execute(
        """
        SELECT source_type, support_relation
        FROM memory_candidate_sources
        WHERE candidate_id = ?
        ORDER BY candidate_source_id
        """,
        (candidate_id,),
    ).fetchall()
    return tuple(rows)


def _authoritative_matches(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    duplicate_rows = connection.execute(
        """
        SELECT memory_id
        FROM memories
        WHERE content_sha256 = ?
          AND deletion_state = 'active'
          AND retention_state <> 'archived'
          AND data_classification <> 'HIGHLY_SENSITIVE'
        ORDER BY memory_id
        """,
        (candidate.content_sha256,),
    ).fetchall()
    duplicate_ids = tuple(str(row["memory_id"]) for row in duplicate_rows)

    if candidate.memory_key is None:
        return duplicate_ids, ()

    key_rows = connection.execute(
        """
        SELECT memory_id
        FROM memories
        WHERE memory_key = ?
          AND content_sha256 <> ?
          AND deletion_state = 'active'
          AND retention_state <> 'archived'
          AND validity_state IN ('current', 'disputed')
          AND data_classification <> 'HIGHLY_SENSITIVE'
        ORDER BY memory_id
        """,
        (
            candidate.memory_key,
            candidate.content_sha256,
        ),
    ).fetchall()
    key_ids = tuple(str(row["memory_id"]) for row in key_rows)
    return duplicate_ids, key_ids


def _duplicate_candidate_ids(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
) -> tuple[str, ...]:
    rows = connection.execute(
        """
        SELECT candidate_id, created_at
        FROM memory_candidates
        WHERE content_sha256 = ?
          AND candidate_state <> 'rejected'
        ORDER BY created_at, candidate_id
        """,
        (candidate.content_sha256,),
    ).fetchall()
    ordered_ids = tuple(str(row["candidate_id"]) for row in rows)
    if not ordered_ids or ordered_ids[0] == candidate.candidate_id:
        return ()
    return tuple(
        candidate_id
        for candidate_id in ordered_ids
        if candidate_id != candidate.candidate_id
    )


def _assessment_decision(
    connection: sqlite3.Connection,
    *,
    candidate: MemoryCandidateRecord,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    content = _candidate_plaintext(
        connection,
        candidate_id=candidate.candidate_id,
    )
    sources = _candidate_sources(
        connection,
        candidate_id=candidate.candidate_id,
    )
    source_types = {str(row["source_type"]) for row in sources}
    support_relations = [str(row["support_relation"]) for row in sources]

    exact_memory_ids, same_key_memory_ids = _authoritative_matches(
        connection,
        candidate=candidate,
    )
    duplicate_candidate_ids = _duplicate_candidate_ids(
        connection,
        candidate=candidate,
    )

    reject_reasons: set[str] = set()
    review_reasons: set[str] = set()

    if re.search(r"\w", content, flags=re.UNICODE) is None:
        reject_reasons.add("content_has_no_alphanumeric_text")

    if candidate.retention_state == "archived":
        reject_reasons.add("candidate_starts_archived")

    if support_relations and all(
        relation == "contradicts" for relation in support_relations
    ):
        reject_reasons.add("provenance_only_contradicts")
    elif "contradicts" in support_relations:
        review_reasons.add("contradictory_provenance_present")

    if exact_memory_ids:
        reject_reasons.add("duplicates_authoritative_memory")

    if duplicate_candidate_ids:
        reject_reasons.add("duplicates_earlier_candidate")

    if reject_reasons:
        return (
            "rejected",
            tuple(sorted(reject_reasons)),
            tuple(sorted(set(exact_memory_ids + same_key_memory_ids))),
            tuple(sorted(duplicate_candidate_ids)),
        )

    if same_key_memory_ids:
        review_reasons.add("current_memory_exists_for_key")

    if not any(
        relation in {"supports", "derived_from"}
        for relation in support_relations
    ):
        review_reasons.add("supporting_provenance_missing")

    if candidate.validity_state != "current":
        review_reasons.add("candidate_not_current")

    if candidate.retention_state == "review_due":
        review_reasons.add("retention_review_due")

    if candidate.confidence < 0.80:
        review_reasons.add("confidence_below_eligibility_threshold")

    if candidate.knowledge_status not in {
        "verified_fact",
        "rayan_statement",
    }:
        review_reasons.add("knowledge_status_requires_review")

    if (
        candidate.knowledge_status == "verified_fact"
        and candidate.verified_at is None
    ):
        review_reasons.add("verified_fact_missing_verified_at")

    if candidate.origin == "model_proposed":
        review_reasons.add("model_proposals_require_user_review")

    elif candidate.origin == "explicit_user":
        if not candidate.rayan_confirmed:
            review_reasons.add("explicit_user_candidate_not_confirmed")
        if not source_types.intersection(
            {"rayan_direct_statement", "approved_manual_entry"}
        ):
            review_reasons.add("explicit_user_provenance_missing")

    elif candidate.origin == "deterministic_import":
        if candidate.knowledge_status != "verified_fact":
            review_reasons.add("deterministic_import_not_verified_fact")
        if candidate.confidence < 0.90:
            review_reasons.add("deterministic_import_confidence_below_threshold")
        if not source_types.intersection(
            {"phase1_chunk", "phase1_source", "approved_manual_entry"}
        ):
            review_reasons.add("deterministic_import_provenance_missing")

    if review_reasons:
        outcome = "review_required"
        reasons = tuple(sorted(review_reasons))
    else:
        outcome = "promotion_eligible"
        reasons = ("deterministic_eligibility_rules_passed",)

    matched_memory_ids = tuple(
        sorted(set(exact_memory_ids + same_key_memory_ids))
    )
    return outcome, reasons, matched_memory_ids, ()


def _row_to_assessment(row: sqlite3.Row) -> MemoryCandidateAssessment:
    try:
        details = json.loads(str(row["details_json"]))
    except (TypeError, ValueError) as exc:
        raise MemoryCandidateAssessmentError(
            "Candidate assessment event contains invalid JSON."
        ) from exc

    outcome = str(details.get("outcome", ""))
    if outcome not in ASSESSMENT_OUTCOMES:
        raise MemoryCandidateAssessmentError(
            "Candidate assessment event contains an unsupported outcome."
        )

    return MemoryCandidateAssessment(
        candidate_id=str(row["candidate_id"]),
        outcome=outcome,
        reason_codes=tuple(str(value) for value in details.get("reason_codes", ())),
        matched_memory_ids=tuple(
            str(value) for value in details.get("matched_memory_ids", ())
        ),
        matched_candidate_ids=tuple(
            str(value) for value in details.get("matched_candidate_ids", ())
        ),
        assessed_by=str(row["actor"]),
        assessed_at=str(row["created_at"]),
        assessment_version=str(details.get("assessment_version", "")),
    )


def load_latest_candidate_assessment(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
) -> MemoryCandidateAssessment:
    """Load the latest metadata-safe assessment event for one candidate."""
    if connection.execute(
        "SELECT 1 FROM memory_candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone() is None:
        raise MemoryCandidateNotFoundError(
            f"Memory candidate not found: {candidate_id}"
        )

    row = connection.execute(
        """
        SELECT candidate_id, actor, details_json, created_at
        FROM memory_candidate_events
        WHERE candidate_id = ?
          AND event_type IN ('validated', 'rejected')
        ORDER BY created_at DESC, candidate_event_id DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise MemoryCandidateAssessmentStateError(
            "Memory candidate has not been assessed."
        )
    return _row_to_assessment(row)


def assess_memory_candidate(
    connection: sqlite3.Connection,
    *,
    candidate_id: str,
    authorization: MemoryCandidateAssessmentAuthorization,
    assessed_at: str,
) -> MemoryCandidateAssessment:
    """Assess one proposal without promoting it into authoritative memory."""
    _require_assessment_authorization(authorization)
    assessed_at = _normalize_timestamp(
        assessed_at,
        field_name="assessed_at",
    )
    candidate = load_memory_candidate(
        connection,
        candidate_id=candidate_id,
    )

    if candidate.candidate_state in {"validated", "rejected"}:
        return load_latest_candidate_assessment(
            connection,
            candidate_id=candidate_id,
        )
    if candidate.candidate_state != "proposed":
        raise MemoryCandidateAssessmentStateError(
            "Only proposed candidates can be assessed."
        )

    outcome, reason_codes, matched_memory_ids, matched_candidate_ids = (
        _assessment_decision(
            connection,
            candidate=candidate,
        )
    )
    candidate_state = "rejected" if outcome == "rejected" else "validated"
    event_type = "rejected" if outcome == "rejected" else "validated"
    rejection_reason = (
        ";".join(reason_codes) if outcome == "rejected" else None
    )
    details_json = json.dumps(
        {
            "assessment_version": ASSESSMENT_VERSION,
            "outcome": outcome,
            "reason_codes": list(reason_codes),
            "matched_memory_ids": list(matched_memory_ids),
            "matched_candidate_ids": list(matched_candidate_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    with transaction(connection):
        cursor = connection.execute(
            """
            UPDATE memory_candidates
            SET candidate_state = ?,
                rejection_reason = ?,
                updated_at = ?
            WHERE candidate_id = ?
              AND candidate_state = 'proposed'
            """,
            (
                candidate_state,
                rejection_reason,
                assessed_at,
                candidate_id,
            ),
        )
        if cursor.rowcount != 1:
            raise MemoryCandidateAssessmentStateError(
                "Candidate state changed before assessment could commit."
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
                event_type,
                authorization.actor,
                details_json,
                assessed_at,
            ),
        )

    return load_latest_candidate_assessment(
        connection,
        candidate_id=candidate_id,
    )
