"""Deterministic source-cited Memory Core answer packets for P2.9b.

This module does not call a language model. It converts explicitly selected,
authorized, authoritative memories into a structured personal-answer packet
whose claims remain traceable to exact ``memory_sources`` rows. Phase 3 may
later render these packets conversationally without weakening the citation
contract established here.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .evaluation_contract import MemoryEvaluationCase
from .provenance import AttachedMemorySource, list_memory_sources
from .service import (
    MemoryContentAccessAuthorization,
    MemoryNotFoundError,
    MemoryRecord,
    load_memory,
    load_memory_content,
)

ORDINARY_ANSWER_CLASSIFICATIONS = (
    "PUBLIC",
    "INTERNAL",
    "PRIVATE",
)

_CLASSIFICATION_RANK = {
    value: index
    for index, value in enumerate(ORDINARY_ANSWER_CLASSIFICATIONS)
}

_SUPPORTING_RELATIONS = {
    "supports",
    "derived_from",
}

_CLAIM_OUTCOMES = {
    "answerable",
    "conflict",
    "uncertain",
}

_EMPTY_OUTCOMES = {
    "insufficient_evidence",
    "denied",
}


class MemoryCitedAnswerError(RuntimeError):
    """Base error for deterministic source-cited answer construction."""


class MemoryCitedAnswerAuthorizationError(MemoryCitedAnswerError):
    """Raised when answer construction is not explicitly authorized."""


class MemoryCitedAnswerValidationError(MemoryCitedAnswerError):
    """Raised when selected memory cannot safely support an answer."""


@dataclass(frozen=True)
class MemoryAnswerAuthorization:
    """Read authorization for one private, source-cited answer packet."""

    actor: str
    allowed: bool
    purpose: str
    max_classification: str = "PRIVATE"


@dataclass(frozen=True)
class MemoryAnswerCitation:
    """One exact citation to an authoritative ``memory_sources`` row."""

    memory_source_id: str
    memory_id: str
    source_type: str
    source_ref: str
    source_content_sha256: str | None
    source_text_sha256: str | None
    chunk_id: str | None
    file_id: str | None
    support_relation: str

    @property
    def token(self) -> str:
        return (
            f"[memory:{self.memory_id} "
            f"source:{self.source_ref} "
            f"source_id:{self.memory_source_id}]"
        )


@dataclass(frozen=True)
class MemoryAnswerClaim:
    """One authoritative memory claim and its source citations."""

    claim_id: str
    memory_id: str
    text: str
    content_sha256: str
    knowledge_status: str
    confidence: float
    data_classification: str
    citations: tuple[MemoryAnswerCitation, ...]


@dataclass(frozen=True)
class MemoryAnswerSubmission:
    """Deterministic personal-answer packet for one evaluation case."""

    case_id: str
    outcome: str
    answer_text: str
    claims: tuple[MemoryAnswerClaim, ...]


def _require_authorization(
    authorization: MemoryAnswerAuthorization,
    *,
    case: MemoryEvaluationCase,
) -> None:
    if not authorization.allowed:
        raise MemoryCitedAnswerAuthorizationError(
            "Source-cited memory answer denied by explicit authorization."
        )
    if not authorization.actor.strip():
        raise MemoryCitedAnswerAuthorizationError(
            "Authorized memory answers require a non-empty actor."
        )
    if not authorization.purpose.strip():
        raise MemoryCitedAnswerAuthorizationError(
            "Authorized memory answers require a non-empty purpose."
        )
    if authorization.max_classification not in _CLASSIFICATION_RANK:
        raise MemoryCitedAnswerAuthorizationError(
            "Ordinary cited answers cannot authorize HIGHLY_SENSITIVE data."
        )
    if case.max_classification not in _CLASSIFICATION_RANK:
        raise MemoryCitedAnswerAuthorizationError(
            "Evaluation case exceeds the ordinary answer boundary."
        )
    if (
        _CLASSIFICATION_RANK[authorization.max_classification]
        < _CLASSIFICATION_RANK[case.max_classification]
    ):
        raise MemoryCitedAnswerAuthorizationError(
            "Answer authorization is narrower than the evaluation case."
        )


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MemoryCitedAnswerValidationError(
            "Evaluation timestamp must be valid ISO-8601."
        ) from exc
    if parsed.tzinfo is None:
        raise MemoryCitedAnswerValidationError(
            "Evaluation timestamp must include a timezone offset."
        )
    return parsed.astimezone(timezone.utc)


def _optional_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _parse_timestamp(value)


def _temporally_eligible(
    record: MemoryRecord,
    *,
    case: MemoryEvaluationCase,
) -> bool:
    if record.deletion_state != "active":
        return False
    if record.retention_state == "archived" and not case.include_historical:
        return False

    if case.at is None:
        if not case.include_historical and (
            record.validity_state == "historical"
            or record.knowledge_status in {"historical", "superseded"}
        ):
            return False
        return record.validity_state in {"current", "disputed"} or (
            case.include_historical
            and record.validity_state == "historical"
        )

    at = _parse_timestamp(case.at)
    valid_from = _optional_timestamp(record.valid_from)
    valid_to = _optional_timestamp(record.valid_to)
    if valid_from is not None and at < valid_from:
        return False
    if valid_to is not None and at >= valid_to:
        return False
    if not case.include_historical and record.validity_state == "historical":
        return False
    return True


def _corrected_target_ids(connection: sqlite3.Connection) -> set[str]:
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


def _citation(source: AttachedMemorySource) -> MemoryAnswerCitation:
    return MemoryAnswerCitation(
        memory_source_id=source.memory_source_id,
        memory_id=source.memory_id,
        source_type=source.source_type,
        source_ref=source.source_ref,
        source_content_sha256=source.source_content_sha256,
        source_text_sha256=source.source_text_sha256,
        chunk_id=source.chunk_id,
        file_id=source.file_id,
        support_relation=source.support_relation,
    )


def _supporting_citations(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
) -> tuple[MemoryAnswerCitation, ...]:
    citations = tuple(
        _citation(source)
        for source in list_memory_sources(
            connection,
            memory_id=memory_id,
        )
        if source.support_relation in _SUPPORTING_RELATIONS
    )
    if not citations:
        raise MemoryCitedAnswerValidationError(
            f"Authoritative memory lacks supporting provenance: {memory_id}"
        )
    return citations


def _claim_id(
    *,
    case_id: str,
    memory_id: str,
    content_sha256: str,
) -> str:
    value = f"{case_id}|{memory_id}|{content_sha256}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_conflict_selection(
    connection: sqlite3.Connection,
    *,
    memory_ids: tuple[str, ...],
) -> None:
    if len(memory_ids) < 2:
        raise MemoryCitedAnswerValidationError(
            "Conflict answers require at least two authoritative memories."
        )
    selected = set(memory_ids)
    rows = connection.execute(
        """
        SELECT from_memory_id, to_memory_id
        FROM memory_relations
        WHERE relation_type = 'conflicts_with'
        """
    ).fetchall()
    related = {
        frozenset((str(row["from_memory_id"]), str(row["to_memory_id"])))
        for row in rows
    }
    for memory_id in selected:
        if not any(memory_id in pair and pair <= selected for pair in related):
            raise MemoryCitedAnswerValidationError(
                "Conflict answer selection lacks an authoritative conflict "
                f"relation for memory: {memory_id}"
            )


def _render_answer(
    *,
    outcome: str,
    claims: tuple[MemoryAnswerClaim, ...],
) -> str:
    if outcome == "insufficient_evidence":
        return "I do not have enough authoritative memory evidence to answer this reliably."
    if outcome == "denied":
        return "Access denied by deterministic memory authorization."

    rendered_claims = []
    for claim in claims:
        tokens = " ".join(citation.token for citation in claim.citations)
        rendered_claims.append(f"{claim.text} {tokens}")

    if outcome == "conflict":
        prefix = "Authoritative memory contains a material conflict:"
    elif outcome == "uncertain":
        prefix = "The available authoritative memory is uncertain:"
    else:
        prefix = "Authoritative memory supports the following answer:"
    return prefix + "\n" + "\n".join(
        f"- {item}" for item in rendered_claims
    )


def build_memory_answer_submission(
    connection: sqlite3.Connection,
    *,
    case: MemoryEvaluationCase,
    memory_ids: tuple[str, ...],
    authorization: MemoryAnswerAuthorization,
) -> MemoryAnswerSubmission:
    """Build one private structured answer from explicitly selected memory IDs."""
    _require_authorization(authorization, case=case)

    if len(set(memory_ids)) != len(memory_ids):
        raise MemoryCitedAnswerValidationError(
            "Answer selection contains duplicate memory IDs."
        )
    if case.expected_outcome in _EMPTY_OUTCOMES:
        if memory_ids:
            raise MemoryCitedAnswerValidationError(
                "Denied or insufficient-evidence answers cannot contain memories."
            )
        claims: tuple[MemoryAnswerClaim, ...] = ()
        return MemoryAnswerSubmission(
            case_id=case.case_id,
            outcome=case.expected_outcome,
            answer_text=_render_answer(
                outcome=case.expected_outcome,
                claims=claims,
            ),
            claims=claims,
        )

    if case.expected_outcome not in _CLAIM_OUTCOMES:
        raise MemoryCitedAnswerValidationError(
            f"Unsupported answer outcome: {case.expected_outcome!r}"
        )
    if not memory_ids:
        raise MemoryCitedAnswerValidationError(
            "Answerable cases require at least one selected memory."
        )
    if case.expected_outcome == "conflict":
        _validate_conflict_selection(
            connection,
            memory_ids=memory_ids,
        )

    corrected_targets = _corrected_target_ids(connection)
    claims_list: list[MemoryAnswerClaim] = []
    content_authorization = MemoryContentAccessAuthorization(
        actor=authorization.actor,
        allowed=True,
        reason=authorization.purpose,
    )

    for memory_id in memory_ids:
        try:
            record = load_memory(connection, memory_id=memory_id)
        except MemoryNotFoundError as exc:
            raise MemoryCitedAnswerValidationError(
                f"Selected authoritative memory does not exist: {memory_id}"
            ) from exc

        rank = _CLASSIFICATION_RANK.get(record.data_classification)
        if rank is None:
            raise MemoryCitedAnswerValidationError(
                "HIGHLY_SENSITIVE memory cannot enter ordinary cited answers."
            )
        if rank > _CLASSIFICATION_RANK[authorization.max_classification]:
            raise MemoryCitedAnswerAuthorizationError(
                f"Answer authorization does not allow memory: {memory_id}"
            )
        if rank > _CLASSIFICATION_RANK[case.max_classification]:
            raise MemoryCitedAnswerValidationError(
                f"Evaluation case does not allow memory: {memory_id}"
            )
        if memory_id in corrected_targets:
            raise MemoryCitedAnswerValidationError(
                f"Corrected memory cannot support a current answer: {memory_id}"
            )
        if not _temporally_eligible(record, case=case):
            raise MemoryCitedAnswerValidationError(
                f"Memory is not temporally eligible for the case: {memory_id}"
            )

        content = load_memory_content(
            connection,
            memory_id=memory_id,
            authorization=content_authorization,
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != record.content_sha256:
            raise MemoryCitedAnswerValidationError(
                f"Authoritative content digest mismatch: {memory_id}"
            )

        citations = _supporting_citations(
            connection,
            memory_id=memory_id,
        )
        claims_list.append(
            MemoryAnswerClaim(
                claim_id=_claim_id(
                    case_id=case.case_id,
                    memory_id=memory_id,
                    content_sha256=digest,
                ),
                memory_id=memory_id,
                text=content,
                content_sha256=digest,
                knowledge_status=record.knowledge_status,
                confidence=record.confidence,
                data_classification=record.data_classification,
                citations=citations,
            )
        )

    claims = tuple(claims_list)
    return MemoryAnswerSubmission(
        case_id=case.case_id,
        outcome=case.expected_outcome,
        answer_text=_render_answer(
            outcome=case.expected_outcome,
            claims=claims,
        ),
        claims=claims,
    )
