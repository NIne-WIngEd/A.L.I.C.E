"""Metadata-safe inspection for controlled response repair in A.L.I.C.E. P3.9."""

from __future__ import annotations

from dataclasses import dataclass

from .repair_policy import ConversationResponseRepairPolicy
from .state_inspection import (
    inspect_conversation_session,
    verify_conversation_session_integrity,
)
from .state_service import ConversationStateError
from .state_store import ConversationStateStore


class ConversationResponseRepairInspectionError(RuntimeError):
    """Raised when response-repair metadata cannot be inspected safely."""


@dataclass(frozen=True)
class ConversationResponseRepairInspection:
    policy_version: str
    turn_status: str
    repair_attempted: bool
    attempt_count: int
    original_status: str | None
    original_validation_outcome: str | None
    original_response_sha256: str | None
    repair_status: str | None
    repair_validation_outcome: str | None
    repair_response_sha256: str | None
    repair_request_sha256: str | None
    same_provider_model: bool | None


def inspect_conversation_response_repair(
    store: ConversationStateStore,
    *,
    session_id: str,
    turn_id: str,
    policy: ConversationResponseRepairPolicy,
) -> ConversationResponseRepairInspection:
    policy.validate()
    try:
        integrity = verify_conversation_session_integrity(store, session_id=session_id)
        inspection = inspect_conversation_session(
            store,
            session_id=session_id,
            include_content=False,
        )
    except ConversationStateError as exc:
        raise ConversationResponseRepairInspectionError(
            "Response-repair state could not be inspected."
        ) from exc
    if not integrity.valid:
        raise ConversationResponseRepairInspectionError(
            "Response-repair inspection requires valid conversation state."
        )
    turns = [turn for turn in inspection.turns if turn.turn_id == turn_id]
    if len(turns) != 1:
        raise ConversationResponseRepairInspectionError(
            "Response-repair inspection requires exactly one matching turn."
        )
    turn = turns[0]
    attempts = turn.generations
    repair_attempts = tuple(
        attempt
        for attempt in attempts
        if attempt.request_id.startswith("repair-request:")
    )
    if len(repair_attempts) > policy.max_repair_attempts:
        raise ConversationResponseRepairInspectionError(
            "Response-repair attempt count exceeds policy."
        )
    repair = repair_attempts[0] if repair_attempts else None
    original = None
    if repair is not None:
        prior_rejected = [
            attempt
            for attempt in attempts
            if attempt.attempt_index < repair.attempt_index
            and attempt.validation_outcome == "rejected"
        ]
        if len(prior_rejected) != 1:
            raise ConversationResponseRepairInspectionError(
                "A repair attempt requires exactly one prior rejected generation."
            )
        original = prior_rejected[0]
    elif attempts:
        original = attempts[-1]
    repair_digest = None
    if repair is not None:
        prefix = "repair-request:"
        if not repair.request_id.startswith(prefix):
            raise ConversationResponseRepairInspectionError(
                "Repair generation request identity is not deterministic."
            )
        candidate = repair.request_id[len(prefix):]
        if len(candidate) != 64 or any(c not in "0123456789abcdef" for c in candidate):
            raise ConversationResponseRepairInspectionError(
                "Repair generation request digest is invalid."
            )
        repair_digest = candidate
    same_identity = None
    if original is not None and repair is not None:
        same_identity = (
            original.provider == repair.provider and original.model == repair.model
        )
    return ConversationResponseRepairInspection(
        policy_version=policy.version,
        turn_status=turn.status,
        repair_attempted=repair is not None,
        attempt_count=len(attempts),
        original_status=original.status if original is not None else None,
        original_validation_outcome=(
            original.validation_outcome if original is not None else None
        ),
        original_response_sha256=(
            original.response_sha256 if original is not None else None
        ),
        repair_status=repair.status if repair is not None else None,
        repair_validation_outcome=(
            repair.validation_outcome if repair is not None else None
        ),
        repair_response_sha256=(
            repair.response_sha256 if repair is not None else None
        ),
        repair_request_sha256=repair_digest,
        same_provider_model=same_identity,
    )


def render_conversation_response_repair_inspection(
    inspection: ConversationResponseRepairInspection,
) -> str:
    """Render only metadata; never render model, user, or grounding content."""
    values = (
        f"policy_version={inspection.policy_version}",
        f"turn_status={inspection.turn_status}",
        f"repair_attempted={str(inspection.repair_attempted).lower()}",
        f"attempt_count={inspection.attempt_count}",
        f"original_status={inspection.original_status or 'none'}",
        f"original_validation_outcome={inspection.original_validation_outcome or 'none'}",
        f"original_response_sha256={inspection.original_response_sha256 or 'none'}",
        f"repair_status={inspection.repair_status or 'none'}",
        f"repair_validation_outcome={inspection.repair_validation_outcome or 'none'}",
        f"repair_response_sha256={inspection.repair_response_sha256 or 'none'}",
        f"repair_request_sha256={inspection.repair_request_sha256 or 'none'}",
        f"same_provider_model={str(inspection.same_provider_model).lower() if inspection.same_provider_model is not None else 'none'}",
    )
    return "\n".join(values)
