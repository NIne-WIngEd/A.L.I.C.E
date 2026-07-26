"""Deterministic inspection and integrity checks for P3.2 conversation state."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import sha256_text
from .state_schema import ORDINARY_CLASSIFICATIONS, REASONING_STATUSES
from .state_service import ConversationSessionTombstone, ConversationStateError
from .state_store import ConversationStateStore, ConversationStateStoreError


@dataclass(frozen=True)
class ConversationMessageInspection:
    message_id: str
    role: str
    content: str | None
    content_sha256: str
    data_classification: str
    created_at: str


@dataclass(frozen=True)
class ConversationReferenceInspection:
    reference_id: str
    source_kind: str
    source_ref: str
    citation_token: str | None
    content_sha256: str | None
    data_classification: str
    created_at: str


@dataclass(frozen=True)
class ConversationGenerationInspection:
    generation_id: str
    attempt_index: int
    request_id: str
    provider: str
    model: str
    status: str
    reasoning_status: str
    validation_outcome: str
    finish_reason: str | None
    response_sha256: str | None
    failure_code: str | None
    started_at: str
    completed_at: str | None


@dataclass(frozen=True)
class ConversationTurnInspection:
    turn_id: str
    turn_index: int
    status: str
    grounding_packet_id: str | None
    grounding_packet_sha256: str | None
    interruption_count: int
    created_at: str
    updated_at: str
    completed_at: str | None
    failure_code: str | None
    messages: tuple[ConversationMessageInspection, ...]
    references: tuple[ConversationReferenceInspection, ...]
    generations: tuple[ConversationGenerationInspection, ...]


@dataclass(frozen=True)
class ConversationSessionInspection:
    session_id: str
    status: str
    retention: str
    data_classification: str
    created_at: str
    updated_at: str
    closed_at: str | None
    turns: tuple[ConversationTurnInspection, ...]


@dataclass(frozen=True)
class ConversationStateIntegrityReport:
    session_id: str
    valid: bool
    errors: tuple[str, ...]
    turn_count: int
    message_count: int
    reference_count: int
    generation_count: int


def inspect_conversation_session(
    store: ConversationStateStore,
    *,
    session_id: str,
    include_content: bool = False,
) -> ConversationSessionInspection:
    with store.read_connection() as connection:
        session = connection.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise ConversationStateError("Conversation session does not exist.")
        turn_rows = connection.execute(
            """
            SELECT * FROM conversation_turns
            WHERE session_id = ? ORDER BY turn_index
            """,
            (session_id,),
        ).fetchall()
        turns: list[ConversationTurnInspection] = []
        for turn in turn_rows:
            message_rows = connection.execute(
                """
                SELECT * FROM conversation_messages
                WHERE turn_id = ? ORDER BY ordinal
                """,
                (turn["turn_id"],),
            ).fetchall()
            reference_rows = connection.execute(
                """
                SELECT * FROM conversation_turn_references
                WHERE turn_id = ? ORDER BY reference_index
                """,
                (turn["turn_id"],),
            ).fetchall()
            generation_rows = connection.execute(
                """
                SELECT * FROM conversation_generations
                WHERE turn_id = ? ORDER BY attempt_index
                """,
                (turn["turn_id"],),
            ).fetchall()
            messages = tuple(
                ConversationMessageInspection(
                    message_id=row["message_id"],
                    role=row["role"],
                    content=row["content"] if include_content else None,
                    content_sha256=row["content_sha256"],
                    data_classification=row["data_classification"],
                    created_at=row["created_at"],
                )
                for row in message_rows
            )
            references = tuple(
                ConversationReferenceInspection(
                    reference_id=row["reference_id"],
                    source_kind=row["source_kind"],
                    source_ref=row["source_ref"],
                    citation_token=row["citation_token"],
                    content_sha256=row["content_sha256"],
                    data_classification=row["data_classification"],
                    created_at=row["created_at"],
                )
                for row in reference_rows
            )
            generations = tuple(
                ConversationGenerationInspection(
                    generation_id=row["generation_id"],
                    attempt_index=int(row["attempt_index"]),
                    request_id=row["request_id"],
                    provider=row["provider"],
                    model=row["model"],
                    status=row["status"],
                    reasoning_status=row["reasoning_status"],
                    validation_outcome=row["validation_outcome"],
                    finish_reason=row["finish_reason"],
                    response_sha256=row["response_sha256"],
                    failure_code=row["failure_code"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                )
                for row in generation_rows
            )
            turns.append(
                ConversationTurnInspection(
                    turn_id=turn["turn_id"],
                    turn_index=int(turn["turn_index"]),
                    status=turn["status"],
                    grounding_packet_id=turn["grounding_packet_id"],
                    grounding_packet_sha256=turn["grounding_packet_sha256"],
                    interruption_count=int(turn["interruption_count"]),
                    created_at=turn["created_at"],
                    updated_at=turn["updated_at"],
                    completed_at=turn["completed_at"],
                    failure_code=turn["failure_code"],
                    messages=messages,
                    references=references,
                    generations=generations,
                )
            )
        return ConversationSessionInspection(
            session_id=session["session_id"],
            status=session["status"],
            retention=session["retention"],
            data_classification=session["data_classification"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            closed_at=session["closed_at"],
            turns=tuple(turns),
        )


def inspect_conversation_tombstone(
    store: ConversationStateStore,
    *,
    session_id: str,
) -> ConversationSessionTombstone:
    with store.read_connection() as connection:
        row = connection.execute(
            "SELECT * FROM conversation_session_tombstones WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ConversationStateError("Conversation tombstone does not exist.")
        return ConversationSessionTombstone(
            session_id=row["session_id"],
            retention=row["retention"],
            deleted_at=row["deleted_at"],
            turn_count=int(row["turn_count"]),
            message_count=int(row["message_count"]),
            reference_count=int(row["reference_count"]),
            generation_count=int(row["generation_count"]),
        )


def verify_conversation_session_integrity(
    store: ConversationStateStore,
    *,
    session_id: str,
) -> ConversationStateIntegrityReport:
    errors: list[str] = []
    try:
        store.verify_schema()
    except ConversationStateStoreError as exc:
        errors.append(str(exc))
    inspection = inspect_conversation_session(
        store,
        session_id=session_id,
        include_content=True,
    )
    session_rank = ORDINARY_CLASSIFICATIONS.index(inspection.data_classification)
    if tuple(turn.turn_index for turn in inspection.turns) != tuple(
        range(len(inspection.turns))
    ):
        errors.append("Turn indexes are not contiguous from zero.")
    nonterminal = [
        turn
        for turn in inspection.turns
        if turn.status in {"received", "context_ready", "generating", "interrupted"}
    ]
    if len(nonterminal) > 1:
        errors.append("A session contains more than one nonterminal turn.")
    interrupted_turns = [turn for turn in inspection.turns if turn.status == "interrupted"]
    if inspection.status == "interrupted" and len(interrupted_turns) != 1:
        errors.append("Interrupted session must contain exactly one interrupted turn.")
    if inspection.status != "interrupted" and interrupted_turns:
        errors.append("Interrupted turn requires an interrupted session.")

    message_count = 0
    reference_count = 0
    generation_count = 0
    for turn in inspection.turns:
        message_count += len(turn.messages)
        reference_count += len(turn.references)
        generation_count += len(turn.generations)
        if not turn.messages or turn.messages[0].role != "user":
            errors.append(f"Turn {turn.turn_id} does not begin with one user message.")
        roles = tuple(message.role for message in turn.messages)
        if len(roles) != len(set(roles)):
            errors.append(f"Turn {turn.turn_id} contains duplicate message roles.")
        assistant_messages = [
            message for message in turn.messages if message.role == "assistant"
        ]
        if turn.status == "completed" and len(assistant_messages) != 1:
            errors.append(f"Completed turn {turn.turn_id} lacks one assistant message.")
        if turn.status != "completed" and assistant_messages:
            errors.append(
                f"Non-completed turn {turn.turn_id} contains an assistant message."
            )
        for message in turn.messages:
            if message.content is None:
                errors.append(f"Integrity inspection lost content for {message.message_id}.")
                continue
            if sha256_text(message.content) != message.content_sha256:
                errors.append(f"Message digest mismatch: {message.message_id}.")
            if message.data_classification not in ORDINARY_CLASSIFICATIONS:
                errors.append(f"Prohibited message classification: {message.message_id}.")
            elif ORDINARY_CLASSIFICATIONS.index(message.data_classification) > session_rank:
                errors.append(
                    f"Message classification exceeds session: {message.message_id}."
                )
        if len(turn.references) > store.policy.max_references_per_turn:
            errors.append(f"Turn reference limit exceeded: {turn.turn_id}.")
        for reference in turn.references:
            if reference.data_classification not in ORDINARY_CLASSIFICATIONS:
                errors.append(
                    f"Prohibited reference classification: {reference.reference_id}."
                )
            elif ORDINARY_CLASSIFICATIONS.index(reference.data_classification) > session_rank:
                errors.append(
                    f"Reference classification exceeds session: {reference.reference_id}."
                )
        if tuple(generation.attempt_index for generation in turn.generations) != tuple(
            range(len(turn.generations))
        ):
            errors.append(f"Generation attempts are not contiguous: {turn.turn_id}.")
        for generation in turn.generations:
            if generation.reasoning_status not in REASONING_STATUSES:
                errors.append(
                    f"Invalid reasoning status: {generation.generation_id}."
                )
        completed_generations = [
            generation for generation in turn.generations if generation.status == "completed"
        ]
        if turn.status == "completed":
            if len(completed_generations) != 1:
                errors.append(
                    f"Completed turn must have one completed generation: {turn.turn_id}."
                )
            elif assistant_messages and (
                completed_generations[0].response_sha256
                != assistant_messages[0].content_sha256
            ):
                errors.append(
                    f"Generation response digest does not match assistant message: {turn.turn_id}."
                )
        if turn.status == "generating":
            started = [
                generation for generation in turn.generations if generation.status == "started"
            ]
            if len(started) != 1:
                errors.append(
                    f"Generating turn must have one active generation: {turn.turn_id}."
                )

    return ConversationStateIntegrityReport(
        session_id=session_id,
        valid=not errors,
        errors=tuple(errors),
        turn_count=len(inspection.turns),
        message_count=message_count,
        reference_count=reference_count,
        generation_count=generation_count,
    )
