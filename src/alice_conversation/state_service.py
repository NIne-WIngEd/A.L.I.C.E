"""Deterministic lifecycle service for private A.L.I.C.E. conversation state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable
from uuid import uuid4

from .contracts import ConversationMessage, ModelResponse, sha256_text
from .state_schema import (
    ORDINARY_CLASSIFICATIONS,
    REASONING_STATUSES,
    REFERENCE_KINDS,
    SESSION_RETENTIONS,
    TURN_STATUSES,
    VALIDATION_OUTCOMES,
)
from .state_store import ConversationStateStore


class ConversationStateError(RuntimeError):
    """Raised when a conversation-state operation violates P3.2 policy."""


_CLASSIFICATION_RANK = {
    value: index for index, value in enumerate(ORDINARY_CLASSIFICATIONS)
}
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_TERMINAL_TURN_STATUSES = {"completed", "cancelled", "failed"}
_NONTERMINAL_TURN_STATUSES = set(TURN_STATUSES) - _TERMINAL_TURN_STATUSES


@dataclass(frozen=True)
class ConversationStateReference:
    reference_id: str
    source_kind: str
    source_ref: str
    data_classification: str
    created_at: str
    citation_token: str | None = None
    content_sha256: str | None = None

    def validate(self) -> None:
        _require_text(self.reference_id, field="reference_id")
        if self.source_kind not in REFERENCE_KINDS:
            raise ConversationStateError(
                f"Unsupported conversation-state reference kind: {self.source_kind!r}"
            )
        _require_text(self.source_ref, field="source_ref")
        if self.data_classification not in _CLASSIFICATION_RANK:
            raise ConversationStateError(
                "Ordinary conversation state cannot reference HIGHLY_SENSITIVE "
                "or SECRETS content."
            )
        _parse_timestamp(self.created_at, field="reference created_at")
        if self.citation_token is not None:
            _require_text(self.citation_token, field="citation_token")
        if self.content_sha256 is not None:
            _require_digest(self.content_sha256, field="reference content_sha256")


@dataclass(frozen=True)
class ConversationSessionTombstone:
    session_id: str
    retention: str
    deleted_at: str
    turn_count: int
    message_count: int
    reference_count: int
    generation_count: int


class ConversationStateService:
    """Atomic state transitions over the private conversation database."""

    def __init__(
        self,
        store: ConversationStateStore,
        *,
        event_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.policy = store.policy
        self._event_id_factory = event_id_factory or (lambda: uuid4().hex)

    def create_session(
        self,
        *,
        session_id: str,
        created_at: str,
        retention: str | None = None,
        data_classification: str = "PRIVATE",
    ) -> None:
        _require_text(session_id, field="session_id")
        _parse_timestamp(created_at, field="session created_at")
        selected_retention = retention or self.policy.default_retention
        if selected_retention not in SESSION_RETENTIONS:
            raise ConversationStateError(
                f"Unsupported conversation retention: {selected_retention!r}"
            )
        if data_classification not in _CLASSIFICATION_RANK:
            raise ConversationStateError(
                "Ordinary conversation sessions cannot use HIGHLY_SENSITIVE or SECRETS."
            )
        with self.store.transaction() as connection:
            tombstone = connection.execute(
                "SELECT 1 FROM conversation_session_tombstones WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if tombstone is not None:
                raise ConversationStateError(
                    "A purged conversation session ID cannot be reused."
                )
            existing = connection.execute(
                "SELECT * FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["created_at"] == created_at
                    and existing["retention"] == selected_retention
                    and existing["data_classification"] == data_classification
                ):
                    return
                raise ConversationStateError(
                    "Conversation session already exists with different attributes."
                )
            connection.execute(
                """
                INSERT INTO conversation_sessions(
                    session_id, status, retention, data_classification,
                    created_at, updated_at, closed_at
                ) VALUES (?, 'active', ?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    selected_retention,
                    data_classification,
                    created_at,
                    created_at,
                ),
            )
            self._insert_event(
                connection,
                session_id=session_id,
                turn_id=None,
                event_type="session_created",
                detail_code=selected_retention,
                occurred_at=created_at,
            )

    def start_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_message: ConversationMessage,
    ) -> int:
        user_message.validate()
        if user_message.role != "user" or user_message.turn_id != turn_id:
            raise ConversationStateError(
                "A new turn requires a matching user-role ConversationMessage."
            )
        if len(user_message.content) > self.policy.max_message_chars:
            raise ConversationStateError("User message exceeds the approved character limit.")
        with self.store.transaction() as connection:
            session = self._require_session(connection, session_id)
            if session["status"] != "active":
                raise ConversationStateError(
                    "New turns require an active conversation session."
                )
            self._require_classification_within_session(
                message_classification=user_message.data_classification,
                session_classification=session["data_classification"],
            )
            existing = connection.execute(
                "SELECT * FROM conversation_turns WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if existing is not None:
                existing_message = connection.execute(
                    "SELECT * FROM conversation_messages WHERE turn_id = ? AND role = 'user'",
                    (turn_id,),
                ).fetchone()
                if (
                    existing["session_id"] == session_id
                    and existing_message is not None
                    and existing_message["message_id"] == user_message.message_id
                    and existing_message["content_sha256"] == user_message.content_sha256
                ):
                    return int(existing["turn_index"])
                raise ConversationStateError(
                    "Conversation turn already exists with different attributes."
                )
            open_turn = connection.execute(
                """
                SELECT turn_id FROM conversation_turns
                WHERE session_id = ?
                  AND status IN ('received', 'context_ready', 'generating', 'interrupted')
                """,
                (session_id,),
            ).fetchone()
            if open_turn is not None:
                raise ConversationStateError(
                    "A session cannot start another turn while one is nonterminal."
                )
            row = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(MAX(turn_index), -1) AS maximum
                FROM conversation_turns WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if int(row["count"]) >= self.policy.max_turns_per_session:
                raise ConversationStateError(
                    "Conversation session reached its approved turn limit."
                )
            turn_index = int(row["maximum"]) + 1
            connection.execute(
                """
                INSERT INTO conversation_turns(
                    turn_id, session_id, turn_index, status,
                    grounding_packet_id, grounding_packet_sha256,
                    interruption_count, created_at, updated_at,
                    completed_at, failure_code
                ) VALUES (?, ?, ?, 'received', NULL, NULL, 0, ?, ?, NULL, NULL)
                """,
                (
                    turn_id,
                    session_id,
                    turn_index,
                    user_message.created_at,
                    user_message.created_at,
                ),
            )
            self._insert_message(connection, user_message, ordinal=0)
            self._touch_session(connection, session_id, user_message.created_at)
            self._insert_event(
                connection,
                session_id=session_id,
                turn_id=turn_id,
                event_type="turn_received",
                detail_code=None,
                occurred_at=user_message.created_at,
            )
            return turn_index

    def set_turn_context(
        self,
        *,
        turn_id: str,
        references: Iterable[ConversationStateReference],
        updated_at: str,
        grounding_packet_id: str | None = None,
        grounding_packet_sha256: str | None = None,
    ) -> None:
        _parse_timestamp(updated_at, field="context updated_at")
        if (grounding_packet_id is None) != (grounding_packet_sha256 is None):
            raise ConversationStateError(
                "Grounding packet ID and digest must be supplied together."
            )
        if grounding_packet_id is not None:
            _require_text(grounding_packet_id, field="grounding_packet_id")
            _require_digest(
                grounding_packet_sha256 or "",
                field="grounding_packet_sha256",
            )
        selected = tuple(references)
        if len(selected) > self.policy.max_references_per_turn:
            raise ConversationStateError(
                "Turn context exceeds the approved reference limit."
            )
        ids: set[str] = set()
        logical_refs: set[tuple[str, str]] = set()
        for reference in selected:
            reference.validate()
            if reference.reference_id in ids:
                raise ConversationStateError("Turn context contains duplicate reference IDs.")
            logical = (reference.source_kind, reference.source_ref)
            if logical in logical_refs:
                raise ConversationStateError(
                    "Turn context contains duplicate logical references."
                )
            ids.add(reference.reference_id)
            logical_refs.add(logical)
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] != "received":
                raise ConversationStateError(
                    "Turn context can only be set from the received state."
                )
            session = self._require_session(connection, turn["session_id"])
            for index, reference in enumerate(selected):
                self._require_classification_within_session(
                    message_classification=reference.data_classification,
                    session_classification=session["data_classification"],
                )
                connection.execute(
                    """
                    INSERT INTO conversation_turn_references(
                        reference_id, turn_id, reference_index, source_kind,
                        source_ref, citation_token, content_sha256,
                        data_classification, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reference.reference_id,
                        turn_id,
                        index,
                        reference.source_kind,
                        reference.source_ref,
                        reference.citation_token,
                        reference.content_sha256,
                        reference.data_classification,
                        reference.created_at,
                    ),
                )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = 'context_ready', grounding_packet_id = ?,
                    grounding_packet_sha256 = ?, updated_at = ?
                WHERE turn_id = ?
                """,
                (
                    grounding_packet_id,
                    grounding_packet_sha256,
                    updated_at,
                    turn_id,
                ),
            )
            self._touch_session(connection, turn["session_id"], updated_at)
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type="context_ready",
                detail_code="with_references" if selected else "without_references",
                occurred_at=updated_at,
            )

    def start_generation(
        self,
        *,
        turn_id: str,
        generation_id: str,
        request_id: str,
        provider: str,
        model: str,
        started_at: str,
        reasoning_status: str = "not_persisted",
    ) -> int:
        for field, value in (
            ("generation_id", generation_id),
            ("request_id", request_id),
            ("provider", provider),
            ("model", model),
        ):
            _require_text(value, field=field)
        _parse_timestamp(started_at, field="generation started_at")
        if reasoning_status not in REASONING_STATUSES:
            raise ConversationStateError(
                f"Unsupported reasoning-status metadata: {reasoning_status!r}"
            )
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] != "context_ready":
                raise ConversationStateError(
                    "Generation can only start after context is ready."
                )
            session = self._require_session(connection, turn["session_id"])
            if session["status"] != "active":
                raise ConversationStateError(
                    "Generation requires an active conversation session."
                )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_index), -1) AS maximum
                FROM conversation_generations WHERE turn_id = ?
                """,
                (turn_id,),
            ).fetchone()
            attempt_index = int(row["maximum"]) + 1
            connection.execute(
                """
                INSERT INTO conversation_generations(
                    generation_id, turn_id, attempt_index, request_id,
                    provider, model, status, reasoning_status,
                    validation_outcome, finish_reason, response_sha256,
                    failure_code, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'started', ?, 'not_evaluated',
                          NULL, NULL, NULL, ?, NULL)
                """,
                (
                    generation_id,
                    turn_id,
                    attempt_index,
                    request_id,
                    provider,
                    model,
                    reasoning_status,
                    started_at,
                ),
            )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = 'generating', updated_at = ?, failure_code = NULL
                WHERE turn_id = ?
                """,
                (started_at, turn_id),
            )
            self._touch_session(connection, turn["session_id"], started_at)
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type="generation_started",
                detail_code=f"attempt_{attempt_index}",
                occurred_at=started_at,
            )
            return attempt_index

    def complete_turn(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: ModelResponse,
        assistant_message: ConversationMessage,
        completed_at: str,
        validation_outcome: str = "accepted",
    ) -> None:
        response.validate()
        assistant_message.validate()
        _parse_timestamp(completed_at, field="turn completed_at")
        if validation_outcome not in {"accepted", "abstained"}:
            raise ConversationStateError(
                "Completed user-visible turns must be accepted or abstained."
            )
        if response.request_id != request_id:
            raise ConversationStateError("Model response request ID does not match.")
        if response.finish_reason == "cancelled":
            raise ConversationStateError(
                "A cancelled model response cannot complete a conversation turn."
            )
        if assistant_message.role != "assistant" or assistant_message.turn_id != turn_id:
            raise ConversationStateError(
                "Turn completion requires a matching assistant-role message."
            )
        if assistant_message.content != response.content:
            raise ConversationStateError(
                "Assistant message content must exactly match the model response."
            )
        if len(assistant_message.content) > self.policy.max_message_chars:
            raise ConversationStateError(
                "Assistant message exceeds the approved character limit."
            )
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] != "generating":
                raise ConversationStateError(
                    "Only a generating turn can be completed."
                )
            generation = connection.execute(
                """
                SELECT * FROM conversation_generations
                WHERE turn_id = ? AND request_id = ?
                """,
                (turn_id, request_id),
            ).fetchone()
            if generation is None or generation["status"] != "started":
                raise ConversationStateError(
                    "Turn completion requires the active generation attempt."
                )
            if generation["provider"] != response.provider or generation["model"] != response.model:
                raise ConversationStateError(
                    "Model response identity does not match the generation attempt."
                )
            session = self._require_session(connection, turn["session_id"])
            self._require_classification_within_session(
                message_classification=assistant_message.data_classification,
                session_classification=session["data_classification"],
            )
            self._insert_message(connection, assistant_message, ordinal=1)
            connection.execute(
                """
                UPDATE conversation_generations
                SET status = 'completed', validation_outcome = ?,
                    finish_reason = ?, response_sha256 = ?,
                    completed_at = ?, failure_code = NULL
                WHERE generation_id = ?
                """,
                (
                    validation_outcome,
                    response.finish_reason,
                    sha256_text(response.content),
                    completed_at,
                    generation["generation_id"],
                ),
            )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = 'completed', updated_at = ?, completed_at = ?,
                    failure_code = NULL
                WHERE turn_id = ?
                """,
                (completed_at, completed_at, turn_id),
            )
            self._touch_session(connection, turn["session_id"], completed_at)
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type="turn_completed",
                detail_code=validation_outcome,
                occurred_at=completed_at,
            )

    def interrupt_turn(
        self,
        *,
        turn_id: str,
        request_id: str,
        interrupted_at: str,
        reason_code: str,
    ) -> None:
        _parse_timestamp(interrupted_at, field="interrupted_at")
        _require_code(reason_code, field="reason_code")
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] != "generating":
                raise ConversationStateError(
                    "Only a generating turn can be interrupted."
                )
            generation = connection.execute(
                """
                SELECT * FROM conversation_generations
                WHERE turn_id = ? AND request_id = ?
                """,
                (turn_id, request_id),
            ).fetchone()
            if generation is None or generation["status"] != "started":
                raise ConversationStateError(
                    "Turn interruption requires the active generation attempt."
                )
            connection.execute(
                """
                UPDATE conversation_generations
                SET status = 'interrupted', completed_at = ?, failure_code = ?
                WHERE generation_id = ?
                """,
                (interrupted_at, reason_code, generation["generation_id"]),
            )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = 'interrupted', updated_at = ?, failure_code = ?,
                    interruption_count = interruption_count + 1
                WHERE turn_id = ?
                """,
                (interrupted_at, reason_code, turn_id),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET status = 'interrupted', updated_at = ?
                WHERE session_id = ?
                """,
                (interrupted_at, turn["session_id"]),
            )
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type="turn_interrupted",
                detail_code=reason_code,
                occurred_at=interrupted_at,
            )

    def resume_turn(self, *, turn_id: str, resumed_at: str) -> None:
        _parse_timestamp(resumed_at, field="resumed_at")
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] != "interrupted":
                raise ConversationStateError(
                    "Only an interrupted turn can be resumed."
                )
            session = self._require_session(connection, turn["session_id"])
            if session["status"] != "interrupted":
                raise ConversationStateError(
                    "Interrupted turn/session state is inconsistent."
                )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = 'context_ready', updated_at = ?, failure_code = NULL
                WHERE turn_id = ?
                """,
                (resumed_at, turn_id),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET status = 'active', updated_at = ?
                WHERE session_id = ?
                """,
                (resumed_at, turn["session_id"]),
            )
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type="turn_resumed",
                detail_code=None,
                occurred_at=resumed_at,
            )

    def cancel_turn(
        self,
        *,
        turn_id: str,
        cancelled_at: str,
        reason_code: str,
    ) -> None:
        self._terminate_turn(
            turn_id=turn_id,
            terminated_at=cancelled_at,
            reason_code=reason_code,
            target_status="cancelled",
        )

    def fail_turn(
        self,
        *,
        turn_id: str,
        failed_at: str,
        failure_code: str,
    ) -> None:
        self._terminate_turn(
            turn_id=turn_id,
            terminated_at=failed_at,
            reason_code=failure_code,
            target_status="failed",
        )

    def close_session(
        self,
        *,
        session_id: str,
        closed_at: str,
    ) -> ConversationSessionTombstone | None:
        _parse_timestamp(closed_at, field="closed_at")
        existing_tombstone = self._read_tombstone(session_id)
        if existing_tombstone is not None:
            return existing_tombstone
        with self.store.transaction() as connection:
            session = self._require_session(connection, session_id)
            open_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversation_turns
                    WHERE session_id = ?
                      AND status IN ('received', 'context_ready', 'generating', 'interrupted')
                    """,
                    (session_id,),
                ).fetchone()[0]
            )
            if open_count:
                raise ConversationStateError(
                    "A conversation session cannot close with a nonterminal turn."
                )
            if session["retention"] == "session_only":
                return self._purge_session_in_transaction(
                    connection,
                    session_id=session_id,
                    deleted_at=closed_at,
                )
            if session["status"] == "completed":
                return None
            connection.execute(
                """
                UPDATE conversation_sessions
                SET status = 'completed', updated_at = ?, closed_at = ?
                WHERE session_id = ?
                """,
                (closed_at, closed_at, session_id),
            )
            self._insert_event(
                connection,
                session_id=session_id,
                turn_id=None,
                event_type="session_completed",
                detail_code="retained",
                occurred_at=closed_at,
            )
            return None

    def delete_session(
        self,
        *,
        session_id: str,
        deleted_at: str,
    ) -> ConversationSessionTombstone:
        _parse_timestamp(deleted_at, field="deleted_at")
        existing = self._read_tombstone(session_id)
        if existing is not None:
            return existing
        with self.store.transaction() as connection:
            self._require_session(connection, session_id)
            open_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversation_turns
                    WHERE session_id = ?
                      AND status IN ('received', 'context_ready', 'generating', 'interrupted')
                    """,
                    (session_id,),
                ).fetchone()[0]
            )
            if open_count:
                raise ConversationStateError(
                    "Conversation deletion requires all turns to be terminal."
                )
            return self._purge_session_in_transaction(
                connection,
                session_id=session_id,
                deleted_at=deleted_at,
            )

    def _terminate_turn(
        self,
        *,
        turn_id: str,
        terminated_at: str,
        reason_code: str,
        target_status: str,
    ) -> None:
        _parse_timestamp(terminated_at, field="terminated_at")
        _require_code(reason_code, field="reason_code")
        if target_status not in {"cancelled", "failed"}:
            raise ConversationStateError("Unsupported terminal turn transition.")
        with self.store.transaction() as connection:
            turn = self._require_turn(connection, turn_id)
            if turn["status"] in _TERMINAL_TURN_STATUSES:
                if turn["status"] == target_status and turn["failure_code"] == reason_code:
                    return
                raise ConversationStateError("Conversation turn is already terminal.")
            if turn["status"] not in _NONTERMINAL_TURN_STATUSES:
                raise ConversationStateError("Conversation turn cannot be terminated.")
            active = connection.execute(
                """
                SELECT * FROM conversation_generations
                WHERE turn_id = ? AND status = 'started'
                ORDER BY attempt_index DESC LIMIT 1
                """,
                (turn_id,),
            ).fetchone()
            if active is not None:
                connection.execute(
                    """
                    UPDATE conversation_generations
                    SET status = ?, completed_at = ?, failure_code = ?
                    WHERE generation_id = ?
                    """,
                    (
                        target_status,
                        terminated_at,
                        reason_code,
                        active["generation_id"],
                    ),
                )
            connection.execute(
                """
                UPDATE conversation_turns
                SET status = ?, updated_at = ?, completed_at = ?, failure_code = ?
                WHERE turn_id = ?
                """,
                (
                    target_status,
                    terminated_at,
                    terminated_at,
                    reason_code,
                    turn_id,
                ),
            )
            connection.execute(
                """
                UPDATE conversation_sessions
                SET status = 'active', updated_at = ?
                WHERE session_id = ?
                """,
                (terminated_at, turn["session_id"]),
            )
            self._insert_event(
                connection,
                session_id=turn["session_id"],
                turn_id=turn_id,
                event_type=f"turn_{target_status}",
                detail_code=reason_code,
                occurred_at=terminated_at,
            )

    def _purge_session_in_transaction(
        self,
        connection,
        *,
        session_id: str,
        deleted_at: str,
    ) -> ConversationSessionTombstone:
        session = self._require_session(connection, session_id)
        counts = {
            "turn_count": int(
                connection.execute(
                    "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            ),
            "message_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversation_messages m
                    JOIN conversation_turns t ON t.turn_id = m.turn_id
                    WHERE t.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()[0]
            ),
            "reference_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversation_turn_references r
                    JOIN conversation_turns t ON t.turn_id = r.turn_id
                    WHERE t.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()[0]
            ),
            "generation_count": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM conversation_generations g
                    JOIN conversation_turns t ON t.turn_id = g.turn_id
                    WHERE t.session_id = ?
                    """,
                    (session_id,),
                ).fetchone()[0]
            ),
        }
        tombstone = ConversationSessionTombstone(
            session_id=session_id,
            retention=session["retention"],
            deleted_at=deleted_at,
            **counts,
        )
        connection.execute(
            "DELETE FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        )
        connection.execute(
            """
            INSERT INTO conversation_session_tombstones(
                session_id, retention, deleted_at, turn_count, message_count,
                reference_count, generation_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tombstone.session_id,
                tombstone.retention,
                tombstone.deleted_at,
                tombstone.turn_count,
                tombstone.message_count,
                tombstone.reference_count,
                tombstone.generation_count,
            ),
        )
        return tombstone

    def _read_tombstone(
        self,
        session_id: str,
    ) -> ConversationSessionTombstone | None:
        with self.store.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_session_tombstones WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return ConversationSessionTombstone(
                session_id=row["session_id"],
                retention=row["retention"],
                deleted_at=row["deleted_at"],
                turn_count=int(row["turn_count"]),
                message_count=int(row["message_count"]),
                reference_count=int(row["reference_count"]),
                generation_count=int(row["generation_count"]),
            )

    @staticmethod
    def _require_session(connection, session_id: str):
        _require_text(session_id, field="session_id")
        row = connection.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ConversationStateError("Conversation session does not exist.")
        return row

    @staticmethod
    def _require_turn(connection, turn_id: str):
        _require_text(turn_id, field="turn_id")
        row = connection.execute(
            "SELECT * FROM conversation_turns WHERE turn_id = ?",
            (turn_id,),
        ).fetchone()
        if row is None:
            raise ConversationStateError("Conversation turn does not exist.")
        return row

    @staticmethod
    def _touch_session(connection, session_id: str, updated_at: str) -> None:
        connection.execute(
            "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?",
            (updated_at, session_id),
        )

    @staticmethod
    def _insert_message(connection, message: ConversationMessage, *, ordinal: int) -> None:
        connection.execute(
            """
            INSERT INTO conversation_messages(
                message_id, turn_id, ordinal, role, content, content_sha256,
                data_classification, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.message_id,
                message.turn_id,
                ordinal,
                message.role,
                message.content,
                message.content_sha256,
                message.data_classification,
                message.created_at,
            ),
        )

    def _insert_event(
        self,
        connection,
        *,
        session_id: str,
        turn_id: str | None,
        event_type: str,
        detail_code: str | None,
        occurred_at: str,
    ) -> None:
        event_id = self._event_id_factory()
        _require_text(event_id, field="event_id")
        _require_code(event_type, field="event_type")
        if detail_code is not None:
            _require_code(detail_code, field="detail_code")
        connection.execute(
            """
            INSERT INTO conversation_state_events(
                event_id, session_id, turn_id, event_type, detail_code, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                turn_id,
                event_type,
                detail_code,
                occurred_at,
            ),
        )

    @staticmethod
    def _require_classification_within_session(
        *,
        message_classification: str,
        session_classification: str,
    ) -> None:
        message_rank = _CLASSIFICATION_RANK.get(message_classification)
        session_rank = _CLASSIFICATION_RANK.get(session_classification)
        if message_rank is None or session_rank is None or message_rank > session_rank:
            raise ConversationStateError(
                "Conversation content exceeds the session classification boundary."
            )


def _require_text(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConversationStateError(f"{field} must be non-empty text.")


def _require_digest(value: str, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ConversationStateError(f"{field} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConversationStateError(
            f"{field} must contain hexadecimal SHA-256 text."
        ) from exc


def _require_code(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _CODE_PATTERN.fullmatch(value) is None:
        raise ConversationStateError(
            f"{field} must be a short sanitized code without free-form content."
        )


def _parse_timestamp(value: str, *, field: str) -> datetime:
    _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConversationStateError(f"{field} must be valid ISO-8601 text.") from exc
    if parsed.tzinfo is None:
        raise ConversationStateError(f"{field} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)
