"""Integrity-checked bounded cross-turn context assembly for A.L.I.C.E. P3.8."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .context_policy import ConversationContextPolicy
from .contracts import ConversationContractError, ConversationMessage
from .state_inspection import (
    ConversationSessionInspection,
    ConversationTurnInspection,
    inspect_conversation_session,
    verify_conversation_session_integrity,
)
from .state_service import ConversationStateError
from .state_store import ConversationStateStore


class ConversationContextAssemblyError(ConversationContractError):
    """Sanitized P3.8 context-assembly failure."""

    def __init__(self, message: str, *, failure_code: str) -> None:
        self.failure_code = failure_code
        super().__init__(message)


@dataclass(frozen=True)
class ConversationContextAssembly:
    """Selected prior messages plus metadata that does not reveal their content."""

    policy_version: str
    messages: tuple[ConversationMessage, ...]
    context_sha256: str
    eligible_turn_count: int
    included_turn_count: int
    omitted_turn_count: int
    excluded_turn_count: int
    included_character_count: int
    truncated: bool

    def validate(self, *, policy: ConversationContextPolicy) -> None:
        if self.policy_version != policy.version:
            raise ConversationContextAssemblyError(
                "Context assembly policy version does not match.",
                failure_code=policy.failure_code("assembly"),
            )
        if len(self.messages) != self.included_turn_count * 2:
            raise ConversationContextAssemblyError(
                "Context message count does not match included turn pairs.",
                failure_code=policy.failure_code("assembly"),
            )
        if len(self.messages) > policy.max_prior_messages:
            raise ConversationContextAssemblyError(
                "Context assembly exceeds its message budget.",
                failure_code=policy.failure_code("assembly"),
            )
        if self.included_turn_count > policy.max_prior_turns:
            raise ConversationContextAssemblyError(
                "Context assembly exceeds its turn budget.",
                failure_code=policy.failure_code("assembly"),
            )
        if self.included_character_count > policy.max_prior_characters:
            raise ConversationContextAssemblyError(
                "Context assembly exceeds its character budget.",
                failure_code=policy.failure_code("assembly"),
            )
        if self.included_character_count != sum(
            len(message.content) for message in self.messages
        ):
            raise ConversationContextAssemblyError(
                "Context character metadata does not match selected messages.",
                failure_code=policy.failure_code("assembly"),
            )
        if self.eligible_turn_count != (
            self.included_turn_count + self.omitted_turn_count
        ):
            raise ConversationContextAssemblyError(
                "Context eligible-turn metadata is inconsistent.",
                failure_code=policy.failure_code("assembly"),
            )
        if self.truncated is not (self.omitted_turn_count > 0):
            raise ConversationContextAssemblyError(
                "Context truncation metadata is inconsistent.",
                failure_code=policy.failure_code("assembly"),
            )
        seen_turns: set[str] = set()
        for index in range(0, len(self.messages), 2):
            user = self.messages[index]
            assistant = self.messages[index + 1]
            user.validate()
            assistant.validate()
            if user.role != "user" or assistant.role != "assistant":
                raise ConversationContextAssemblyError(
                    "Context messages must remain complete user/assistant pairs.",
                    failure_code=policy.failure_code("assembly"),
                )
            if user.turn_id != assistant.turn_id or user.turn_id in seen_turns:
                raise ConversationContextAssemblyError(
                    "Context turn pairing is invalid.",
                    failure_code=policy.failure_code("assembly"),
                )
            seen_turns.add(user.turn_id)
        if conversation_context_sha256(
            self.messages, policy_version=self.policy_version
        ) != self.context_sha256:
            raise ConversationContextAssemblyError(
                "Context digest does not match selected messages.",
                failure_code=policy.failure_code("assembly"),
            )


def conversation_context_sha256(
    messages: tuple[ConversationMessage, ...],
    *,
    policy_version: str,
) -> str:
    """Digest ordered context metadata without embedding raw IDs or content."""

    payload = {
        "policy_version": policy_version,
        "messages": [
            {
                "role": message.role,
                "content_sha256": message.content_sha256,
                "data_classification": message.data_classification,
            }
            for message in messages
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assemble_conversation_context(
    store: ConversationStateStore,
    *,
    session_id: str,
    policy: ConversationContextPolicy,
    current_turn_id: str | None = None,
) -> ConversationContextAssembly:
    """Select the newest contiguous suffix of eligible completed turn pairs."""

    if not isinstance(policy, ConversationContextPolicy):
        raise ConversationContextAssemblyError(
            "Conversation context requires a validated P3.8 policy.",
            failure_code="context_assembly_failed",
        )
    policy.validate()
    try:
        integrity = verify_conversation_session_integrity(
            store,
            session_id=session_id,
        )
    except (ConversationStateError, OSError) as exc:
        raise ConversationContextAssemblyError(
            "Conversation context integrity could not be verified.",
            failure_code=policy.failure_code("integrity"),
        ) from exc
    if not integrity.valid:
        raise ConversationContextAssemblyError(
            "Conversation context integrity verification failed.",
            failure_code=policy.failure_code("integrity"),
        )
    try:
        inspection = inspect_conversation_session(
            store,
            session_id=session_id,
            include_content=True,
        )
    except ConversationStateError as exc:
        raise ConversationContextAssemblyError(
            "Conversation context could not be inspected.",
            failure_code=policy.failure_code("assembly"),
        ) from exc
    return _assemble_from_inspection(
        inspection,
        current_turn_id=current_turn_id,
        policy=policy,
    )


def _assemble_from_inspection(
    inspection: ConversationSessionInspection,
    *,
    current_turn_id: str | None,
    policy: ConversationContextPolicy,
) -> ConversationContextAssembly:
    if inspection.retention not in {"session_only", "retained"}:
        raise ConversationContextAssemblyError(
            "Conversation retention is not eligible for context assembly.",
            failure_code=policy.failure_code("assembly"),
        )

    current_index: int | None = None
    if current_turn_id is not None:
        current = [turn for turn in inspection.turns if turn.turn_id == current_turn_id]
        if len(current) != 1 or current[0].status != "context_ready":
            raise ConversationContextAssemblyError(
                "Current turn is not ready for governed context assembly.",
                failure_code=policy.failure_code("assembly"),
            )
        if inspection.status != "active":
            raise ConversationContextAssemblyError(
                "Cross-turn context requires an active conversation session.",
                failure_code=policy.failure_code("assembly"),
            )
        current_index = current[0].turn_index

    considered = tuple(
        turn
        for turn in inspection.turns
        if current_index is None or turn.turn_index < current_index
    )
    eligible_pairs: list[tuple[ConversationMessage, ConversationMessage]] = []
    excluded_turn_count = 0
    for turn in considered:
        pair = _eligible_pair(turn)
        if pair is None:
            excluded_turn_count += 1
        else:
            eligible_pairs.append(pair)

    selected_reverse: list[tuple[ConversationMessage, ConversationMessage]] = []
    selected_characters = 0
    for pair in reversed(eligible_pairs):
        pair_characters = len(pair[0].content) + len(pair[1].content)
        next_turn_count = len(selected_reverse) + 1
        next_message_count = next_turn_count * 2
        if (
            next_turn_count > policy.max_prior_turns
            or next_message_count > policy.max_prior_messages
            or selected_characters + pair_characters > policy.max_prior_characters
        ):
            break
        selected_reverse.append(pair)
        selected_characters += pair_characters

    selected_pairs = tuple(reversed(selected_reverse))
    messages = tuple(message for pair in selected_pairs for message in pair)
    eligible_turn_count = len(eligible_pairs)
    omitted_turn_count = eligible_turn_count - len(selected_pairs)
    assembly = ConversationContextAssembly(
        policy_version=policy.version,
        messages=messages,
        context_sha256=conversation_context_sha256(
            messages,
            policy_version=policy.version,
        ),
        eligible_turn_count=eligible_turn_count,
        included_turn_count=len(selected_pairs),
        omitted_turn_count=omitted_turn_count,
        excluded_turn_count=excluded_turn_count,
        included_character_count=selected_characters,
        truncated=omitted_turn_count > 0,
    )
    assembly.validate(policy=policy)
    return assembly


def _eligible_pair(
    turn: ConversationTurnInspection,
) -> tuple[ConversationMessage, ConversationMessage] | None:
    if turn.status != "completed":
        return None
    completed = [
        generation
        for generation in turn.generations
        if generation.status == "completed"
        and generation.validation_outcome in {"accepted", "abstained"}
    ]
    if len(completed) != 1:
        return None
    if len(turn.messages) != 2:
        return None
    user_inspection, assistant_inspection = turn.messages
    if (
        user_inspection.role != "user"
        or assistant_inspection.role != "assistant"
        or user_inspection.content is None
        or assistant_inspection.content is None
    ):
        return None
    user = ConversationMessage(
        message_id=user_inspection.message_id,
        turn_id=turn.turn_id,
        role="user",
        content=user_inspection.content,
        content_sha256=user_inspection.content_sha256,
        created_at=user_inspection.created_at,
        data_classification=user_inspection.data_classification,
    )
    assistant = ConversationMessage(
        message_id=assistant_inspection.message_id,
        turn_id=turn.turn_id,
        role="assistant",
        content=assistant_inspection.content,
        content_sha256=assistant_inspection.content_sha256,
        created_at=assistant_inspection.created_at,
        data_classification=assistant_inspection.data_classification,
    )
    user.validate()
    assistant.validate()
    return user, assistant
