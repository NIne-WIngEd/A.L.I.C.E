"""Versioned governed cross-turn context policy for A.L.I.C.E. P3.8."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError

DEFAULT_CONTEXT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_context_policy.json"
)
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_EXPECTED_BOUNDARIES = {
    "same_session_only": True,
    "completed_turns_only": True,
    "accepted_or_abstained_only": True,
    "whole_turn_pairs_only": True,
    "exclude_current_turn": True,
    "integrity_verification_required": True,
    "hidden_reasoning_allowed": False,
    "rejected_output_allowed": False,
    "failed_turn_content_allowed": False,
    "cross_session_content_allowed": False,
    "message_identifiers_rendered_to_model": False,
    "semantic_summarization_allowed": False,
    "memory_write_allowed": False,
}
_EXPECTED_TRUNCATION = {
    "strategy": "recent_contiguous_suffix",
    "drop_oldest_first": True,
    "partial_turn_allowed": False,
    "partial_message_allowed": False,
}
_EXPECTED_FAILURE_KEYS = {"integrity", "assembly"}


class ConversationContextPolicyError(ConversationContractError):
    """Raised when the P3.8 context policy is invalid or weakened."""


@dataclass(frozen=True)
class ConversationContextPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    boundaries: tuple[tuple[str, bool], ...]
    max_prior_turns: int
    max_prior_messages: int
    max_prior_characters: int
    truncation_strategy: str
    drop_oldest_first: bool
    partial_turn_allowed: bool
    partial_message_allowed: bool
    failure_codes: tuple[tuple[str, str], ...]

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]

    def failure_code(self, name: str) -> str:
        return dict(self.failure_codes)[name]

    def validate(self) -> None:
        if (
            self.policy_name != "alice_conversation_context_policy"
            or self.phase != "3"
            or self.milestone != "P3.8"
            or self.status != "governed_cross_turn_context"
            or not isinstance(self.version, str)
            or not self.version.strip()
        ):
            raise ConversationContextPolicyError(
                "Context policy identity does not match the P3.8 contract."
            )
        if dict(self.boundaries) != _EXPECTED_BOUNDARIES:
            raise ConversationContextPolicyError(
                "Context boundaries do not match the P3.8 contract."
            )
        if not 1 <= self.max_prior_turns <= 128:
            raise ConversationContextPolicyError(
                "Context turn budget is outside the P3.8 range."
            )
        if (
            self.max_prior_messages % 2 != 0
            or not 2 <= self.max_prior_messages <= 256
        ):
            raise ConversationContextPolicyError(
                "Context message budget must preserve complete turn pairs."
            )
        if not 1_024 <= self.max_prior_characters <= 100_000:
            raise ConversationContextPolicyError(
                "Context character budget is outside the P3.8 range."
            )
        if (
            self.truncation_strategy != "recent_contiguous_suffix"
            or self.drop_oldest_first is not True
            or self.partial_turn_allowed is not False
            or self.partial_message_allowed is not False
        ):
            raise ConversationContextPolicyError(
                "Context truncation does not match the P3.8 contract."
            )
        if set(dict(self.failure_codes)) != _EXPECTED_FAILURE_KEYS:
            raise ConversationContextPolicyError(
                "Context failure codes do not match the P3.8 contract."
            )
        values = tuple(dict(self.failure_codes).values())
        if len(values) != len(set(values)) or any(
            not _CODE_PATTERN.fullmatch(value) for value in values
        ):
            raise ConversationContextPolicyError(
                "Context failure codes must be unique safe codes."
            )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationContextPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationContextPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationContextPolicyError(
            f"{field} must remain {str(expected).lower()} in P3.8."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationContextPolicyError(f"{field} must be a positive integer.")
    return value


def _safe_code(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not _CODE_PATTERN.fullmatch(text):
        raise ConversationContextPolicyError(
            f"{field} must be a safe context failure code."
        )
    return text


def parse_conversation_context_policy(
    payload: dict[str, Any],
) -> ConversationContextPolicy:
    root = _mapping(payload, field="context policy")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "boundaries",
        "limits",
        "truncation",
        "failure_codes",
    }
    if set(root) != expected_root:
        raise ConversationContextPolicyError(
            "Context policy fields do not match the P3.8 contract."
        )
    if _text(root["policy_name"], field="policy_name") != (
        "alice_conversation_context_policy"
    ):
        raise ConversationContextPolicyError("Unexpected context policy name.")
    if _text(root["phase"], field="phase") != "3":
        raise ConversationContextPolicyError("Context policy phase must be 3.")
    if _text(root["milestone"], field="milestone") != "P3.8":
        raise ConversationContextPolicyError("Context policy milestone must be P3.8.")
    if _text(root["status"], field="status") != "governed_cross_turn_context":
        raise ConversationContextPolicyError("Unexpected context policy status.")

    boundaries = _mapping(root["boundaries"], field="boundaries")
    if set(boundaries) != set(_EXPECTED_BOUNDARIES):
        raise ConversationContextPolicyError(
            "Context boundaries do not match the P3.8 contract."
        )
    parsed_boundaries = tuple(
        (
            name,
            _strict_bool(
                boundaries[name], expected=expected, field=f"boundaries.{name}"
            ),
        )
        for name, expected in _EXPECTED_BOUNDARIES.items()
    )

    limits = _mapping(root["limits"], field="limits")
    if set(limits) != {
        "max_prior_turns",
        "max_prior_messages",
        "max_prior_characters",
    }:
        raise ConversationContextPolicyError(
            "Context limits do not match the P3.8 contract."
        )
    max_prior_turns = _positive_int(
        limits["max_prior_turns"], field="limits.max_prior_turns"
    )
    max_prior_messages = _positive_int(
        limits["max_prior_messages"], field="limits.max_prior_messages"
    )
    max_prior_characters = _positive_int(
        limits["max_prior_characters"], field="limits.max_prior_characters"
    )
    if not 1 <= max_prior_turns <= 128:
        raise ConversationContextPolicyError(
            "limits.max_prior_turns must be between 1 and 128."
        )
    if max_prior_messages % 2 != 0 or not 2 <= max_prior_messages <= 256:
        raise ConversationContextPolicyError(
            "limits.max_prior_messages must be an even value between 2 and 256."
        )
    if not 1_024 <= max_prior_characters <= 100_000:
        raise ConversationContextPolicyError(
            "limits.max_prior_characters must be between 1024 and 100000."
        )

    truncation = _mapping(root["truncation"], field="truncation")
    if set(truncation) != set(_EXPECTED_TRUNCATION):
        raise ConversationContextPolicyError(
            "Context truncation fields do not match the P3.8 contract."
        )
    strategy = _text(truncation["strategy"], field="truncation.strategy")
    if strategy != _EXPECTED_TRUNCATION["strategy"]:
        raise ConversationContextPolicyError(
            "P3.8 truncation must use the recent contiguous suffix strategy."
        )
    drop_oldest_first = _strict_bool(
        truncation["drop_oldest_first"],
        expected=True,
        field="truncation.drop_oldest_first",
    )
    partial_turn_allowed = _strict_bool(
        truncation["partial_turn_allowed"],
        expected=False,
        field="truncation.partial_turn_allowed",
    )
    partial_message_allowed = _strict_bool(
        truncation["partial_message_allowed"],
        expected=False,
        field="truncation.partial_message_allowed",
    )

    failure_codes = _mapping(root["failure_codes"], field="failure_codes")
    if set(failure_codes) != _EXPECTED_FAILURE_KEYS:
        raise ConversationContextPolicyError(
            "Context failure-code fields do not match the P3.8 contract."
        )
    parsed_codes = tuple(
        (name, _safe_code(failure_codes[name], field=f"failure_codes.{name}"))
        for name in sorted(_EXPECTED_FAILURE_KEYS)
    )
    values = [value for _, value in parsed_codes]
    if len(values) != len(set(values)):
        raise ConversationContextPolicyError(
            "Context failure codes cannot be duplicated."
        )

    policy = ConversationContextPolicy(
        policy_name="alice_conversation_context_policy",
        version=_text(root["version"], field="version"),
        phase="3",
        milestone="P3.8",
        status="governed_cross_turn_context",
        boundaries=parsed_boundaries,
        max_prior_turns=max_prior_turns,
        max_prior_messages=max_prior_messages,
        max_prior_characters=max_prior_characters,
        truncation_strategy=strategy,
        drop_oldest_first=drop_oldest_first,
        partial_turn_allowed=partial_turn_allowed,
        partial_message_allowed=partial_message_allowed,
        failure_codes=parsed_codes,
    )
    policy.validate()
    return policy


def load_conversation_context_policy(
    path: str | Path = DEFAULT_CONTEXT_POLICY_PATH,
) -> ConversationContextPolicy:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationContextPolicyError(
            f"Unable to read conversation context policy: {selected}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationContextPolicyError(
            f"Conversation context policy is not valid JSON: {selected}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationContextPolicyError(
            "Conversation context policy JSON root must be an object."
        )
    return parse_conversation_context_policy(payload)
