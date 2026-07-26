"""Versioned fail-closed orchestration policy for A.L.I.C.E. P3.5."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError


DEFAULT_ORCHESTRATION_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_orchestration_policy.json"
)
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_EXPECTED_BOUNDARIES = {
    "web_access_allowed": False,
    "tool_calling_allowed": False,
    "external_action_allowed": False,
    "memory_write_allowed": False,
    "memory_promotion_allowed": False,
    "highly_sensitive_grounding_allowed": False,
    "chain_of_thought_persistence_allowed": False,
}
_EXPECTED_LIFECYCLE = {
    "constitutional_contract_required": True,
    "prebuilt_grounding_only": True,
    "live_retrieval_allowed": False,
    "model_registry_resolution_required": True,
    "generation_attempt_recording_required": True,
    "atomic_state_transitions_required": True,
    "response_identity_match_required": True,
    "duplicate_assistant_messages_allowed": False,
    "automatic_retry_count": 0,
    "provider_fallback_allowed": False,
    "final_grounding_validation_enabled": True,
}
_EXPECTED_FAILURE_KEYS = {
    "cancelled",
    "interrupted",
    "timeout",
    "budget",
    "provider",
    "configuration",
    "protocol",
    "internal",
    "validation",
}


class ConversationOrchestrationPolicyError(ConversationContractError):
    """Raised when P3.5 orchestration policy is missing or weakened."""


@dataclass(frozen=True)
class ConversationOrchestrationPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    boundaries: tuple[tuple[str, bool], ...]
    lifecycle: tuple[tuple[str, bool | int], ...]
    max_output_tokens: int
    temperature: float
    failure_codes: tuple[tuple[str, str], ...]

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]

    def lifecycle_value(self, name: str) -> bool | int:
        return dict(self.lifecycle)[name]

    def failure_code(self, name: str) -> str:
        return dict(self.failure_codes)[name]


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationOrchestrationPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationOrchestrationPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationOrchestrationPolicyError(
            f"{field} must remain {str(expected).lower()} in P3.5."
        )
    return expected


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversationOrchestrationPolicyError(
            f"{field} must be a non-negative integer."
        )
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationOrchestrationPolicyError(
            f"{field} must be a positive integer."
        )
    return value


def _number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversationOrchestrationPolicyError(f"{field} must be numeric.")
    return float(value)


def _failure_code(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not _CODE_PATTERN.fullmatch(text):
        raise ConversationOrchestrationPolicyError(
            f"{field} must be a safe orchestration code."
        )
    return text


def parse_conversation_orchestration_policy(
    payload: dict[str, Any],
) -> ConversationOrchestrationPolicy:
    root = _mapping(payload, field="orchestration policy")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "boundaries",
        "lifecycle",
        "request",
        "failure_codes",
    }
    if set(root) != expected_root:
        raise ConversationOrchestrationPolicyError(
            "Orchestration policy fields do not match the P3.5 contract."
        )
    if _text(root["policy_name"], field="policy_name") != (
        "alice_conversation_orchestration_policy"
    ):
        raise ConversationOrchestrationPolicyError(
            "Unexpected orchestration policy name."
        )
    if _text(root["phase"], field="phase") != "3":
        raise ConversationOrchestrationPolicyError(
            "Orchestration policy phase must be 3."
        )
    if _text(root["milestone"], field="milestone") != "P3.5":
        raise ConversationOrchestrationPolicyError(
            "Orchestration policy milestone must be P3.5."
        )
    if _text(root["status"], field="status") != (
        "controlled_turn_orchestration"
    ):
        raise ConversationOrchestrationPolicyError(
            "Unexpected orchestration policy status."
        )

    boundaries = _mapping(root["boundaries"], field="boundaries")
    if set(boundaries) != set(_EXPECTED_BOUNDARIES):
        raise ConversationOrchestrationPolicyError(
            "Orchestration boundaries do not match the P3.5 contract."
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

    lifecycle = _mapping(root["lifecycle"], field="lifecycle")
    if set(lifecycle) != set(_EXPECTED_LIFECYCLE):
        raise ConversationOrchestrationPolicyError(
            "Orchestration lifecycle fields do not match the P3.5 contract."
        )
    parsed_lifecycle: list[tuple[str, bool | int]] = []
    for name, expected in _EXPECTED_LIFECYCLE.items():
        value = lifecycle[name]
        if isinstance(expected, bool):
            parsed = _strict_bool(
                value, expected=expected, field=f"lifecycle.{name}"
            )
        else:
            parsed = _nonnegative_int(value, field=f"lifecycle.{name}")
            if parsed != expected:
                raise ConversationOrchestrationPolicyError(
                    f"lifecycle.{name} must remain {expected} in P3.5."
                )
        parsed_lifecycle.append((name, parsed))

    request = _mapping(root["request"], field="request")
    if set(request) != {"max_output_tokens", "temperature"}:
        raise ConversationOrchestrationPolicyError(
            "Request policy fields do not match the P3.5 contract."
        )
    max_output_tokens = _positive_int(
        request["max_output_tokens"], field="request.max_output_tokens"
    )
    if not 1 <= max_output_tokens <= 8192:
        raise ConversationOrchestrationPolicyError(
            "request.max_output_tokens must be between 1 and 8192."
        )
    temperature = _number(request["temperature"], field="request.temperature")
    if temperature != 0.0:
        raise ConversationOrchestrationPolicyError(
            "P3.5 request.temperature must remain 0.0 for deterministic orchestration."
        )

    failure_codes = _mapping(root["failure_codes"], field="failure_codes")
    if set(failure_codes) != _EXPECTED_FAILURE_KEYS:
        raise ConversationOrchestrationPolicyError(
            "Failure-code fields do not match the P3.5 contract."
        )
    parsed_codes = tuple(
        (name, _failure_code(failure_codes[name], field=f"failure_codes.{name}"))
        for name in sorted(_EXPECTED_FAILURE_KEYS)
    )
    codes = [value for _, value in parsed_codes]
    if len(codes) != len(set(codes)):
        raise ConversationOrchestrationPolicyError(
            "Orchestration failure codes cannot be duplicated."
        )

    return ConversationOrchestrationPolicy(
        policy_name="alice_conversation_orchestration_policy",
        version=_text(root["version"], field="version"),
        phase="3",
        milestone="P3.5",
        status="controlled_turn_orchestration",
        boundaries=parsed_boundaries,
        lifecycle=tuple(parsed_lifecycle),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        failure_codes=parsed_codes,
    )


def load_conversation_orchestration_policy(
    path: str | Path = DEFAULT_ORCHESTRATION_POLICY_PATH,
) -> ConversationOrchestrationPolicy:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationOrchestrationPolicyError(
            f"Unable to read conversation orchestration policy: {selected}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationOrchestrationPolicyError(
            f"Conversation orchestration policy is not valid JSON: {selected}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationOrchestrationPolicyError(
            "Conversation orchestration policy JSON root must be an object."
        )
    return parse_conversation_orchestration_policy(payload)
