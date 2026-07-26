"""Versioned controlled response-repair policy for A.L.I.C.E. P3.9."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError

DEFAULT_RESPONSE_REPAIR_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_response_repair_policy.json"
)
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_EXPECTED_BOUNDARIES = {
    "same_provider_model_required": True,
    "same_grounding_required": True,
    "same_context_required": True,
    "issue_codes_only": True,
    "rejected_response_text_in_prompt_allowed": False,
    "rejected_response_text_persistence_allowed": False,
    "hidden_reasoning_allowed": False,
    "provider_fallback_allowed": False,
    "live_retrieval_allowed": False,
    "tool_calling_allowed": False,
    "external_action_allowed": False,
    "memory_write_allowed": False,
}
_EXPECTED_FAILURE_KEYS = {"unavailable", "exhausted", "timeout", "budget", "internal"}


class ConversationResponseRepairPolicyError(ConversationContractError):
    """Raised when P3.9 response-repair policy is invalid or weakened."""


@dataclass(frozen=True)
class ConversationResponseRepairPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    enabled: bool
    boundaries: tuple[tuple[str, bool], ...]
    max_repair_attempts: int
    max_issue_codes: int
    max_repair_prompt_chars: int
    max_repair_output_tokens: int
    max_total_output_tokens: int
    max_total_elapsed_seconds: float
    failure_codes: tuple[tuple[str, str], ...]

    @classmethod
    def disabled(cls) -> "ConversationResponseRepairPolicy":
        return cls(
            policy_name="alice_conversation_response_repair_policy",
            version="disabled",
            phase="3",
            milestone="P3.9",
            status="controlled_response_repair",
            enabled=False,
            boundaries=tuple(_EXPECTED_BOUNDARIES.items()),
            max_repair_attempts=1,
            max_issue_codes=64,
            max_repair_prompt_chars=4096,
            max_repair_output_tokens=1024,
            max_total_output_tokens=2048,
            max_total_elapsed_seconds=900.0,
            failure_codes=tuple(
                sorted(
                    {
                        "unavailable": "response_repair_unavailable",
                        "exhausted": "response_repair_exhausted",
                        "timeout": "response_repair_timeout",
                        "budget": "response_repair_budget",
                        "internal": "response_repair_internal",
                    }.items()
                )
            ),
        )

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]

    def failure_code(self, name: str) -> str:
        return dict(self.failure_codes)[name]

    def validate(self) -> None:
        if self.policy_name != "alice_conversation_response_repair_policy":
            raise ConversationResponseRepairPolicyError("Unexpected response-repair policy name.")
        if self.phase != "3" or self.milestone != "P3.9":
            raise ConversationResponseRepairPolicyError(
                "Response-repair policy must identify Phase 3 milestone P3.9."
            )
        if self.status != "controlled_response_repair":
            raise ConversationResponseRepairPolicyError("Unexpected response-repair policy status.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ConversationResponseRepairPolicyError(
                "Response-repair policy version must be non-empty text."
            )
        if not isinstance(self.enabled, bool):
            raise ConversationResponseRepairPolicyError(
                "Response-repair policy enabled must be boolean."
            )
        if dict(self.boundaries) != _EXPECTED_BOUNDARIES:
            raise ConversationResponseRepairPolicyError(
                "Response-repair boundaries do not match the P3.9 contract."
            )
        if self.max_repair_attempts != 1:
            raise ConversationResponseRepairPolicyError(
                "P3.9 permits exactly one response-repair attempt."
            )
        if not 1 <= self.max_issue_codes <= 256:
            raise ConversationResponseRepairPolicyError("max_issue_codes must be between 1 and 256.")
        if not 256 <= self.max_repair_prompt_chars <= 20000:
            raise ConversationResponseRepairPolicyError(
                "max_repair_prompt_chars must be between 256 and 20000."
            )
        if not 1 <= self.max_repair_output_tokens <= 8192:
            raise ConversationResponseRepairPolicyError(
                "max_repair_output_tokens must be between 1 and 8192."
            )
        if not self.max_repair_output_tokens <= self.max_total_output_tokens <= 16384:
            raise ConversationResponseRepairPolicyError(
                "max_total_output_tokens must cover repair and remain at most 16384."
            )
        if not 0.1 <= self.max_total_elapsed_seconds <= 3600.0:
            raise ConversationResponseRepairPolicyError(
                "max_total_elapsed_seconds must be between 0.1 and 3600."
            )
        codes = dict(self.failure_codes)
        if set(codes) != _EXPECTED_FAILURE_KEYS:
            raise ConversationResponseRepairPolicyError(
                "Response-repair failure codes do not match the P3.9 contract."
            )
        values = list(codes.values())
        if len(values) != len(set(values)):
            raise ConversationResponseRepairPolicyError(
                "Response-repair failure codes cannot be duplicated."
            )
        for name, value in codes.items():
            if not _CODE_PATTERN.fullmatch(value):
                raise ConversationResponseRepairPolicyError(
                    f"failure_codes.{name} must be a safe code."
                )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationResponseRepairPolicyError(f"{field} must be an object.")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationResponseRepairPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConversationResponseRepairPolicyError(f"{field} must be boolean.")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationResponseRepairPolicyError(f"{field} must be a positive integer.")
    return value


def _positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConversationResponseRepairPolicyError(f"{field} must be a positive number.")
    return float(value)


def parse_conversation_response_repair_policy(
    payload: dict[str, Any],
) -> ConversationResponseRepairPolicy:
    root = _mapping(payload, field="response-repair policy")
    expected = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "enabled",
        "boundaries",
        "limits",
        "failure_codes",
    }
    if set(root) != expected:
        raise ConversationResponseRepairPolicyError(
            "Response-repair policy fields do not match the P3.9 contract."
        )
    boundaries = _mapping(root["boundaries"], field="boundaries")
    if set(boundaries) != set(_EXPECTED_BOUNDARIES):
        raise ConversationResponseRepairPolicyError(
            "Response-repair boundary fields do not match the P3.9 contract."
        )
    parsed_boundaries = tuple(
        (
            name,
            _bool(boundaries[name], field=f"boundaries.{name}"),
        )
        for name in _EXPECTED_BOUNDARIES
    )
    if dict(parsed_boundaries) != _EXPECTED_BOUNDARIES:
        raise ConversationResponseRepairPolicyError(
            "Response-repair boundaries cannot be weakened."
        )
    limits = _mapping(root["limits"], field="limits")
    expected_limits = {
        "max_repair_attempts",
        "max_issue_codes",
        "max_repair_prompt_chars",
        "max_repair_output_tokens",
        "max_total_output_tokens",
        "max_total_elapsed_seconds",
    }
    if set(limits) != expected_limits:
        raise ConversationResponseRepairPolicyError(
            "Response-repair limits do not match the P3.9 contract."
        )
    failure_codes = _mapping(root["failure_codes"], field="failure_codes")
    if set(failure_codes) != _EXPECTED_FAILURE_KEYS:
        raise ConversationResponseRepairPolicyError(
            "Response-repair failure codes do not match the P3.9 contract."
        )
    policy = ConversationResponseRepairPolicy(
        policy_name=_text(root["policy_name"], field="policy_name"),
        version=_text(root["version"], field="version"),
        phase=_text(root["phase"], field="phase"),
        milestone=_text(root["milestone"], field="milestone"),
        status=_text(root["status"], field="status"),
        enabled=_bool(root["enabled"], field="enabled"),
        boundaries=parsed_boundaries,
        max_repair_attempts=_positive_int(
            limits["max_repair_attempts"], field="limits.max_repair_attempts"
        ),
        max_issue_codes=_positive_int(
            limits["max_issue_codes"], field="limits.max_issue_codes"
        ),
        max_repair_prompt_chars=_positive_int(
            limits["max_repair_prompt_chars"], field="limits.max_repair_prompt_chars"
        ),
        max_repair_output_tokens=_positive_int(
            limits["max_repair_output_tokens"], field="limits.max_repair_output_tokens"
        ),
        max_total_output_tokens=_positive_int(
            limits["max_total_output_tokens"], field="limits.max_total_output_tokens"
        ),
        max_total_elapsed_seconds=_positive_number(
            limits["max_total_elapsed_seconds"], field="limits.max_total_elapsed_seconds"
        ),
        failure_codes=tuple(
            sorted(
                (
                    name,
                    _text(failure_codes[name], field=f"failure_codes.{name}"),
                )
                for name in _EXPECTED_FAILURE_KEYS
            )
        ),
    )
    policy.validate()
    return policy


def load_conversation_response_repair_policy(
    path: str | Path = DEFAULT_RESPONSE_REPAIR_POLICY_PATH,
) -> ConversationResponseRepairPolicy:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationResponseRepairPolicyError(
            f"Unable to read conversation response-repair policy: {selected}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationResponseRepairPolicyError(
            f"Conversation response-repair policy is not valid JSON: {selected}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationResponseRepairPolicyError(
            "Conversation response-repair policy JSON root must be an object."
        )
    return parse_conversation_response_repair_policy(payload)
