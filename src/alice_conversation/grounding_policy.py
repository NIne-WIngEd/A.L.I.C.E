"""Versioned policy for the Phase 3 read-only grounding bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError, ORDINARY_CONVERSATION_CLASSIFICATIONS


class ConversationGroundingPolicyError(ConversationContractError):
    """Raised when the P3.3 grounding policy is invalid or weakened."""


@dataclass(frozen=True)
class ConversationGroundingPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    ordinary_classifications: tuple[str, ...]
    maximum_phase1_sources: int
    phase1_default_knowledge_status: str
    phase1_default_confidence: float
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    highly_sensitive_grounding_allowed: bool
    source_text_is_untrusted_data: bool
    preserve_conflicts: bool
    preserve_uncertainty: bool
    require_exact_citation_tokens: bool


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationGroundingPolicyError(f"{field} must be an object.")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationGroundingPolicyError(f"{field} must be non-empty text.")
    return value


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationGroundingPolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return bool(value)


def parse_conversation_grounding_policy(
    payload: dict[str, Any],
) -> ConversationGroundingPolicy:
    root = _mapping(payload, field="grounding policy")
    boundaries = _mapping(root.get("boundaries"), field="boundaries")
    phase1 = _mapping(root.get("phase1"), field="phase1")
    integrity = _mapping(root.get("integrity"), field="integrity")

    classifications_value = root.get("ordinary_classifications")
    if not isinstance(classifications_value, list):
        raise ConversationGroundingPolicyError(
            "ordinary_classifications must be an array."
        )
    classifications = tuple(str(value) for value in classifications_value)
    if classifications != ORDINARY_CONVERSATION_CLASSIFICATIONS:
        raise ConversationGroundingPolicyError(
            "Grounding policy must preserve the exact ordinary classification order."
        )

    maximum_phase1_sources = phase1.get("maximum_sources")
    if not isinstance(maximum_phase1_sources, int) or not 1 <= maximum_phase1_sources <= 12:
        raise ConversationGroundingPolicyError(
            "phase1.maximum_sources must be an integer between 1 and 12."
        )
    knowledge_status = _text(
        phase1.get("default_knowledge_status"),
        field="phase1.default_knowledge_status",
    )
    if knowledge_status != "external_claim":
        raise ConversationGroundingPolicyError(
            "Direct Phase 1 evidence must remain labeled external_claim."
        )
    try:
        default_confidence = float(phase1["default_confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversationGroundingPolicyError(
            "phase1.default_confidence must be numeric."
        ) from exc
    if default_confidence != 0.5:
        raise ConversationGroundingPolicyError(
            "Direct Phase 1 evidence confidence must remain 0.5 pending claim validation."
        )

    policy = ConversationGroundingPolicy(
        policy_name=_text(root.get("policy_name"), field="policy_name"),
        version=_text(root.get("version"), field="version"),
        phase=_text(root.get("phase"), field="phase"),
        milestone=_text(root.get("milestone"), field="milestone"),
        ordinary_classifications=classifications,
        maximum_phase1_sources=maximum_phase1_sources,
        phase1_default_knowledge_status=knowledge_status,
        phase1_default_confidence=default_confidence,
        memory_write_allowed=_strict_bool(
            boundaries.get("memory_write_allowed"),
            expected=False,
            field="boundaries.memory_write_allowed",
        ),
        external_action_allowed=_strict_bool(
            boundaries.get("external_action_allowed"),
            expected=False,
            field="boundaries.external_action_allowed",
        ),
        tool_calling_allowed=_strict_bool(
            boundaries.get("tool_calling_allowed"),
            expected=False,
            field="boundaries.tool_calling_allowed",
        ),
        web_access_allowed=_strict_bool(
            boundaries.get("web_access_allowed"),
            expected=False,
            field="boundaries.web_access_allowed",
        ),
        highly_sensitive_grounding_allowed=_strict_bool(
            boundaries.get("highly_sensitive_grounding_allowed"),
            expected=False,
            field="boundaries.highly_sensitive_grounding_allowed",
        ),
        source_text_is_untrusted_data=_strict_bool(
            integrity.get("source_text_is_untrusted_data"),
            expected=True,
            field="integrity.source_text_is_untrusted_data",
        ),
        preserve_conflicts=_strict_bool(
            integrity.get("preserve_conflicts"),
            expected=True,
            field="integrity.preserve_conflicts",
        ),
        preserve_uncertainty=_strict_bool(
            integrity.get("preserve_uncertainty"),
            expected=True,
            field="integrity.preserve_uncertainty",
        ),
        require_exact_citation_tokens=_strict_bool(
            integrity.get("require_exact_citation_tokens"),
            expected=True,
            field="integrity.require_exact_citation_tokens",
        ),
    )
    if policy.phase != "3" or policy.milestone != "P3.3":
        raise ConversationGroundingPolicyError(
            "Grounding policy must target Phase 3 milestone P3.3."
        )
    return policy


def default_conversation_grounding_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "conversation_grounding_policy.json"
    )


def load_conversation_grounding_policy(
    path: Path | None = None,
) -> ConversationGroundingPolicy:
    source = (path or default_conversation_grounding_policy_path()).expanduser().resolve(
        strict=True
    )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversationGroundingPolicyError(
            f"Conversation grounding policy could not be loaded: {source}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationGroundingPolicyError(
            "Conversation grounding policy root must be an object."
        )
    return parse_conversation_grounding_policy(payload)
