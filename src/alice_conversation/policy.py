"""Versioned policy loading for conversational A.L.I.C.E. P3.0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationCapabilities, ConversationContractError

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_policy.json"
)


class ConversationPolicyError(ConversationContractError):
    """Raised when the public conversation policy is invalid."""


@dataclass(frozen=True)
class ConversationPolicy:
    """Validated P3.0 policy projection used by deterministic application code."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    system_contract_version: str
    capabilities: ConversationCapabilities
    default_data_classification: str
    default_retention: str
    durable_memory_promotion_path: str
    ordinary_grounding_classifications: tuple[str, ...]
    personal_claims_require_citations: bool
    prompt_injection_content_is_data: bool
    conflicts_must_be_preserved: bool
    uncertainty_must_be_visible: bool


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationPolicyError(f"{field} must be non-empty text.")
    return value


def _strict_false(value: Any, *, field: str) -> bool:
    if value is not False:
        raise ConversationPolicyError(f"{field} must remain false in P3.0.")
    return False


def _strict_true(value: Any, *, field: str) -> bool:
    if value is not True:
        raise ConversationPolicyError(f"{field} must remain true in P3.0.")
    return True


def parse_conversation_policy(payload: dict[str, Any]) -> ConversationPolicy:
    """Validate and project one decoded conversation policy."""

    boundaries = _mapping(payload.get("boundaries"), field="boundaries")
    state = _mapping(
        payload.get("conversation_state"),
        field="conversation_state",
    )
    grounding = _mapping(payload.get("grounding"), field="grounding")
    allowed_tools = payload.get("allowed_tools")
    if allowed_tools != []:
        raise ConversationPolicyError(
            "P3.0 allowed_tools must be an empty list."
        )
    capabilities = ConversationCapabilities(
        web_access_allowed=_strict_false(
            boundaries.get("web_access_allowed"),
            field="boundaries.web_access_allowed",
        ),
        tool_calling_allowed=_strict_false(
            boundaries.get("tool_calling_allowed"),
            field="boundaries.tool_calling_allowed",
        ),
        external_action_allowed=_strict_false(
            boundaries.get("external_action_allowed"),
            field="boundaries.external_action_allowed",
        ),
        memory_write_allowed=_strict_false(
            boundaries.get("memory_write_allowed"),
            field="boundaries.memory_write_allowed",
        ),
        highly_sensitive_grounding_allowed=_strict_false(
            boundaries.get("highly_sensitive_grounding_allowed"),
            field="boundaries.highly_sensitive_grounding_allowed",
        ),
        chain_of_thought_persistence_allowed=_strict_false(
            boundaries.get("chain_of_thought_persistence_allowed"),
            field="boundaries.chain_of_thought_persistence_allowed",
        ),
    )
    capabilities.validate()
    if state.get("persist_chain_of_thought") is not False:
        raise ConversationPolicyError(
            "conversation_state.persist_chain_of_thought must remain false."
        )
    classifications = grounding.get("ordinary_classifications")
    if classifications != ["PUBLIC", "INTERNAL", "PRIVATE"]:
        raise ConversationPolicyError(
            "grounding.ordinary_classifications must preserve the ordinary "
            "classification boundary."
        )
    policy = ConversationPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        system_contract_version=_text(
            payload.get("system_contract_version"),
            field="system_contract_version",
        ),
        capabilities=capabilities,
        default_data_classification=_text(
            state.get("default_data_classification"),
            field="conversation_state.default_data_classification",
        ),
        default_retention=_text(
            state.get("default_retention"),
            field="conversation_state.default_retention",
        ),
        durable_memory_promotion_path=_text(
            state.get("durable_memory_promotion_path"),
            field="conversation_state.durable_memory_promotion_path",
        ),
        ordinary_grounding_classifications=tuple(classifications),
        personal_claims_require_citations=_strict_true(
            grounding.get("personal_claims_require_citations"),
            field="grounding.personal_claims_require_citations",
        ),
        prompt_injection_content_is_data=_strict_true(
            grounding.get("prompt_injection_content_is_data"),
            field="grounding.prompt_injection_content_is_data",
        ),
        conflicts_must_be_preserved=_strict_true(
            grounding.get("conflicts_must_be_preserved"),
            field="grounding.conflicts_must_be_preserved",
        ),
        uncertainty_must_be_visible=_strict_true(
            grounding.get("uncertainty_must_be_visible"),
            field="grounding.uncertainty_must_be_visible",
        ),
    )
    if policy.phase != "3" or policy.milestone != "P3.0":
        raise ConversationPolicyError(
            "Conversation policy must be bound to Phase 3 milestone P3.0."
        )
    if policy.status != "foundation":
        raise ConversationPolicyError(
            "P3.0 conversation policy status must be foundation."
        )
    if policy.default_data_classification != "PRIVATE":
        raise ConversationPolicyError(
            "Unknown conversational content must default to PRIVATE."
        )
    if policy.default_retention != "session_only":
        raise ConversationPolicyError(
            "P3.0 conversation state must default to session-only retention."
        )
    if policy.durable_memory_promotion_path != "phase2_candidate_authorization":
        raise ConversationPolicyError(
            "Durable memory must remain behind the Phase 2 candidate and "
            "authorization boundary."
        )
    return policy


def load_conversation_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> ConversationPolicy:
    """Load and validate the versioned public conversation policy."""

    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversationPolicyError(
            f"Unable to load conversation policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationPolicyError(
            "Conversation policy root must be a JSON object."
        )
    return parse_conversation_policy(payload)
