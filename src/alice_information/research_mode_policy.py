"""Explicit local research-mode policy for Phase 4 P4.7a."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicy,
)

from .contracts import InformationContractError
from .conversation_bridge_policy import InformationConversationBridgePolicy
from .research_evidence_policy import InformationResearchEvidencePolicy

DEFAULT_RESEARCH_MODE_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_research_mode_policy.json"
)
ALLOWED_MODES = ("local_only", "research")
ALLOWED_AVAILABILITY_STATES = (
    "not_requested",
    "available",
    "offline",
    "unavailable",
)
ALLOWED_RESULT_STATUSES = ("completed", "unavailable")
SOURCE_SUMMARY_FIELDS = (
    "citation_token",
    "source_id",
    "canonical_url",
    "source_content_sha256",
    "freshness_verdict",
)
APPROVED_MAX_SOURCE_SUMMARIES = 288


class InformationResearchModePolicyError(InformationContractError):
    """Raised when the public P4.7a research-mode policy is invalid."""


@dataclass(frozen=True)
class InformationResearchModePolicy:
    """Fail-closed policy for explicit local-only and research turns."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_modes: tuple[str, ...]
    allowed_availability_states: tuple[str, ...]
    allowed_result_statuses: tuple[str, ...]
    source_summary_fields: tuple[str, ...]
    max_source_summaries: int
    explicit_mode_required: bool
    unavailable_preflight_required: bool
    evidence_revalidation_required: bool
    exact_projection_required: bool
    p36_precommit_validation_required: bool
    metadata_only_source_summaries_required: bool
    silent_web_activation_allowed: bool
    local_only_web_grounding_allowed: bool
    research_without_evidence_allowed: bool
    research_execution_allowed: bool
    live_provider_registration_allowed: bool
    source_body_persistence_allowed: bool
    memory_write_allowed: bool
    phase5_storage_runtime_allowed: bool
    external_action_allowed: bool
    retry_allowed: bool
    background_execution_allowed: bool

    def validate(
        self,
        *,
        evidence_policy: InformationResearchEvidencePolicy | None = None,
        bridge_policy: InformationConversationBridgePolicy | None = None,
        response_validation_policy: ConversationResponseValidationPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_research_mode_policy":
            raise InformationResearchModePolicyError(
                "Unexpected P4.7a research-mode policy name."
            )
        if self.version != "1.0.0":
            raise InformationResearchModePolicyError(
                "P4.7a research-mode policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.7a",
            "explicit_local_research_mode",
        ):
            raise InformationResearchModePolicyError(
                "Research-mode policy milestone binding is invalid."
            )
        if self.permission_id != "web.search":
            raise InformationResearchModePolicyError(
                "P4.7a must remain bound to web.search."
            )
        if self.allowed_modes != ALLOWED_MODES:
            raise InformationResearchModePolicyError(
                "P4.7a mode vocabulary changed."
            )
        if self.allowed_availability_states != ALLOWED_AVAILABILITY_STATES:
            raise InformationResearchModePolicyError(
                "P4.7a availability vocabulary changed."
            )
        if self.allowed_result_statuses != ALLOWED_RESULT_STATUSES:
            raise InformationResearchModePolicyError(
                "P4.7a result-status vocabulary changed."
            )
        if self.source_summary_fields != SOURCE_SUMMARY_FIELDS:
            raise InformationResearchModePolicyError(
                "P4.7a source-summary schema changed."
            )
        if self.max_source_summaries != APPROVED_MAX_SOURCE_SUMMARIES:
            raise InformationResearchModePolicyError(
                "P4.7a source-summary limit changed without a policy-version change."
            )
        required_true = (
            self.explicit_mode_required,
            self.unavailable_preflight_required,
            self.evidence_revalidation_required,
            self.exact_projection_required,
            self.p36_precommit_validation_required,
            self.metadata_only_source_summaries_required,
        )
        if not all(value is True for value in required_true):
            raise InformationResearchModePolicyError(
                "Required P4.7a controls must remain enabled."
            )
        required_false = (
            self.silent_web_activation_allowed,
            self.local_only_web_grounding_allowed,
            self.research_without_evidence_allowed,
            self.research_execution_allowed,
            self.live_provider_registration_allowed,
            self.source_body_persistence_allowed,
            self.memory_write_allowed,
            self.phase5_storage_runtime_allowed,
            self.external_action_allowed,
            self.retry_allowed,
            self.background_execution_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationResearchModePolicyError(
                "Prohibited P4.7a capabilities must remain disabled."
            )
        if evidence_policy is not None:
            evidence_policy.validate()
        if bridge_policy is not None:
            bridge_policy.validate(
                response_validation_policy=response_validation_policy,
            )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InformationResearchModePolicyError(
                f"Duplicate research-mode policy key: {key}"
            )
        result[key] = value
    return result


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationResearchModePolicyError(
            f"{field} contains missing or unknown keys."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchModePolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationResearchModePolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def _exact_int(value: Any, field: str, expected: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise InformationResearchModePolicyError(
            f"{field} must equal the approved value {expected}."
        )
    return value


def parse_information_research_mode_policy(
    payload: dict[str, Any],
    *,
    evidence_policy: InformationResearchEvidencePolicy | None = None,
    bridge_policy: InformationConversationBridgePolicy | None = None,
    response_validation_policy: ConversationResponseValidationPolicy | None = None,
) -> InformationResearchModePolicy:
    """Validate and project one decoded P4.7a policy."""

    if not isinstance(payload, dict):
        raise InformationResearchModePolicyError(
            "Research-mode policy root must be an object."
        )
    expected = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "allowed_modes",
        "allowed_availability_states",
        "allowed_result_statuses",
        "source_summary_fields",
        "limits",
        "explicit_mode_required",
        "unavailable_preflight_required",
        "evidence_revalidation_required",
        "exact_projection_required",
        "p36_precommit_validation_required",
        "metadata_only_source_summaries_required",
        "silent_web_activation_allowed",
        "local_only_web_grounding_allowed",
        "research_without_evidence_allowed",
        "research_execution_allowed",
        "live_provider_registration_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "phase5_storage_runtime_allowed",
        "external_action_allowed",
        "retry_allowed",
        "background_execution_allowed",
    }
    _exact_keys(payload, expected, "policy")
    limits = payload["limits"]
    if not isinstance(limits, dict):
        raise InformationResearchModePolicyError("limits must be an object.")
    _exact_keys(limits, {"max_source_summaries"}, "limits")
    for field, expected_values in (
        ("allowed_modes", list(ALLOWED_MODES)),
        ("allowed_availability_states", list(ALLOWED_AVAILABILITY_STATES)),
        ("allowed_result_statuses", list(ALLOWED_RESULT_STATUSES)),
        ("source_summary_fields", list(SOURCE_SUMMARY_FIELDS)),
    ):
        if payload[field] != expected_values:
            raise InformationResearchModePolicyError(
                f"{field} must match the approved vocabulary."
            )
    policy = InformationResearchModePolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_modes=tuple(payload["allowed_modes"]),
        allowed_availability_states=tuple(payload["allowed_availability_states"]),
        allowed_result_statuses=tuple(payload["allowed_result_statuses"]),
        source_summary_fields=tuple(payload["source_summary_fields"]),
        max_source_summaries=_exact_int(
            limits["max_source_summaries"],
            "limits.max_source_summaries",
            APPROVED_MAX_SOURCE_SUMMARIES,
        ),
        explicit_mode_required=_strict_bool(
            payload["explicit_mode_required"],
            "explicit_mode_required",
            True,
        ),
        unavailable_preflight_required=_strict_bool(
            payload["unavailable_preflight_required"],
            "unavailable_preflight_required",
            True,
        ),
        evidence_revalidation_required=_strict_bool(
            payload["evidence_revalidation_required"],
            "evidence_revalidation_required",
            True,
        ),
        exact_projection_required=_strict_bool(
            payload["exact_projection_required"],
            "exact_projection_required",
            True,
        ),
        p36_precommit_validation_required=_strict_bool(
            payload["p36_precommit_validation_required"],
            "p36_precommit_validation_required",
            True,
        ),
        metadata_only_source_summaries_required=_strict_bool(
            payload["metadata_only_source_summaries_required"],
            "metadata_only_source_summaries_required",
            True,
        ),
        silent_web_activation_allowed=_strict_bool(
            payload["silent_web_activation_allowed"],
            "silent_web_activation_allowed",
            False,
        ),
        local_only_web_grounding_allowed=_strict_bool(
            payload["local_only_web_grounding_allowed"],
            "local_only_web_grounding_allowed",
            False,
        ),
        research_without_evidence_allowed=_strict_bool(
            payload["research_without_evidence_allowed"],
            "research_without_evidence_allowed",
            False,
        ),
        research_execution_allowed=_strict_bool(
            payload["research_execution_allowed"],
            "research_execution_allowed",
            False,
        ),
        live_provider_registration_allowed=_strict_bool(
            payload["live_provider_registration_allowed"],
            "live_provider_registration_allowed",
            False,
        ),
        source_body_persistence_allowed=_strict_bool(
            payload["source_body_persistence_allowed"],
            "source_body_persistence_allowed",
            False,
        ),
        memory_write_allowed=_strict_bool(
            payload["memory_write_allowed"],
            "memory_write_allowed",
            False,
        ),
        phase5_storage_runtime_allowed=_strict_bool(
            payload["phase5_storage_runtime_allowed"],
            "phase5_storage_runtime_allowed",
            False,
        ),
        external_action_allowed=_strict_bool(
            payload["external_action_allowed"],
            "external_action_allowed",
            False,
        ),
        retry_allowed=_strict_bool(
            payload["retry_allowed"],
            "retry_allowed",
            False,
        ),
        background_execution_allowed=_strict_bool(
            payload["background_execution_allowed"],
            "background_execution_allowed",
            False,
        ),
    )
    policy.validate(
        evidence_policy=evidence_policy,
        bridge_policy=bridge_policy,
        response_validation_policy=response_validation_policy,
    )
    return policy


def load_information_research_mode_policy(
    path: Path | str = DEFAULT_RESEARCH_MODE_POLICY_PATH,
    **dependencies: object,
) -> InformationResearchModePolicy:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    return parse_information_research_mode_policy(payload, **dependencies)
