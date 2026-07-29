"""Governed end-to-end research execution policy for Phase 4 P4.7b."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_RESEARCH_EXECUTION_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_research_execution_policy.json"
)

ALLOWED_MODES = ("local_only", "research")
ALLOWED_REQUESTED_AVAILABILITY_STATES = (
    "not_requested",
    "available",
    "offline",
    "unavailable",
)
ALLOWED_RESULT_AVAILABILITY_STATES = (
    "not_requested",
    "available",
    "offline",
    "unavailable",
)
ALLOWED_RESULT_STATUSES = ("completed", "unavailable")
ALLOWED_UNAVAILABLE_REASONS = (
    "offline",
    "unavailable",
    "research_cancelled",
    "research_failed",
    "insufficient_sources",
    "insufficient_evidence",
)


class InformationResearchExecutionPolicyError(ValueError):
    """Raised when the public P4.7b execution policy is invalid."""


@dataclass(frozen=True)
class InformationResearchExecutionPolicy:
    """Fail-closed policy for composing P4.6a, P4.6b, and P4.7a."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_modes: tuple[str, ...]
    allowed_requested_availability_states: tuple[str, ...]
    allowed_result_availability_states: tuple[str, ...]
    allowed_result_statuses: tuple[str, ...]
    allowed_unavailable_reasons: tuple[str, ...]
    explicit_mode_required: bool
    exact_provider_selection_required: bool
    deterministic_fixture_execution_required: bool
    orchestration_revalidation_required: bool
    evidence_revalidation_required: bool
    mode_adapter_revalidation_required: bool
    preconversation_failure_handling_required: bool
    local_only_provider_execution_allowed: bool
    silent_web_activation_allowed: bool
    provider_fallback_allowed: bool
    live_provider_registration_allowed: bool
    source_body_persistence_allowed: bool
    memory_write_allowed: bool
    phase5_storage_runtime_allowed: bool
    external_action_allowed: bool
    retry_allowed: bool
    recursive_browsing_allowed: bool
    background_execution_allowed: bool

    def validate(
        self,
        *,
        orchestration_policy: object | None = None,
        evidence_policy: object | None = None,
        mode_policy: object | None = None,
    ) -> None:
        if self.policy_name != "alice_information_research_execution_policy":
            raise InformationResearchExecutionPolicyError(
                "Unexpected P4.7b research-execution policy name."
            )
        if self.version != "1.0.0":
            raise InformationResearchExecutionPolicyError(
                "P4.7b research-execution policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.7b",
            "governed_research_execution",
        ):
            raise InformationResearchExecutionPolicyError(
                "Research-execution policy milestone binding is invalid."
            )
        if self.permission_id != "web.search":
            raise InformationResearchExecutionPolicyError(
                "P4.7b must remain bound to web.search."
            )
        expected_vocabularies = (
            (self.allowed_modes, ALLOWED_MODES, "mode"),
            (
                self.allowed_requested_availability_states,
                ALLOWED_REQUESTED_AVAILABILITY_STATES,
                "requested-availability",
            ),
            (
                self.allowed_result_availability_states,
                ALLOWED_RESULT_AVAILABILITY_STATES,
                "result-availability",
            ),
            (self.allowed_result_statuses, ALLOWED_RESULT_STATUSES, "result-status"),
            (
                self.allowed_unavailable_reasons,
                ALLOWED_UNAVAILABLE_REASONS,
                "unavailable-reason",
            ),
        )
        for supplied, expected, label in expected_vocabularies:
            if supplied != expected:
                raise InformationResearchExecutionPolicyError(
                    f"P4.7b {label} vocabulary changed."
                )
        required_true = (
            self.explicit_mode_required,
            self.exact_provider_selection_required,
            self.deterministic_fixture_execution_required,
            self.orchestration_revalidation_required,
            self.evidence_revalidation_required,
            self.mode_adapter_revalidation_required,
            self.preconversation_failure_handling_required,
        )
        if not all(value is True for value in required_true):
            raise InformationResearchExecutionPolicyError(
                "Required P4.7b controls must remain enabled."
            )
        required_false = (
            self.local_only_provider_execution_allowed,
            self.silent_web_activation_allowed,
            self.provider_fallback_allowed,
            self.live_provider_registration_allowed,
            self.source_body_persistence_allowed,
            self.memory_write_allowed,
            self.phase5_storage_runtime_allowed,
            self.external_action_allowed,
            self.retry_allowed,
            self.recursive_browsing_allowed,
            self.background_execution_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationResearchExecutionPolicyError(
                "Prohibited P4.7b capabilities must remain disabled."
            )
        if orchestration_policy is not None:
            validator = getattr(orchestration_policy, "validate", None)
            if not callable(validator):
                raise InformationResearchExecutionPolicyError(
                    "P4.7b requires a valid orchestration policy."
                )
            validator()
            if (
                getattr(orchestration_policy, "deterministic_fixture_only", None)
                is not True
                or getattr(orchestration_policy, "provider_fallback_allowed", None)
                is not False
                or getattr(orchestration_policy, "live_provider_registration_allowed", None)
                is not False
            ):
                raise InformationResearchExecutionPolicyError(
                    "P4.7b requires the exact fixture-only no-fallback orchestration profile."
                )
        if evidence_policy is not None:
            validator = getattr(evidence_policy, "validate", None)
            if not callable(validator):
                raise InformationResearchExecutionPolicyError(
                    "P4.7b requires a valid evidence policy."
                )
            validator(orchestration_policy=orchestration_policy)
        if mode_policy is not None:
            validator = getattr(mode_policy, "validate", None)
            if not callable(validator):
                raise InformationResearchExecutionPolicyError(
                    "P4.7b requires a valid research-mode policy."
                )
            validator(evidence_policy=evidence_policy)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InformationResearchExecutionPolicyError(
                f"Duplicate research-execution policy key: {key}"
            )
        result[key] = value
    return result


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationResearchExecutionPolicyError(
            f"{field} contains missing or unknown keys."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchExecutionPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationResearchExecutionPolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def _exact_list(value: Any, expected: tuple[str, ...], field: str) -> tuple[str, ...]:
    if value != list(expected):
        raise InformationResearchExecutionPolicyError(
            f"{field} must match the approved vocabulary."
        )
    return expected


def parse_information_research_execution_policy(
    payload: dict[str, Any],
    *,
    orchestration_policy: object | None = None,
    evidence_policy: object | None = None,
    mode_policy: object | None = None,
) -> InformationResearchExecutionPolicy:
    """Validate and project one decoded P4.7b policy."""

    if not isinstance(payload, dict):
        raise InformationResearchExecutionPolicyError(
            "Research-execution policy root must be an object."
        )
    expected = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "allowed_modes",
        "allowed_requested_availability_states",
        "allowed_result_availability_states",
        "allowed_result_statuses",
        "allowed_unavailable_reasons",
        "explicit_mode_required",
        "exact_provider_selection_required",
        "deterministic_fixture_execution_required",
        "orchestration_revalidation_required",
        "evidence_revalidation_required",
        "mode_adapter_revalidation_required",
        "preconversation_failure_handling_required",
        "local_only_provider_execution_allowed",
        "silent_web_activation_allowed",
        "provider_fallback_allowed",
        "live_provider_registration_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "phase5_storage_runtime_allowed",
        "external_action_allowed",
        "retry_allowed",
        "recursive_browsing_allowed",
        "background_execution_allowed",
    }
    _exact_keys(payload, expected, "policy")
    policy = InformationResearchExecutionPolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_modes=_exact_list(payload["allowed_modes"], ALLOWED_MODES, "allowed_modes"),
        allowed_requested_availability_states=_exact_list(
            payload["allowed_requested_availability_states"],
            ALLOWED_REQUESTED_AVAILABILITY_STATES,
            "allowed_requested_availability_states",
        ),
        allowed_result_availability_states=_exact_list(
            payload["allowed_result_availability_states"],
            ALLOWED_RESULT_AVAILABILITY_STATES,
            "allowed_result_availability_states",
        ),
        allowed_result_statuses=_exact_list(
            payload["allowed_result_statuses"],
            ALLOWED_RESULT_STATUSES,
            "allowed_result_statuses",
        ),
        allowed_unavailable_reasons=_exact_list(
            payload["allowed_unavailable_reasons"],
            ALLOWED_UNAVAILABLE_REASONS,
            "allowed_unavailable_reasons",
        ),
        explicit_mode_required=_strict_bool(payload["explicit_mode_required"], "explicit_mode_required", True),
        exact_provider_selection_required=_strict_bool(payload["exact_provider_selection_required"], "exact_provider_selection_required", True),
        deterministic_fixture_execution_required=_strict_bool(payload["deterministic_fixture_execution_required"], "deterministic_fixture_execution_required", True),
        orchestration_revalidation_required=_strict_bool(payload["orchestration_revalidation_required"], "orchestration_revalidation_required", True),
        evidence_revalidation_required=_strict_bool(payload["evidence_revalidation_required"], "evidence_revalidation_required", True),
        mode_adapter_revalidation_required=_strict_bool(payload["mode_adapter_revalidation_required"], "mode_adapter_revalidation_required", True),
        preconversation_failure_handling_required=_strict_bool(payload["preconversation_failure_handling_required"], "preconversation_failure_handling_required", True),
        local_only_provider_execution_allowed=_strict_bool(payload["local_only_provider_execution_allowed"], "local_only_provider_execution_allowed", False),
        silent_web_activation_allowed=_strict_bool(payload["silent_web_activation_allowed"], "silent_web_activation_allowed", False),
        provider_fallback_allowed=_strict_bool(payload["provider_fallback_allowed"], "provider_fallback_allowed", False),
        live_provider_registration_allowed=_strict_bool(payload["live_provider_registration_allowed"], "live_provider_registration_allowed", False),
        source_body_persistence_allowed=_strict_bool(payload["source_body_persistence_allowed"], "source_body_persistence_allowed", False),
        memory_write_allowed=_strict_bool(payload["memory_write_allowed"], "memory_write_allowed", False),
        phase5_storage_runtime_allowed=_strict_bool(payload["phase5_storage_runtime_allowed"], "phase5_storage_runtime_allowed", False),
        external_action_allowed=_strict_bool(payload["external_action_allowed"], "external_action_allowed", False),
        retry_allowed=_strict_bool(payload["retry_allowed"], "retry_allowed", False),
        recursive_browsing_allowed=_strict_bool(payload["recursive_browsing_allowed"], "recursive_browsing_allowed", False),
        background_execution_allowed=_strict_bool(payload["background_execution_allowed"], "background_execution_allowed", False),
    )
    policy.validate(
        orchestration_policy=orchestration_policy,
        evidence_policy=evidence_policy,
        mode_policy=mode_policy,
    )
    return policy


def load_information_research_execution_policy(
    path: Path | str = DEFAULT_RESEARCH_EXECUTION_POLICY_PATH,
    **dependencies: object,
) -> InformationResearchExecutionPolicy:
    """Load one duplicate-key-safe P4.7b policy document."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except InformationResearchExecutionPolicyError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationResearchExecutionPolicyError(
            "Research-execution policy could not be loaded."
        ) from exc
    return parse_information_research_execution_policy(payload, **dependencies)
