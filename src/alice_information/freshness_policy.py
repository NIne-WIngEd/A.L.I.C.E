"""Versioned deterministic freshness policy for Phase 4 P4.4a."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy

DEFAULT_FRESHNESS_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_freshness_policy.json"
)

ALLOWED_TEMPORAL_INTENTS = (
    "current",
    "latest",
    "recent",
    "historical",
    "time_insensitive",
)
ALLOWED_FRESHNESS_VERDICTS = (
    "fresh",
    "stale",
    "unknown",
    "historical_match",
    "historical_mismatch",
    "time_insensitive",
)

APPROVED_AGE_LIMITS_SECONDS = {
    "current": 86400,
    "latest": 604800,
    "recent": 2592000,
}
APPROVED_MAX_CLOCK_SKEW_SECONDS = 300


class InformationFreshnessPolicyError(InformationContractError):
    """Raised when the public P4.4a freshness policy is invalid."""


@dataclass(frozen=True)
class InformationFreshnessPolicy:
    """Fail-closed deterministic freshness and temporal policy projection."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_intents: tuple[str, ...]
    allowed_verdicts: tuple[str, ...]
    current_max_age_seconds: int
    latest_max_age_seconds: int
    recent_max_age_seconds: int
    max_clock_skew_seconds: int
    deterministic_query_classification_required: bool
    ambiguous_temporal_queries_fail_closed: bool
    retrieval_time_is_freshness_evidence: bool
    require_source_time_for_time_sensitive_claims: bool
    updated_time_preferred: bool
    future_source_time_allowed: bool
    historical_window_required: bool
    time_insensitive_without_source_time_allowed: bool
    model_temporal_inference_allowed: bool
    raw_temporal_metadata_logging_allowed: bool
    source_digest_binding_required: bool
    firewall_clear_required: bool

    def validate(
        self,
        *,
        information_policy: InformationPolicy | None = None,
        firewall_policy: InformationInjectionFirewallPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_freshness_policy":
            raise InformationFreshnessPolicyError("Unexpected P4.4a freshness policy name.")
        if self.version != "1.0.0":
            raise InformationFreshnessPolicyError("P4.4a freshness policy version must be 1.0.0.")
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.4a",
            "deterministic_freshness_temporal_contracts",
        ):
            raise InformationFreshnessPolicyError("Freshness policy milestone binding is invalid.")
        if self.permission_id != "web.search":
            raise InformationFreshnessPolicyError("P4.4 must remain bound to web.search.")
        if self.allowed_intents != ALLOWED_TEMPORAL_INTENTS:
            raise InformationFreshnessPolicyError("P4.4a temporal intent vocabulary changed.")
        if self.allowed_verdicts != ALLOWED_FRESHNESS_VERDICTS:
            raise InformationFreshnessPolicyError("P4.4 freshness verdict vocabulary changed.")
        _bounded_int(self.current_max_age_seconds, "current_max_age_seconds", 60, 604800)
        _bounded_int(self.latest_max_age_seconds, "latest_max_age_seconds", 60, 2592000)
        _bounded_int(self.recent_max_age_seconds, "recent_max_age_seconds", 60, 7776000)
        _bounded_int(self.max_clock_skew_seconds, "max_clock_skew_seconds", 0, 3600)
        if not (
            self.current_max_age_seconds
            <= self.latest_max_age_seconds
            <= self.recent_max_age_seconds
        ):
            raise InformationFreshnessPolicyError("P4.4a age limits must be monotonically increasing.")
        actual_age_limits = {
            "current": self.current_max_age_seconds,
            "latest": self.latest_max_age_seconds,
            "recent": self.recent_max_age_seconds,
        }
        if actual_age_limits != APPROVED_AGE_LIMITS_SECONDS:
            raise InformationFreshnessPolicyError(
                "P4.4a age limits changed without a policy-version change."
            )
        if self.max_clock_skew_seconds != APPROVED_MAX_CLOCK_SKEW_SECONDS:
            raise InformationFreshnessPolicyError(
                "P4.4a clock skew changed without a policy-version change."
            )
        required_true = (
            self.deterministic_query_classification_required,
            self.ambiguous_temporal_queries_fail_closed,
            self.require_source_time_for_time_sensitive_claims,
            self.updated_time_preferred,
            self.historical_window_required,
            self.time_insensitive_without_source_time_allowed,
            self.source_digest_binding_required,
            self.firewall_clear_required,
        )
        if not all(value is True for value in required_true):
            raise InformationFreshnessPolicyError("Required P4.4a controls must remain enabled.")
        required_false = (
            self.retrieval_time_is_freshness_evidence,
            self.future_source_time_allowed,
            self.model_temporal_inference_allowed,
            self.raw_temporal_metadata_logging_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationFreshnessPolicyError("Prohibited P4.4a capabilities must remain disabled.")
        if information_policy is not None:
            information_policy.validate()
            if information_policy.raw_content_logging_allowed is not False:
                raise InformationFreshnessPolicyError("Base policy must prohibit raw source logging.")
            if information_policy.capabilities.external_action_allowed is not False:
                raise InformationFreshnessPolicyError("External actions must remain disabled.")
            if information_policy.capabilities.memory_write_allowed is not False:
                raise InformationFreshnessPolicyError("Memory writes must remain disabled.")
        if firewall_policy is not None:
            firewall_policy.validate(information_policy=information_policy)
            if firewall_policy.clear_sources_renderable is not True:
                raise InformationFreshnessPolicyError("P4.4a requires clear firewall sources.")
            if firewall_policy.flagged_sources_renderable is not False:
                raise InformationFreshnessPolicyError("P4.4a cannot admit flagged sources.")

    def max_age_seconds(self, intent: str) -> int:
        self.validate()
        if intent == "current":
            return self.current_max_age_seconds
        if intent == "latest":
            return self.latest_max_age_seconds
        if intent == "recent":
            return self.recent_max_age_seconds
        raise InformationFreshnessPolicyError("The requested intent does not use an age limit.")


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InformationFreshnessPolicyError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise InformationFreshnessPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationFreshnessPolicyError(f"{field} must be an object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationFreshnessPolicyError(f"{field} contains missing or unknown keys.")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationFreshnessPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationFreshnessPolicyError(f"{field} must remain {str(expected).lower()}.")
    return expected


def parse_information_freshness_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
    firewall_policy: InformationInjectionFirewallPolicy | None = None,
) -> InformationFreshnessPolicy:
    """Validate and project one decoded P4.4a freshness policy."""

    if not isinstance(payload, dict):
        raise InformationFreshnessPolicyError("Freshness policy root must be an object.")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "allowed_intents",
        "allowed_verdicts",
        "age_limits_seconds",
        "max_clock_skew_seconds",
        "deterministic_query_classification_required",
        "ambiguous_temporal_queries_fail_closed",
        "retrieval_time_is_freshness_evidence",
        "require_source_time_for_time_sensitive_claims",
        "updated_time_preferred",
        "future_source_time_allowed",
        "historical_window_required",
        "time_insensitive_without_source_time_allowed",
        "model_temporal_inference_allowed",
        "raw_temporal_metadata_logging_allowed",
        "source_digest_binding_required",
        "firewall_clear_required",
    }
    _exact_keys(payload, expected_root, "policy")
    age_limits = _mapping(payload["age_limits_seconds"], "age_limits_seconds")
    _exact_keys(age_limits, {"current", "latest", "recent"}, "age_limits_seconds")
    if payload["allowed_intents"] != list(ALLOWED_TEMPORAL_INTENTS):
        raise InformationFreshnessPolicyError("allowed_intents must match the approved vocabulary.")
    if payload["allowed_verdicts"] != list(ALLOWED_FRESHNESS_VERDICTS):
        raise InformationFreshnessPolicyError("allowed_verdicts must match the approved vocabulary.")
    policy = InformationFreshnessPolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_intents=tuple(payload["allowed_intents"]),
        allowed_verdicts=tuple(payload["allowed_verdicts"]),
        current_max_age_seconds=_bounded_int(age_limits["current"], "age_limits_seconds.current", 60, 604800),
        latest_max_age_seconds=_bounded_int(age_limits["latest"], "age_limits_seconds.latest", 60, 2592000),
        recent_max_age_seconds=_bounded_int(age_limits["recent"], "age_limits_seconds.recent", 60, 7776000),
        max_clock_skew_seconds=_bounded_int(payload["max_clock_skew_seconds"], "max_clock_skew_seconds", 0, 3600),
        deterministic_query_classification_required=_strict_bool(payload["deterministic_query_classification_required"], "deterministic_query_classification_required", True),
        ambiguous_temporal_queries_fail_closed=_strict_bool(payload["ambiguous_temporal_queries_fail_closed"], "ambiguous_temporal_queries_fail_closed", True),
        retrieval_time_is_freshness_evidence=_strict_bool(payload["retrieval_time_is_freshness_evidence"], "retrieval_time_is_freshness_evidence", False),
        require_source_time_for_time_sensitive_claims=_strict_bool(payload["require_source_time_for_time_sensitive_claims"], "require_source_time_for_time_sensitive_claims", True),
        updated_time_preferred=_strict_bool(payload["updated_time_preferred"], "updated_time_preferred", True),
        future_source_time_allowed=_strict_bool(payload["future_source_time_allowed"], "future_source_time_allowed", False),
        historical_window_required=_strict_bool(payload["historical_window_required"], "historical_window_required", True),
        time_insensitive_without_source_time_allowed=_strict_bool(payload["time_insensitive_without_source_time_allowed"], "time_insensitive_without_source_time_allowed", True),
        model_temporal_inference_allowed=_strict_bool(payload["model_temporal_inference_allowed"], "model_temporal_inference_allowed", False),
        raw_temporal_metadata_logging_allowed=_strict_bool(payload["raw_temporal_metadata_logging_allowed"], "raw_temporal_metadata_logging_allowed", False),
        source_digest_binding_required=_strict_bool(payload["source_digest_binding_required"], "source_digest_binding_required", True),
        firewall_clear_required=_strict_bool(payload["firewall_clear_required"], "firewall_clear_required", True),
    )
    policy.validate(information_policy=information_policy, firewall_policy=firewall_policy)
    return policy


def load_information_freshness_policy(
    path: Path | str = DEFAULT_FRESHNESS_POLICY_PATH,
    *,
    information_policy: InformationPolicy | None = None,
    firewall_policy: InformationInjectionFirewallPolicy | None = None,
) -> InformationFreshnessPolicy:
    """Load the exact public P4.4a freshness policy from disk."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationFreshnessPolicyError("Could not load the P4.4a freshness policy.") from exc
    return parse_information_freshness_policy(
        payload,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
    )
