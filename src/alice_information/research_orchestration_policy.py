"""Versioned P4.6a policy for deterministic foreground research orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError

DEFAULT_RESEARCH_ORCHESTRATION_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_research_orchestration_policy.json"
)


class InformationResearchOrchestrationPolicyError(InformationContractError):
    """Raised when the P4.6a orchestration policy is invalid."""


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchOrchestrationPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationResearchOrchestrationPolicyError(
            f"{field} must be an object."
        )
    return value


def _strict_bool(value: Any, *, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationResearchOrchestrationPolicyError(
            f"{field} must remain {str(expected).lower()} in P4.6a."
        )
    return expected


def _exact_int(value: Any, *, field: str, expected: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise InformationResearchOrchestrationPolicyError(
            f"{field} must equal {expected} in P4.6a."
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InformationResearchOrchestrationPolicyError(
                f"Duplicate orchestration policy key: {key}"
            )
        result[key] = value
    return result


@dataclass(frozen=True)
class InformationResearchOrchestrationPolicy:
    """Exact fixture-only P4.6a orchestration policy."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    exact_search_provider_required: bool
    exact_fetch_provider_required: bool
    deterministic_fixture_only: bool
    live_provider_registration_allowed: bool
    provider_fallback_allowed: bool
    foreground_only: bool
    query_rewriting_allowed: bool
    recursive_browsing_allowed: bool
    arbitrary_link_following_allowed: bool
    retries_allowed: bool
    canonical_url_deduplication_required: bool
    partial_results_preserved: bool
    activity_records_required: bool
    raw_query_in_activity_allowed: bool
    raw_source_content_in_activity_allowed: bool
    max_search_calls: int
    max_fetch_calls: int
    max_sources: int
    max_response_bytes: int
    max_request_timeout_seconds: int
    max_total_timeout_seconds: int

    def validate(self) -> None:
        if self.policy_name != "alice_information_research_orchestration_policy":
            raise InformationResearchOrchestrationPolicyError(
                "P4.6a policy_name must be alice_information_research_orchestration_policy."
            )
        if self.version != "1.0.0":
            raise InformationResearchOrchestrationPolicyError(
                "P4.6a orchestration policy version must be 1.0.0."
            )
        if self.phase != "4" or self.milestone != "P4.6a":
            raise InformationResearchOrchestrationPolicyError(
                "Orchestration policy must be bound to Phase 4 milestone P4.6a."
            )
        if self.status != "deterministic_research_orchestration_foundation":
            raise InformationResearchOrchestrationPolicyError(
                "P4.6a status must be deterministic_research_orchestration_foundation."
            )
        for field_name, value, expected in (
            ("exact_search_provider_required", self.exact_search_provider_required, True),
            ("exact_fetch_provider_required", self.exact_fetch_provider_required, True),
            ("deterministic_fixture_only", self.deterministic_fixture_only, True),
            ("live_provider_registration_allowed", self.live_provider_registration_allowed, False),
            ("provider_fallback_allowed", self.provider_fallback_allowed, False),
            ("foreground_only", self.foreground_only, True),
            ("query_rewriting_allowed", self.query_rewriting_allowed, False),
            ("recursive_browsing_allowed", self.recursive_browsing_allowed, False),
            ("arbitrary_link_following_allowed", self.arbitrary_link_following_allowed, False),
            ("retries_allowed", self.retries_allowed, False),
            ("canonical_url_deduplication_required", self.canonical_url_deduplication_required, True),
            ("partial_results_preserved", self.partial_results_preserved, True),
            ("activity_records_required", self.activity_records_required, True),
            ("raw_query_in_activity_allowed", self.raw_query_in_activity_allowed, False),
            ("raw_source_content_in_activity_allowed", self.raw_source_content_in_activity_allowed, False),
        ):
            if value is not expected:
                raise InformationResearchOrchestrationPolicyError(
                    f"{field_name} must remain {str(expected).lower()} in P4.6a."
                )
        for field_name, value, expected in (
            ("max_search_calls", self.max_search_calls, 1),
            ("max_fetch_calls", self.max_fetch_calls, 8),
            ("max_sources", self.max_sources, 8),
            ("max_response_bytes", self.max_response_bytes, 2_000_000),
            ("max_request_timeout_seconds", self.max_request_timeout_seconds, 10),
            ("max_total_timeout_seconds", self.max_total_timeout_seconds, 45),
        ):
            _exact_int(value, field=field_name, expected=expected)

    def validate_request_budget(self, request: object) -> None:
        """Reject a request that exceeds the exact P4.6a orchestration envelope."""

        self.validate()
        limits = (
            ("max_search_calls", self.max_search_calls),
            ("max_fetch_calls", self.max_fetch_calls),
            ("max_sources", self.max_sources),
            ("request_timeout_seconds", self.max_request_timeout_seconds),
            ("total_timeout_seconds", self.max_total_timeout_seconds),
        )
        for field_name, maximum in limits:
            value = getattr(request, field_name, None)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InformationResearchOrchestrationPolicyError(
                    f"Research request {field_name} must be numeric."
                )
            if value > maximum:
                raise InformationResearchOrchestrationPolicyError(
                    f"Research request {field_name} exceeds the P4.6a policy limit."
                )


def parse_information_research_orchestration_policy(
    payload: dict[str, Any],
) -> InformationResearchOrchestrationPolicy:
    """Validate and project one decoded P4.6a orchestration policy."""

    provider = _mapping(payload.get("provider_selection"), field="provider_selection")
    execution = _mapping(payload.get("execution"), field="execution")
    budgets = _mapping(payload.get("budgets"), field="budgets")
    policy = InformationResearchOrchestrationPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        exact_search_provider_required=_strict_bool(
            provider.get("exact_search_provider_required"),
            field="provider_selection.exact_search_provider_required",
            expected=True,
        ),
        exact_fetch_provider_required=_strict_bool(
            provider.get("exact_fetch_provider_required"),
            field="provider_selection.exact_fetch_provider_required",
            expected=True,
        ),
        deterministic_fixture_only=_strict_bool(
            provider.get("deterministic_fixture_only"),
            field="provider_selection.deterministic_fixture_only",
            expected=True,
        ),
        live_provider_registration_allowed=_strict_bool(
            provider.get("live_provider_registration_allowed"),
            field="provider_selection.live_provider_registration_allowed",
            expected=False,
        ),
        provider_fallback_allowed=_strict_bool(
            provider.get("provider_fallback_allowed"),
            field="provider_selection.provider_fallback_allowed",
            expected=False,
        ),
        foreground_only=_strict_bool(
            execution.get("foreground_only"),
            field="execution.foreground_only",
            expected=True,
        ),
        query_rewriting_allowed=_strict_bool(
            execution.get("query_rewriting_allowed"),
            field="execution.query_rewriting_allowed",
            expected=False,
        ),
        recursive_browsing_allowed=_strict_bool(
            execution.get("recursive_browsing_allowed"),
            field="execution.recursive_browsing_allowed",
            expected=False,
        ),
        arbitrary_link_following_allowed=_strict_bool(
            execution.get("arbitrary_link_following_allowed"),
            field="execution.arbitrary_link_following_allowed",
            expected=False,
        ),
        retries_allowed=_strict_bool(
            execution.get("retries_allowed"),
            field="execution.retries_allowed",
            expected=False,
        ),
        canonical_url_deduplication_required=_strict_bool(
            execution.get("canonical_url_deduplication_required"),
            field="execution.canonical_url_deduplication_required",
            expected=True,
        ),
        partial_results_preserved=_strict_bool(
            execution.get("partial_results_preserved"),
            field="execution.partial_results_preserved",
            expected=True,
        ),
        activity_records_required=_strict_bool(
            execution.get("activity_records_required"),
            field="execution.activity_records_required",
            expected=True,
        ),
        raw_query_in_activity_allowed=_strict_bool(
            execution.get("raw_query_in_activity_allowed"),
            field="execution.raw_query_in_activity_allowed",
            expected=False,
        ),
        raw_source_content_in_activity_allowed=_strict_bool(
            execution.get("raw_source_content_in_activity_allowed"),
            field="execution.raw_source_content_in_activity_allowed",
            expected=False,
        ),
        max_search_calls=_exact_int(
            budgets.get("max_search_calls"), field="budgets.max_search_calls", expected=1
        ),
        max_fetch_calls=_exact_int(
            budgets.get("max_fetch_calls"), field="budgets.max_fetch_calls", expected=8
        ),
        max_sources=_exact_int(
            budgets.get("max_sources"), field="budgets.max_sources", expected=8
        ),
        max_response_bytes=_exact_int(
            budgets.get("max_response_bytes"),
            field="budgets.max_response_bytes",
            expected=2_000_000,
        ),
        max_request_timeout_seconds=_exact_int(
            budgets.get("max_request_timeout_seconds"),
            field="budgets.max_request_timeout_seconds",
            expected=10,
        ),
        max_total_timeout_seconds=_exact_int(
            budgets.get("max_total_timeout_seconds"),
            field="budgets.max_total_timeout_seconds",
            expected=45,
        ),
    )
    policy.validate()
    return policy


def load_information_research_orchestration_policy(
    path: str | Path = DEFAULT_RESEARCH_ORCHESTRATION_POLICY_PATH,
) -> InformationResearchOrchestrationPolicy:
    """Load and validate the repository P4.6a orchestration policy."""

    policy_path = Path(path)
    try:
        payload = json.loads(
            policy_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except InformationResearchOrchestrationPolicyError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationResearchOrchestrationPolicyError(
            f"Unable to load research orchestration policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise InformationResearchOrchestrationPolicyError(
            "Research orchestration policy root must be a JSON object."
        )
    return parse_information_research_orchestration_policy(payload)
