"""Versioned public policy loading for A.L.I.C.E. Phase 4 P4.0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationCapabilities, InformationContractError

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "policies" / "information_policy.json"
)


class InformationPolicyError(InformationContractError):
    """Raised when the public P4.0 information policy is invalid."""


@dataclass(frozen=True)
class InformationPolicy:
    """Validated P4.0 policy projection for deterministic application code."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    capabilities: InformationCapabilities
    allowed_operations: tuple[str, ...]
    approved_live_providers: tuple[str, ...]
    deterministic_fixture_mode_allowed: bool
    allowed_query_classifications: tuple[str, ...]
    allowed_schemes: tuple[str, ...]
    max_search_calls: int
    max_fetch_calls: int
    max_sources: int
    max_redirects: int
    request_timeout_seconds: float
    total_timeout_seconds: float
    max_response_bytes: int
    foreground_only: bool
    retrieved_content_is_untrusted_data: bool
    activity_logging_required: bool
    raw_query_logging_allowed: bool
    raw_content_logging_allowed: bool

    def validate(self) -> None:
        """Revalidate a projected policy at every trust-boundary entry."""

        if self.policy_name != "alice_information_policy":
            raise InformationPolicyError(
                "P4.0 policy_name must be alice_information_policy."
            )
        if self.version != "1.0.0":
            raise InformationPolicyError(
                "P4.0 information policy version must be 1.0.0."
            )
        if self.phase != "4" or self.milestone != "P4.0":
            raise InformationPolicyError(
                "Information policy must be bound to Phase 4 milestone P4.0."
            )
        if self.status != "foundation":
            raise InformationPolicyError(
                "P4.0 information policy status must be foundation."
            )
        if self.permission_id != "web.search":
            raise InformationPolicyError(
                "Phase 4 read-only information access must map to web.search."
            )
        self.capabilities.validate()
        if self.allowed_operations != ("search", "fetch"):
            raise InformationPolicyError(
                "P4.0 allowed_operations must remain search and fetch."
            )
        if self.approved_live_providers != ():
            raise InformationPolicyError(
                "P4.0 approved_live_providers must remain empty."
            )
        if self.deterministic_fixture_mode_allowed is not True:
            raise InformationPolicyError(
                "P4.0 deterministic fixture mode must remain enabled."
            )
        if self.allowed_query_classifications != ("PUBLIC",):
            raise InformationPolicyError(
                "P4.0 external query classification must remain PUBLIC-only."
            )
        if self.allowed_schemes != ("http", "https"):
            raise InformationPolicyError(
                "P4.0 allowed URL schemes must remain HTTP and HTTPS."
            )
        _bounded_int(
            self.max_search_calls,
            field="max_search_calls",
            minimum=1,
            maximum=10,
        )
        _bounded_int(
            self.max_fetch_calls,
            field="max_fetch_calls",
            minimum=1,
            maximum=20,
        )
        _bounded_int(
            self.max_sources,
            field="max_sources",
            minimum=1,
            maximum=20,
        )
        _bounded_int(
            self.max_redirects,
            field="max_redirects",
            minimum=0,
            maximum=5,
        )
        _bounded_number(
            self.request_timeout_seconds,
            field="request_timeout_seconds",
            minimum=1,
            maximum=30,
        )
        _bounded_number(
            self.total_timeout_seconds,
            field="total_timeout_seconds",
            minimum=1,
            maximum=120,
        )
        _bounded_int(
            self.max_response_bytes,
            field="max_response_bytes",
            minimum=1,
            maximum=5_000_000,
        )
        if self.total_timeout_seconds < self.request_timeout_seconds:
            raise InformationPolicyError(
                "Total research timeout cannot be smaller than request timeout."
            )
        if self.foreground_only is not True:
            raise InformationPolicyError(
                "P4.0 information access must remain foreground-only."
            )
        if self.retrieved_content_is_untrusted_data is not True:
            raise InformationPolicyError(
                "P4.0 retrieved content must remain untrusted data."
            )
        if self.activity_logging_required is not True:
            raise InformationPolicyError(
                "P4.0 activity logging must remain required."
            )
        if self.raw_query_logging_allowed is not False:
            raise InformationPolicyError(
                "P4.0 raw query logging must remain disabled."
            )
        if self.raw_content_logging_allowed is not False:
            raise InformationPolicyError(
                "P4.0 raw content logging must remain disabled."
            )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_false(value: Any, *, field: str) -> bool:
    if value is not False:
        raise InformationPolicyError(f"{field} must remain false in P4.0.")
    return False


def _strict_true(value: Any, *, field: str) -> bool:
    if value is not True:
        raise InformationPolicyError(f"{field} must remain true in P4.0.")
    return True


def _exact_text_list(value: Any, *, field: str, expected: list[str]) -> tuple[str, ...]:
    if value != expected:
        raise InformationPolicyError(
            f"{field} must equal the approved P4.0 value: {expected!r}."
        )
    return tuple(expected)


def _bounded_int(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InformationPolicyError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise InformationPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def _bounded_number(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InformationPolicyError(f"{field} must be numeric.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise InformationPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return result


def parse_information_policy(payload: dict[str, Any]) -> InformationPolicy:
    """Validate and project one decoded P4.0 information policy."""

    boundaries = _mapping(payload.get("boundaries"), field="boundaries")
    transmission = _mapping(
        payload.get("query_transmission"),
        field="query_transmission",
    )
    retrieval = _mapping(payload.get("source_retrieval"), field="source_retrieval")
    untrusted = _mapping(payload.get("untrusted_content"), field="untrusted_content")
    budgets = _mapping(payload.get("budgets"), field="budgets")
    logging = _mapping(payload.get("logging"), field="logging")

    capabilities = InformationCapabilities(
        live_network_access_allowed=_strict_false(
            boundaries.get("live_network_access_allowed"),
            field="boundaries.live_network_access_allowed",
        ),
        external_action_allowed=_strict_false(
            boundaries.get("external_action_allowed"),
            field="boundaries.external_action_allowed",
        ),
        memory_write_allowed=_strict_false(
            boundaries.get("memory_write_allowed"),
            field="boundaries.memory_write_allowed",
        ),
        background_monitoring_allowed=_strict_false(
            boundaries.get("background_monitoring_allowed"),
            field="boundaries.background_monitoring_allowed",
        ),
        authenticated_browsing_allowed=_strict_false(
            retrieval.get("authenticated_browsing_allowed"),
            field="source_retrieval.authenticated_browsing_allowed",
        ),
        javascript_execution_allowed=_strict_false(
            retrieval.get("javascript_execution_allowed"),
            field="source_retrieval.javascript_execution_allowed",
        ),
        form_submission_allowed=_strict_false(
            retrieval.get("form_submission_allowed"),
            field="source_retrieval.form_submission_allowed",
        ),
        arbitrary_code_execution_allowed=_strict_false(
            boundaries.get("arbitrary_code_execution_allowed"),
            field="boundaries.arbitrary_code_execution_allowed",
        ),
        provider_fallback_allowed=_strict_false(
            boundaries.get("provider_fallback_allowed"),
            field="boundaries.provider_fallback_allowed",
        ),
        chain_of_thought_persistence_allowed=_strict_false(
            boundaries.get("chain_of_thought_persistence_allowed"),
            field="boundaries.chain_of_thought_persistence_allowed",
        ),
    )
    capabilities.validate()

    approved_live_providers = payload.get("approved_live_providers")
    if approved_live_providers != []:
        raise InformationPolicyError(
            "P4.0 approved_live_providers must remain an empty list."
        )

    for field_name in (
        "private_context_allowed",
        "highly_sensitive_allowed",
        "secrets_allowed",
    ):
        _strict_false(
            transmission.get(field_name),
            field=f"query_transmission.{field_name}",
        )
    for field_name in (
        "cookies_allowed",
        "downloads_allowed",
        "recursive_browsing_allowed",
    ):
        _strict_false(
            retrieval.get(field_name),
            field=f"source_retrieval.{field_name}",
        )
    for field_name in (
        "can_modify_policy",
        "can_grant_permission",
        "can_request_credentials",
        "can_trigger_tools",
        "can_trigger_actions",
        "can_write_memory",
    ):
        _strict_false(
            untrusted.get(field_name),
            field=f"untrusted_content.{field_name}",
        )

    policy = InformationPolicy(
        policy_name=_text(payload.get("policy_name"), field="policy_name"),
        version=_text(payload.get("version"), field="version"),
        phase=_text(payload.get("phase"), field="phase"),
        milestone=_text(payload.get("milestone"), field="milestone"),
        status=_text(payload.get("status"), field="status"),
        permission_id=_text(payload.get("permission_id"), field="permission_id"),
        capabilities=capabilities,
        allowed_operations=_exact_text_list(
            payload.get("allowed_operations"),
            field="allowed_operations",
            expected=["search", "fetch"],
        ),
        approved_live_providers=(),
        deterministic_fixture_mode_allowed=_strict_true(
            payload.get("deterministic_fixture_mode_allowed"),
            field="deterministic_fixture_mode_allowed",
        ),
        allowed_query_classifications=_exact_text_list(
            transmission.get("allowed_classifications"),
            field="query_transmission.allowed_classifications",
            expected=["PUBLIC"],
        ),
        allowed_schemes=_exact_text_list(
            retrieval.get("allowed_schemes"),
            field="source_retrieval.allowed_schemes",
            expected=["http", "https"],
        ),
        max_search_calls=_bounded_int(
            budgets.get("max_search_calls"),
            field="budgets.max_search_calls",
            minimum=1,
            maximum=10,
        ),
        max_fetch_calls=_bounded_int(
            budgets.get("max_fetch_calls"),
            field="budgets.max_fetch_calls",
            minimum=1,
            maximum=20,
        ),
        max_sources=_bounded_int(
            budgets.get("max_sources"),
            field="budgets.max_sources",
            minimum=1,
            maximum=20,
        ),
        max_redirects=_bounded_int(
            budgets.get("max_redirects"),
            field="budgets.max_redirects",
            minimum=0,
            maximum=5,
        ),
        request_timeout_seconds=_bounded_number(
            budgets.get("request_timeout_seconds"),
            field="budgets.request_timeout_seconds",
            minimum=1,
            maximum=30,
        ),
        total_timeout_seconds=_bounded_number(
            budgets.get("total_timeout_seconds"),
            field="budgets.total_timeout_seconds",
            minimum=1,
            maximum=120,
        ),
        max_response_bytes=_bounded_int(
            budgets.get("max_response_bytes"),
            field="budgets.max_response_bytes",
            minimum=1,
            maximum=5_000_000,
        ),
        foreground_only=_strict_true(
            budgets.get("foreground_only"),
            field="budgets.foreground_only",
        ),
        retrieved_content_is_untrusted_data=_strict_true(
            untrusted.get("treated_as_untrusted_data"),
            field="untrusted_content.treated_as_untrusted_data",
        ),
        activity_logging_required=_strict_true(
            logging.get("activity_logging_required"),
            field="logging.activity_logging_required",
        ),
        raw_query_logging_allowed=_strict_false(
            logging.get("raw_query_logging_allowed"),
            field="logging.raw_query_logging_allowed",
        ),
        raw_content_logging_allowed=_strict_false(
            logging.get("raw_content_logging_allowed"),
            field="logging.raw_content_logging_allowed",
        ),
    )

    policy.validate()
    return policy


def load_information_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> InformationPolicy:
    """Load and validate the versioned public P4.0 information policy."""

    policy_path = Path(path)
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationPolicyError(
            f"Unable to load information policy: {policy_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise InformationPolicyError(
            "Information policy root must be a JSON object."
        )
    return parse_information_policy(payload)
