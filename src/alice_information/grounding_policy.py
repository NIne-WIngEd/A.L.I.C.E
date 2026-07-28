"""Versioned deterministic citation-grounding policy for Phase 4 P4.5a."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError
from .freshness_policy import InformationFreshnessPolicy
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy

DEFAULT_GROUNDING_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_grounding_policy.json"
)

ALLOWED_GROUNDING_OUTCOMES = (
    "answerable",
    "conflict",
    "uncertain",
    "insufficient_sources",
)
ALLOWED_GROUNDING_KNOWLEDGE_STATUSES = (
    "external_claim",
    "verified_fact",
    "uncertain",
    "disputed",
    "historical",
)
APPROVED_MAX_SOURCES = 12
APPROVED_MAX_CLAIMS = 24
APPROVED_MAX_SUPPORT_SPAN_CHARACTERS = 2000
APPROVED_MIN_SOURCE_CHARACTERS = 20
APPROVED_VERIFIED_FACT_MIN_DISTINCT_DOMAINS = 2
APPROVED_CONFLICT_MIN_DISTINCT_DOMAINS = 2
APPROVED_CITATION_TOKEN_PREFIX = "[WEB:"
APPROVED_CITATION_TOKEN_SUFFIX = "]"


class InformationGroundingPolicyError(InformationContractError):
    """Raised when the public P4.5a grounding policy is invalid."""


@dataclass(frozen=True)
class InformationGroundingPolicy:
    """Fail-closed policy for extractive, citation-bound web grounding."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_outcomes: tuple[str, ...]
    allowed_knowledge_statuses: tuple[str, ...]
    max_sources: int
    max_claims: int
    max_support_span_characters: int
    min_source_characters: int
    verified_fact_min_distinct_domains: int
    conflict_min_distinct_domains: int
    require_https_sources: bool
    require_clear_firewall: bool
    require_freshness_support: bool
    require_exact_support_span: bool
    require_all_packet_sources_cited: bool
    allow_unused_sources: bool
    allow_model_claim_generation: bool
    allow_semantic_entailment_inference: bool
    allow_publisher_reputation_inference: bool
    raw_support_logging_allowed: bool
    source_digest_binding_required: bool
    query_digest_binding_required: bool
    citation_token_prefix: str
    citation_token_suffix: str

    def validate(
        self,
        *,
        information_policy: InformationPolicy | None = None,
        firewall_policy: InformationInjectionFirewallPolicy | None = None,
        freshness_policy: InformationFreshnessPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_grounding_policy":
            raise InformationGroundingPolicyError(
                "Unexpected P4.5a grounding policy name."
            )
        if self.version != "1.0.0":
            raise InformationGroundingPolicyError(
                "P4.5a grounding policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.5a",
            "deterministic_citation_grounding",
        ):
            raise InformationGroundingPolicyError(
                "Grounding policy milestone binding is invalid."
            )
        if self.permission_id != "web.search":
            raise InformationGroundingPolicyError(
                "P4.5a must remain bound to web.search."
            )
        if self.allowed_outcomes != ALLOWED_GROUNDING_OUTCOMES:
            raise InformationGroundingPolicyError(
                "Grounding outcome vocabulary changed."
            )
        if self.allowed_knowledge_statuses != ALLOWED_GROUNDING_KNOWLEDGE_STATUSES:
            raise InformationGroundingPolicyError(
                "Grounding knowledge-status vocabulary changed."
            )
        approved_numbers = (
            (self.max_sources, APPROVED_MAX_SOURCES, "max_sources"),
            (self.max_claims, APPROVED_MAX_CLAIMS, "max_claims"),
            (
                self.max_support_span_characters,
                APPROVED_MAX_SUPPORT_SPAN_CHARACTERS,
                "max_support_span_characters",
            ),
            (
                self.min_source_characters,
                APPROVED_MIN_SOURCE_CHARACTERS,
                "min_source_characters",
            ),
            (
                self.verified_fact_min_distinct_domains,
                APPROVED_VERIFIED_FACT_MIN_DISTINCT_DOMAINS,
                "verified_fact_min_distinct_domains",
            ),
            (
                self.conflict_min_distinct_domains,
                APPROVED_CONFLICT_MIN_DISTINCT_DOMAINS,
                "conflict_min_distinct_domains",
            ),
        )
        for value, expected, field in approved_numbers:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value != expected
            ):
                raise InformationGroundingPolicyError(
                    f"{field} changed without a grounding-policy version change."
                )
        required_true = (
            self.require_https_sources,
            self.require_clear_firewall,
            self.require_freshness_support,
            self.require_exact_support_span,
            self.require_all_packet_sources_cited,
            self.source_digest_binding_required,
            self.query_digest_binding_required,
        )
        if not all(value is True for value in required_true):
            raise InformationGroundingPolicyError(
                "Required P4.5a controls must remain enabled."
            )
        required_false = (
            self.allow_unused_sources,
            self.allow_model_claim_generation,
            self.allow_semantic_entailment_inference,
            self.allow_publisher_reputation_inference,
            self.raw_support_logging_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationGroundingPolicyError(
                "Prohibited P4.5a capabilities must remain disabled."
            )
        if (
            self.citation_token_prefix != APPROVED_CITATION_TOKEN_PREFIX
            or self.citation_token_suffix != APPROVED_CITATION_TOKEN_SUFFIX
        ):
            raise InformationGroundingPolicyError(
                "Citation token syntax changed without a version change."
            )
        if information_policy is not None:
            information_policy.validate()
            if information_policy.raw_content_logging_allowed is not False:
                raise InformationGroundingPolicyError(
                    "Base policy must prohibit raw content logging."
                )
            if information_policy.capabilities.external_action_allowed is not False:
                raise InformationGroundingPolicyError(
                    "External actions must remain disabled."
                )
            if information_policy.capabilities.memory_write_allowed is not False:
                raise InformationGroundingPolicyError(
                    "Memory writes must remain disabled."
                )
        if firewall_policy is not None:
            firewall_policy.validate(information_policy=information_policy)
            if firewall_policy.clear_sources_renderable is not True:
                raise InformationGroundingPolicyError(
                    "P4.5a requires clear firewall sources."
                )
        if freshness_policy is not None:
            freshness_policy.validate(
                information_policy=information_policy,
                firewall_policy=firewall_policy,
            )
            if (
                freshness_policy.require_source_time_for_time_sensitive_claims
                is not True
            ):
                raise InformationGroundingPolicyError(
                    "P4.5a requires freshness enforcement."
                )


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationGroundingPolicyError(
            f"{field} contains missing or unknown keys."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationGroundingPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _exact_int(value: Any, expected: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise InformationGroundingPolicyError(f"{field} must equal {expected}.")
    return value


def _strict_bool(value: Any, expected: bool, field: str) -> bool:
    if value is not expected:
        raise InformationGroundingPolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def parse_information_grounding_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
    firewall_policy: InformationInjectionFirewallPolicy | None = None,
    freshness_policy: InformationFreshnessPolicy | None = None,
) -> InformationGroundingPolicy:
    """Validate and project one decoded P4.5a grounding policy."""

    if not isinstance(payload, dict):
        raise InformationGroundingPolicyError(
            "Grounding policy root must be an object."
        )
    expected = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "allowed_outcomes",
        "allowed_knowledge_statuses",
        "max_sources",
        "max_claims",
        "max_support_span_characters",
        "min_source_characters",
        "verified_fact_min_distinct_domains",
        "conflict_min_distinct_domains",
        "require_https_sources",
        "require_clear_firewall",
        "require_freshness_support",
        "require_exact_support_span",
        "require_all_packet_sources_cited",
        "allow_unused_sources",
        "allow_model_claim_generation",
        "allow_semantic_entailment_inference",
        "allow_publisher_reputation_inference",
        "raw_support_logging_allowed",
        "source_digest_binding_required",
        "query_digest_binding_required",
        "citation_token_prefix",
        "citation_token_suffix",
    }
    _exact_keys(payload, expected, "policy")
    if payload["allowed_outcomes"] != list(ALLOWED_GROUNDING_OUTCOMES):
        raise InformationGroundingPolicyError(
            "allowed_outcomes must match the approved vocabulary."
        )
    if payload["allowed_knowledge_statuses"] != list(
        ALLOWED_GROUNDING_KNOWLEDGE_STATUSES
    ):
        raise InformationGroundingPolicyError(
            "allowed_knowledge_statuses must match the approved vocabulary."
        )
    policy = InformationGroundingPolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_outcomes=tuple(payload["allowed_outcomes"]),
        allowed_knowledge_statuses=tuple(
            payload["allowed_knowledge_statuses"]
        ),
        max_sources=_exact_int(
            payload["max_sources"], APPROVED_MAX_SOURCES, "max_sources"
        ),
        max_claims=_exact_int(
            payload["max_claims"], APPROVED_MAX_CLAIMS, "max_claims"
        ),
        max_support_span_characters=_exact_int(
            payload["max_support_span_characters"],
            APPROVED_MAX_SUPPORT_SPAN_CHARACTERS,
            "max_support_span_characters",
        ),
        min_source_characters=_exact_int(
            payload["min_source_characters"],
            APPROVED_MIN_SOURCE_CHARACTERS,
            "min_source_characters",
        ),
        verified_fact_min_distinct_domains=_exact_int(
            payload["verified_fact_min_distinct_domains"],
            APPROVED_VERIFIED_FACT_MIN_DISTINCT_DOMAINS,
            "verified_fact_min_distinct_domains",
        ),
        conflict_min_distinct_domains=_exact_int(
            payload["conflict_min_distinct_domains"],
            APPROVED_CONFLICT_MIN_DISTINCT_DOMAINS,
            "conflict_min_distinct_domains",
        ),
        require_https_sources=_strict_bool(
            payload["require_https_sources"], True, "require_https_sources"
        ),
        require_clear_firewall=_strict_bool(
            payload["require_clear_firewall"], True, "require_clear_firewall"
        ),
        require_freshness_support=_strict_bool(
            payload["require_freshness_support"],
            True,
            "require_freshness_support",
        ),
        require_exact_support_span=_strict_bool(
            payload["require_exact_support_span"],
            True,
            "require_exact_support_span",
        ),
        require_all_packet_sources_cited=_strict_bool(
            payload["require_all_packet_sources_cited"],
            True,
            "require_all_packet_sources_cited",
        ),
        allow_unused_sources=_strict_bool(
            payload["allow_unused_sources"], False, "allow_unused_sources"
        ),
        allow_model_claim_generation=_strict_bool(
            payload["allow_model_claim_generation"],
            False,
            "allow_model_claim_generation",
        ),
        allow_semantic_entailment_inference=_strict_bool(
            payload["allow_semantic_entailment_inference"],
            False,
            "allow_semantic_entailment_inference",
        ),
        allow_publisher_reputation_inference=_strict_bool(
            payload["allow_publisher_reputation_inference"],
            False,
            "allow_publisher_reputation_inference",
        ),
        raw_support_logging_allowed=_strict_bool(
            payload["raw_support_logging_allowed"],
            False,
            "raw_support_logging_allowed",
        ),
        source_digest_binding_required=_strict_bool(
            payload["source_digest_binding_required"],
            True,
            "source_digest_binding_required",
        ),
        query_digest_binding_required=_strict_bool(
            payload["query_digest_binding_required"],
            True,
            "query_digest_binding_required",
        ),
        citation_token_prefix=_text(
            payload["citation_token_prefix"], "citation_token_prefix"
        ),
        citation_token_suffix=_text(
            payload["citation_token_suffix"], "citation_token_suffix"
        ),
    )
    policy.validate(
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
    )
    return policy


def load_information_grounding_policy(
    path: str | Path | None = None,
    *,
    information_policy: InformationPolicy | None = None,
    firewall_policy: InformationInjectionFirewallPolicy | None = None,
    freshness_policy: InformationFreshnessPolicy | None = None,
) -> InformationGroundingPolicy:
    """Load the P4.5a policy from disk with strict decoding and validation."""

    policy_path = Path(path) if path is not None else DEFAULT_GROUNDING_POLICY_PATH
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationGroundingPolicyError(
            "Grounding policy could not be loaded."
        ) from exc
    return parse_information_grounding_policy(
        payload,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
    )
