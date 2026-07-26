"""Versioned fail-closed generated-response validation policy for P3.6."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ConversationContractError


DEFAULT_RESPONSE_VALIDATION_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "conversation_response_validation_policy.json"
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
    "automatic_repair_allowed": False,
    "provider_fallback_allowed": False,
}
_EXPECTED_CITATION_RULES = {
    "require_exact_tokens": True,
    "reject_unknown_tokens": True,
    "require_grounded_personal_claims": True,
    "require_supported_factual_claims": True,
}
_EXPECTED_EPISTEMIC_RULES = {
    "preserve_conflict": True,
    "preserve_uncertainty": True,
    "require_abstention_on_insufficient_evidence": True,
    "require_abstention_on_denied": True,
    "require_abstention_on_not_applicable": True,
    "reject_certainty_language_for_conflict": True,
    "reject_certainty_language_for_uncertainty": True,
}
_EXPECTED_SAFETY_RULES = {
    "reject_action_completion_claims": True,
    "reject_capability_claims": True,
    "reject_invented_personal_facts": True,
    "reject_dependency_language": True,
    "reject_hidden_reasoning_disclosure": True,
    "reject_truncated_responses": True,
}
_EXPECTED_FAILURE_KEYS = {"rejected", "internal"}


class ConversationResponseValidationPolicyError(ConversationContractError):
    """Raised when the P3.6 response-validation policy is weakened."""


@dataclass(frozen=True)
class ConversationResponseValidationPolicy:
    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    boundaries: tuple[tuple[str, bool], ...]
    citation_rules: tuple[tuple[str, bool], ...]
    minimum_answerable_claims_cited: int
    minimum_conflict_claims_cited: int
    epistemic_rules: tuple[tuple[str, bool], ...]
    safety_rules: tuple[tuple[str, bool], ...]
    max_response_chars: int
    max_issues: int
    failure_codes: tuple[tuple[str, str], ...]

    def boundary(self, name: str) -> bool:
        return dict(self.boundaries)[name]

    def citation_rule(self, name: str) -> bool:
        return dict(self.citation_rules)[name]

    def epistemic_rule(self, name: str) -> bool:
        return dict(self.epistemic_rules)[name]

    def safety_rule(self, name: str) -> bool:
        return dict(self.safety_rules)[name]

    def failure_code(self, name: str) -> str:
        return dict(self.failure_codes)[name]


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationResponseValidationPolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationResponseValidationPolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise ConversationResponseValidationPolicyError(
            f"{field} must remain {str(expected).lower()} in P3.6."
        )
    return expected


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationResponseValidationPolicyError(
            f"{field} must be a positive integer."
        )
    return value


def _safe_code(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not _CODE_PATTERN.fullmatch(text):
        raise ConversationResponseValidationPolicyError(
            f"{field} must be a safe validation code."
        )
    return text


def _exact_boolean_map(
    value: Any,
    *,
    field: str,
    expected: dict[str, bool],
) -> tuple[tuple[str, bool], ...]:
    selected = _mapping(value, field=field)
    if set(selected) != set(expected):
        raise ConversationResponseValidationPolicyError(
            f"{field} fields do not match the P3.6 contract."
        )
    return tuple(
        (
            name,
            _strict_bool(
                selected[name], expected=required, field=f"{field}.{name}"
            ),
        )
        for name, required in expected.items()
    )


def parse_conversation_response_validation_policy(
    payload: dict[str, Any],
) -> ConversationResponseValidationPolicy:
    root = _mapping(payload, field="response-validation policy")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "boundaries",
        "citations",
        "epistemic",
        "safety",
        "limits",
        "failure_codes",
    }
    if set(root) != expected_root:
        raise ConversationResponseValidationPolicyError(
            "Response-validation policy fields do not match the P3.6 contract."
        )
    if _text(root["policy_name"], field="policy_name") != (
        "alice_conversation_response_validation_policy"
    ):
        raise ConversationResponseValidationPolicyError(
            "Unexpected response-validation policy name."
        )
    if _text(root["phase"], field="phase") != "3":
        raise ConversationResponseValidationPolicyError(
            "Response-validation policy phase must be 3."
        )
    if _text(root["milestone"], field="milestone") != "P3.6":
        raise ConversationResponseValidationPolicyError(
            "Response-validation policy milestone must be P3.6."
        )
    if _text(root["status"], field="status") != "generated_response_validation":
        raise ConversationResponseValidationPolicyError(
            "Unexpected response-validation policy status."
        )

    boundaries = _exact_boolean_map(
        root["boundaries"], field="boundaries", expected=_EXPECTED_BOUNDARIES
    )

    citations = _mapping(root["citations"], field="citations")
    citation_keys = set(_EXPECTED_CITATION_RULES) | {
        "minimum_answerable_claims_cited",
        "minimum_conflict_claims_cited",
    }
    if set(citations) != citation_keys:
        raise ConversationResponseValidationPolicyError(
            "Citation policy fields do not match the P3.6 contract."
        )
    citation_rules = tuple(
        (
            name,
            _strict_bool(
                citations[name],
                expected=required,
                field=f"citations.{name}",
            ),
        )
        for name, required in _EXPECTED_CITATION_RULES.items()
    )
    minimum_answerable = _positive_int(
        citations["minimum_answerable_claims_cited"],
        field="citations.minimum_answerable_claims_cited",
    )
    if minimum_answerable != 1:
        raise ConversationResponseValidationPolicyError(
            "P3.6 requires exactly one minimum cited claim for answerable grounding."
        )
    minimum_conflict = _positive_int(
        citations["minimum_conflict_claims_cited"],
        field="citations.minimum_conflict_claims_cited",
    )
    if minimum_conflict != 2:
        raise ConversationResponseValidationPolicyError(
            "P3.6 requires at least two distinct cited claims for conflict grounding."
        )

    epistemic_rules = _exact_boolean_map(
        root["epistemic"], field="epistemic", expected=_EXPECTED_EPISTEMIC_RULES
    )
    safety_rules = _exact_boolean_map(
        root["safety"], field="safety", expected=_EXPECTED_SAFETY_RULES
    )

    limits = _mapping(root["limits"], field="limits")
    if set(limits) != {"max_response_chars", "max_issues"}:
        raise ConversationResponseValidationPolicyError(
            "Response-validation limits do not match the P3.6 contract."
        )
    max_response_chars = _positive_int(
        limits["max_response_chars"], field="limits.max_response_chars"
    )
    if not 1024 <= max_response_chars <= 100_000:
        raise ConversationResponseValidationPolicyError(
            "limits.max_response_chars must be between 1024 and 100000."
        )
    max_issues = _positive_int(limits["max_issues"], field="limits.max_issues")
    if not 1 <= max_issues <= 256:
        raise ConversationResponseValidationPolicyError(
            "limits.max_issues must be between 1 and 256."
        )

    failure_codes = _mapping(root["failure_codes"], field="failure_codes")
    if set(failure_codes) != _EXPECTED_FAILURE_KEYS:
        raise ConversationResponseValidationPolicyError(
            "Response-validation failure codes do not match the P3.6 contract."
        )
    parsed_failure_codes = tuple(
        (
            name,
            _safe_code(failure_codes[name], field=f"failure_codes.{name}"),
        )
        for name in sorted(_EXPECTED_FAILURE_KEYS)
    )
    values = [value for _, value in parsed_failure_codes]
    if len(values) != len(set(values)):
        raise ConversationResponseValidationPolicyError(
            "Response-validation failure codes cannot be duplicated."
        )

    return ConversationResponseValidationPolicy(
        policy_name="alice_conversation_response_validation_policy",
        version=_text(root["version"], field="version"),
        phase="3",
        milestone="P3.6",
        status="generated_response_validation",
        boundaries=boundaries,
        citation_rules=citation_rules,
        minimum_answerable_claims_cited=minimum_answerable,
        minimum_conflict_claims_cited=minimum_conflict,
        epistemic_rules=epistemic_rules,
        safety_rules=safety_rules,
        max_response_chars=max_response_chars,
        max_issues=max_issues,
        failure_codes=parsed_failure_codes,
    )


def load_conversation_response_validation_policy(
    path: str | Path = DEFAULT_RESPONSE_VALIDATION_POLICY_PATH,
) -> ConversationResponseValidationPolicy:
    selected = Path(path)
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationResponseValidationPolicyError(
            f"Unable to read conversation response-validation policy: {selected}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationResponseValidationPolicyError(
            f"Conversation response-validation policy is not valid JSON: {selected}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationResponseValidationPolicyError(
            "Conversation response-validation policy JSON root must be an object."
        )
    return parse_conversation_response_validation_policy(payload)
