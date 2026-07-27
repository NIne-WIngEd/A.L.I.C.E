"""Versioned deterministic injection-firewall policy for Phase 4 P4.3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError
from .policy import InformationPolicy

DEFAULT_INJECTION_FIREWALL_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_injection_firewall_policy.json"
)

_ALLOWED_FINDING_CODES = (
    "boundary_collision_attempt",
    "credential_request_instruction",
    "encoded_instruction_payload",
    "memory_write_instruction",
    "permission_laundering_instruction",
    "policy_mutation_instruction",
    "policy_override_instruction",
    "private_data_exfiltration_instruction",
    "role_marker_instruction",
    "tool_execution_instruction",
    "unicode_obfuscation_detected",
)


class InformationInjectionPolicyError(InformationContractError):
    """Raised when the public P4.3 firewall policy is invalid."""


@dataclass(frozen=True)
class InformationInjectionFirewallPolicy:
    """Fail-closed deterministic firewall policy projection."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    untrusted_content_required: bool
    deterministic_detection_required: bool
    model_classifier_allowed: bool
    content_mutation_allowed: bool
    preserve_original_source_text: bool
    source_digest_binding_required: bool
    raw_excerpt_logging_allowed: bool
    clear_sources_renderable: bool
    flagged_sources_renderable: bool
    unicode_form: str
    strip_format_characters_for_detection: bool
    collapse_detection_whitespace: bool
    max_source_characters: int
    max_source_lines: int
    max_findings: int
    critical_finding_codes: tuple[str, ...]

    def validate(self, *, information_policy: InformationPolicy | None = None) -> None:
        if self.policy_name != "alice_information_injection_firewall_policy":
            raise InformationInjectionPolicyError("Unexpected P4.3 firewall policy name.")
        if self.version != "1.0.0":
            raise InformationInjectionPolicyError("P4.3 firewall policy version must be 1.0.0.")
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.3",
            "deterministic_injection_firewall",
        ):
            raise InformationInjectionPolicyError("Firewall policy milestone binding is invalid.")
        if self.permission_id != "web.search":
            raise InformationInjectionPolicyError("P4.3 must remain bound to web.search.")
        required_true = (
            self.untrusted_content_required,
            self.deterministic_detection_required,
            self.preserve_original_source_text,
            self.source_digest_binding_required,
            self.clear_sources_renderable,
            self.strip_format_characters_for_detection,
            self.collapse_detection_whitespace,
        )
        if not all(value is True for value in required_true):
            raise InformationInjectionPolicyError("Required P4.3 firewall controls must remain enabled.")
        required_false = (
            self.model_classifier_allowed,
            self.content_mutation_allowed,
            self.raw_excerpt_logging_allowed,
            self.flagged_sources_renderable,
        )
        if not all(value is False for value in required_false):
            raise InformationInjectionPolicyError("Prohibited P4.3 firewall capabilities must remain disabled.")
        if self.unicode_form != "NFKC":
            raise InformationInjectionPolicyError("P4.3 detection normalization must remain NFKC.")
        _bounded_int(self.max_source_characters, "max_source_characters", 1, 2_000_000)
        _bounded_int(self.max_source_lines, "max_source_lines", 1, 50_000)
        _bounded_int(self.max_findings, "max_findings", 1, 100)
        if self.critical_finding_codes != _ALLOWED_FINDING_CODES:
            raise InformationInjectionPolicyError("P4.3 critical finding vocabulary changed.")
        if information_policy is not None:
            information_policy.validate()
            if information_policy.retrieved_content_is_untrusted_data is not True:
                raise InformationInjectionPolicyError("Base policy must classify retrieved content as untrusted.")
            if information_policy.raw_content_logging_allowed is not False:
                raise InformationInjectionPolicyError("Base policy must prohibit raw source logging.")
            if information_policy.capabilities.external_action_allowed is not False:
                raise InformationInjectionPolicyError("External actions must remain disabled.")
            if information_policy.capabilities.memory_write_allowed is not False:
                raise InformationInjectionPolicyError("Memory writes must remain disabled.")


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InformationInjectionPolicyError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise InformationInjectionPolicyError(
            f"{field} must be between {minimum} and {maximum}."
        )
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationInjectionPolicyError(f"{field} must be an object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationInjectionPolicyError(f"{field} contains missing or unknown keys.")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationInjectionPolicyError(f"{field} must be non-empty text.")
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationInjectionPolicyError(f"{field} must remain {str(expected).lower()}.")
    return expected


def parse_information_injection_firewall_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
) -> InformationInjectionFirewallPolicy:
    """Validate and project one decoded P4.3 firewall policy."""

    if not isinstance(payload, dict):
        raise InformationInjectionPolicyError("Firewall policy root must be an object.")
    expected_root = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "permission_id",
        "untrusted_content_required",
        "deterministic_detection_required",
        "model_classifier_allowed",
        "content_mutation_allowed",
        "preserve_original_source_text",
        "source_digest_binding_required",
        "raw_excerpt_logging_allowed",
        "clear_sources_renderable",
        "flagged_sources_renderable",
        "normalization",
        "limits",
        "critical_finding_codes",
    }
    _exact_keys(payload, expected_root, "policy")
    normalization = _mapping(payload["normalization"], "normalization")
    limits = _mapping(payload["limits"], "limits")
    _exact_keys(
        normalization,
        {
            "unicode_form",
            "strip_format_characters_for_detection",
            "collapse_detection_whitespace",
        },
        "normalization",
    )
    _exact_keys(
        limits,
        {"max_source_characters", "max_source_lines", "max_findings"},
        "limits",
    )
    codes = payload["critical_finding_codes"]
    if codes != list(_ALLOWED_FINDING_CODES):
        raise InformationInjectionPolicyError("critical_finding_codes must match the approved vocabulary.")
    policy = InformationInjectionFirewallPolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        untrusted_content_required=_strict_bool(
            payload["untrusted_content_required"], "untrusted_content_required", True
        ),
        deterministic_detection_required=_strict_bool(
            payload["deterministic_detection_required"],
            "deterministic_detection_required",
            True,
        ),
        model_classifier_allowed=_strict_bool(
            payload["model_classifier_allowed"], "model_classifier_allowed", False
        ),
        content_mutation_allowed=_strict_bool(
            payload["content_mutation_allowed"], "content_mutation_allowed", False
        ),
        preserve_original_source_text=_strict_bool(
            payload["preserve_original_source_text"],
            "preserve_original_source_text",
            True,
        ),
        source_digest_binding_required=_strict_bool(
            payload["source_digest_binding_required"],
            "source_digest_binding_required",
            True,
        ),
        raw_excerpt_logging_allowed=_strict_bool(
            payload["raw_excerpt_logging_allowed"],
            "raw_excerpt_logging_allowed",
            False,
        ),
        clear_sources_renderable=_strict_bool(
            payload["clear_sources_renderable"], "clear_sources_renderable", True
        ),
        flagged_sources_renderable=_strict_bool(
            payload["flagged_sources_renderable"],
            "flagged_sources_renderable",
            False,
        ),
        unicode_form=_text(normalization["unicode_form"], "normalization.unicode_form"),
        strip_format_characters_for_detection=_strict_bool(
            normalization["strip_format_characters_for_detection"],
            "normalization.strip_format_characters_for_detection",
            True,
        ),
        collapse_detection_whitespace=_strict_bool(
            normalization["collapse_detection_whitespace"],
            "normalization.collapse_detection_whitespace",
            True,
        ),
        max_source_characters=_bounded_int(
            limits["max_source_characters"], "limits.max_source_characters", 1, 2_000_000
        ),
        max_source_lines=_bounded_int(
            limits["max_source_lines"], "limits.max_source_lines", 1, 50_000
        ),
        max_findings=_bounded_int(limits["max_findings"], "limits.max_findings", 1, 100),
        critical_finding_codes=tuple(codes),
    )
    policy.validate(information_policy=information_policy)
    return policy


def load_information_injection_firewall_policy(
    path: str | Path = DEFAULT_INJECTION_FIREWALL_POLICY_PATH,
    *,
    information_policy: InformationPolicy | None = None,
) -> InformationInjectionFirewallPolicy:
    """Load one exact public P4.3 firewall policy file."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationInjectionPolicyError("Unable to load P4.3 firewall policy.") from exc
    return parse_information_injection_firewall_policy(
        payload,
        information_policy=information_policy,
    )
