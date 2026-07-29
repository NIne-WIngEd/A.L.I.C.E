"""Versioned controlled research-evidence policy for Phase 4 P4.6b."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import InformationContractError
from .freshness_policy import InformationFreshnessPolicy
from .grounding_policy import InformationGroundingPolicy
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy
from .research_orchestration_policy import InformationResearchOrchestrationPolicy

DEFAULT_RESEARCH_EVIDENCE_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_research_evidence_policy.json"
)
ALLOWED_RESEARCH_INPUT_OUTCOMES = ("completed", "partial")
ALLOWED_PIPELINE_OUTCOMES = (
    "answerable",
    "conflict",
    "uncertain",
    "insufficient_sources",
)
ALLOWED_SOURCE_DISPOSITIONS = (
    "qualified",
    "blocked_injection",
    "freshness_rejected",
)
APPROVED_MAX_SOURCES = 12
APPROVED_MAX_CLAIMS = 24


class InformationResearchEvidencePolicyError(InformationContractError):
    """Raised when the public P4.6b evidence-pipeline policy is invalid."""


@dataclass(frozen=True)
class InformationResearchEvidencePolicy:
    """Fail-closed policy for composing verified Phase 4 evidence gates."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    permission_id: str
    allowed_research_input_outcomes: tuple[str, ...]
    allowed_pipeline_outcomes: tuple[str, ...]
    allowed_source_dispositions: tuple[str, ...]
    max_sources: int
    max_claims: int
    research_run_revalidation_required: bool
    injection_inspection_required: bool
    freshness_assessment_required: bool
    grounding_revalidation_required: bool
    partial_research_preserved: bool
    rejected_source_metadata_preserved: bool
    raw_source_logging_allowed: bool
    source_body_persistence_allowed: bool
    live_provider_registration_allowed: bool
    phase3_runtime_registration_allowed: bool
    memory_write_allowed: bool
    phase5_storage_runtime_allowed: bool
    external_action_allowed: bool
    background_execution_allowed: bool
    model_claim_generation_allowed: bool
    semantic_entailment_inference_allowed: bool

    def validate(
        self,
        *,
        information_policy: InformationPolicy | None = None,
        orchestration_policy: InformationResearchOrchestrationPolicy | None = None,
        firewall_policy: InformationInjectionFirewallPolicy | None = None,
        freshness_policy: InformationFreshnessPolicy | None = None,
        grounding_policy: InformationGroundingPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_research_evidence_policy":
            raise InformationResearchEvidencePolicyError(
                "Unexpected P4.6b research-evidence policy name."
            )
        if self.version != "1.0.0":
            raise InformationResearchEvidencePolicyError(
                "P4.6b research-evidence policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.6b",
            "controlled_research_evidence_pipeline",
        ):
            raise InformationResearchEvidencePolicyError(
                "Research-evidence policy milestone binding is invalid."
            )
        if self.permission_id != "web.search":
            raise InformationResearchEvidencePolicyError(
                "P4.6b must remain bound to web.search."
            )
        if self.allowed_research_input_outcomes != ALLOWED_RESEARCH_INPUT_OUTCOMES:
            raise InformationResearchEvidencePolicyError(
                "P4.6b research-input outcome vocabulary changed."
            )
        if self.allowed_pipeline_outcomes != ALLOWED_PIPELINE_OUTCOMES:
            raise InformationResearchEvidencePolicyError(
                "P4.6b pipeline outcome vocabulary changed."
            )
        if self.allowed_source_dispositions != ALLOWED_SOURCE_DISPOSITIONS:
            raise InformationResearchEvidencePolicyError(
                "P4.6b source disposition vocabulary changed."
            )
        if self.max_sources != APPROVED_MAX_SOURCES or self.max_claims != APPROVED_MAX_CLAIMS:
            raise InformationResearchEvidencePolicyError(
                "P4.6b approved evidence limits changed without a policy-version change."
            )
        required_true = (
            self.research_run_revalidation_required,
            self.injection_inspection_required,
            self.freshness_assessment_required,
            self.grounding_revalidation_required,
            self.partial_research_preserved,
            self.rejected_source_metadata_preserved,
        )
        if not all(value is True for value in required_true):
            raise InformationResearchEvidencePolicyError(
                "Required P4.6b controls must remain enabled."
            )
        required_false = (
            self.raw_source_logging_allowed,
            self.source_body_persistence_allowed,
            self.live_provider_registration_allowed,
            self.phase3_runtime_registration_allowed,
            self.memory_write_allowed,
            self.phase5_storage_runtime_allowed,
            self.external_action_allowed,
            self.background_execution_allowed,
            self.model_claim_generation_allowed,
            self.semantic_entailment_inference_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationResearchEvidencePolicyError(
                "Prohibited P4.6b capabilities must remain disabled."
            )
        if information_policy is not None:
            information_policy.validate()
            if information_policy.raw_content_logging_allowed is not False:
                raise InformationResearchEvidencePolicyError(
                    "Base information policy must prohibit raw source logging."
                )
        if orchestration_policy is not None:
            orchestration_policy.validate()
        if firewall_policy is not None:
            firewall_policy.validate(information_policy=information_policy)
        if freshness_policy is not None:
            freshness_policy.validate(
                information_policy=information_policy,
                firewall_policy=firewall_policy,
            )
        if grounding_policy is not None:
            grounding_policy.validate(
                information_policy=information_policy,
                firewall_policy=firewall_policy,
                freshness_policy=freshness_policy,
            )



def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InformationResearchEvidencePolicyError(
                f"Duplicate research-evidence policy key: {key}"
            )
        result[key] = value
    return result

def _exact_keys(mapping: dict[str, Any], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise InformationResearchEvidencePolicyError(
            f"{field} contains missing or unknown keys."
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationResearchEvidencePolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, field: str, expected: bool) -> bool:
    if value is not expected:
        raise InformationResearchEvidencePolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def _exact_int(value: Any, field: str, expected: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise InformationResearchEvidencePolicyError(
            f"{field} must equal the approved value {expected}."
        )
    return value


def parse_information_research_evidence_policy(
    payload: dict[str, Any],
    *,
    information_policy: InformationPolicy | None = None,
    orchestration_policy: InformationResearchOrchestrationPolicy | None = None,
    firewall_policy: InformationInjectionFirewallPolicy | None = None,
    freshness_policy: InformationFreshnessPolicy | None = None,
    grounding_policy: InformationGroundingPolicy | None = None,
) -> InformationResearchEvidencePolicy:
    """Validate and project one decoded P4.6b policy."""

    if not isinstance(payload, dict):
        raise InformationResearchEvidencePolicyError(
            "Research-evidence policy root must be an object."
        )
    expected = {
        "policy_name", "version", "phase", "milestone", "status",
        "permission_id", "allowed_research_input_outcomes",
        "allowed_pipeline_outcomes", "allowed_source_dispositions", "limits",
        "research_run_revalidation_required", "injection_inspection_required",
        "freshness_assessment_required", "grounding_revalidation_required",
        "partial_research_preserved", "rejected_source_metadata_preserved",
        "raw_source_logging_allowed", "source_body_persistence_allowed",
        "live_provider_registration_allowed", "phase3_runtime_registration_allowed",
        "memory_write_allowed", "phase5_storage_runtime_allowed",
        "external_action_allowed", "background_execution_allowed",
        "model_claim_generation_allowed", "semantic_entailment_inference_allowed",
    }
    _exact_keys(payload, expected, "policy")
    limits = payload["limits"]
    if not isinstance(limits, dict):
        raise InformationResearchEvidencePolicyError("limits must be an object.")
    _exact_keys(limits, {"max_sources", "max_claims"}, "limits")
    for field, expected_values in (
        ("allowed_research_input_outcomes", list(ALLOWED_RESEARCH_INPUT_OUTCOMES)),
        ("allowed_pipeline_outcomes", list(ALLOWED_PIPELINE_OUTCOMES)),
        ("allowed_source_dispositions", list(ALLOWED_SOURCE_DISPOSITIONS)),
    ):
        if payload[field] != expected_values:
            raise InformationResearchEvidencePolicyError(
                f"{field} must match the approved vocabulary."
            )
    policy = InformationResearchEvidencePolicy(
        policy_name=_text(payload["policy_name"], "policy_name"),
        version=_text(payload["version"], "version"),
        phase=_text(payload["phase"], "phase"),
        milestone=_text(payload["milestone"], "milestone"),
        status=_text(payload["status"], "status"),
        permission_id=_text(payload["permission_id"], "permission_id"),
        allowed_research_input_outcomes=tuple(payload["allowed_research_input_outcomes"]),
        allowed_pipeline_outcomes=tuple(payload["allowed_pipeline_outcomes"]),
        allowed_source_dispositions=tuple(payload["allowed_source_dispositions"]),
        max_sources=_exact_int(limits["max_sources"], "limits.max_sources", APPROVED_MAX_SOURCES),
        max_claims=_exact_int(limits["max_claims"], "limits.max_claims", APPROVED_MAX_CLAIMS),
        research_run_revalidation_required=_strict_bool(payload["research_run_revalidation_required"], "research_run_revalidation_required", True),
        injection_inspection_required=_strict_bool(payload["injection_inspection_required"], "injection_inspection_required", True),
        freshness_assessment_required=_strict_bool(payload["freshness_assessment_required"], "freshness_assessment_required", True),
        grounding_revalidation_required=_strict_bool(payload["grounding_revalidation_required"], "grounding_revalidation_required", True),
        partial_research_preserved=_strict_bool(payload["partial_research_preserved"], "partial_research_preserved", True),
        rejected_source_metadata_preserved=_strict_bool(payload["rejected_source_metadata_preserved"], "rejected_source_metadata_preserved", True),
        raw_source_logging_allowed=_strict_bool(payload["raw_source_logging_allowed"], "raw_source_logging_allowed", False),
        source_body_persistence_allowed=_strict_bool(payload["source_body_persistence_allowed"], "source_body_persistence_allowed", False),
        live_provider_registration_allowed=_strict_bool(payload["live_provider_registration_allowed"], "live_provider_registration_allowed", False),
        phase3_runtime_registration_allowed=_strict_bool(payload["phase3_runtime_registration_allowed"], "phase3_runtime_registration_allowed", False),
        memory_write_allowed=_strict_bool(payload["memory_write_allowed"], "memory_write_allowed", False),
        phase5_storage_runtime_allowed=_strict_bool(payload["phase5_storage_runtime_allowed"], "phase5_storage_runtime_allowed", False),
        external_action_allowed=_strict_bool(payload["external_action_allowed"], "external_action_allowed", False),
        background_execution_allowed=_strict_bool(payload["background_execution_allowed"], "background_execution_allowed", False),
        model_claim_generation_allowed=_strict_bool(payload["model_claim_generation_allowed"], "model_claim_generation_allowed", False),
        semantic_entailment_inference_allowed=_strict_bool(payload["semantic_entailment_inference_allowed"], "semantic_entailment_inference_allowed", False),
    )
    policy.validate(
        information_policy=information_policy,
        orchestration_policy=orchestration_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
        grounding_policy=grounding_policy,
    )
    return policy


def load_information_research_evidence_policy(
    path: Path | str = DEFAULT_RESEARCH_EVIDENCE_POLICY_PATH,
    **dependencies: object,
) -> InformationResearchEvidencePolicy:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    return parse_information_research_evidence_policy(payload, **dependencies)
