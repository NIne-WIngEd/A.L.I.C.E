"""Versioned Phase 4 to Phase 3 grounding-projection policy for P4.5b."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alice_conversation.contracts import GROUNDING_SOURCE_KINDS
from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicy,
)
from alice_conversation.state_schema import REFERENCE_KINDS

from .contracts import InformationContractError
from .grounding_policy import InformationGroundingPolicy

DEFAULT_CONVERSATION_BRIDGE_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_conversation_bridge_policy.json"
)

APPROVED_OUTCOME_MAPPING = (
    ("answerable", "answerable"),
    ("conflict", "conflict"),
    ("uncertain", "uncertain"),
    ("insufficient_sources", "insufficient_evidence"),
)
APPROVED_SOURCE_KIND = "web_source"
APPROVED_STATE_REFERENCE_KIND = "grounding_packet"
_APPROVED_P36_BOUNDARIES = {
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
_APPROVED_P36_CITATION_RULES = {
    "require_exact_tokens": True,
    "reject_unknown_tokens": True,
    "require_grounded_personal_claims": True,
    "require_supported_factual_claims": True,
}
_APPROVED_P36_EPISTEMIC_RULES = {
    "preserve_conflict": True,
    "preserve_uncertainty": True,
    "require_abstention_on_insufficient_evidence": True,
    "require_abstention_on_denied": True,
    "require_abstention_on_not_applicable": True,
    "reject_certainty_language_for_conflict": True,
    "reject_certainty_language_for_uncertainty": True,
}
_APPROVED_P36_SAFETY_RULES = {
    "reject_action_completion_claims": True,
    "reject_capability_claims": True,
    "reject_invented_personal_facts": True,
    "reject_dependency_language": True,
    "reject_hidden_reasoning_disclosure": True,
    "reject_truncated_responses": True,
}


def _exact_pairs_match(
    pairs: tuple[tuple[str, bool], ...],
    expected: dict[str, bool],
) -> bool:
    keys = tuple(key for key, _ in pairs)
    return (
        len(pairs) == len(expected)
        and len(set(keys)) == len(keys)
        and dict(pairs) == expected
    )


class InformationConversationBridgePolicyError(InformationContractError):
    """Raised when the P4.5b conversation-bridge policy is weakened."""


@dataclass(frozen=True)
class InformationConversationBridgePolicy:
    """Fail-closed policy for projecting verified web grounding into Phase 3."""

    policy_name: str
    version: str
    phase: str
    milestone: str
    status: str
    source_kind: str
    state_reference_kind: str
    outcome_mapping: tuple[tuple[str, str], ...]
    require_verified_information_grounding: bool
    require_exact_projection: bool
    require_exact_citation_tokens: bool
    require_source_version_bindings: bool
    require_freshness_metadata: bool
    require_p3_response_validation: bool
    require_metadata_only_state_reference: bool
    raw_source_persistence_allowed: bool
    raw_support_persistence_allowed: bool
    conversation_runtime_registration_allowed: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    model_claim_generation_allowed: bool
    semantic_entailment_inference_allowed: bool

    def validate(
        self,
        *,
        grounding_policy: InformationGroundingPolicy | None = None,
        response_validation_policy: ConversationResponseValidationPolicy | None = None,
    ) -> None:
        if self.policy_name != "alice_information_conversation_bridge_policy":
            raise InformationConversationBridgePolicyError(
                "Unexpected P4.5b conversation-bridge policy name."
            )
        if self.version != "1.0.0":
            raise InformationConversationBridgePolicyError(
                "P4.5b conversation-bridge policy version must be 1.0.0."
            )
        if (self.phase, self.milestone, self.status) != (
            "4",
            "P4.5b",
            "deterministic_phase3_grounding_projection",
        ):
            raise InformationConversationBridgePolicyError(
                "Conversation-bridge milestone binding is invalid."
            )
        if self.source_kind != APPROVED_SOURCE_KIND:
            raise InformationConversationBridgePolicyError(
                "P4.5b source kind changed without a policy-version change."
            )
        if self.state_reference_kind != APPROVED_STATE_REFERENCE_KIND:
            raise InformationConversationBridgePolicyError(
                "P4.5b state-reference kind changed without a policy-version change."
            )
        if self.outcome_mapping != APPROVED_OUTCOME_MAPPING:
            raise InformationConversationBridgePolicyError(
                "P4.5b outcome mapping changed without a policy-version change."
            )
        required_true = (
            self.require_verified_information_grounding,
            self.require_exact_projection,
            self.require_exact_citation_tokens,
            self.require_source_version_bindings,
            self.require_freshness_metadata,
            self.require_p3_response_validation,
            self.require_metadata_only_state_reference,
        )
        if not all(value is True for value in required_true):
            raise InformationConversationBridgePolicyError(
                "Required P4.5b bridge controls must remain enabled."
            )
        required_false = (
            self.raw_source_persistence_allowed,
            self.raw_support_persistence_allowed,
            self.conversation_runtime_registration_allowed,
            self.memory_write_allowed,
            self.external_action_allowed,
            self.model_claim_generation_allowed,
            self.semantic_entailment_inference_allowed,
        )
        if not all(value is False for value in required_false):
            raise InformationConversationBridgePolicyError(
                "Prohibited P4.5b capabilities must remain disabled."
            )
        if APPROVED_SOURCE_KIND not in GROUNDING_SOURCE_KINDS:
            raise InformationConversationBridgePolicyError(
                "Phase 3 does not recognize the approved web grounding source kind."
            )
        if APPROVED_STATE_REFERENCE_KIND not in REFERENCE_KINDS:
            raise InformationConversationBridgePolicyError(
                "Phase 3 does not recognize the grounding-packet reference kind."
            )
        if grounding_policy is not None:
            if not isinstance(grounding_policy, InformationGroundingPolicy):
                raise InformationConversationBridgePolicyError(
                    "A validated P4.5a grounding policy is required."
                )
            try:
                grounding_policy.validate()
            except InformationContractError as exc:
                raise InformationConversationBridgePolicyError(
                    "P4.5a grounding policy validation failed."
                ) from exc
            if tuple(grounding_policy.allowed_outcomes) != tuple(
                source for source, _ in APPROVED_OUTCOME_MAPPING
            ):
                raise InformationConversationBridgePolicyError(
                    "P4.5a and P4.5b outcome vocabularies are inconsistent."
                )
            if grounding_policy.source_digest_binding_required is not True:
                raise InformationConversationBridgePolicyError(
                    "P4.5b requires P4.5a source-version binding."
                )
            if grounding_policy.query_digest_binding_required is not True:
                raise InformationConversationBridgePolicyError(
                    "P4.5b requires P4.5a query binding."
                )
        if response_validation_policy is not None:
            if not isinstance(
                response_validation_policy,
                ConversationResponseValidationPolicy,
            ):
                raise InformationConversationBridgePolicyError(
                    "A validated P3.6 response-validation policy is required."
                )
            if (
                response_validation_policy.policy_name,
                response_validation_policy.version,
                response_validation_policy.phase,
                response_validation_policy.milestone,
                response_validation_policy.status,
            ) != (
                "alice_conversation_response_validation_policy",
                "1.0.0",
                "3",
                "P3.6",
                "generated_response_validation",
            ):
                raise InformationConversationBridgePolicyError(
                    "P3.6 response-validation identity is invalid."
                )
            if not _exact_pairs_match(
                response_validation_policy.boundaries,
                _APPROVED_P36_BOUNDARIES,
            ):
                raise InformationConversationBridgePolicyError(
                    "P3.6 response-validation boundaries are invalid."
                )
            if (
                not _exact_pairs_match(
                    response_validation_policy.citation_rules,
                    _APPROVED_P36_CITATION_RULES,
                )
                or response_validation_policy.minimum_answerable_claims_cited != 1
                or response_validation_policy.minimum_conflict_claims_cited != 2
            ):
                raise InformationConversationBridgePolicyError(
                    "P3.6 citation controls are invalid."
                )
            if (
                not _exact_pairs_match(
                    response_validation_policy.epistemic_rules,
                    _APPROVED_P36_EPISTEMIC_RULES,
                )
            ):
                raise InformationConversationBridgePolicyError(
                    "P3.6 epistemic controls are invalid."
                )
            if (
                not _exact_pairs_match(
                    response_validation_policy.safety_rules,
                    _APPROVED_P36_SAFETY_RULES,
                )
            ):
                raise InformationConversationBridgePolicyError(
                    "P3.6 safety controls are invalid."
                )

    def map_outcome(self, information_outcome: str) -> str:
        self.validate()
        try:
            return dict(self.outcome_mapping)[information_outcome]
        except KeyError as exc:
            raise InformationConversationBridgePolicyError(
                "Information outcome is not projectable through P4.5b."
            ) from exc


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InformationConversationBridgePolicyError(f"{field} must be an object.")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationConversationBridgePolicyError(
            f"{field} must be non-empty text."
        )
    return value.strip()


def _strict_bool(value: Any, *, expected: bool, field: str) -> bool:
    if value is not expected:
        raise InformationConversationBridgePolicyError(
            f"{field} must remain {str(expected).lower()}."
        )
    return expected


def parse_information_conversation_bridge_policy(
    payload: dict[str, Any],
    *,
    grounding_policy: InformationGroundingPolicy | None = None,
    response_validation_policy: ConversationResponseValidationPolicy | None = None,
) -> InformationConversationBridgePolicy:
    root = _mapping(payload, field="conversation-bridge policy")
    expected = {
        "policy_name",
        "version",
        "phase",
        "milestone",
        "status",
        "source_kind",
        "state_reference_kind",
        "outcome_mapping",
        "requirements",
        "boundaries",
    }
    if set(root) != expected:
        raise InformationConversationBridgePolicyError(
            "Conversation-bridge policy fields do not match the P4.5b contract."
        )
    outcome_mapping = _mapping(root["outcome_mapping"], field="outcome_mapping")
    if tuple(outcome_mapping.items()) != APPROVED_OUTCOME_MAPPING:
        raise InformationConversationBridgePolicyError(
            "outcome_mapping must match the approved ordered mapping."
        )
    requirements = _mapping(root["requirements"], field="requirements")
    expected_requirements = {
        "verified_information_grounding",
        "exact_projection",
        "exact_citation_tokens",
        "source_version_bindings",
        "freshness_metadata",
        "p3_response_validation",
        "metadata_only_state_reference",
    }
    if set(requirements) != expected_requirements:
        raise InformationConversationBridgePolicyError(
            "Requirement fields do not match the P4.5b contract."
        )
    boundaries = _mapping(root["boundaries"], field="boundaries")
    expected_boundaries = {
        "raw_source_persistence_allowed",
        "raw_support_persistence_allowed",
        "conversation_runtime_registration_allowed",
        "memory_write_allowed",
        "external_action_allowed",
        "model_claim_generation_allowed",
        "semantic_entailment_inference_allowed",
    }
    if set(boundaries) != expected_boundaries:
        raise InformationConversationBridgePolicyError(
            "Boundary fields do not match the P4.5b contract."
        )
    policy = InformationConversationBridgePolicy(
        policy_name=_text(root["policy_name"], field="policy_name"),
        version=_text(root["version"], field="version"),
        phase=_text(root["phase"], field="phase"),
        milestone=_text(root["milestone"], field="milestone"),
        status=_text(root["status"], field="status"),
        source_kind=_text(root["source_kind"], field="source_kind"),
        state_reference_kind=_text(
            root["state_reference_kind"], field="state_reference_kind"
        ),
        outcome_mapping=tuple(outcome_mapping.items()),
        require_verified_information_grounding=_strict_bool(
            requirements["verified_information_grounding"],
            expected=True,
            field="requirements.verified_information_grounding",
        ),
        require_exact_projection=_strict_bool(
            requirements["exact_projection"],
            expected=True,
            field="requirements.exact_projection",
        ),
        require_exact_citation_tokens=_strict_bool(
            requirements["exact_citation_tokens"],
            expected=True,
            field="requirements.exact_citation_tokens",
        ),
        require_source_version_bindings=_strict_bool(
            requirements["source_version_bindings"],
            expected=True,
            field="requirements.source_version_bindings",
        ),
        require_freshness_metadata=_strict_bool(
            requirements["freshness_metadata"],
            expected=True,
            field="requirements.freshness_metadata",
        ),
        require_p3_response_validation=_strict_bool(
            requirements["p3_response_validation"],
            expected=True,
            field="requirements.p3_response_validation",
        ),
        require_metadata_only_state_reference=_strict_bool(
            requirements["metadata_only_state_reference"],
            expected=True,
            field="requirements.metadata_only_state_reference",
        ),
        raw_source_persistence_allowed=_strict_bool(
            boundaries["raw_source_persistence_allowed"],
            expected=False,
            field="boundaries.raw_source_persistence_allowed",
        ),
        raw_support_persistence_allowed=_strict_bool(
            boundaries["raw_support_persistence_allowed"],
            expected=False,
            field="boundaries.raw_support_persistence_allowed",
        ),
        conversation_runtime_registration_allowed=_strict_bool(
            boundaries["conversation_runtime_registration_allowed"],
            expected=False,
            field="boundaries.conversation_runtime_registration_allowed",
        ),
        memory_write_allowed=_strict_bool(
            boundaries["memory_write_allowed"],
            expected=False,
            field="boundaries.memory_write_allowed",
        ),
        external_action_allowed=_strict_bool(
            boundaries["external_action_allowed"],
            expected=False,
            field="boundaries.external_action_allowed",
        ),
        model_claim_generation_allowed=_strict_bool(
            boundaries["model_claim_generation_allowed"],
            expected=False,
            field="boundaries.model_claim_generation_allowed",
        ),
        semantic_entailment_inference_allowed=_strict_bool(
            boundaries["semantic_entailment_inference_allowed"],
            expected=False,
            field="boundaries.semantic_entailment_inference_allowed",
        ),
    )
    policy.validate(
        grounding_policy=grounding_policy,
        response_validation_policy=response_validation_policy,
    )
    return policy


def load_information_conversation_bridge_policy(
    path: str | Path | None = None,
    *,
    grounding_policy: InformationGroundingPolicy | None = None,
    response_validation_policy: ConversationResponseValidationPolicy | None = None,
) -> InformationConversationBridgePolicy:
    selected = (
        Path(path)
        if path is not None
        else DEFAULT_CONVERSATION_BRIDGE_POLICY_PATH
    )
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InformationConversationBridgePolicyError(
            "Conversation-bridge policy could not be loaded."
        ) from exc
    if not isinstance(payload, dict):
        raise InformationConversationBridgePolicyError(
            "Conversation-bridge policy root must be an object."
        )
    return parse_information_conversation_bridge_policy(
        payload,
        grounding_policy=grounding_policy,
        response_validation_policy=response_validation_policy,
    )
