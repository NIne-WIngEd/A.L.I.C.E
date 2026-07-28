from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alice_conversation.response_validation_policy import (
    parse_conversation_response_validation_policy,
)
from alice_information.conversation_bridge_policy import (
    APPROVED_OUTCOME_MAPPING,
    InformationConversationBridgePolicyError,
    load_information_conversation_bridge_policy,
    parse_information_conversation_bridge_policy,
)
from alice_information.grounding_policy import InformationGroundingPolicy


def _grounding_policy() -> InformationGroundingPolicy:
    return InformationGroundingPolicy(
        policy_name="alice_information_grounding_policy",
        version="1.0.0",
        phase="4",
        milestone="P4.5a",
        status="deterministic_citation_grounding",
        permission_id="web.search",
        allowed_outcomes=(
            "answerable",
            "conflict",
            "uncertain",
            "insufficient_sources",
        ),
        allowed_knowledge_statuses=(
            "external_claim",
            "verified_fact",
            "uncertain",
            "disputed",
            "historical",
        ),
        max_sources=12,
        max_claims=24,
        max_support_span_characters=2000,
        min_source_characters=20,
        verified_fact_min_distinct_domains=2,
        conflict_min_distinct_domains=2,
        require_https_sources=True,
        require_clear_firewall=True,
        require_freshness_support=True,
        require_exact_support_span=True,
        require_all_packet_sources_cited=True,
        allow_unused_sources=False,
        allow_model_claim_generation=False,
        allow_semantic_entailment_inference=False,
        allow_publisher_reputation_inference=False,
        raw_support_logging_allowed=False,
        source_digest_binding_required=True,
        query_digest_binding_required=True,
        citation_token_prefix="[WEB:",
        citation_token_suffix="]",
    )


def _response_policy():
    return parse_conversation_response_validation_policy(
        {
            "policy_name": "alice_conversation_response_validation_policy",
            "version": "1.0.0",
            "phase": "3",
            "milestone": "P3.6",
            "status": "generated_response_validation",
            "boundaries": {
                "web_access_allowed": False,
                "tool_calling_allowed": False,
                "external_action_allowed": False,
                "memory_write_allowed": False,
                "memory_promotion_allowed": False,
                "highly_sensitive_grounding_allowed": False,
                "chain_of_thought_persistence_allowed": False,
                "automatic_repair_allowed": False,
                "provider_fallback_allowed": False,
            },
            "citations": {
                "require_exact_tokens": True,
                "reject_unknown_tokens": True,
                "require_grounded_personal_claims": True,
                "require_supported_factual_claims": True,
                "minimum_answerable_claims_cited": 1,
                "minimum_conflict_claims_cited": 2,
            },
            "epistemic": {
                "preserve_conflict": True,
                "preserve_uncertainty": True,
                "require_abstention_on_insufficient_evidence": True,
                "require_abstention_on_denied": True,
                "require_abstention_on_not_applicable": True,
                "reject_certainty_language_for_conflict": True,
                "reject_certainty_language_for_uncertainty": True,
            },
            "safety": {
                "reject_action_completion_claims": True,
                "reject_capability_claims": True,
                "reject_invented_personal_facts": True,
                "reject_dependency_language": True,
                "reject_hidden_reasoning_disclosure": True,
                "reject_truncated_responses": True,
            },
            "limits": {"max_response_chars": 20000, "max_issues": 64},
            "failure_codes": {
                "rejected": "response_validation_rejected",
                "internal": "response_validation_internal",
            },
        }
    )


def _payload() -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "policies" / "information_conversation_bridge_policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_default_policy_loads_with_p45a_and_p36_dependencies() -> None:
    policy = load_information_conversation_bridge_policy(
        grounding_policy=_grounding_policy(),
        response_validation_policy=_response_policy(),
    )
    assert policy.version == "1.0.0"
    assert policy.milestone == "P4.5b"
    assert policy.outcome_mapping == APPROVED_OUTCOME_MAPPING
    assert policy.map_outcome("insufficient_sources") == "insufficient_evidence"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("requirements", "verified_information_grounding", False),
        ("requirements", "exact_projection", False),
        ("requirements", "exact_citation_tokens", False),
        ("requirements", "source_version_bindings", False),
        ("requirements", "freshness_metadata", False),
        ("requirements", "p3_response_validation", False),
        ("requirements", "metadata_only_state_reference", False),
        ("boundaries", "raw_source_persistence_allowed", True),
        ("boundaries", "raw_support_persistence_allowed", True),
        ("boundaries", "conversation_runtime_registration_allowed", True),
        ("boundaries", "memory_write_allowed", True),
        ("boundaries", "external_action_allowed", True),
        ("boundaries", "model_claim_generation_allowed", True),
        ("boundaries", "semantic_entailment_inference_allowed", True),
    ],
)
def test_policy_rejects_weakened_controls(section: str, field: str, value: bool) -> None:
    payload = _payload()
    payload[section][field] = value  # type: ignore[index]
    with pytest.raises(InformationConversationBridgePolicyError):
        parse_information_conversation_bridge_policy(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "1.0.1"),
        ("phase", "3"),
        ("milestone", "P4.5a"),
        ("status", "runtime_enabled"),
        ("source_kind", "phase1_source"),
        ("state_reference_kind", "web_source"),
    ],
)
def test_policy_rejects_identity_changes(field: str, value: str) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationConversationBridgePolicyError):
        parse_information_conversation_bridge_policy(payload)


def test_policy_rejects_unknown_root_key() -> None:
    payload = _payload()
    payload["unknown"] = True
    with pytest.raises(InformationConversationBridgePolicyError):
        parse_information_conversation_bridge_policy(payload)


def test_policy_rejects_changed_outcome_mapping() -> None:
    payload = _payload()
    payload["outcome_mapping"]["insufficient_sources"] = "answerable"  # type: ignore[index]
    with pytest.raises(InformationConversationBridgePolicyError):
        parse_information_conversation_bridge_policy(payload)


def test_policy_rejects_p45a_without_source_digest_binding() -> None:
    weakened = replace(_grounding_policy(), source_digest_binding_required=False)
    policy = load_information_conversation_bridge_policy()
    with pytest.raises(InformationConversationBridgePolicyError):
        policy.validate(grounding_policy=weakened)


def test_policy_rejects_unprojectable_outcome() -> None:
    policy = load_information_conversation_bridge_policy()
    with pytest.raises(InformationConversationBridgePolicyError):
        policy.map_outcome("failed")


def test_policy_rejects_forged_p36_identity() -> None:
    forged = replace(_response_policy(), version="9.9.9")
    policy = load_information_conversation_bridge_policy()
    with pytest.raises(InformationConversationBridgePolicyError):
        policy.validate(response_validation_policy=forged)


def test_policy_rejects_forged_p36_safety_rules() -> None:
    rules = dict(_response_policy().safety_rules)
    rules["reject_action_completion_claims"] = False
    forged = replace(_response_policy(), safety_rules=tuple(rules.items()))
    policy = load_information_conversation_bridge_policy()
    with pytest.raises(InformationConversationBridgePolicyError):
        policy.validate(response_validation_policy=forged)


def test_policy_rejects_duplicate_p36_boundary_keys() -> None:
    response_policy = _response_policy()
    forged = replace(
        response_policy,
        boundaries=response_policy.boundaries
        + (("web_access_allowed", False),),
    )
    policy = load_information_conversation_bridge_policy()
    with pytest.raises(InformationConversationBridgePolicyError):
        policy.validate(response_validation_policy=forged)
