from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alice_information.freshness_policy import load_information_freshness_policy
from alice_information.grounding_policy import (
    InformationGroundingPolicyError,
    load_information_grounding_policy,
    parse_information_grounding_policy,
)
from alice_information.injection_policy import (
    load_information_injection_firewall_policy,
)
from alice_information.policy import load_information_policy


def _payload() -> dict[str, object]:
    path = "policies/information_grounding_policy.json"
    return json.loads(open(path, encoding="utf-8").read())


def _boundaries():
    base = load_information_policy()
    firewall = load_information_injection_firewall_policy(
        information_policy=base
    )
    freshness = load_information_freshness_policy(
        information_policy=base,
        firewall_policy=firewall,
    )
    return base, firewall, freshness


def test_grounding_policy_loads_with_exact_approved_values() -> None:
    base, firewall, freshness = _boundaries()
    policy = load_information_grounding_policy(
        information_policy=base,
        firewall_policy=firewall,
        freshness_policy=freshness,
    )
    assert policy.version == "1.0.0"
    assert policy.max_sources == 12
    assert policy.max_claims == 24
    assert policy.require_exact_support_span is True
    assert policy.allow_model_claim_generation is False
    assert policy.allow_publisher_reputation_inference is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_name", "changed"),
        ("version", "1.0.1"),
        ("phase", "5"),
        ("milestone", "P4.5b"),
        ("status", "enabled"),
        ("permission_id", "external.action"),
        ("max_sources", 11),
        ("max_claims", 23),
        ("max_support_span_characters", 1999),
        ("min_source_characters", 19),
        ("verified_fact_min_distinct_domains", 1),
        ("conflict_min_distinct_domains", 1),
        ("require_https_sources", False),
        ("require_clear_firewall", False),
        ("require_freshness_support", False),
        ("require_exact_support_span", False),
        ("require_all_packet_sources_cited", False),
        ("allow_unused_sources", True),
        ("allow_model_claim_generation", True),
        ("allow_semantic_entailment_inference", True),
        ("allow_publisher_reputation_inference", True),
        ("raw_support_logging_allowed", True),
        ("source_digest_binding_required", False),
        ("query_digest_binding_required", False),
        ("citation_token_prefix", "[CITE:"),
        ("citation_token_suffix", ")"),
    ],
)
def test_grounding_policy_rejects_changed_controls(
    field: str, value: object
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationGroundingPolicyError):
        parse_information_grounding_policy(payload)


def test_grounding_policy_rejects_unknown_root_key() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(InformationGroundingPolicyError):
        parse_information_grounding_policy(payload)


def test_grounding_policy_rejects_changed_outcome_vocabulary() -> None:
    payload = _payload()
    payload["allowed_outcomes"] = ["answerable"]
    with pytest.raises(InformationGroundingPolicyError):
        parse_information_grounding_policy(payload)


def test_grounding_policy_rejects_changed_knowledge_vocabulary() -> None:
    payload = _payload()
    payload["allowed_knowledge_statuses"] = ["verified_fact"]
    with pytest.raises(InformationGroundingPolicyError):
        parse_information_grounding_policy(payload)


def test_projected_policy_revalidates_exact_numbers() -> None:
    policy = load_information_grounding_policy()
    with pytest.raises(InformationGroundingPolicyError):
        replace(policy, max_sources=1).validate()
