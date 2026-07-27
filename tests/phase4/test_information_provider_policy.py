"""P4.1 provider-policy validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alice_information.provider_policy import (
    InformationProviderPolicyError,
    load_information_provider_policy,
    parse_information_provider_policy,
)

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_provider_policy.json"
)


def _payload() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_repository_provider_policy_allows_only_exact_deterministic_fixture() -> None:
    policy = load_information_provider_policy(POLICY_PATH)
    assert policy.phase == "4"
    assert policy.milestone == "P4.1"
    assert policy.live_network_access_allowed is False
    assert policy.provider_fallback_allowed is False
    assert policy.allows(
        provider="deterministic-fixture-v1",
        provider_type="deterministic_fixture",
        operation="search",
    )
    assert not policy.allows(
        provider="other-fixture",
        provider_type="deterministic_fixture",
        operation="search",
    )


def test_provider_policy_identity_and_version_are_exact() -> None:
    payload = _payload()
    payload["policy_name"] = "lookalike"
    with pytest.raises(InformationProviderPolicyError, match="policy_name"):
        parse_information_provider_policy(payload)
    payload = _payload()
    payload["version"] = "2.0.0"
    with pytest.raises(InformationProviderPolicyError, match="version"):
        parse_information_provider_policy(payload)


def test_provider_policy_rejects_live_network_and_fallback() -> None:
    payload = _payload()
    payload["live_network_access_allowed"] = True
    with pytest.raises(InformationProviderPolicyError, match="must remain false"):
        parse_information_provider_policy(payload)
    payload = _payload()
    payload["provider_fallback_allowed"] = True
    with pytest.raises(InformationProviderPolicyError, match="must remain false"):
        parse_information_provider_policy(payload)


def test_provider_policy_rejects_live_or_duplicate_providers() -> None:
    payload = _payload()
    payload["approved_providers"][0]["provider_type"] = "live"
    with pytest.raises(InformationProviderPolicyError, match="deterministic"):
        parse_information_provider_policy(payload)
    payload = _payload()
    payload["approved_providers"].append(
        dict(payload["approved_providers"][0])
    )
    with pytest.raises(InformationProviderPolicyError, match="unique"):
        parse_information_provider_policy(payload)


def test_provider_policy_requires_sanitized_failure_boundaries() -> None:
    payload = _payload()
    payload["sanitized_failures_required"] = False
    with pytest.raises(InformationProviderPolicyError, match="must remain true"):
        parse_information_provider_policy(payload)
    payload = _payload()
    payload["raw_query_in_failures_allowed"] = True
    with pytest.raises(InformationProviderPolicyError, match="must remain false"):
        parse_information_provider_policy(payload)
    payload = _payload()
    payload["raw_source_content_in_failures_allowed"] = True
    with pytest.raises(InformationProviderPolicyError, match="must remain false"):
        parse_information_provider_policy(payload)
