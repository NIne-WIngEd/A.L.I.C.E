"""P4.0 public information-policy validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from alice_information.policy import (
    InformationPolicyError,
    load_information_policy,
    parse_information_policy,
)

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_policy.json"
)


def _payload() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_repository_policy_loads_with_no_live_provider_or_network() -> None:
    policy = load_information_policy(POLICY_PATH)
    assert policy.phase == "4"
    assert policy.milestone == "P4.0"
    assert policy.permission_id == "web.search"
    assert policy.approved_live_providers == ()
    assert policy.allowed_query_classifications == ("PUBLIC",)
    assert policy.allowed_schemes == ("http", "https")
    assert policy.foreground_only is True
    policy.capabilities.validate()


def test_policy_identity_and_version_are_exactly_bound() -> None:
    payload = _payload()
    payload["policy_name"] = "lookalike_information_policy"
    with pytest.raises(InformationPolicyError, match="policy_name"):
        parse_information_policy(payload)
    payload = _payload()
    payload["version"] = "9.9.9"
    with pytest.raises(InformationPolicyError, match="version"):
        parse_information_policy(payload)


def test_policy_rejects_live_network_or_provider_fallback() -> None:
    payload = _payload()
    payload["boundaries"]["live_network_access_allowed"] = True
    with pytest.raises(InformationPolicyError, match="must remain false"):
        parse_information_policy(payload)
    payload = _payload()
    payload["boundaries"]["provider_fallback_allowed"] = True
    with pytest.raises(InformationPolicyError, match="must remain false"):
        parse_information_policy(payload)


def test_policy_rejects_live_providers_in_foundation() -> None:
    payload = _payload()
    payload["approved_live_providers"] = ["example-search"]
    with pytest.raises(InformationPolicyError, match="empty list"):
        parse_information_policy(payload)


def test_policy_allows_only_public_query_transmission() -> None:
    payload = _payload()
    payload["query_transmission"]["allowed_classifications"].append("PRIVATE")
    with pytest.raises(InformationPolicyError, match="approved P4.0 value"):
        parse_information_policy(payload)
    payload = _payload()
    payload["query_transmission"]["private_context_allowed"] = True
    with pytest.raises(InformationPolicyError, match="must remain false"):
        parse_information_policy(payload)


def test_policy_preserves_untrusted_content_boundary() -> None:
    payload = _payload()
    payload["untrusted_content"]["can_grant_permission"] = True
    with pytest.raises(InformationPolicyError, match="must remain false"):
        parse_information_policy(payload)
    payload = _payload()
    payload["untrusted_content"]["treated_as_untrusted_data"] = False
    with pytest.raises(InformationPolicyError, match="must remain true"):
        parse_information_policy(payload)


def test_policy_rejects_authenticated_or_active_browsing() -> None:
    for field in (
        "authenticated_browsing_allowed",
        "cookies_allowed",
        "javascript_execution_allowed",
        "form_submission_allowed",
        "downloads_allowed",
        "recursive_browsing_allowed",
    ):
        payload = _payload()
        payload["source_retrieval"][field] = True
        with pytest.raises(InformationPolicyError, match="must remain false"):
            parse_information_policy(payload)


def test_policy_enforces_resource_budgets_and_sanitized_logging() -> None:
    payload = _payload()
    payload["budgets"]["max_fetch_calls"] = 21
    with pytest.raises(InformationPolicyError, match="between 1 and 20"):
        parse_information_policy(payload)
    payload = _payload()
    payload["logging"]["raw_query_logging_allowed"] = True
    with pytest.raises(InformationPolicyError, match="must remain false"):
        parse_information_policy(payload)
