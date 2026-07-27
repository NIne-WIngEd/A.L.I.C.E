from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_information.freshness_policy import (
    ALLOWED_FRESHNESS_VERDICTS,
    ALLOWED_TEMPORAL_INTENTS,
    DEFAULT_FRESHNESS_POLICY_PATH,
    InformationFreshnessPolicyError,
    load_information_freshness_policy,
    parse_information_freshness_policy,
)
from alice_information.injection_policy import load_information_injection_firewall_policy
from alice_information.policy import load_information_policy


def _payload() -> dict[str, object]:
    return json.loads(Path(DEFAULT_FRESHNESS_POLICY_PATH).read_text(encoding="utf-8"))


def test_default_freshness_policy_loads_and_binds_to_prior_boundaries() -> None:
    base = load_information_policy()
    firewall = load_information_injection_firewall_policy(information_policy=base)
    policy = load_information_freshness_policy(
        information_policy=base,
        firewall_policy=firewall,
    )
    assert policy.version == "1.0.0"
    assert policy.milestone == "P4.4a"
    assert policy.allowed_intents == ALLOWED_TEMPORAL_INTENTS
    assert policy.allowed_verdicts == ALLOWED_FRESHNESS_VERDICTS
    assert policy.current_max_age_seconds == 86400
    assert policy.latest_max_age_seconds == 604800
    assert policy.recent_max_age_seconds == 2592000


def test_unknown_root_key_is_rejected() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_unknown_age_limit_key_is_rejected() -> None:
    payload = _payload()
    limits = dict(payload["age_limits_seconds"])
    limits["unexpected"] = 1
    payload["age_limits_seconds"] = limits
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic_query_classification_required", False),
        ("ambiguous_temporal_queries_fail_closed", False),
        ("retrieval_time_is_freshness_evidence", True),
        ("future_source_time_allowed", True),
        ("model_temporal_inference_allowed", True),
        ("raw_temporal_metadata_logging_allowed", True),
        ("source_digest_binding_required", False),
        ("firewall_clear_required", False),
    ],
)
def test_security_boundary_mutations_are_rejected(field: str, value: bool) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_intent_vocabulary_must_be_exact() -> None:
    payload = _payload()
    payload["allowed_intents"] = list(payload["allowed_intents"])[1:]
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_verdict_vocabulary_must_be_exact() -> None:
    payload = _payload()
    payload["allowed_verdicts"] = list(payload["allowed_verdicts"])[1:]
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_age_limits_must_increase_monotonically() -> None:
    payload = _payload()
    limits = dict(payload["age_limits_seconds"])
    limits["current"] = 700000
    payload["age_limits_seconds"] = limits
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_max_age_lookup_rejects_non_age_intent() -> None:
    policy = load_information_freshness_policy()
    with pytest.raises(InformationFreshnessPolicyError):
        policy.max_age_seconds("historical")

@pytest.mark.parametrize(
    ("intent", "value"),
    [
        ("current", 86401),
        ("latest", 604801),
        ("recent", 2592001),
    ],
)
def test_versioned_age_limits_are_exact(intent: str, value: int) -> None:
    payload = _payload()
    limits = dict(payload["age_limits_seconds"])
    limits[intent] = value
    payload["age_limits_seconds"] = limits
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)


def test_versioned_clock_skew_is_exact() -> None:
    payload = _payload()
    payload["max_clock_skew_seconds"] = 301
    with pytest.raises(InformationFreshnessPolicyError):
        parse_information_freshness_policy(payload)
