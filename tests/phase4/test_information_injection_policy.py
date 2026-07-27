from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alice_information.injection_policy import (
    DEFAULT_INJECTION_FIREWALL_POLICY_PATH,
    InformationInjectionPolicyError,
    load_information_injection_firewall_policy,
    parse_information_injection_firewall_policy,
)
from alice_information.policy import load_information_policy


def _payload() -> dict[str, object]:
    return json.loads(Path(DEFAULT_INJECTION_FIREWALL_POLICY_PATH).read_text(encoding="utf-8"))


def test_default_firewall_policy_loads_and_binds_to_base_policy() -> None:
    base = load_information_policy()
    policy = load_information_injection_firewall_policy(information_policy=base)
    assert policy.version == "1.0.0"
    assert policy.milestone == "P4.3"
    assert policy.flagged_sources_renderable is False
    assert policy.model_classifier_allowed is False


def test_unknown_root_key_is_rejected() -> None:
    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(InformationInjectionPolicyError):
        parse_information_injection_firewall_policy(payload)


def test_unknown_nested_key_is_rejected() -> None:
    payload = _payload()
    normalization = dict(payload["normalization"])
    normalization["unexpected"] = True
    payload["normalization"] = normalization
    with pytest.raises(InformationInjectionPolicyError):
        parse_information_injection_firewall_policy(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_classifier_allowed", True),
        ("content_mutation_allowed", True),
        ("raw_excerpt_logging_allowed", True),
        ("flagged_sources_renderable", True),
        ("preserve_original_source_text", False),
        ("source_digest_binding_required", False),
    ],
)
def test_security_boundary_mutations_are_rejected(field: str, value: bool) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationInjectionPolicyError):
        parse_information_injection_firewall_policy(payload)


def test_finding_vocabulary_must_be_exact() -> None:
    payload = _payload()
    payload["critical_finding_codes"] = list(payload["critical_finding_codes"])[1:]
    with pytest.raises(InformationInjectionPolicyError):
        parse_information_injection_firewall_policy(payload)


def test_policy_revalidates_after_dataclass_mutation() -> None:
    policy = load_information_injection_firewall_policy()
    mutated = replace(policy, max_findings=101)
    with pytest.raises(InformationInjectionPolicyError):
        mutated.validate()
