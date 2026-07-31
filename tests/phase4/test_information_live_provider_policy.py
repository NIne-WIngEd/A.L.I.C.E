import json
from copy import deepcopy
from pathlib import Path

import pytest

from alice_information.live_provider_policy import (
    InformationLiveProviderPolicyError,
    InformationLiveProviderRuntimePolicy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "policies/information_live_provider_runtime_policy.json"


def load_mapping():
    return json.loads(POLICY.read_text(encoding="utf-8"))


def test_canonical_policy_loads_and_binds_digest():
    policy = InformationLiveProviderRuntimePolicy.load(POLICY)
    policy.validate()
    assert policy.search_provider_id == "brave-search-v1"
    assert policy.fetch_provider_id == "controlled-live-http-v1"
    assert len(policy.policy_sha256) == 64
    assert policy.binding.startswith("alice_information_live_provider_runtime_policy@1.0.0:")
    registration = (ROOT / "scripts/register_phase4_live_scope.py").read_text(encoding="utf-8")
    assert "active_successor_profile" not in registration
    assert '"scope_kind": "active_milestone_guard"' in registration
    assert '"scope_kind": "compatibility_test"' in registration
    assert '"scope_kind": "historical_or_phase_local"' in registration
    assert '"scope_kind": "phase_local_compatibility"' in registration


@pytest.mark.parametrize(
    "path,value",
    [
        (("phase",), "5"),
        (("milestone",), "P4.9"),
        (("query_classifications",), ["PRIVATE"]),
        (("capability_ceiling",), True),
        (("search_provider", "provider_id"), "other"),
        (("search_provider", "host"), "example.com"),
        (("search_provider", "offset"), 1),
        (("search_provider", "spellcheck"), True),
        (("fetch_provider", "credential_headers_allowed"), True),
        (("execution_controls", "provider_fallback_allowed"), True),
        (("execution_controls", "retry_allowed"), True),
        (("execution_controls", "source_body_persistence_allowed"), True),
        (("execution_controls", "phase5_storage_allowed"), True),
        (("execution_controls", "memory_write_allowed"), True),
        (("execution_controls", "external_action_allowed"), True),
        (("execution_controls", "background_execution_allowed"), True),
    ],
)
def test_policy_mutations_fail_closed(path, value):
    data = deepcopy(load_mapping())
    current = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
    with pytest.raises(InformationLiveProviderPolicyError):
        InformationLiveProviderRuntimePolicy.from_mapping(data)


def test_unknown_field_is_rejected():
    data = load_mapping()
    data["implicit_fallback"] = True
    with pytest.raises(InformationLiveProviderPolicyError):
        InformationLiveProviderRuntimePolicy.from_mapping(data)
