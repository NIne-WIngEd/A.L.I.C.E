from __future__ import annotations

import json

import pytest

from alice_information.live_research_policy import (
    InformationLiveResearchPolicy,
    InformationLiveResearchPolicyError,
)
from _information_live_research_helpers import POLICY_PATH, policy


def test_exact_policy_loads_with_required_path_and_bounds():
    value = policy()
    assert value.required_mode == "research"
    assert value.required_availability == "available"
    assert value.required_operations == ("search", "fetch")
    assert value.search_provider == "brave-search-v1"
    assert value.fetch_provider == "controlled-live-http-v1"
    assert value.maximum_search_calls == 1
    assert value.maximum_fetch_calls == 5
    assert value.maximum_grounded_sources == 2
    assert value.skippable_fetch_failure_codes == (
        "http_status_rejected",
        "response_header_invalid",
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("controls", "no_retry"), False),
        (("controls", "no_silent_fallback"), False),
        (("controls", "no_source_body_persistence"), False),
        (("controls", "no_phase5_storage"), False),
        (("controls", "no_memory_write"), False),
        (("controls", "no_external_action"), False),
        (("controls", "no_background_execution"), False),
        (("controls", "continue_after_skippable_source_failure"), False),
        (("skippable_fetch_failure_codes",), []),
        (("capability_ceiling",), True),
        (("maximum_search_calls",), 2),
        (("search_provider",), "other-provider"),
    ],
)
def test_policy_mutations_are_rejected(path, value):
    data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    cursor = data
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(InformationLiveResearchPolicyError):
        InformationLiveResearchPolicy.from_mapping(data)


def test_provider_binding_must_match_exact_p410a_policy():
    class ProviderPolicy:
        search_provider_id = "wrong"
        fetch_provider_id = "controlled-live-http-v1"
        def validate(self): return None
    with pytest.raises(InformationLiveResearchPolicyError):
        policy().validate(provider_policy=ProviderPolicy())
