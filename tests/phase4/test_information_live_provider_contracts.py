from dataclasses import replace

import pytest

from alice_information.brave_search import BraveInformationSearchProvider
from alice_information.live_provider_contracts import (
    InformationLiveProviderContractError,
    InformationLiveProviderFailure,
    InformationLiveRateLimitState,
)

from _information_live_provider_helpers import FixtureBraveTransport, configuration, policy, query


def test_failure_is_sanitized_and_metadata_only():
    failure = InformationLiveProviderFailure.create(
        provider="brave-search-v1", operation="search", code="live_provider_timeout"
    )
    failure.validate()
    assert failure.metadata_record()["code"] == "live_provider_timeout"
    assert "query" not in failure.metadata_record()


def test_rate_limit_metadata_rejects_controls():
    with pytest.raises(InformationLiveProviderContractError):
        InformationLiveRateLimitState("1\n2", None, None, None).validate()


def test_search_receipt_detects_tampering():
    provider = BraveInformationSearchProvider(
        policy=policy(), configuration=configuration(), transport=FixtureBraveTransport()
    )
    response = provider.search_with_receipt(query(), max_results=1, timeout_seconds=5)
    response.validate()
    with pytest.raises(InformationLiveProviderContractError):
        replace(response.receipt, item_count=2).validate()
