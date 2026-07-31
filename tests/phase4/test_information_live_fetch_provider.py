import pytest

from alice_information.live_fetch_provider import (
    InformationLiveFetchProviderError,
    LiveControlledInformationFetchProvider,
)

from _information_live_provider_helpers import policy


def test_fetch_provider_rejects_substituted_retriever():
    with pytest.raises(InformationLiveFetchProviderError):
        LiveControlledInformationFetchProvider(policy=policy(), retriever=object())
