from dataclasses import dataclass
from pathlib import Path

import pytest

from alice_information.live_provider_policy import InformationLiveProviderRuntimePolicy
from alice_information.live_provider_registry import (
    ExactInformationLiveProviderRegistry,
    InformationLiveProviderRegistryError,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = InformationLiveProviderRuntimePolicy.load(
    ROOT / "policies/information_live_provider_runtime_policy.json"
)


@dataclass
class Provider:
    provider: str
    provider_type: str = "live"
    def search(self, *args, **kwargs): return ()
    def fetch(self, *args, **kwargs): return None


def test_exact_registry_has_no_default_or_fallback_path():
    registry = ExactInformationLiveProviderRegistry(
        policy=POLICY,
        search_provider=Provider("brave-search-v1"),
        fetch_provider=Provider("controlled-live-http-v1"),
    )
    assert registry.resolve_search("brave-search-v1").provider == "brave-search-v1"
    assert registry.resolve_fetch("controlled-live-http-v1").provider == "controlled-live-http-v1"
    assert registry.metadata_record()["fallback_allowed"] is False
    with pytest.raises(InformationLiveProviderRegistryError):
        registry.resolve_search("other")


@pytest.mark.parametrize(
    "search,fetch",
    [
        (Provider("fixture"), Provider("controlled-live-http-v1")),
        (Provider("brave-search-v1", "deterministic_fixture"), Provider("controlled-live-http-v1")),
        (Provider("brave-search-v1"), Provider("other")),
    ],
)
def test_registry_substitution_is_rejected(search, fetch):
    with pytest.raises((InformationLiveProviderRegistryError, ValueError)):
        ExactInformationLiveProviderRegistry(policy=POLICY, search_provider=search, fetch_provider=fetch)
