"""Exact no-fallback live provider registry for P4.10a."""

from __future__ import annotations

from dataclasses import dataclass

from .live_provider_policy import InformationLiveProviderRuntimePolicy
from .providers import validate_provider_identity


class InformationLiveProviderRegistryError(ValueError):
    """Raised when the P4.10a registry is incomplete or substitutable."""


@dataclass(frozen=True)
class ExactInformationLiveProviderRegistry:
    policy: InformationLiveProviderRuntimePolicy
    search_provider: object
    fetch_provider: object

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.policy.validate()
        search_id = validate_provider_identity(getattr(self.search_provider, "provider", None))
        fetch_id = validate_provider_identity(getattr(self.fetch_provider, "provider", None))
        if search_id != self.policy.search_provider_id:
            raise InformationLiveProviderRegistryError("Exact live search provider is missing.")
        if fetch_id != self.policy.fetch_provider_id:
            raise InformationLiveProviderRegistryError("Exact live fetch provider is missing.")
        if getattr(self.search_provider, "provider_type", None) != "live":
            raise InformationLiveProviderRegistryError("Search provider must be live.")
        if getattr(self.fetch_provider, "provider_type", None) != "live":
            raise InformationLiveProviderRegistryError("Fetch provider must be live.")
        if not callable(getattr(self.search_provider, "search", None)):
            raise InformationLiveProviderRegistryError("Search operation is unavailable.")
        if not callable(getattr(self.fetch_provider, "fetch", None)):
            raise InformationLiveProviderRegistryError("Fetch operation is unavailable.")

    def resolve_search(self, provider: str) -> object:
        if validate_provider_identity(provider) != self.policy.search_provider_id:
            raise InformationLiveProviderRegistryError("No fallback search provider is registered.")
        self.validate()
        return self.search_provider

    def resolve_fetch(self, provider: str) -> object:
        if validate_provider_identity(provider) != self.policy.fetch_provider_id:
            raise InformationLiveProviderRegistryError("No fallback fetch provider is registered.")
        self.validate()
        return self.fetch_provider

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "policy_binding": self.policy.binding,
            "search_provider": self.policy.search_provider_id,
            "fetch_provider": self.policy.fetch_provider_id,
            "fallback_allowed": False,
            "provider_count": 2,
        }
