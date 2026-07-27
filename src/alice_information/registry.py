"""Exact, no-fallback provider registry for Phase 4 P4.1."""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy import InformationPolicy
from .provider_policy import InformationProviderPolicy
from .providers import (
    InformationFetchProvider,
    InformationProviderConfigurationError,
    InformationSearchProvider,
    validate_provider_identity,
)


@dataclass
class InformationProviderRegistry:
    """Explicit registry keyed by operation and exact provider identity."""

    information_policy: InformationPolicy
    provider_policy: InformationProviderPolicy
    _search_providers: dict[str, InformationSearchProvider] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _fetch_providers: dict[str, InformationFetchProvider] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.information_policy.validate()
        self.provider_policy.validate()
        if self.information_policy.approved_live_providers:
            raise InformationProviderConfigurationError(
                "P4.1 cannot use a foundation policy with live providers."
            )
        if not self.information_policy.deterministic_fixture_mode_allowed:
            raise InformationProviderConfigurationError(
                "P4.1 requires deterministic fixture mode."
            )
        if self.provider_policy.live_network_access_allowed:
            raise InformationProviderConfigurationError(
                "P4.1 provider registry cannot enable live network access."
            )
        if self.provider_policy.provider_fallback_allowed:
            raise InformationProviderConfigurationError(
                "P4.1 provider registry cannot enable provider fallback."
            )

    def _validate_adapter(
        self,
        adapter: object,
        *,
        operation: str,
        method_name: str,
    ) -> tuple[str, str]:
        provider = validate_provider_identity(
            getattr(adapter, "provider", None),
        )
        provider_type = getattr(adapter, "provider_type", None)
        if not isinstance(provider_type, str) or not provider_type.strip():
            raise InformationProviderConfigurationError(
                "Information provider_type must be non-empty text."
            )
        provider_type = provider_type.strip()
        method = getattr(adapter, method_name, None)
        if not callable(method):
            raise InformationProviderConfigurationError(
                f"Information provider must provide a callable {method_name} method."
            )
        if not self.provider_policy.allows(
            provider=provider,
            provider_type=provider_type,
            operation=operation,
        ):
            raise InformationProviderConfigurationError(
                f"Information provider is not approved for {operation}: {provider}"
            )
        return provider, provider_type

    def register_search(self, adapter: InformationSearchProvider) -> None:
        provider, _ = self._validate_adapter(
            adapter,
            operation="search",
            method_name="search",
        )
        if provider in self._search_providers:
            raise InformationProviderConfigurationError(
                f"Search provider already registered: {provider}"
            )
        self._search_providers[provider] = adapter

    def register_fetch(self, adapter: InformationFetchProvider) -> None:
        provider, _ = self._validate_adapter(
            adapter,
            operation="fetch",
            method_name="fetch",
        )
        if provider in self._fetch_providers:
            raise InformationProviderConfigurationError(
                f"Fetch provider already registered: {provider}"
            )
        self._fetch_providers[provider] = adapter

    def resolve_search(self, *, provider: str) -> InformationSearchProvider:
        exact = validate_provider_identity(provider)
        try:
            adapter = self._search_providers[exact]
        except KeyError as exc:
            raise InformationProviderConfigurationError(
                f"Search provider is not registered: {exact}"
            ) from exc
        current, _ = self._validate_adapter(
            adapter,
            operation="search",
            method_name="search",
        )
        if current != exact:
            raise InformationProviderConfigurationError(
                "Registered search provider identity changed after registration."
            )
        return adapter

    def resolve_fetch(self, *, provider: str) -> InformationFetchProvider:
        exact = validate_provider_identity(provider)
        try:
            adapter = self._fetch_providers[exact]
        except KeyError as exc:
            raise InformationProviderConfigurationError(
                f"Fetch provider is not registered: {exact}"
            ) from exc
        current, _ = self._validate_adapter(
            adapter,
            operation="fetch",
            method_name="fetch",
        )
        if current != exact:
            raise InformationProviderConfigurationError(
                "Registered fetch provider identity changed after registration."
            )
        return adapter

    def identities(self) -> tuple[tuple[str, str], ...]:
        identities = [
            ("search", provider) for provider in self._search_providers
        ] + [("fetch", provider) for provider in self._fetch_providers]
        return tuple(sorted(identities))
