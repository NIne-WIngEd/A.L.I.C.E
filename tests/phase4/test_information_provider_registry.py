"""P4.1 exact provider-registry tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from alice_information.contracts import InformationQuery, sha256_text
from alice_information.policy import InformationPolicyError, load_information_policy
from alice_information.provider_policy import load_information_provider_policy
from alice_information.providers import (
    DeterministicInformationFetchProvider,
    DeterministicInformationSearchProvider,
    InformationProviderConfigurationError,
    InformationSearchFixture,
    InformationSourceFixture,
)
from alice_information.registry import InformationProviderRegistry

NOW = "2026-07-26T22:00:00Z"
PROVIDER = "deterministic-fixture-v1"


def _query() -> InformationQuery:
    return InformationQuery.create(
        query_id="query-registry-001",
        text="registry fixture",
        created_at=NOW,
    )


def _search_provider() -> DeterministicInformationSearchProvider:
    return DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={
            _query().content_sha256: (
                InformationSearchFixture(
                    result_id="result-registry-001",
                    rank=1,
                    title="Registry source",
                    canonical_url="https://example.com/registry",
                    snippet="Registry fixture.",
                    retrieved_at=NOW,
                ),
            )
        },
    )


def _fetch_provider() -> DeterministicInformationFetchProvider:
    return DeterministicInformationFetchProvider(
        provider=PROVIDER,
        fixtures={
            "https://example.com/registry": InformationSourceFixture(
                source_id="source-registry-001",
                canonical_url="https://example.com/registry",
                title="Registry source",
                normalized_text="Registry normalized fixture.",
                retrieved_at=NOW,
            )
        },
    )


def _registry() -> InformationProviderRegistry:
    return InformationProviderRegistry(
        information_policy=load_information_policy(),
        provider_policy=load_information_provider_policy(),
    )


def test_registry_registers_and_resolves_exact_operation_identity() -> None:
    registry = _registry()
    search = _search_provider()
    fetch = _fetch_provider()
    registry.register_search(search)
    registry.register_fetch(fetch)
    assert registry.resolve_search(provider=PROVIDER) is search
    assert registry.resolve_fetch(provider=PROVIDER) is fetch
    assert registry.identities() == (
        ("fetch", PROVIDER),
        ("search", PROVIDER),
    )


def test_registry_has_no_provider_fallback() -> None:
    registry = _registry()
    registry.register_search(_search_provider())
    with pytest.raises(InformationProviderConfigurationError, match="not registered"):
        registry.resolve_search(provider="missing-provider")
    with pytest.raises(InformationProviderConfigurationError, match="not registered"):
        registry.resolve_fetch(provider=PROVIDER)


def test_registry_rejects_duplicates_and_unapproved_identity() -> None:
    registry = _registry()
    provider = _search_provider()
    registry.register_search(provider)
    with pytest.raises(InformationProviderConfigurationError, match="already"):
        registry.register_search(provider)
    unapproved = DeterministicInformationSearchProvider(
        provider="unapproved-fixture",
        fixtures={
            sha256_text("other"): (
                InformationSearchFixture(
                    result_id="result-other",
                    rank=1,
                    title="Other",
                    canonical_url="https://example.com/other",
                    snippet="Other",
                    retrieved_at=NOW,
                ),
            )
        },
    )
    with pytest.raises(InformationProviderConfigurationError, match="not approved"):
        registry.register_search(unapproved)


def test_registry_rejects_live_provider_even_with_callable_method() -> None:
    @dataclass
    class FakeLiveProvider:
        provider: str = PROVIDER
        provider_type: str = "live"

        def search(self, *args, **kwargs):  # pragma: no cover - never called
            raise AssertionError("must not execute")

    with pytest.raises(InformationProviderConfigurationError, match="not approved"):
        _registry().register_search(FakeLiveProvider())


def test_registry_rejects_missing_protocol_method() -> None:
    @dataclass
    class MissingSearch:
        provider: str = PROVIDER
        provider_type: str = "deterministic_fixture"

    with pytest.raises(InformationProviderConfigurationError, match="callable search"):
        _registry().register_search(MissingSearch())


def test_registry_revalidates_foundation_policy_projection() -> None:
    policy = replace(load_information_policy(), permission_id="web.write")
    with pytest.raises(InformationPolicyError, match="web.search"):
        InformationProviderRegistry(
            information_policy=policy,
            provider_policy=load_information_provider_policy(),
        )


def test_registry_revalidates_adapter_identity_on_resolution() -> None:
    registry = _registry()
    search = _search_provider()
    fetch = _fetch_provider()
    registry.register_search(search)
    registry.register_fetch(fetch)

    search.provider = "unapproved-fixture"
    fetch.provider = "unapproved-fixture"

    with pytest.raises(InformationProviderConfigurationError, match="not approved"):
        registry.resolve_search(provider=PROVIDER)
    with pytest.raises(InformationProviderConfigurationError, match="not approved"):
        registry.resolve_fetch(provider=PROVIDER)
