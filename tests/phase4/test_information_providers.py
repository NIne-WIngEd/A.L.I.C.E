"""P4.1 provider protocol and deterministic fixture tests."""

from __future__ import annotations

from dataclasses import replace
import socket
from urllib import request as urllib_request

import pytest
from alice_information.contracts import InformationQuery, InformationSearchResult
from alice_information.providers import (
    DeterministicInformationFetchProvider,
    DeterministicInformationSearchProvider,
    InformationCancellationToken,
    InformationProviderCancelledError,
    InformationProviderConfigurationError,
    InformationProviderExecutionError,
    InformationProviderFailure,
    InformationSearchFixture,
    InformationSourceFixture,
)

NOW = "2026-07-26T22:00:00Z"
QUERY_TEXT = "phase 4 deterministic provider"
PROVIDER = "deterministic-fixture-v1"


def _query() -> InformationQuery:
    return InformationQuery.create(
        query_id="query-p41-001",
        text=QUERY_TEXT,
        created_at=NOW,
    )


def _search_fixture(
    *,
    result_id: str = "result-p41-001",
    rank: int = 1,
    url: str = "https://example.com/research",
) -> InformationSearchFixture:
    return InformationSearchFixture(
        result_id=result_id,
        rank=rank,
        title="Deterministic source",
        canonical_url=url,
        snippet="Fixture snippet that remains untrusted.",
        retrieved_at=NOW,
    )


def _source_fixture(
    *,
    source_id: str = "source-p41-001",
    url: str = "https://example.com/research",
    text: str = "Deterministic normalized source content.",
) -> InformationSourceFixture:
    return InformationSourceFixture(
        source_id=source_id,
        canonical_url=url,
        title="Deterministic source",
        normalized_text=text,
        retrieved_at=NOW,
        published_at="2026-07-25T10:00:00Z",
        updated_at="2026-07-26T10:00:00Z",
    )


def _search_provider() -> DeterministicInformationSearchProvider:
    query = _query()
    return DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={query.content_sha256: (_search_fixture(),)},
    )


def _search_result() -> InformationSearchResult:
    return _search_provider().search(
        _query(),
        max_results=3,
        timeout_seconds=5,
    )[0]


def test_deterministic_search_replays_identical_digest_bound_results() -> None:
    query = _query()
    provider = _search_provider()
    first = provider.search(query, max_results=3, timeout_seconds=5)
    second = provider.search(query, max_results=3, timeout_seconds=5)
    assert first == second
    assert first[0].query_id == query.query_id
    assert first[0].provider == PROVIDER
    assert provider.query_digests == [query.content_sha256, query.content_sha256]


def test_deterministic_search_enforces_order_and_result_budget() -> None:
    query = _query()
    provider = DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={
            query.content_sha256: (
                _search_fixture(
                    result_id="result-p41-002",
                    rank=2,
                    url="https://example.com/two",
                ),
                _search_fixture(
                    result_id="result-p41-001",
                    rank=1,
                    url="https://example.com/one",
                ),
            )
        },
    )
    results = provider.search(query, max_results=1, timeout_seconds=5)
    assert tuple(item.rank for item in results) == (1,)
    with pytest.raises(InformationProviderConfigurationError, match="max_results"):
        provider.search(query, max_results=21, timeout_seconds=5)


def test_search_fixture_configuration_fails_closed() -> None:
    query = _query()
    with pytest.raises(InformationProviderConfigurationError, match="canonical"):
        DeterministicInformationSearchProvider(
            provider=PROVIDER,
            fixtures={
                query.content_sha256: (
                    _search_fixture(url="HTTPS://Example.COM/research"),
                )
            },
        )
    with pytest.raises(InformationProviderConfigurationError, match="ranks"):
        DeterministicInformationSearchProvider(
            provider=PROVIDER,
            fixtures={
                query.content_sha256: (
                    _search_fixture(),
                    _search_fixture(
                        result_id="result-p41-002",
                        url="https://example.com/two",
                    ),
                )
            },
        )


def test_missing_search_fixture_uses_sanitized_failure() -> None:
    provider = DeterministicInformationSearchProvider(
        provider=PROVIDER,
        fixtures={"0" * 64: (_search_fixture(),)},
    )
    with pytest.raises(InformationProviderExecutionError) as raised:
        provider.search(_query(), max_results=3, timeout_seconds=5)
    failure = raised.value.failure
    assert failure.code == "provider_fixture_missing"
    assert QUERY_TEXT not in str(raised.value)
    assert QUERY_TEXT not in failure.message


def test_cancellation_is_checked_before_search_and_fetch() -> None:
    token = InformationCancellationToken()
    token.cancel()
    with pytest.raises(InformationProviderCancelledError):
        _search_provider().search(
            _query(),
            max_results=3,
            timeout_seconds=5,
            cancellation=token,
        )
    fetch = DeterministicInformationFetchProvider(
        provider=PROVIDER,
        fixtures={
            "https://example.com/research": _source_fixture(),
        },
    )
    with pytest.raises(InformationProviderCancelledError):
        fetch.fetch(
            _search_result(),
            timeout_seconds=5,
            max_response_bytes=1000,
            cancellation=token,
        )


def test_deterministic_fetch_returns_exact_source_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("deterministic provider attempted network access")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib_request, "urlopen", fail_network)
    search_result = _search_provider().search(
        _query(),
        max_results=3,
        timeout_seconds=5,
    )[0]
    provider = DeterministicInformationFetchProvider(
        provider=PROVIDER,
        fixtures={
            "https://example.com/research": _source_fixture(),
        },
    )
    result = search_result
    source = provider.fetch(
        result,
        timeout_seconds=5,
        max_response_bytes=1000,
    )
    source.validate()
    assert source.provider == PROVIDER
    assert source.canonical_url == result.canonical_url
    assert source.normalized_text == "Deterministic normalized source content."
    assert provider.result_ids == [result.result_id]


def test_fetch_enforces_exact_url_fixture_and_byte_budget() -> None:
    result = _search_result()
    provider = DeterministicInformationFetchProvider(
        provider=PROVIDER,
        fixtures={
            result.canonical_url: _source_fixture(text="x" * 20),
        },
    )
    with pytest.raises(InformationProviderExecutionError) as raised:
        provider.fetch(
            result,
            timeout_seconds=5,
            max_response_bytes=10,
        )
    assert raised.value.failure.code == "response_too_large"

    other = replace(result, canonical_url="https://example.com/missing")
    other.validate()
    with pytest.raises(InformationProviderExecutionError) as missing:
        provider.fetch(
            other,
            timeout_seconds=5,
            max_response_bytes=1000,
        )
    assert missing.value.failure.code == "provider_fixture_missing"


def test_provider_identity_timeout_and_failure_vocabularies_are_strict() -> None:
    with pytest.raises(InformationProviderConfigurationError, match="lowercase"):
        DeterministicInformationSearchProvider(
            provider="Bad Provider",
            fixtures={_query().content_sha256: (_search_fixture(),)},
        )
    with pytest.raises(InformationProviderConfigurationError, match="timeout"):
        _search_provider().search(_query(), max_results=1, timeout_seconds=0)
    with pytest.raises(InformationProviderConfigurationError, match="vocabulary"):
        InformationProviderFailure(
            provider=PROVIDER,
            operation="search",
            code="raw-query-text",
            message="not approved",
            retryable=False,
        ).validate()
    with pytest.raises(InformationProviderConfigurationError, match="sanitized text"):
        InformationProviderFailure(
            provider=PROVIDER,
            operation="search",
            code="provider_timeout",
            message=QUERY_TEXT,
            retryable=True,
        ).validate()
