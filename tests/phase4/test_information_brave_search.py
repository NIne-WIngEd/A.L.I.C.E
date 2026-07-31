import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from alice_information.brave_search import (
    BraveInformationSearchProvider,
    BraveSearchProviderError,
    build_brave_search_url,
    parse_brave_search_response,
)
from alice_information.brave_search_live import BraveSearchRawResponse
from alice_information.contracts import InformationQuery
from alice_information.live_provider_config import (
    InformationLiveProviderConfiguration,
    InformationSecretValue,
)
from alice_information.live_provider_contracts import InformationLiveProviderExecutionError
from alice_information.live_provider_policy import (
    InformationLiveProviderRuntimePolicy,
    canonical_json_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = InformationLiveProviderRuntimePolicy.load(
    ROOT / "policies/information_live_provider_runtime_policy.json"
)
META = {
    "provider": "brave-search-v1",
    "country": "US",
    "search_lang": "en",
    "ui_lang": "en-US",
    "safesearch": "off",
}
CONFIG = InformationLiveProviderConfiguration(
    **META,
    configuration_path="C:/ALICE_Vault/config/phase4-live-provider.json",
    configuration_sha256=hashlib.sha256(canonical_json_bytes(META)).hexdigest(),
    credential=InformationSecretValue("token"),
    metadata=META,
)
QUERY = InformationQuery.create(
    query_id="q1",
    text="current OpenAI news",
    created_at="2026-07-30T12:00:00Z",
)


def response(payload, *, status=200, headers=()):
    return BraveSearchRawResponse(
        status_code=status,
        headers=(("Content-Type", "application/json"),) + tuple(headers),
        body=json.dumps(payload).encode(),
        peer_address="1.1.1.1",
    )


def test_request_is_exact_and_disables_provider_mutation_features():
    url = build_brave_search_url(QUERY, configuration=CONFIG, policy=POLICY, max_results=3)
    assert url.startswith("https://api.search.brave.com/res/v1/web/search?")
    assert "offset=0" in url
    assert "spellcheck=false" in url
    assert "summary=false" in url
    assert "enable_rich_callback=false" in url
    assert "count=3" in url


def test_query_boundaries_are_enforced():
    long_query = InformationQuery.create(
        query_id="q2",
        text="x" * 401,
        created_at="2026-07-30T12:00:00Z",
    )
    with pytest.raises(BraveSearchProviderError):
        build_brave_search_url(long_query, configuration=CONFIG, policy=POLICY, max_results=1)


def test_parser_creates_https_results_and_deduplicates_urls():
    raw = response(
        {
            "query": {"original": QUERY.text},
            "web": {
                "results": [
                    {"title": "One", "url": "https://example.com/a", "description": "Alpha"},
                    {"title": "Duplicate", "url": "https://example.com/a", "description": "Again"},
                    {"title": "Two", "url": "https://example.org/b", "description": "Beta"},
                ]
            },
        }
    )
    results = parse_brave_search_response(
        raw,
        query=QUERY,
        policy=POLICY,
        retrieved_at="2026-07-30T12:00:01Z",
        max_results=3,
    )
    assert [item.rank for item in results] == [1, 2]
    assert [item.canonical_url for item in results] == [
        "https://example.com/a",
        "https://example.org/b",
    ]


@pytest.mark.parametrize(
    "raw,code",
    [
        (response({}, status=401), "live_provider_authentication_failed"),
        (response({}, status=429, headers=(("X-RateLimit-Remaining", "1"),)), "live_provider_rate_limited"),
        (response({}, status=503), "live_provider_unavailable"),
    ],
)
def test_http_failures_are_sanitized(raw, code):
    with pytest.raises(InformationLiveProviderExecutionError) as caught:
        parse_brave_search_response(
            raw,
            query=QUERY,
            policy=POLICY,
            retrieved_at="2026-07-30T12:00:01Z",
            max_results=1,
        )
    assert caught.value.failure.code == code


def test_provider_receipt_never_contains_query_text_or_token():
    @dataclass
    class FakeTransport:
        transport_type: str = "deterministic_fixture"
        def perform(self, **kwargs):
            assert kwargs["credential"].reveal_for_exact_header() == "token"
            return response(
                {"query": {"original": QUERY.text}, "web": {"results": [
                    {"title": "One", "url": "https://example.com/a", "description": "Alpha"}
                ]}},
                headers=(("X-RateLimit-Remaining", "10"),),
            )
    provider = BraveInformationSearchProvider(
        policy=POLICY,
        configuration=CONFIG,
        transport=FakeTransport(),
        clock=iter(("2026-07-30T12:00:00Z", "2026-07-30T12:00:01Z")).__next__,
    )
    result = provider.search_with_receipt(QUERY, max_results=1, timeout_seconds=5)
    rendered = json.dumps(result.receipt.to_metadata_record())
    assert QUERY.text not in rendered
    assert "token" not in rendered
    assert result.receipt.item_count == 1
