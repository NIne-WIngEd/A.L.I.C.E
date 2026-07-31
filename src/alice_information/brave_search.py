"""Brave Search API request, parsing, provider, and metadata receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Callable
from urllib.parse import urlencode, urlsplit

from .brave_search_live import BraveSearchRawResponse, BraveSearchTransport
from .contracts import InformationQuery, InformationSearchResult, canonicalize_public_url
from .live_provider_config import InformationLiveProviderConfiguration
from .live_provider_contracts import (
    InformationLiveEgressReceipt,
    InformationLiveProviderExecutionError,
    InformationLiveProviderFailure,
    InformationLiveRateLimitState,
    InformationLiveSearchResponse,
    sequence_sha256,
)
from .live_provider_policy import InformationLiveProviderRuntimePolicy
from .providers import InformationCancellationToken, validate_provider_identity


class BraveSearchProviderError(ValueError):
    """Raised when a Brave request or response violates the exact profile."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _failure(code: str) -> InformationLiveProviderExecutionError:
    return InformationLiveProviderExecutionError(
        InformationLiveProviderFailure.create(
            provider="brave-search-v1", operation="search", code=code
        )
    )


def _header_map(response: BraveSearchRawResponse) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in response.headers:
        result.setdefault(name.casefold(), value.strip())
    return result


def _rate_limit(headers: dict[str, str]) -> InformationLiveRateLimitState:
    state = InformationLiveRateLimitState(
        limit=headers.get("x-ratelimit-limit"),
        policy=headers.get("x-ratelimit-policy"),
        remaining=headers.get("x-ratelimit-remaining"),
        reset=headers.get("x-ratelimit-reset"),
    )
    state.validate()
    return state


def build_brave_search_url(
    query: InformationQuery,
    *,
    configuration: InformationLiveProviderConfiguration,
    policy: InformationLiveProviderRuntimePolicy,
    max_results: int,
) -> str:
    query.validate()
    configuration.validate(policy=policy)
    if query.data_classification != "PUBLIC":
        raise BraveSearchProviderError("Brave Search accepts PUBLIC queries only.")
    if len(query.text) > policy.max_query_characters or len(query.text.split()) > policy.max_query_words:
        raise BraveSearchProviderError("Query exceeds the exact Brave Search limit.")
    if not 1 <= max_results <= policy.max_results:
        raise BraveSearchProviderError("Requested result count exceeds P4.10a policy.")
    parameters = (
        ("q", query.text),
        ("count", str(max_results)),
        ("offset", "0"),
        ("result_filter", "web"),
        ("country", configuration.country),
        ("search_lang", configuration.search_lang),
        ("ui_lang", configuration.ui_lang),
        ("safesearch", configuration.safesearch),
        ("spellcheck", "false"),
        ("text_decorations", "false"),
        ("extra_snippets", "false"),
        ("summary", "false"),
        ("enable_rich_callback", "false"),
        ("include_fetch_metadata", "false"),
    )
    return f"https://{policy.search_host}{policy.search_path}?{urlencode(parameters)}"


def _clean_text(value: object, *, maximum: int, fallback: str | None = None) -> str:
    if not isinstance(value, str):
        if fallback is not None:
            return fallback
        raise BraveSearchProviderError("Brave result text is missing.")
    normalized = " ".join(value.replace("\x00", " ").split()).strip()
    if not normalized:
        if fallback is not None:
            return fallback
        raise BraveSearchProviderError("Brave result text is empty.")
    if len(normalized) > maximum:
        raise BraveSearchProviderError("Brave result text exceeded the public contract.")
    return normalized


def parse_brave_search_response(
    response: BraveSearchRawResponse,
    *,
    query: InformationQuery,
    policy: InformationLiveProviderRuntimePolicy,
    retrieved_at: str,
    max_results: int,
) -> tuple[InformationSearchResult, ...]:
    response.validate(maximum_bytes=policy.max_response_bytes)
    headers = _header_map(response)
    if response.status_code in {401, 403}:
        raise _failure("live_provider_authentication_failed")
    if response.status_code == 429:
        remaining = headers.get("x-ratelimit-remaining", "")
        code = "live_provider_quota_exhausted" if remaining.startswith("0") else "live_provider_rate_limited"
        raise _failure(code)
    if 500 <= response.status_code <= 599:
        raise _failure("live_provider_unavailable")
    if response.status_code != 200:
        raise _failure("live_provider_protocol_error")
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        raise _failure("live_provider_protocol_error")
    try:
        payload = json.loads(response.body.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise _failure("live_provider_protocol_error") from exc
    if not isinstance(payload, dict):
        raise _failure("live_provider_protocol_error")
    query_record = payload.get("query")
    if isinstance(query_record, dict):
        original = query_record.get("original")
        if original is not None and original != query.text:
            raise _failure("live_provider_protocol_error")
    web = payload.get("web", {})
    if not isinstance(web, dict):
        raise _failure("live_provider_protocol_error")
    entries = web.get("results", [])
    if not isinstance(entries, list) or len(entries) > max_results:
        raise _failure("live_provider_protocol_error")
    results: list[InformationSearchResult] = []
    seen_urls: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise _failure("live_provider_protocol_error")
        raw_url = entry.get("url")
        if not isinstance(raw_url, str):
            raise _failure("live_provider_protocol_error")
        try:
            canonical = canonicalize_public_url(raw_url)
        except ValueError as exc:
            raise _failure("live_provider_protocol_error") from exc
        parsed = urlsplit(canonical)
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
            raise _failure("live_provider_protocol_error")
        if canonical in seen_urls:
            continue
        seen_urls.add(canonical)
        title = _clean_text(entry.get("title"), maximum=500, fallback=parsed.hostname or "External source")
        snippet = _clean_text(entry.get("description"), maximum=10_000, fallback="No provider snippet supplied.")
        rank = len(results) + 1
        result_id = "brave-result-" + hashlib.sha256(
            f"{query.query_id}\n{rank}\n{canonical}".encode("utf-8")
        ).hexdigest()[:20]
        results.append(
            InformationSearchResult.create(
                result_id=result_id,
                query_id=query.query_id,
                provider=policy.search_provider_id,
                rank=rank,
                title=title,
                url=canonical,
                snippet=snippet,
                retrieved_at=retrieved_at,
            )
        )
    return tuple(results)


@dataclass
class BraveInformationSearchProvider:
    """Exact one-call Brave Search provider with metadata-only egress receipts."""

    policy: InformationLiveProviderRuntimePolicy
    configuration: InformationLiveProviderConfiguration
    transport: BraveSearchTransport
    clock: Callable[[], str] = _now
    monotonic_clock: Callable[[], float] = monotonic
    provider: str = field(default="brave-search-v1", init=False)
    provider_type: str = field(default="live", init=False)
    last_response: InformationLiveSearchResponse | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.policy.validate()
        self.configuration.validate(policy=self.policy)
        if validate_provider_identity(self.provider) != self.policy.search_provider_id:
            raise BraveSearchProviderError("Brave provider identity changed.")
        if getattr(self.transport, "transport_type", None) not in {
            "brave-direct-https-v1",
            "deterministic_fixture",
        }:
            raise BraveSearchProviderError("Brave transport identity is not approved.")
        if not callable(self.clock) or not callable(self.monotonic_clock):
            raise BraveSearchProviderError("Brave provider clocks must be callable.")

    def search(
        self,
        query: InformationQuery,
        *,
        max_results: int,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[InformationSearchResult, ...]:
        return self.search_with_receipt(
            query,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
        ).results  # type: ignore[return-value]

    def search_with_receipt(
        self,
        query: InformationQuery,
        *,
        max_results: int,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
        operation: str = "search",
    ) -> InformationLiveSearchResponse:
        query.validate()
        url = build_brave_search_url(
            query,
            configuration=self.configuration,
            policy=self.policy,
            max_results=max_results,
        )
        started = self.clock()
        started_mono = self.monotonic_clock()
        raw = self.transport.perform(
            canonical_url=url,
            credential_header=self.policy.credential_header,
            credential=self.configuration.credential,
            timeout_seconds=timeout_seconds,
            maximum_response_bytes=self.policy.max_response_bytes,
            cancellation=cancellation,
        )
        completed = self.clock()
        results = parse_brave_search_response(
            raw,
            query=query,
            policy=self.policy,
            retrieved_at=completed,
            max_results=max_results,
        )
        headers = _header_map(raw)
        receipt = InformationLiveEgressReceipt.create(
            provider=self.provider,
            operation=operation,
            endpoint_host=self.policy.search_host,
            endpoint_path=self.policy.search_path,
            query_id=query.query_id,
            query_sha256=query.content_sha256,
            configuration_sha256=self.configuration.configuration_sha256,
            started_at=started,
            completed_at=completed,
            elapsed_milliseconds=max(
                0, int((self.monotonic_clock() - started_mono) * 1000)
            ),
            status_code=raw.status_code,
            peer_address=raw.peer_address,
            item_count=len(results),
            response_sha256=hashlib.sha256(raw.body).hexdigest(),
            item_sequence_sha256=sequence_sha256(result.content_sha256 for result in results),
            rate_limit=_rate_limit(headers),
            policy_binding=self.policy.binding,
        )
        response = InformationLiveSearchResponse(results=results, receipt=receipt)
        response.validate()
        self.last_response = response
        return response
