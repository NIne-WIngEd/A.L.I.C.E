"""Provider-neutral, network-free information adapters for Phase 4 P4.1.

P4.1 establishes search and fetch protocols plus deterministic fixture
providers. It intentionally contains no HTTP client and cannot open a network
connection. Provider failures expose only approved metadata and messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Event
from types import MappingProxyType
from typing import Mapping, Protocol

from .contracts import (
    InformationContractError,
    InformationQuery,
    InformationSearchResult,
    InformationSourceDocument,
    canonicalize_public_url,
)

INFORMATION_PROVIDER_TYPES = ("deterministic_fixture", "live")
INFORMATION_PROVIDER_FAILURE_MESSAGES = {
    "provider_fixture_missing": "Deterministic provider fixture is unavailable.",
    "provider_protocol_error": "Information provider response was invalid.",
    "provider_timeout": "Information provider exceeded its approved timeout.",
    "response_too_large": "Information source exceeded the approved byte limit.",
}
INFORMATION_PROVIDER_FAILURE_CODES = tuple(INFORMATION_PROVIDER_FAILURE_MESSAGES)
_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class InformationProviderError(RuntimeError):
    """Base error for Phase 4 provider adapters."""


class InformationProviderConfigurationError(InformationProviderError):
    """Raised when a provider or registry configuration is invalid."""


class InformationProviderTimeoutError(InformationProviderError):
    """Raised when a provider exceeds its approved timeout."""


class InformationProviderCancelledError(InformationProviderError):
    """Raised when an information operation is cooperatively cancelled."""


class InformationProviderProtocolError(InformationProviderError):
    """Raised when a provider result violates the public contract."""


def validate_provider_identity(value: object, *, field: str = "provider") -> str:
    """Return one normalized exact provider identifier."""

    if not isinstance(value, str) or not value.strip():
        raise InformationProviderConfigurationError(
            f"Information {field} must be non-empty text."
        )
    normalized = value.strip()
    if _PROVIDER_ID_PATTERN.fullmatch(normalized) is None:
        raise InformationProviderConfigurationError(
            f"Information {field} must use lowercase token characters only."
        )
    return normalized


def _validate_timeout(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InformationProviderConfigurationError(
            "Provider timeout must be numeric."
        )
    timeout = float(value)
    if not 0 < timeout <= 30:
        raise InformationProviderConfigurationError(
            "Provider timeout must be between 0 and 30 seconds."
        )
    return timeout


def _validate_limit(value: object, *, field: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InformationProviderConfigurationError(f"{field} must be an integer.")
    if not 1 <= value <= maximum:
        raise InformationProviderConfigurationError(
            f"{field} must be between 1 and {maximum}."
        )
    return value


def _validate_fixture_digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise InformationProviderConfigurationError(
            "Fixture query keys must be lowercase SHA-256 digests."
        )
    return value


@dataclass(frozen=True)
class InformationProviderFailure:
    """Sanitized provider-side failure metadata."""

    provider: str
    operation: str
    code: str
    message: str
    retryable: bool

    def validate(self) -> None:
        validate_provider_identity(self.provider)
        if self.operation not in {"search", "fetch"}:
            raise InformationProviderConfigurationError(
                "Provider failure operation must be search or fetch."
            )
        if self.code not in INFORMATION_PROVIDER_FAILURE_CODES:
            raise InformationProviderConfigurationError(
                "Provider failure code is not in the approved vocabulary."
            )
        expected_message = INFORMATION_PROVIDER_FAILURE_MESSAGES.get(self.code)
        if self.message != expected_message:
            raise InformationProviderConfigurationError(
                "Provider failure message must match the approved sanitized text."
            )
        if not isinstance(self.retryable, bool):
            raise InformationProviderConfigurationError(
                "Provider failure retryable must be boolean."
            )


class InformationProviderExecutionError(InformationProviderError):
    """Raised for one sanitized deterministic provider failure."""

    def __init__(self, failure: InformationProviderFailure) -> None:
        failure.validate()
        self.failure = failure
        super().__init__(
            f"{failure.provider} {failure.operation} {failure.code}: "
            f"{failure.message}"
        )


class InformationCancellationToken:
    """Thread-safe cooperative cancellation token for information operations."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise InformationProviderCancelledError(
                "Information provider operation was cancelled."
            )


class InformationSearchProvider(Protocol):
    """Minimal read-only search-provider interface."""

    provider: str
    provider_type: str

    def search(
        self,
        query: InformationQuery,
        *,
        max_results: int,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[InformationSearchResult, ...]:
        """Return validated untrusted search results for one public query."""


class InformationFetchProvider(Protocol):
    """Minimal read-only source-fetch interface."""

    provider: str
    provider_type: str

    def fetch(
        self,
        result: InformationSearchResult,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationSourceDocument:
        """Return one validated normalized public source document."""


@dataclass(frozen=True)
class InformationSearchFixture:
    """One deterministic search result template."""

    result_id: str
    rank: int
    title: str
    canonical_url: str
    snippet: str
    retrieved_at: str

    def validate(self) -> None:
        canonical = canonicalize_public_url(self.canonical_url)
        if canonical != self.canonical_url:
            raise InformationProviderConfigurationError(
                "Search fixture URL must already be canonical."
            )
        try:
            InformationSearchResult.create(
                result_id=self.result_id,
                query_id="fixture-query",
                provider="fixture-validation",
                rank=self.rank,
                title=self.title,
                url=self.canonical_url,
                snippet=self.snippet,
                retrieved_at=self.retrieved_at,
            )
        except InformationContractError as exc:
            raise InformationProviderConfigurationError(
                "Search fixture violates the information-result contract."
            ) from exc

    def to_result(self, *, query_id: str, provider: str) -> InformationSearchResult:
        self.validate()
        return InformationSearchResult.create(
            result_id=self.result_id,
            query_id=query_id,
            provider=provider,
            rank=self.rank,
            title=self.title,
            url=self.canonical_url,
            snippet=self.snippet,
            retrieved_at=self.retrieved_at,
        )


@dataclass(frozen=True)
class InformationSourceFixture:
    """One deterministic normalized source template."""

    source_id: str
    canonical_url: str
    title: str
    normalized_text: str
    retrieved_at: str
    published_at: str | None = None
    updated_at: str | None = None

    def validate(self) -> None:
        canonical = canonicalize_public_url(self.canonical_url)
        if canonical != self.canonical_url:
            raise InformationProviderConfigurationError(
                "Source fixture URL must already be canonical."
            )
        try:
            InformationSourceDocument.create(
                source_id=self.source_id,
                provider="fixture-validation",
                url=self.canonical_url,
                title=self.title,
                normalized_text=self.normalized_text,
                retrieved_at=self.retrieved_at,
                published_at=self.published_at,
                updated_at=self.updated_at,
            )
        except InformationContractError as exc:
            raise InformationProviderConfigurationError(
                "Source fixture violates the information-source contract."
            ) from exc

    def to_source(self, *, provider: str) -> InformationSourceDocument:
        self.validate()
        return InformationSourceDocument.create(
            source_id=self.source_id,
            provider=provider,
            url=self.canonical_url,
            title=self.title,
            normalized_text=self.normalized_text,
            retrieved_at=self.retrieved_at,
            published_at=self.published_at,
            updated_at=self.updated_at,
        )


@dataclass
class DeterministicInformationSearchProvider:
    """Network-free search adapter keyed by exact public-query digest."""

    provider: str
    fixtures: Mapping[str, tuple[InformationSearchFixture, ...]]
    provider_type: str = field(default="deterministic_fixture", init=False)
    query_digests: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.provider = validate_provider_identity(self.provider)
        normalized: dict[str, tuple[InformationSearchFixture, ...]] = {}
        for digest, fixtures in self.fixtures.items():
            fixture_digest = _validate_fixture_digest(digest)
            entries = tuple(fixtures)
            if not entries:
                raise InformationProviderConfigurationError(
                    "Each deterministic search fixture set must be non-empty."
                )
            ranks: set[int] = set()
            result_ids: set[str] = set()
            for fixture in entries:
                if not isinstance(fixture, InformationSearchFixture):
                    raise InformationProviderConfigurationError(
                        "Search fixture sets must contain InformationSearchFixture values."
                    )
                fixture.validate()
                if fixture.rank in ranks:
                    raise InformationProviderConfigurationError(
                        "Search fixture ranks must be unique per query."
                    )
                if fixture.result_id in result_ids:
                    raise InformationProviderConfigurationError(
                        "Search fixture result IDs must be unique per query."
                    )
                ranks.add(fixture.rank)
                result_ids.add(fixture.result_id)
            normalized[fixture_digest] = tuple(sorted(entries, key=lambda item: item.rank))
        self.fixtures = MappingProxyType(normalized)

    def search(
        self,
        query: InformationQuery,
        *,
        max_results: int,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[InformationSearchResult, ...]:
        query.validate()
        _validate_limit(max_results, field="max_results", maximum=20)
        _validate_timeout(timeout_seconds)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        entries = self.fixtures.get(query.content_sha256)
        if entries is None:
            raise InformationProviderExecutionError(
                InformationProviderFailure(
                    provider=self.provider,
                    operation="search",
                    code="provider_fixture_missing",
                    message=INFORMATION_PROVIDER_FAILURE_MESSAGES["provider_fixture_missing"],
                    retryable=False,
                )
            )
        self.query_digests.append(query.content_sha256)
        results = tuple(
            fixture.to_result(query_id=query.query_id, provider=self.provider)
            for fixture in entries[:max_results]
        )
        for result in results:
            try:
                result.validate()
            except InformationContractError as exc:
                raise InformationProviderProtocolError(
                    "Deterministic search provider returned an invalid result."
                ) from exc
            if result.provider != self.provider or result.query_id != query.query_id:
                raise InformationProviderProtocolError(
                    "Deterministic search provider changed result identity."
                )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return results


@dataclass
class DeterministicInformationFetchProvider:
    """Network-free source adapter keyed by exact canonical URL."""

    provider: str
    fixtures: Mapping[str, InformationSourceFixture]
    provider_type: str = field(default="deterministic_fixture", init=False)
    result_ids: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.provider = validate_provider_identity(self.provider)
        normalized: dict[str, InformationSourceFixture] = {}
        source_ids: set[str] = set()
        for url, fixture in self.fixtures.items():
            canonical = canonicalize_public_url(url)
            if canonical != url:
                raise InformationProviderConfigurationError(
                    "Fetch fixture map keys must already be canonical URLs."
                )
            if not isinstance(fixture, InformationSourceFixture):
                raise InformationProviderConfigurationError(
                    "Fetch fixtures must contain InformationSourceFixture values."
                )
            fixture.validate()
            if fixture.canonical_url != canonical:
                raise InformationProviderConfigurationError(
                    "Fetch fixture key must match its source canonical URL."
                )
            if fixture.source_id in source_ids:
                raise InformationProviderConfigurationError(
                    "Fetch fixture source IDs must be unique."
                )
            source_ids.add(fixture.source_id)
            normalized[canonical] = fixture
        self.fixtures = MappingProxyType(normalized)

    def fetch(
        self,
        result: InformationSearchResult,
        *,
        timeout_seconds: float,
        max_response_bytes: int,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationSourceDocument:
        result.validate()
        _validate_timeout(timeout_seconds)
        _validate_limit(
            max_response_bytes,
            field="max_response_bytes",
            maximum=5_000_000,
        )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        fixture = self.fixtures.get(result.canonical_url)
        if fixture is None:
            raise InformationProviderExecutionError(
                InformationProviderFailure(
                    provider=self.provider,
                    operation="fetch",
                    code="provider_fixture_missing",
                    message=INFORMATION_PROVIDER_FAILURE_MESSAGES["provider_fixture_missing"],
                    retryable=False,
                )
            )
        if len(fixture.normalized_text.encode("utf-8")) > max_response_bytes:
            raise InformationProviderExecutionError(
                InformationProviderFailure(
                    provider=self.provider,
                    operation="fetch",
                    code="response_too_large",
                    message=INFORMATION_PROVIDER_FAILURE_MESSAGES["response_too_large"],
                    retryable=False,
                )
            )
        self.result_ids.append(result.result_id)
        source = fixture.to_source(provider=self.provider)
        try:
            source.validate()
        except InformationContractError as exc:
            raise InformationProviderProtocolError(
                "Deterministic fetch provider returned an invalid source."
            ) from exc
        if source.provider != self.provider:
            raise InformationProviderProtocolError(
                "Deterministic fetch provider changed source identity."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return source
