"""Network-security primitives for Phase 4 P4.2 controlled retrieval.

The milestone deliberately provides deterministic resolver and transport
fixtures only. No socket, urllib, requests, proxy, or operating-system DNS API
is imported here. A later live adapter must implement these protocols while
preserving address pinning and every P4.2 policy gate.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol
from urllib.parse import urlsplit

from .contracts import canonicalize_public_url
from .providers import InformationCancellationToken
from .retrieval_policy import InformationHttpRetrievalPolicy


class InformationHttpError(RuntimeError):
    """Base error for the controlled HTTP boundary."""


class InformationHttpConfigurationError(InformationHttpError):
    """Raised when deterministic HTTP fixtures or policy are invalid."""


HTTP_FAILURE_MESSAGES = {
    "invalid_source_url": "Information source URL was rejected.",
    "dns_resolution_failed": "Information source hostname could not be resolved safely.",
    "private_network_blocked": "Information source resolved to a prohibited network.",
    "peer_address_mismatch": "Information transport peer did not match the approved address set.",
    "redirect_blocked": "Information source redirect was rejected.",
    "response_header_invalid": "Information source response headers were invalid.",
    "http_status_rejected": "Information source returned a rejected HTTP status.",
    "unsupported_content_type": "Information source content type was not approved.",
    "response_too_large": "Information source exceeded the approved byte limit.",
    "content_decode_failed": "Information source content could not be decoded safely.",
    "normalization_failed": "Information source content could not be normalized safely.",
    "network_connection_failed": "Information source connection could not be established safely.",
    "network_timeout": "Information source network operation exceeded the approved timeout.",
    "tls_validation_failed": "Information source TLS validation failed.",
    "http_protocol_invalid": "Information source HTTP response was invalid.",
}
HTTP_FAILURE_CODES = tuple(HTTP_FAILURE_MESSAGES)
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


@dataclass(frozen=True)
class InformationHttpFailure:
    """Sanitized P4.2 retrieval failure without raw URL or body text."""

    code: str
    message: str
    retryable: bool = False

    def validate(self) -> None:
        expected = HTTP_FAILURE_MESSAGES.get(self.code)
        if expected is None:
            raise InformationHttpConfigurationError(
                "HTTP failure code is not approved."
            )
        if self.message != expected:
            raise InformationHttpConfigurationError(
                "HTTP failure message must match the approved sanitized text."
            )
        if not isinstance(self.retryable, bool):
            raise InformationHttpConfigurationError(
                "HTTP failure retryable must be boolean."
            )


class InformationHttpExecutionError(InformationHttpError):
    """Raised for one sanitized controlled-retrieval failure."""

    def __init__(self, failure: InformationHttpFailure) -> None:
        failure.validate()
        self.failure = failure
        super().__init__(f"{failure.code}: {failure.message}")


def http_failure(code: str, *, retryable: bool = False) -> InformationHttpExecutionError:
    return InformationHttpExecutionError(
        InformationHttpFailure(
            code=code,
            message=HTTP_FAILURE_MESSAGES[code],
            retryable=retryable,
        )
    )


def validate_global_address(value: object) -> str:
    """Return one canonical globally routable IP address or fail closed."""

    if not isinstance(value, str) or not value.strip():
        raise http_failure("dns_resolution_failed")
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError as exc:
        raise http_failure("dns_resolution_failed") from exc
    if not address.is_global or address.is_multicast:
        raise http_failure("private_network_blocked")
    return address.compressed


@dataclass(frozen=True)
class InformationResolvedTarget:
    """One canonical URL pinned to a validated public address set."""

    canonical_url: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]

    def validate(self, *, policy: InformationHttpRetrievalPolicy) -> None:
        canonical = canonicalize_public_url(self.canonical_url)
        if canonical != self.canonical_url:
            raise http_failure("invalid_source_url")
        parsed = urlsplit(canonical)
        if parsed.scheme != self.scheme or parsed.hostname != self.hostname:
            raise InformationHttpConfigurationError(
                "Resolved target identity does not match its canonical URL."
            )
        if self.hostname.endswith(".") or "%" in self.hostname:
            raise http_failure("invalid_source_url")
        expected_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if self.port != expected_port:
            raise InformationHttpConfigurationError(
                "Resolved target port does not match its canonical URL."
            )
        if self.port not in policy.allowed_ports_for(self.scheme):
            raise http_failure("invalid_source_url")
        if not self.addresses:
            raise http_failure("dns_resolution_failed")
        normalized = tuple(validate_global_address(item) for item in self.addresses)
        if normalized != self.addresses:
            raise InformationHttpConfigurationError(
                "Resolved addresses must already be canonical."
            )
        if len(set(normalized)) != len(normalized):
            raise InformationHttpConfigurationError(
                "Resolved addresses cannot contain duplicates."
            )


class InformationNameResolver(Protocol):
    """Resolver interface that must return all addresses for one hostname."""

    resolver_type: str

    def resolve(
        self,
        canonical_url: str,
        *,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationResolvedTarget:
        """Resolve and validate one canonical URL at point of use."""


@dataclass
class DeterministicInformationNameResolver:
    """Network-free resolver fixture keyed by exact lowercase hostname."""

    fixtures: Mapping[str, tuple[str, ...]]
    resolver_type: str = field(default="deterministic_fixture", init=False)
    resolved_hosts: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        for host, addresses in self.fixtures.items():
            if not isinstance(host, str) or not host.strip():
                raise InformationHttpConfigurationError(
                    "Resolver fixture host must be non-empty text."
                )
            exact_host = host.strip().lower()
            if (
                exact_host == "localhost"
                or exact_host.endswith(".localhost")
                or exact_host.endswith(".")
                or "%" in exact_host
            ):
                raise InformationHttpConfigurationError(
                    "Resolver fixture host is not canonical."
                )
            if exact_host in normalized:
                raise InformationHttpConfigurationError(
                    "Resolver fixture hosts must be unique after normalization."
                )
            entries = tuple(validate_global_address(item) for item in addresses)
            if not entries:
                raise InformationHttpConfigurationError(
                    "Resolver fixtures require at least one public address."
                )
            if len(set(entries)) != len(entries):
                raise InformationHttpConfigurationError(
                    "Resolver fixture addresses cannot contain duplicates."
                )
            normalized[exact_host] = entries
        self.fixtures = MappingProxyType(normalized)

    def resolve(
        self,
        canonical_url: str,
        *,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationResolvedTarget:
        policy.validate()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        try:
            canonical = canonicalize_public_url(canonical_url)
        except ValueError as exc:
            raise http_failure("invalid_source_url") from exc
        if canonical != canonical_url:
            raise http_failure("invalid_source_url")
        parsed = urlsplit(canonical)
        host = parsed.hostname
        if host is None or host.endswith(".") or "%" in host:
            raise http_failure("invalid_source_url")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in policy.allowed_ports_for(parsed.scheme):
            raise http_failure("invalid_source_url")
        addresses = self.fixtures.get(host)
        if addresses is None:
            raise http_failure("dns_resolution_failed")
        target = InformationResolvedTarget(
            canonical_url=canonical,
            scheme=parsed.scheme,
            hostname=host,
            port=port,
            addresses=addresses,
        )
        target.validate(policy=policy)
        self.resolved_hosts.append(host)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return target


@dataclass(frozen=True)
class InformationHttpRequest:
    """Fixed read-only GET request without credentials or user-controlled headers."""

    canonical_url: str
    headers: tuple[tuple[str, str], ...]
    timeout_seconds: float

    def validate(self, *, policy: InformationHttpRetrievalPolicy) -> None:
        if canonicalize_public_url(self.canonical_url) != self.canonical_url:
            raise http_failure("invalid_source_url")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not 0 < float(self.timeout_seconds) <= policy.request_timeout_seconds
        ):
            raise InformationHttpConfigurationError(
                "HTTP request timeout must be numeric and within policy."
            )
        expected = (
            ("accept", "text/html, application/xhtml+xml, text/plain;q=0.9"),
            ("accept-encoding", "gzip, deflate"),
            ("connection", "close"),
            ("user-agent", "ALICE-Information/0.3"),
        )
        if self.headers != expected:
            raise InformationHttpConfigurationError(
                "HTTP request headers must match the fixed P4.2 allowlist."
            )


@dataclass(frozen=True)
class InformationRawHttpResponse:
    """Deterministic raw response returned by a transport fixture."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body_chunks: tuple[bytes, ...]
    peer_address: str

    def validate_shape(self) -> None:
        if not isinstance(self.status_code, int) or isinstance(self.status_code, bool):
            raise InformationHttpConfigurationError(
                "HTTP response status must be an integer."
            )
        if not 100 <= self.status_code <= 599:
            raise InformationHttpConfigurationError(
                "HTTP response status is outside the valid range."
            )
        for name, value in self.headers:
            if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
                raise InformationHttpConfigurationError(
                    "HTTP response header names must use the HTTP token grammar."
                )
            if not isinstance(value, str):
                raise InformationHttpConfigurationError(
                    "HTTP response header values must be text."
                )
            if any(
                (ord(char) < 32 and char != "\t") or ord(char) == 127
                for char in value
            ):
                raise InformationHttpConfigurationError(
                    "HTTP response header values cannot contain control characters."
                )
            if "\r" in name + value or "\n" in name + value:
                raise InformationHttpConfigurationError(
                    "HTTP response headers cannot contain line breaks."
                )
        if any(not isinstance(chunk, bytes) for chunk in self.body_chunks):
            raise InformationHttpConfigurationError(
                "HTTP response body chunks must be bytes."
            )
        validate_global_address(self.peer_address)


class InformationHttpTransport(Protocol):
    """Transport interface that must connect only to the pinned target."""

    transport_type: str

    def get(
        self,
        request: InformationHttpRequest,
        *,
        target: InformationResolvedTarget,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationRawHttpResponse:
        """Return one raw response from the already validated target."""


@dataclass
class DeterministicInformationHttpTransport:
    """Network-free HTTP fixture keyed by exact canonical URL."""

    fixtures: Mapping[str, InformationRawHttpResponse]
    transport_type: str = field(default="deterministic_fixture", init=False)
    requested_urls: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        normalized: dict[str, InformationRawHttpResponse] = {}
        for url, response in self.fixtures.items():
            canonical = canonicalize_public_url(url)
            if canonical != url:
                raise InformationHttpConfigurationError(
                    "HTTP transport fixture URL must already be canonical."
                )
            if canonical in normalized:
                raise InformationHttpConfigurationError(
                    "HTTP transport fixture URLs must be unique after canonicalization."
                )
            if not isinstance(response, InformationRawHttpResponse):
                raise InformationHttpConfigurationError(
                    "HTTP transport fixtures must contain raw responses."
                )
            response.validate_shape()
            normalized[canonical] = response
        self.fixtures = MappingProxyType(normalized)

    def get(
        self,
        request: InformationHttpRequest,
        *,
        target: InformationResolvedTarget,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationRawHttpResponse:
        policy.validate()
        request.validate(policy=policy)
        target.validate(policy=policy)
        if request.canonical_url != target.canonical_url:
            raise InformationHttpConfigurationError(
                "HTTP request and resolved target URLs must match."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        response = self.fixtures.get(request.canonical_url)
        if response is None:
            raise http_failure("http_status_rejected")
        response.validate_shape()
        peer = validate_global_address(response.peer_address)
        if peer not in target.addresses:
            raise http_failure("peer_address_mismatch")
        self.requested_urls.append(request.canonical_url)
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return response
