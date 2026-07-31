"""Metadata-only contracts for the additive P4.10 live PUBLIC provider profile."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlsplit

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PROVIDER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")

LIVE_PROVIDER_FAILURE_MESSAGES = {
    "live_provider_configuration_missing": "Live provider configuration is unavailable.",
    "live_provider_credential_missing": "Live provider credential is unavailable.",
    "live_provider_authentication_failed": "Live provider authentication failed.",
    "live_provider_unavailable": "Live provider is unavailable.",
    "live_provider_rate_limited": "Live provider rate limit was reached.",
    "live_provider_quota_exhausted": "Live provider quota was exhausted.",
    "live_provider_timeout": "Live provider operation timed out.",
    "live_provider_cancelled": "Live provider operation was cancelled.",
    "live_provider_protocol_error": "Live provider response was invalid.",
    "live_provider_response_too_large": "Live provider response exceeded the approved limit.",
    "live_provider_network_boundary_failed": "Live provider network boundary failed.",
}


class InformationLiveProviderContractError(ValueError):
    """Raised when live-provider evidence is malformed or tampered."""


class InformationLiveProviderExecutionError(RuntimeError):
    """One sanitized live-provider failure without raw request or response data."""

    def __init__(self, failure: "InformationLiveProviderFailure") -> None:
        failure.validate()
        self.failure = failure
        super().__init__(f"{failure.code}: {failure.message}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sequence_sha256(values: Iterable[str]) -> str:
    return canonical_sha256(list(values))


def _text(value: object, field: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InformationLiveProviderContractError(f"{field} must be non-empty text.")
    normalized = value.strip()
    if len(normalized) > maximum or "\x00" in normalized:
        raise InformationLiveProviderContractError(f"{field} is invalid.")
    return normalized


def _digest(value: object, field: str) -> str:
    normalized = _text(value, field, maximum=64).lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise InformationLiveProviderContractError(
            f"{field} must be a lowercase SHA-256 digest."
        )
    return normalized


def _provider(value: object) -> str:
    normalized = _text(value, "provider", maximum=100)
    if _PROVIDER_PATTERN.fullmatch(normalized) is None:
        raise InformationLiveProviderContractError("provider identity is invalid.")
    return normalized


def _timestamp(value: object, field: str) -> tuple[str, datetime]:
    normalized = _text(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InformationLiveProviderContractError(
            f"{field} must be valid ISO-8601 text."
        ) from exc
    if parsed.tzinfo is None:
        raise InformationLiveProviderContractError(
            f"{field} must include a timezone offset."
        )
    return normalized, parsed.astimezone(timezone.utc)


def _public_address(value: object) -> str:
    normalized = _text(value, "peer_address", maximum=64)
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError as exc:
        raise InformationLiveProviderContractError(
            "peer_address must be a canonical IP address."
        ) from exc
    prohibited = (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )
    if prohibited or address.compressed != normalized:
        raise InformationLiveProviderContractError(
            "peer_address must be one canonical global address."
        )
    return normalized


@dataclass(frozen=True)
class InformationLiveProviderFailure:
    provider: str
    operation: str
    code: str
    message: str
    retryable: bool = False

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        operation: str,
        code: str,
    ) -> "InformationLiveProviderFailure":
        message = LIVE_PROVIDER_FAILURE_MESSAGES.get(code)
        if message is None:
            raise InformationLiveProviderContractError(
                "Live provider failure code is not approved."
            )
        failure = cls(
            provider=provider,
            operation=operation,
            code=code,
            message=message,
            retryable=False,
        )
        failure.validate()
        return failure

    def validate(self) -> None:
        _provider(self.provider)
        if self.operation not in {"search", "fetch", "preflight", "acceptance"}:
            raise InformationLiveProviderContractError(
                "Live provider failure operation is invalid."
            )
        if LIVE_PROVIDER_FAILURE_MESSAGES.get(self.code) != self.message:
            raise InformationLiveProviderContractError(
                "Live provider failure message changed."
            )
        if self.retryable is not False:
            raise InformationLiveProviderContractError(
                "P4.10 never performs an automatic retry."
            )

    def metadata_record(self) -> dict[str, object]:
        self.validate()
        return {
            "provider": self.provider,
            "operation": self.operation,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class InformationLiveRateLimitState:
    limit: str | None
    policy: str | None
    remaining: str | None
    reset: str | None

    def validate(self) -> None:
        for value in (self.limit, self.policy, self.remaining, self.reset):
            if value is None:
                continue
            if (
                not isinstance(value, str)
                or len(value) > 500
                or "\r" in value
                or "\n" in value
                or "\x00" in value
            ):
                raise InformationLiveProviderContractError(
                    "Rate-limit metadata is invalid."
                )

    def metadata_record(self) -> dict[str, str | None]:
        self.validate()
        return {
            "limit": self.limit,
            "policy": self.policy,
            "remaining": self.remaining,
            "reset": self.reset,
        }


@dataclass(frozen=True)
class InformationLiveEgressReceipt:
    """Raw-query-free and body-free evidence for one network egress."""

    receipt_id: str
    provider: str
    operation: str
    endpoint_host: str
    endpoint_path: str
    query_id: str
    query_sha256: str
    configuration_sha256: str
    started_at: str
    completed_at: str
    elapsed_milliseconds: int
    status_code: int
    peer_address: str
    item_count: int
    response_sha256: str
    item_sequence_sha256: str
    rate_limit: InformationLiveRateLimitState
    policy_binding: str
    receipt_sha256: str

    @classmethod
    def create(cls, **values: object) -> "InformationLiveEgressReceipt":
        draft = cls(
            receipt_id="egress-pending",
            receipt_sha256="0" * 64,
            **values,
        )  # type: ignore[arg-type]
        receipt_id = f"egress-{canonical_sha256(draft._payload(include_id=False))[:20]}"
        identified = cls(**{**draft.__dict__, "receipt_id": receipt_id})
        receipt = cls(
            **{
                **identified.__dict__,
                "receipt_sha256": canonical_sha256(identified._payload()),
            }
        )
        receipt.validate()
        return receipt

    def _payload(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "receipt_id": self.receipt_id,
            "provider": self.provider,
            "operation": self.operation,
            "endpoint_host": self.endpoint_host,
            "endpoint_path": self.endpoint_path,
            "query_id": self.query_id,
            "query_sha256": self.query_sha256,
            "configuration_sha256": self.configuration_sha256,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "status_code": self.status_code,
            "peer_address": self.peer_address,
            "item_count": self.item_count,
            "response_sha256": self.response_sha256,
            "item_sequence_sha256": self.item_sequence_sha256,
            "rate_limit": self.rate_limit.metadata_record(),
            "policy_binding": self.policy_binding,
        }
        if not include_id:
            result.pop("receipt_id")
        return result

    def validate(self) -> None:
        _text(self.receipt_id, "receipt_id", maximum=64)
        _provider(self.provider)
        if self.operation not in {"search", "fetch", "preflight"}:
            raise InformationLiveProviderContractError("Egress operation is invalid.")
        host = _text(self.endpoint_host, "endpoint_host", maximum=253)
        if host != host.casefold() or any(character.isspace() for character in host):
            raise InformationLiveProviderContractError("Egress endpoint host is invalid.")
        path = _text(self.endpoint_path, "endpoint_path", maximum=2048)
        if not path.startswith("/") or "?" in path or "#" in path:
            raise InformationLiveProviderContractError(
                "Egress endpoint path must omit query text and fragments."
            )
        _text(self.query_id, "query_id", maximum=512)
        _digest(self.query_sha256, "query_sha256")
        _digest(self.configuration_sha256, "configuration_sha256")
        _, started = _timestamp(self.started_at, "started_at")
        _, completed = _timestamp(self.completed_at, "completed_at")
        if completed < started:
            raise InformationLiveProviderContractError(
                "Egress completion cannot precede its start."
            )
        if (
            not isinstance(self.elapsed_milliseconds, int)
            or isinstance(self.elapsed_milliseconds, bool)
            or self.elapsed_milliseconds < 0
        ):
            raise InformationLiveProviderContractError(
                "elapsed_milliseconds is invalid."
            )
        if (
            not isinstance(self.status_code, int)
            or isinstance(self.status_code, bool)
            or not 100 <= self.status_code <= 599
        ):
            raise InformationLiveProviderContractError("status_code is invalid.")
        _public_address(self.peer_address)
        if (
            not isinstance(self.item_count, int)
            or isinstance(self.item_count, bool)
            or self.item_count < 0
        ):
            raise InformationLiveProviderContractError("item_count is invalid.")
        _digest(self.response_sha256, "response_sha256")
        _digest(self.item_sequence_sha256, "item_sequence_sha256")
        self.rate_limit.validate()
        _text(self.policy_binding, "policy_binding", maximum=512)
        expected_id = f"egress-{canonical_sha256(self._payload(include_id=False))[:20]}"
        if self.receipt_id != expected_id:
            raise InformationLiveProviderContractError(
                "Egress receipt ID does not match its metadata."
            )
        if _digest(self.receipt_sha256, "receipt_sha256") != canonical_sha256(
            self._payload()
        ):
            raise InformationLiveProviderContractError(
                "Egress receipt digest does not match its metadata."
            )

    def to_metadata_record(self) -> dict[str, object]:
        self.validate()
        return {**self._payload(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True)
class InformationLiveSearchResponse:
    results: tuple[object, ...]
    receipt: InformationLiveEgressReceipt

    def validate(self) -> None:
        self.receipt.validate()
        if self.receipt.operation not in {"search", "preflight"}:
            raise InformationLiveProviderContractError(
                "Search response receipt operation changed."
            )
        if len(self.results) != self.receipt.item_count:
            raise InformationLiveProviderContractError(
                "Search result count does not match its egress receipt."
            )
        digests: list[str] = []
        ranks: list[int] = []
        result_ids: set[str] = set()
        urls: set[str] = set()
        for result in self.results:
            validator = getattr(result, "validate", None)
            if not callable(validator):
                raise InformationLiveProviderContractError(
                    "Search response contains an invalid result."
                )
            validator()
            if getattr(result, "query_id", None) != self.receipt.query_id:
                raise InformationLiveProviderContractError(
                    "Search result query identity changed."
                )
            if getattr(result, "provider", None) != self.receipt.provider:
                raise InformationLiveProviderContractError(
                    "Search result provider identity changed."
                )
            result_id = _text(getattr(result, "result_id", None), "result_id")
            canonical_url = _text(
                getattr(result, "canonical_url", None), "canonical_url", maximum=2048
            )
            if result_id in result_ids or canonical_url in urls:
                raise InformationLiveProviderContractError(
                    "Search response contains duplicate identities."
                )
            result_ids.add(result_id)
            urls.add(canonical_url)
            ranks.append(int(getattr(result, "rank", 0)))
            digests.append(_digest(getattr(result, "content_sha256", ""), "content_sha256"))
        if ranks != list(range(1, len(ranks) + 1)):
            raise InformationLiveProviderContractError(
                "Search result ranks must be contiguous and provider ordered."
            )
        if sequence_sha256(digests) != self.receipt.item_sequence_sha256:
            raise InformationLiveProviderContractError(
                "Search result sequence does not match its egress receipt."
            )


@dataclass(frozen=True)
class InformationLiveFetchResponse:
    search_result: object
    source_document: object
    resource: object
    receipt: InformationLiveEgressReceipt

    def validate(self) -> None:
        self.receipt.validate()
        if self.receipt.operation != "fetch" or self.receipt.item_count != 1:
            raise InformationLiveProviderContractError(
                "Fetch egress receipt metadata is invalid."
            )
        for value, field in (
            (self.search_result, "search_result"),
            (self.source_document, "source_document"),
            (self.resource, "resource"),
        ):
            validator = getattr(value, "validate", None)
            if not callable(validator):
                raise InformationLiveProviderContractError(f"{field} is invalid.")
            validator()
        if getattr(self.search_result, "query_id", None) != self.receipt.query_id:
            raise InformationLiveProviderContractError(
                "Fetch query identity changed."
            )
        source_digest = _digest(
            getattr(self.source_document, "content_sha256", ""),
            "source_content_sha256",
        )
        resource_digest = _digest(
            getattr(self.resource, "content_sha256", ""),
            "resource_content_sha256",
        )
        if source_digest != resource_digest:
            raise InformationLiveProviderContractError(
                "Fetch source document is not bound to the retrieved resource."
            )
        final_url = _text(
            getattr(self.resource, "final_url", None),
            "resource final_url",
            maximum=2048,
        )
        source_url = _text(
            getattr(self.source_document, "canonical_url", None),
            "source canonical_url",
            maximum=2048,
        )
        if final_url != source_url:
            raise InformationLiveProviderContractError(
                "Fetch source URL is not the final retrieved URL."
            )
        parsed = urlsplit(final_url)
        if (
            parsed.hostname != self.receipt.endpoint_host
            or (parsed.path or "/") != self.receipt.endpoint_path
        ):
            raise InformationLiveProviderContractError(
                "Fetch endpoint metadata does not match the retrieved resource."
            )
        if self.receipt.response_sha256 != resource_digest:
            raise InformationLiveProviderContractError(
                "Fetch response digest does not match the retrieved resource."
            )
        if sequence_sha256([source_digest]) != self.receipt.item_sequence_sha256:
            raise InformationLiveProviderContractError(
                "Fetch source sequence does not match its egress receipt."
            )
