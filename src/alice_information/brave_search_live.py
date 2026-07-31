"""Exact credential-bearing Brave Search HTTPS transport for P4.10a."""

from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol
from urllib.parse import urlsplit

from .http_transport import validate_global_address
from .live_provider_config import InformationSecretValue
from .live_provider_contracts import (
    InformationLiveProviderExecutionError,
    InformationLiveProviderFailure,
)
from .live_provider_policy import InformationLiveProviderRuntimePolicy
from .providers import InformationCancellationToken


class BraveSearchTransportError(RuntimeError):
    """Internal transport configuration error. Runtime failures are sanitized."""


@dataclass(frozen=True)
class BraveSearchRawResponse:
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    peer_address: str

    def validate(self, *, maximum_bytes: int) -> None:
        if not isinstance(self.status_code, int) or not 100 <= self.status_code <= 599:
            raise BraveSearchTransportError("Brave response status is invalid.")
        if len(self.body) > maximum_bytes:
            raise _failure("live_provider_response_too_large")
        validate_global_address(self.peer_address)
        names: set[str] = set()
        for name, value in self.headers:
            if not isinstance(name, str) or not isinstance(value, str):
                raise BraveSearchTransportError("Brave response headers are invalid.")
            lowered = name.casefold()
            if lowered in names and lowered in {"content-length", "content-type", "transfer-encoding"}:
                raise BraveSearchTransportError("Brave response singleton header was duplicated.")
            names.add(lowered)
            if "\r" in name + value or "\n" in name + value:
                raise BraveSearchTransportError("Brave response header contains line breaks.")

    def header_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, value in self.headers:
            result.setdefault(name.casefold(), value.strip())
        return result


class BraveSearchTransport(Protocol):
    transport_type: str

    def perform(
        self,
        *,
        canonical_url: str,
        credential_header: str,
        credential: InformationSecretValue,
        timeout_seconds: float,
        maximum_response_bytes: int,
        cancellation: InformationCancellationToken | None,
    ) -> BraveSearchRawResponse: ...


def _failure(code: str) -> InformationLiveProviderExecutionError:
    return InformationLiveProviderExecutionError(
        InformationLiveProviderFailure.create(
            provider="brave-search-v1",
            operation="search",
            code=code,
        )
    )


def _remaining(deadline: float) -> float:
    value = deadline - monotonic()
    if value <= 0:
        raise _failure("live_provider_timeout")
    return value


def _cancel(cancellation: InformationCancellationToken | None) -> None:
    if cancellation is None:
        return
    try:
        cancellation.raise_if_cancelled()
    except Exception as exc:
        raise _failure("live_provider_cancelled") from exc


@dataclass(frozen=True)
class StrictBraveSearchHttpsTransport:
    """One direct pinned-address TLS request with no redirect, retry, or proxy."""

    resolver: object
    retrieval_policy: object
    socket_backend: object
    policy: InformationLiveProviderRuntimePolicy
    transport_type: str = field(default="brave-direct-https-v1", init=False)

    def __post_init__(self) -> None:
        self.policy.validate()
        if not callable(getattr(self.resolver, "resolve", None)):
            raise BraveSearchTransportError("Brave transport requires a resolver.")
        if not callable(getattr(self.socket_backend, "open", None)):
            raise BraveSearchTransportError("Brave transport requires a TLS socket backend.")

    def validate_live_boundary(self) -> None:
        """Require exact production P4.2b resolver/backend identities for real egress."""
        from .live_http import StdlibInformationSocketBackend, SystemInformationNameResolver

        if type(self.resolver) is not SystemInformationNameResolver:
            raise BraveSearchTransportError("Live Brave egress requires the exact system resolver.")
        if type(self.socket_backend) is not StdlibInformationSocketBackend:
            raise BraveSearchTransportError("Live Brave egress requires the exact stdlib TLS backend.")
        validator = getattr(self.resolver, "validate_live_boundary", None)
        if not callable(validator):
            raise BraveSearchTransportError("Live resolver validation is unavailable.")
        validator()

    def perform(
        self,
        *,
        canonical_url: str,
        credential_header: str,
        credential: InformationSecretValue,
        timeout_seconds: float,
        maximum_response_bytes: int,
        cancellation: InformationCancellationToken | None,
    ) -> BraveSearchRawResponse:
        self.policy.validate()
        _cancel(cancellation)
        parsed = urlsplit(canonical_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != self.policy.search_host
            or (parsed.port or 443) != 443
            or parsed.path != self.policy.search_path
            or parsed.fragment
        ):
            raise BraveSearchTransportError("Brave request endpoint changed.")
        if credential_header != self.policy.credential_header:
            raise BraveSearchTransportError("Brave credential header changed.")
        if not 0 < float(timeout_seconds) <= 30:
            raise BraveSearchTransportError("Brave timeout is outside the approved range.")
        if not 1 <= maximum_response_bytes <= self.policy.max_response_bytes:
            raise BraveSearchTransportError("Brave response budget changed.")
        try:
            target = self.resolver.resolve(
                canonical_url,
                policy=self.retrieval_policy,
                cancellation=cancellation,
            )
        except InformationLiveProviderExecutionError:
            raise
        except Exception as exc:
            raise _failure("live_provider_network_boundary_failed") from exc
        _cancel(cancellation)
        addresses = tuple(getattr(target, "addresses", ()))
        if not addresses:
            raise _failure("live_provider_network_boundary_failed")
        address = validate_global_address(addresses[0])
        deadline = monotonic() + float(timeout_seconds)
        connection = None
        token = credential.reveal_for_exact_header()
        try:
            connection = self.socket_backend.open(
                target=target,
                address=address,
                timeout_seconds=_remaining(deadline),
            )
            _cancel(cancellation)
            path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            request = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {self.policy.search_host}\r\n"
                "Accept: application/json\r\n"
                "Accept-Encoding: identity\r\n"
                "Connection: close\r\n"
                "User-Agent: ALICE-Information/0.16\r\n"
                f"{credential_header}: {token}\r\n"
                "\r\n"
            ).encode("ascii")
            connection.settimeout(_remaining(deadline))
            connection.sendall(request)
            raw = self._read_response(
                connection,
                deadline=deadline,
                maximum=maximum_response_bytes,
                cancellation=cancellation,
            )
            peer = connection.getpeername()
            if not isinstance(peer, tuple) or not peer or not isinstance(peer[0], str):
                raise _failure("live_provider_network_boundary_failed")
            peer_address = validate_global_address(peer[0])
            if peer_address not in addresses:
                raise _failure("live_provider_network_boundary_failed")
            response = BraveSearchRawResponse(
                status_code=raw[0],
                headers=raw[1],
                body=raw[2],
                peer_address=peer_address,
            )
            response.validate(maximum_bytes=maximum_response_bytes)
            return response
        except InformationLiveProviderExecutionError:
            raise
        except (TimeoutError, socket.timeout) as exc:
            raise _failure("live_provider_timeout") from exc
        except Exception as exc:
            # Never include exception text because it could contain a credential-bearing request.
            raise _failure("live_provider_network_boundary_failed") from exc
        finally:
            token = ""
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def _read_response(
        self,
        connection: object,
        *,
        deadline: float,
        maximum: int,
        cancellation: InformationCancellationToken | None,
    ) -> tuple[int, tuple[tuple[str, str], ...], bytes]:
        buffer = bytearray()
        header_limit = 64 * 1024
        while b"\r\n\r\n" not in buffer:
            _cancel(cancellation)
            connection.settimeout(_remaining(deadline))
            chunk = connection.recv(8192)
            if not chunk:
                raise BraveSearchTransportError("Brave response ended before headers.")
            buffer.extend(chunk)
            if len(buffer) > header_limit:
                raise BraveSearchTransportError("Brave response headers are too large.")
        head, body_start = bytes(buffer).split(b"\r\n\r\n", 1)
        try:
            lines = head.decode("iso-8859-1").split("\r\n")
            version, status_text, _reason = lines[0].split(" ", 2)
            if version != "HTTP/1.1":
                raise ValueError
            status = int(status_text)
            headers: list[tuple[str, str]] = []
            for line in lines[1:]:
                name, value = line.split(":", 1)
                headers.append((name.strip(), value.strip()))
        except (UnicodeError, ValueError) as exc:
            raise BraveSearchTransportError("Brave response headers are malformed.") from exc
        header_map: dict[str, str] = {}
        for name, value in headers:
            key = name.casefold()
            if key in header_map and key in {"content-length", "content-type", "transfer-encoding"}:
                raise BraveSearchTransportError("Brave response singleton header was duplicated.")
            header_map.setdefault(key, value)
        transfer = header_map.get("transfer-encoding", "").casefold()
        if transfer and transfer != "chunked":
            raise BraveSearchTransportError("Brave transfer encoding is unsupported.")
        if transfer == "chunked":
            body = self._read_chunked(
                connection,
                initial=body_start,
                deadline=deadline,
                maximum=maximum,
                cancellation=cancellation,
            )
        elif "content-length" in header_map:
            try:
                length = int(header_map["content-length"])
            except ValueError as exc:
                raise BraveSearchTransportError("Brave content length is invalid.") from exc
            if length < 0 or length > maximum:
                raise _failure("live_provider_response_too_large")
            body = self._read_exact(
                connection,
                initial=body_start,
                length=length,
                deadline=deadline,
                cancellation=cancellation,
            )
        else:
            body = self._read_to_eof(
                connection,
                initial=body_start,
                deadline=deadline,
                maximum=maximum,
                cancellation=cancellation,
            )
        if len(body) > maximum:
            raise _failure("live_provider_response_too_large")
        return status, tuple(headers), body

    @staticmethod
    def _read_exact(connection: object, *, initial: bytes, length: int, deadline: float, cancellation: InformationCancellationToken | None) -> bytes:
        output = bytearray(initial[:length])
        while len(output) < length:
            _cancel(cancellation)
            connection.settimeout(_remaining(deadline))
            chunk = connection.recv(min(8192, length - len(output)))
            if not chunk:
                raise BraveSearchTransportError("Brave response body ended early.")
            output.extend(chunk)
        return bytes(output)

    @staticmethod
    def _read_to_eof(connection: object, *, initial: bytes, deadline: float, maximum: int, cancellation: InformationCancellationToken | None) -> bytes:
        output = bytearray(initial)
        while True:
            if len(output) > maximum:
                raise _failure("live_provider_response_too_large")
            _cancel(cancellation)
            connection.settimeout(_remaining(deadline))
            chunk = connection.recv(8192)
            if not chunk:
                return bytes(output)
            output.extend(chunk)

    @staticmethod
    def _read_chunked(connection: object, *, initial: bytes, deadline: float, maximum: int, cancellation: InformationCancellationToken | None) -> bytes:
        buffer = bytearray(initial)
        output = bytearray()

        def ensure(marker: bytes) -> None:
            while marker not in buffer:
                _cancel(cancellation)
                connection.settimeout(_remaining(deadline))
                chunk = connection.recv(8192)
                if not chunk:
                    raise BraveSearchTransportError("Brave chunked body ended early.")
                buffer.extend(chunk)
                if len(buffer) + len(output) > maximum + 64 * 1024:
                    raise _failure("live_provider_response_too_large")

        while True:
            ensure(b"\r\n")
            line, remainder = bytes(buffer).split(b"\r\n", 1)
            buffer[:] = remainder
            try:
                size = int(line.split(b";", 1)[0], 16)
            except ValueError as exc:
                raise BraveSearchTransportError("Brave chunk size is invalid.") from exc
            if size == 0:
                return bytes(output)
            if len(output) + size > maximum:
                raise _failure("live_provider_response_too_large")
            while len(buffer) < size + 2:
                _cancel(cancellation)
                connection.settimeout(_remaining(deadline))
                chunk = connection.recv(8192)
                if not chunk:
                    raise BraveSearchTransportError("Brave chunked body ended early.")
                buffer.extend(chunk)
            output.extend(buffer[:size])
            if bytes(buffer[size:size + 2]) != b"\r\n":
                raise BraveSearchTransportError("Brave chunk delimiter is invalid.")
            del buffer[:size + 2]
