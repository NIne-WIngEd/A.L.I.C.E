"""Direct, policy-bound live HTTPS primitives for Phase 4 P4.2b.

This module contains the first operating-system DNS and socket implementation.
It is deliberately not registered as an information provider and is not wired
into the Phase 3 runtime. Callers must explicitly construct the live retriever
with the P4.2b activation policy.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import ssl
from queue import Empty, Queue
from threading import BoundedSemaphore, Thread
from dataclasses import dataclass, field
from hashlib import sha256
from time import monotonic
from typing import Protocol
from urllib.parse import urlsplit

from .contracts import canonicalize_public_url
from .http_transport import (
    InformationHttpConfigurationError,
    InformationHttpRequest,
    InformationRawHttpResponse,
    InformationResolvedTarget,
    http_failure,
    validate_global_address,
)
from .live_policy import InformationLiveHttpPolicy
from .policy import InformationPolicy
from .providers import InformationCancellationToken
from .retrieval_policy import InformationHttpRetrievalPolicy

_HEADER_NAME = re.compile(rb"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_STATUS_LINE = re.compile(rb"^HTTP/(1\.0|1\.1) ([1-5][0-9]{2})(?:[ \t][\t\x20-\x7e]*)?$")


def _parse_content_length(value: str, *, maximum: int) -> int:
    if not value or not value.isdigit():
        raise http_failure("response_header_invalid")
    normalized = value.lstrip("0") or "0"
    maximum_text = str(maximum)
    if len(normalized) > len(maximum_text) or (
        len(normalized) == len(maximum_text) and normalized > maximum_text
    ):
        raise http_failure("response_too_large")
    return int(normalized)


def _validated_request_target(canonical_url: str) -> str:
    parsed = urlsplit(canonical_url)
    path = parsed.path or "/"
    request_target = path + (f"?{parsed.query}" if parsed.query else "")
    try:
        target_bytes = request_target.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise http_failure("invalid_source_url") from exc
    if (
        len(target_bytes) > 4096
        or path.startswith("//")
        or b"\\" in target_bytes
        or any(byte < 33 or byte > 126 for byte in target_bytes)
    ):
        raise http_failure("invalid_source_url")
    return request_target


@dataclass(frozen=True)
class InformationDnsAnswer:
    """One address-family and canonical IP pair returned by a DNS backend."""

    family: int
    address: str

    def validate(self) -> None:
        if self.family not in (socket.AF_INET, socket.AF_INET6):
            raise InformationHttpConfigurationError(
                "DNS answer family must be AF_INET or AF_INET6."
            )
        canonical = validate_global_address(self.address)
        version = ipaddress.ip_address(canonical).version
        expected = 4 if self.family == socket.AF_INET else 6
        if version != expected:
            raise InformationHttpConfigurationError(
                "DNS answer family does not match its IP address."
            )
        if canonical != self.address:
            raise InformationHttpConfigurationError(
                "DNS answers must already contain canonical IP addresses."
            )


class InformationDnsBackend(Protocol):
    """Small backend boundary around one operating-system DNS call."""

    backend_type: str

    def resolve(
        self,
        hostname: str,
        port: int,
        *,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[InformationDnsAnswer, ...]:
        """Return all stream TCP addresses for one exact host and port."""


@dataclass(frozen=True)
class StdlibInformationDnsBackend:
    """Bounded operating-system resolver using one daemon worker at a time."""

    backend_type: str = field(default="stdlib_getaddrinfo", init=False)
    _gate: BoundedSemaphore = field(
        default_factory=lambda: BoundedSemaphore(1),
        init=False,
        repr=False,
        compare=False,
    )

    def resolve(
        self,
        hostname: str,
        port: int,
        *,
        timeout_seconds: float,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[InformationDnsAnswer, ...]:
        if not self._gate.acquire(blocking=False):
            raise http_failure("network_timeout", retryable=True)
        result_queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                records = socket.getaddrinfo(
                    hostname,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                    proto=socket.IPPROTO_TCP,
                )
                result_queue.put_nowait(("ok", records))
            except (socket.gaierror, OSError) as exc:
                result_queue.put_nowait(("error", exc))
            finally:
                self._gate.release()

        Thread(target=worker, name="alice-information-dns", daemon=True).start()
        deadline = monotonic() + timeout_seconds
        while True:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise http_failure("network_timeout", retryable=True)
            try:
                outcome, payload = result_queue.get(timeout=min(0.05, remaining))
            except Empty:
                continue
            if outcome == "error":
                raise http_failure("dns_resolution_failed", retryable=True) from payload
            records = payload
            if not isinstance(records, list):
                raise http_failure("dns_resolution_failed")
            answers: list[InformationDnsAnswer] = []
            for family, socket_type, protocol, _canonical_name, socket_address in records:
                if socket_type != socket.SOCK_STREAM or protocol not in (0, socket.IPPROTO_TCP):
                    continue
                if family not in (socket.AF_INET, socket.AF_INET6):
                    continue
                if not isinstance(socket_address, tuple) or not socket_address:
                    continue
                raw_address = socket_address[0]
                if not isinstance(raw_address, str):
                    continue
                try:
                    canonical = ipaddress.ip_address(raw_address).compressed
                except ValueError as exc:
                    raise http_failure("dns_resolution_failed") from exc
                answers.append(InformationDnsAnswer(family=family, address=canonical))
            return tuple(answers)


@dataclass(frozen=True)
class SystemInformationNameResolver:
    """Point-of-use system resolver with public-address-only output."""

    information_policy: InformationPolicy
    retrieval_policy: InformationHttpRetrievalPolicy
    live_policy: InformationLiveHttpPolicy
    backend: InformationDnsBackend = field(default_factory=StdlibInformationDnsBackend)
    resolver_type: str = field(default="system_getaddrinfo", init=False)

    def __post_init__(self) -> None:
        self._validate_configuration(require_stdlib_backend=False)

    def _validate_configuration(self, *, require_stdlib_backend: bool) -> None:
        self.live_policy.validate(
            information_policy=self.information_policy,
            retrieval_policy=self.retrieval_policy,
        )
        if self.resolver_type != self.live_policy.approved_resolver_type:
            raise InformationHttpConfigurationError(
                "Live resolver identity is not approved by policy."
            )
        if not isinstance(getattr(self.backend, "backend_type", None), str):
            raise InformationHttpConfigurationError(
                "Live resolver backend must declare a backend_type."
            )
        if require_stdlib_backend and type(self.backend) is not StdlibInformationDnsBackend:
            raise InformationHttpConfigurationError(
                "Live retrieval requires the exact stdlib DNS backend."
            )

    def validate_live_boundary(self) -> None:
        """Require the exact production backend before a live retrieval."""

        self._validate_configuration(require_stdlib_backend=True)

    def resolve(
        self,
        canonical_url: str,
        *,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationResolvedTarget:
        self._validate_configuration(require_stdlib_backend=False)
        policy.validate(information_policy=self.information_policy)
        if policy != self.retrieval_policy:
            raise InformationHttpConfigurationError(
                "Live resolver received a different retrieval policy."
            )
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        try:
            canonical = canonicalize_public_url(canonical_url)
        except ValueError as exc:
            raise http_failure("invalid_source_url") from exc
        if canonical != canonical_url:
            raise http_failure("invalid_source_url")
        parsed = urlsplit(canonical)
        if parsed.scheme not in self.live_policy.allowed_schemes:
            raise http_failure("invalid_source_url")
        hostname = parsed.hostname
        if hostname is None or hostname.endswith(".") or "%" in hostname:
            raise http_failure("invalid_source_url")
        port = parsed.port or 443
        if port != 443 or port not in policy.allowed_ports_for("https"):
            raise http_failure("invalid_source_url")
        _validated_request_target(canonical)
        try:
            raw_answers = self.backend.resolve(
                hostname,
                port,
                timeout_seconds=policy.request_timeout_seconds,
                cancellation=cancellation,
            )
        except InformationHttpConfigurationError:
            raise
        except Exception as exc:
            if hasattr(exc, "failure"):
                raise
            raise http_failure("dns_resolution_failed", retryable=True) from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if not raw_answers:
            raise http_failure("dns_resolution_failed", retryable=True)
        if len(raw_answers) > self.live_policy.max_dns_addresses:
            raise http_failure("dns_resolution_failed")
        normalized: list[InformationDnsAnswer] = []
        seen: set[tuple[int, str]] = set()
        for answer in raw_answers:
            if not isinstance(answer, InformationDnsAnswer):
                raise InformationHttpConfigurationError(
                    "DNS backend returned an unsupported answer object."
                )
            answer.validate()
            key = (answer.family, answer.address)
            if key not in seen:
                normalized.append(answer)
                seen.add(key)
        if not normalized:
            raise http_failure("dns_resolution_failed", retryable=True)
        normalized.sort(
            key=lambda item: (
                0 if item.family == socket.AF_INET else 1,
                ipaddress.ip_address(item.address).packed,
            )
        )
        addresses = tuple(item.address for item in normalized)
        target = InformationResolvedTarget(
            canonical_url=canonical,
            scheme="https",
            hostname=hostname,
            port=443,
            addresses=addresses,
        )
        target.validate(policy=policy)
        return target


class InformationLiveConnection(Protocol):
    """Minimal connected TLS socket surface used by the strict parser."""

    def sendall(self, data: bytes) -> None: ...

    def recv(self, size: int) -> bytes: ...

    def settimeout(self, value: float | None) -> None: ...

    def getpeername(self) -> object: ...

    def close(self) -> None: ...


class InformationSocketBackend(Protocol):
    """Connection factory that must connect directly to one pinned address."""

    backend_type: str

    def open(
        self,
        *,
        target: InformationResolvedTarget,
        address: str,
        timeout_seconds: float,
    ) -> InformationLiveConnection:
        """Open one certificate-validating TLS connection."""


@dataclass(frozen=True)
class StdlibInformationSocketBackend:
    """Direct socket plus default trust-store TLS implementation."""

    backend_type: str = field(default="stdlib_direct_tls", init=False)

    def open(
        self,
        *,
        target: InformationResolvedTarget,
        address: str,
        timeout_seconds: float,
    ) -> InformationLiveConnection:
        if any(
            os.environ.get(name)
            for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE")
        ):
            raise http_failure("tls_validation_failed")
        canonical_address = validate_global_address(address)
        parsed_address = ipaddress.ip_address(canonical_address)
        family = socket.AF_INET if parsed_address.version == 4 else socket.AF_INET6
        raw_socket = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        deadline = monotonic() + timeout_seconds

        def remaining() -> float:
            value = deadline - monotonic()
            if value <= 0:
                raise http_failure("network_timeout", retryable=True)
            return value

        try:
            raw_socket.settimeout(remaining())
            destination: object
            if family == socket.AF_INET:
                destination = (canonical_address, target.port)
            else:
                destination = (canonical_address, target.port, 0, 0)
            raw_socket.connect(destination)  # type: ignore[arg-type]
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            if getattr(context, "keylog_filename", None):
                raise http_failure("tls_validation_failed")
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            if hasattr(ssl, "OP_NO_COMPRESSION"):
                context.options |= ssl.OP_NO_COMPRESSION
            try:
                context.set_alpn_protocols(["http/1.1"])
            except NotImplementedError as exc:
                raise http_failure("tls_validation_failed") from exc
            raw_socket.settimeout(remaining())
            connection = context.wrap_socket(
                raw_socket,
                server_hostname=target.hostname,
            )
            connection.settimeout(remaining())
            return connection
        except Exception:
            raw_socket.close()
            raise


@dataclass
class _BufferedConnectionReader:
    connection: InformationLiveConnection
    deadline: float
    chunk_size: int
    cancellation: InformationCancellationToken | None
    buffer: bytearray = field(default_factory=bytearray)

    def _remaining(self) -> float:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise http_failure("network_timeout", retryable=True)
        return remaining

    def _recv(self) -> bytes:
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()
        try:
            self.connection.settimeout(self._remaining())
            data = self.connection.recv(self.chunk_size)
        except (TimeoutError, socket.timeout) as exc:
            raise http_failure("network_timeout", retryable=True) from exc
        except ssl.SSLError as exc:
            raise http_failure("tls_validation_failed") from exc
        except OSError as exc:
            raise http_failure("network_connection_failed", retryable=True) from exc
        if self.cancellation is not None:
            self.cancellation.raise_if_cancelled()
        return data

    def readline(self, *, maximum: int) -> bytes:
        while True:
            marker = self.buffer.find(b"\r\n")
            if marker >= 0:
                end = marker + 2
                if end > maximum:
                    raise http_failure("http_protocol_invalid")
                result = bytes(self.buffer[:end])
                del self.buffer[:end]
                return result
            if len(self.buffer) >= maximum:
                raise http_failure("http_protocol_invalid")
            chunk = self._recv()
            if not chunk:
                raise http_failure("http_protocol_invalid")
            self.buffer.extend(chunk)
            if len(self.buffer) > maximum and b"\r\n" not in self.buffer[:maximum]:
                raise http_failure("http_protocol_invalid")

    def read_exact(self, length: int, *, maximum: int) -> tuple[bytes, ...]:
        if length < 0 or length > maximum:
            raise http_failure("response_too_large")
        chunks: list[bytes] = []
        remaining = length
        if self.buffer:
            take = min(remaining, len(self.buffer))
            if take:
                chunks.append(bytes(self.buffer[:take]))
                del self.buffer[:take]
                remaining -= take
        while remaining:
            chunk = self._recv()
            if not chunk:
                raise http_failure("http_protocol_invalid")
            take = min(remaining, len(chunk))
            chunks.append(chunk[:take])
            remaining -= take
            if take < len(chunk):
                self.buffer.extend(chunk[take:])
        return tuple(chunks)

    def read_to_eof(self, *, maximum: int) -> tuple[bytes, ...]:
        chunks: list[bytes] = []
        total = 0
        if self.buffer:
            initial = bytes(self.buffer)
            self.buffer.clear()
            total += len(initial)
            if total > maximum:
                raise http_failure("response_too_large")
            chunks.append(initial)
        while True:
            chunk = self._recv()
            if not chunk:
                return tuple(chunks)
            total += len(chunk)
            if total > maximum:
                raise http_failure("response_too_large")
            chunks.append(chunk)


@dataclass(frozen=True)
class DirectInformationHttpsTransport:
    """One direct HTTPS/1.1 request to the first pinned public address."""

    information_policy: InformationPolicy
    retrieval_policy: InformationHttpRetrievalPolicy
    live_policy: InformationLiveHttpPolicy
    backend: InformationSocketBackend = field(default_factory=StdlibInformationSocketBackend)
    transport_type: str = field(default="direct_https_socket", init=False)

    def __post_init__(self) -> None:
        self._validate_configuration(require_stdlib_backend=False)

    def _validate_configuration(self, *, require_stdlib_backend: bool) -> None:
        self.live_policy.validate(
            information_policy=self.information_policy,
            retrieval_policy=self.retrieval_policy,
        )
        if self.transport_type != self.live_policy.approved_transport_type:
            raise InformationHttpConfigurationError(
                "Live transport identity is not approved by policy."
            )
        if not isinstance(getattr(self.backend, "backend_type", None), str):
            raise InformationHttpConfigurationError(
                "Live socket backend must declare a backend_type."
            )
        if require_stdlib_backend and type(self.backend) is not StdlibInformationSocketBackend:
            raise InformationHttpConfigurationError(
                "Live retrieval requires the exact stdlib socket backend."
            )

    def validate_live_boundary(self) -> None:
        """Require the exact production backend before a live retrieval."""

        self._validate_configuration(require_stdlib_backend=True)

    def get(
        self,
        request: InformationHttpRequest,
        *,
        target: InformationResolvedTarget,
        policy: InformationHttpRetrievalPolicy,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationRawHttpResponse:
        self._validate_configuration(require_stdlib_backend=False)
        policy.validate(information_policy=self.information_policy)
        if policy != self.retrieval_policy:
            raise InformationHttpConfigurationError(
                "Live transport received a different retrieval policy."
            )
        request.validate(policy=policy)
        target.validate(policy=policy)
        if request.canonical_url != target.canonical_url:
            raise InformationHttpConfigurationError(
                "HTTP request and resolved target URLs must match."
            )
        if target.scheme != "https" or target.port != 443:
            raise http_failure("invalid_source_url")
        if not target.addresses:
            raise http_failure("dns_resolution_failed")
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        selected_address = validate_global_address(target.addresses[0])
        deadline = monotonic() + request.timeout_seconds
        try:
            connection = self.backend.open(
                target=target,
                address=selected_address,
                timeout_seconds=self._remaining(deadline),
            )
        except (TimeoutError, socket.timeout) as exc:
            raise http_failure("network_timeout", retryable=True) from exc
        except ssl.SSLCertVerificationError as exc:
            raise http_failure("tls_validation_failed") from exc
        except ssl.SSLError as exc:
            raise http_failure("tls_validation_failed") from exc
        except OSError as exc:
            raise http_failure("network_connection_failed", retryable=True) from exc
        except Exception as exc:
            if hasattr(exc, "failure"):
                raise
            raise http_failure("network_connection_failed", retryable=True) from exc
        try:
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            peer = self._peer_address(connection)
            if peer != selected_address or peer not in target.addresses:
                raise http_failure("peer_address_mismatch")
            payload = self._request_bytes(request, target=target)
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            try:
                connection.settimeout(self._remaining(deadline))
                connection.sendall(payload)
            except (TimeoutError, socket.timeout) as exc:
                raise http_failure("network_timeout", retryable=True) from exc
            except ssl.SSLError as exc:
                raise http_failure("tls_validation_failed") from exc
            except OSError as exc:
                raise http_failure("network_connection_failed", retryable=True) from exc
            response = self._read_response(
                connection,
                peer_address=peer,
                deadline=deadline,
                cancellation=cancellation,
            )
            response.validate_shape()
            return response
        finally:
            try:
                connection.close()
            except Exception:
                pass

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise http_failure("network_timeout", retryable=True)
        return remaining

    def _peer_address(self, connection: InformationLiveConnection) -> str:
        try:
            peer = connection.getpeername()
        except OSError as exc:
            raise http_failure("network_connection_failed", retryable=True) from exc
        raw_address = peer[0] if isinstance(peer, tuple) and peer else peer
        if not isinstance(raw_address, str):
            raise http_failure("peer_address_mismatch")
        return validate_global_address(raw_address)

    def _request_bytes(
        self,
        request: InformationHttpRequest,
        *,
        target: InformationResolvedTarget,
    ) -> bytes:
        request_target = _validated_request_target(request.canonical_url)
        host = target.hostname
        host_header = f"[{host}]" if ":" in host else host
        lines = [f"GET {request_target} HTTP/1.1", f"Host: {host_header}"]
        lines.extend(f"{name}: {value}" for name, value in request.headers)
        payload = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
        if len(payload) > 8192:
            raise InformationHttpConfigurationError(
                "Fixed live HTTP request exceeded its internal byte limit."
            )
        return payload

    def _read_response(
        self,
        connection: InformationLiveConnection,
        *,
        peer_address: str,
        deadline: float,
        cancellation: InformationCancellationToken | None,
    ) -> InformationRawHttpResponse:
        reader = _BufferedConnectionReader(
            connection=connection,
            deadline=deadline,
            chunk_size=self.live_policy.socket_read_chunk_bytes,
            cancellation=cancellation,
        )
        status_line = reader.readline(maximum=self.live_policy.max_status_line_bytes)
        stripped_status = status_line[:-2]
        match = _STATUS_LINE.fullmatch(stripped_status)
        if match is None:
            raise http_failure("http_protocol_invalid")
        status_code = int(match.group(2))
        if status_code < 200:
            raise http_failure("http_protocol_invalid")
        headers: list[tuple[str, str]] = []
        header_bytes = 0
        while True:
            remaining = self.retrieval_policy.max_header_bytes - header_bytes
            if remaining <= 0:
                raise http_failure("response_header_invalid")
            line = reader.readline(maximum=remaining)
            header_bytes += len(line)
            if header_bytes > self.retrieval_policy.max_header_bytes:
                raise http_failure("response_header_invalid")
            if line == b"\r\n":
                break
            if line.startswith((b" ", b"\t")):
                raise http_failure("response_header_invalid")
            raw = line[:-2]
            name, separator, value = raw.partition(b":")
            if not separator or not _HEADER_NAME.fullmatch(name):
                raise http_failure("response_header_invalid")
            if len(headers) >= self.live_policy.max_header_count:
                raise http_failure("response_header_invalid")
            try:
                decoded_name = name.decode("ascii").lower()
                decoded_value = value.decode("latin-1").strip(" \t")
            except UnicodeDecodeError as exc:
                raise http_failure("response_header_invalid") from exc
            if any(
                (ord(character) < 32 and character != "\t")
                or ord(character) == 127
                for character in decoded_value
            ):
                raise http_failure("response_header_invalid")
            headers.append((decoded_name, decoded_value))
        transfer_values = [
            value for name, value in headers if name == "transfer-encoding"
        ]
        if transfer_values:
            raise http_failure("response_header_invalid")
        content_lengths = [value for name, value in headers if name == "content-length"]
        if len(content_lengths) > 1:
            raise http_failure("response_header_invalid")
        if content_lengths:
            length = _parse_content_length(
                content_lengths[0],
                maximum=self.retrieval_policy.max_wire_bytes,
            )
            body_chunks = reader.read_exact(
                length,
                maximum=self.retrieval_policy.max_wire_bytes,
            )
        else:
            body_chunks = reader.read_to_eof(
                maximum=self.retrieval_policy.max_wire_bytes,
            )
        return InformationRawHttpResponse(
            status_code=status_code,
            headers=tuple(headers),
            body_chunks=body_chunks,
            peer_address=peer_address,
        )


def live_component_digest(*, resolver: SystemInformationNameResolver, transport: DirectInformationHttpsTransport) -> str:
    """Return a metadata-only digest for inspection without host or URL content."""

    text = "|".join(
        (
            resolver.resolver_type,
            resolver.backend.backend_type,
            transport.transport_type,
            transport.backend.backend_type,
            resolver.live_policy.version,
        )
    )
    return sha256(text.encode("utf-8")).hexdigest()
