"""P4.2b system DNS, direct HTTPS transport, and live-boundary tests."""

from __future__ import annotations

import socket
import ssl
from threading import Event
from dataclasses import dataclass
from pathlib import Path

import pytest

from alice_information.http_transport import (
    InformationHttpExecutionError,
    InformationHttpRequest,
    InformationResolvedTarget,
)
from alice_information.live_http import (
    DirectInformationHttpsTransport,
    InformationDnsAnswer,
    StdlibInformationDnsBackend,
    StdlibInformationSocketBackend,
    SystemInformationNameResolver,
    live_component_digest,
)
from alice_information.live_policy import load_information_live_http_policy
from alice_information.policy import load_information_policy
from alice_information.providers import (
    InformationCancellationToken,
    InformationProviderCancelledError,
)
from alice_information.retrieval import LiveControlledInformationHttpRetriever
from alice_information.retrieval_policy import load_information_http_retrieval_policy

ROOT = Path(__file__).resolve().parents[2]
ADDRESS = "93.184.216.34"
URL = "https://example.com/report?q=public"


def _policies():
    information = load_information_policy(ROOT / "policies" / "information_policy.json")
    retrieval = load_information_http_retrieval_policy(
        ROOT / "policies" / "information_http_retrieval_policy.json",
        information_policy=information,
    )
    live = load_information_live_http_policy(
        information_policy=information,
        retrieval_policy=retrieval,
        path=ROOT / "policies" / "information_live_http_policy.json",
    )
    return information, retrieval, live


def _target() -> InformationResolvedTarget:
    information, retrieval, _live = _policies()
    del information
    target = InformationResolvedTarget(
        canonical_url=URL,
        scheme="https",
        hostname="example.com",
        port=443,
        addresses=(ADDRESS,),
    )
    target.validate(policy=retrieval)
    return target


def _request() -> InformationHttpRequest:
    _information, retrieval, _live = _policies()
    request = InformationHttpRequest(
        canonical_url=URL,
        headers=(
            ("accept", "text/html, application/xhtml+xml, text/plain;q=0.9"),
            ("accept-encoding", "gzip, deflate"),
            ("connection", "close"),
            ("user-agent", "ALICE-Information/0.3"),
        ),
        timeout_seconds=retrieval.request_timeout_seconds,
    )
    request.validate(policy=retrieval)
    return request


@dataclass
class ScriptedDnsBackend:
    answers: tuple[InformationDnsAnswer, ...]
    error: Exception | None = None
    backend_type: str = "scripted_dns"

    def resolve(
        self,
        hostname: str,
        port: int,
        *,
        timeout_seconds: float,
        cancellation=None,
    ) -> tuple[InformationDnsAnswer, ...]:
        assert hostname == "example.com"
        assert port == 443
        assert 0 < timeout_seconds <= 10
        if self.error is not None:
            raise self.error
        return self.answers


class ScriptedConnection:
    def __init__(
        self,
        response: bytes,
        *,
        peer: str = ADDRESS,
        recv_error: Exception | None = None,
    ) -> None:
        self._buffer = bytearray(response)
        self._peer = peer
        self._recv_error = recv_error
        self.sent = b""
        self.closed = False
        self.timeouts: list[float | None] = []

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def recv(self, size: int) -> bytes:
        if self._recv_error is not None:
            error = self._recv_error
            self._recv_error = None
            raise error
        if not self._buffer:
            return b""
        take = min(size, len(self._buffer))
        result = bytes(self._buffer[:take])
        del self._buffer[:take]
        return result

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)

    def getpeername(self):
        return (self._peer, 443)

    def close(self) -> None:
        self.closed = True


@dataclass
class ScriptedSocketBackend:
    connection: ScriptedConnection | None = None
    error: Exception | None = None
    backend_type: str = "scripted_socket"

    def open(self, *, target, address: str, timeout_seconds: float):
        assert target.hostname == "example.com"
        assert address == ADDRESS
        assert 0 < timeout_seconds <= 10
        if self.error is not None:
            raise self.error
        assert self.connection is not None
        return self.connection


def _transport(backend: ScriptedSocketBackend) -> DirectInformationHttpsTransport:
    information, retrieval, live = _policies()
    return DirectInformationHttpsTransport(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=backend,
    )


def _response(
    body: bytes = b"hello",
    *,
    status: bytes = b"HTTP/1.1 200 OK",
    extra_headers: bytes = b"",
    include_length: bool = True,
) -> bytes:
    length = f"Content-Length: {len(body)}\r\n".encode() if include_length else b""
    return (
        status
        + b"\r\nContent-Type: text/plain; charset=utf-8\r\n"
        + length
        + extra_headers
        + b"Connection: close\r\n\r\n"
        + body
    )



def test_stdlib_dns_backend_enforces_timeout_and_single_inflight(monkeypatch) -> None:
    release = Event()

    def blocked_getaddrinfo(*args, **kwargs):
        del args, kwargs
        release.wait()
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (ADDRESS, 443),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", blocked_getaddrinfo)
    backend = StdlibInformationDnsBackend()
    with pytest.raises(InformationHttpExecutionError) as first:
        backend.resolve("example.com", 443, timeout_seconds=0.01)
    assert first.value.failure.code == "network_timeout"
    with pytest.raises(InformationHttpExecutionError) as second:
        backend.resolve("example.com", 443, timeout_seconds=0.01)
    assert second.value.failure.code == "network_timeout"
    release.set()


def test_stdlib_socket_backend_uses_direct_tls_validation(monkeypatch) -> None:
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE"):
        monkeypatch.delenv(name, raising=False)
    calls: dict[str, object] = {}

    class RawSocket:
        def settimeout(self, value):
            calls.setdefault("timeouts", []).append(value)

        def connect(self, destination):
            calls["destination"] = destination

        def close(self):
            calls["raw_closed"] = True

    class TlsSocket(RawSocket):
        pass

    class Context:
        def __init__(self):
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.minimum_version = None
            self.options = 0
            self.keylog_filename = None

        def set_alpn_protocols(self, protocols):
            calls["alpn"] = protocols

        def wrap_socket(self, raw_socket, *, server_hostname):
            calls["wrapped_raw"] = raw_socket
            calls["server_hostname"] = server_hostname
            calls["check_hostname"] = self.check_hostname
            calls["verify_mode"] = self.verify_mode
            calls["minimum_version"] = self.minimum_version
            return TlsSocket()

    raw = RawSocket()
    context = Context()
    time_values = iter((100.0, 100.0, 101.0, 102.0))
    monkeypatch.setattr("alice_information.live_http.monotonic", lambda: next(time_values))
    monkeypatch.setattr(socket, "socket", lambda *args: raw)
    monkeypatch.setattr(ssl, "create_default_context", lambda purpose: context)
    backend = StdlibInformationSocketBackend()
    connection = backend.open(
        target=_target(),
        address=ADDRESS,
        timeout_seconds=5,
    )
    assert isinstance(connection, TlsSocket)
    assert calls["destination"] == (ADDRESS, 443)
    assert calls["server_hostname"] == "example.com"
    assert calls["check_hostname"] is True
    assert calls["verify_mode"] == ssl.CERT_REQUIRED
    assert calls["minimum_version"] == ssl.TLSVersion.TLSv1_2
    assert calls["alpn"] == ["http/1.1"]
    assert calls["timeouts"] == [5.0, 4.0, 3.0]

def test_system_resolver_returns_sorted_unique_public_addresses() -> None:
    information, retrieval, live = _policies()
    backend = ScriptedDnsBackend(
        answers=(
            InformationDnsAnswer(socket.AF_INET6, "2606:2800:220:1:248:1893:25c8:1946"),
            InformationDnsAnswer(socket.AF_INET, ADDRESS),
            InformationDnsAnswer(socket.AF_INET, ADDRESS),
        )
    )
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=backend,
    )
    target = resolver.resolve(URL, policy=retrieval)
    assert target.addresses == (
        ADDRESS,
        "2606:2800:220:1:248:1893:25c8:1946",
    )


def test_system_resolver_rejects_http_private_answers_and_excess_results() -> None:
    information, retrieval, live = _policies()
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedDnsBackend((InformationDnsAnswer(socket.AF_INET, ADDRESS),)),
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        resolver.resolve("http://example.com/", policy=retrieval)
    assert error.value.failure.code == "invalid_source_url"

    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedDnsBackend(
            (InformationDnsAnswer(socket.AF_INET, "10.0.0.1"),)
        ),
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        resolver.resolve("https://example.com/", policy=retrieval)
    assert error.value.failure.code == "private_network_blocked"

    many = tuple(
        InformationDnsAnswer(socket.AF_INET, f"8.8.8.{index}")
        for index in range(1, 10)
    )
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedDnsBackend(many),
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        resolver.resolve("https://example.com/", policy=retrieval)
    assert error.value.failure.code == "dns_resolution_failed"


def test_system_resolver_rejects_unsafe_request_target_before_dns() -> None:
    information, retrieval, live = _policies()

    class NeverCalledBackend(ScriptedDnsBackend):
        def resolve(self, *args, **kwargs):
            raise AssertionError("DNS backend must not be called")

    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=NeverCalledBackend(()),
    )
    for url in (
        "https://example.com/a\\b",
        "https://example.com//authority-like",
        "https://example.com/é",
    ):
        with pytest.raises(InformationHttpExecutionError) as error:
            resolver.resolve(url, policy=retrieval)
        assert error.value.failure.code == "invalid_source_url"


def test_system_resolver_sanitizes_backend_failure() -> None:
    information, retrieval, live = _policies()
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedDnsBackend((), error=OSError("secret host detail")),
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        resolver.resolve("https://example.com/", policy=retrieval)
    assert error.value.failure.code == "dns_resolution_failed"
    assert "secret host detail" not in str(error.value)
    assert "example.com" not in str(error.value)


def test_stdlib_socket_backend_rejects_tls_environment_overrides(monkeypatch) -> None:
    backend = StdlibInformationSocketBackend()
    monkeypatch.setenv("SSLKEYLOGFILE", "/tmp/private-key-log")
    with pytest.raises(InformationHttpExecutionError) as error:
        backend.open(target=_target(), address=ADDRESS, timeout_seconds=5)
    assert error.value.failure.code == "tls_validation_failed"
    assert "private-key-log" not in str(error.value)


def test_direct_transport_rejects_backslash_request_target() -> None:
    information, retrieval, live = _policies()
    connection = ScriptedConnection(_response())
    transport = DirectInformationHttpsTransport(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedSocketBackend(connection=connection),
    )
    target = InformationResolvedTarget(
        canonical_url="https://example.com/a\\b",
        scheme="https",
        hostname="example.com",
        port=443,
        addresses=(ADDRESS,),
    )
    request = InformationHttpRequest(
        canonical_url=target.canonical_url,
        headers=_request().headers,
        timeout_seconds=10,
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(request, target=target, policy=retrieval)
    assert error.value.failure.code == "invalid_source_url"


def test_direct_transport_sends_fixed_credential_free_request_and_reads_body() -> None:
    connection = ScriptedConnection(_response())
    transport = _transport(ScriptedSocketBackend(connection=connection))
    _information, retrieval, _live = _policies()
    response = transport.get(_request(), target=_target(), policy=retrieval)
    assert response.status_code == 200
    assert b"".join(response.body_chunks) == b"hello"
    assert connection.sent.startswith(b"GET /report?q=public HTTP/1.1\r\n")
    assert b"Host: example.com\r\n" in connection.sent
    assert b"authorization" not in connection.sent.lower()
    assert b"cookie" not in connection.sent.lower()
    assert connection.closed is True


def test_direct_transport_reads_connection_close_body_without_length() -> None:
    connection = ScriptedConnection(_response(b"close-body", include_length=False))
    transport = _transport(ScriptedSocketBackend(connection=connection))
    _information, retrieval, _live = _policies()
    response = transport.get(_request(), target=_target(), policy=retrieval)
    assert b"".join(response.body_chunks) == b"close-body"


def test_direct_transport_rejects_peer_mismatch_and_transfer_encoding() -> None:
    _information, retrieval, _live = _policies()
    connection = ScriptedConnection(_response(), peer="8.8.8.8")
    transport = _transport(ScriptedSocketBackend(connection=connection))
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "peer_address_mismatch"

    connection = ScriptedConnection(
        _response(extra_headers=b"Transfer-Encoding: chunked\r\n")
    )
    transport = _transport(ScriptedSocketBackend(connection=connection))
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "response_header_invalid"


def test_direct_transport_rejects_invalid_protocol_and_short_body() -> None:
    _information, retrieval, _live = _policies()
    connection = ScriptedConnection(_response(status=b"NOTHTTP 200 OK"))
    transport = _transport(ScriptedSocketBackend(connection=connection))
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "http_protocol_invalid"

    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
        b"Content-Length: 10\r\nConnection: close\r\n\r\nshort"
    )
    connection = ScriptedConnection(response)
    transport = _transport(ScriptedSocketBackend(connection=connection))
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "http_protocol_invalid"


def test_direct_transport_maps_timeout_tls_and_connection_failures() -> None:
    _information, retrieval, _live = _policies()
    for exception, code in (
        (TimeoutError("private timeout"), "network_timeout"),
        (ssl.SSLError("private tls"), "tls_validation_failed"),
        (OSError("private connect"), "network_connection_failed"),
    ):
        transport = _transport(ScriptedSocketBackend(error=exception))
        with pytest.raises(InformationHttpExecutionError) as error:
            transport.get(_request(), target=_target(), policy=retrieval)
        assert error.value.failure.code == code
        assert "private" not in str(error.value)


def test_direct_transport_maps_read_timeout_and_closes_connection() -> None:
    _information, retrieval, _live = _policies()
    connection = ScriptedConnection(b"", recv_error=TimeoutError("secret"))
    transport = _transport(ScriptedSocketBackend(connection=connection))
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "network_timeout"
    assert connection.closed is True


def test_direct_transport_bounds_content_length_before_integer_conversion() -> None:
    _information, retrieval, _live = _policies()
    declared = b"9" * 5000
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
        + b"Content-Length: "
        + declared
        + b"\r\nConnection: close\r\n\r\n"
    )
    transport = _transport(
        ScriptedSocketBackend(connection=ScriptedConnection(response))
    )
    with pytest.raises(InformationHttpExecutionError) as error:
        transport.get(_request(), target=_target(), policy=retrieval)
    assert error.value.failure.code == "response_too_large"


def test_direct_transport_honors_cancellation_after_connect() -> None:
    _information, retrieval, _live = _policies()
    token = InformationCancellationToken()
    connection = ScriptedConnection(_response())

    @dataclass
    class CancellingSocketBackend:
        backend_type: str = "cancelling_socket"

        def open(self, *, target, address: str, timeout_seconds: float):
            del target, address, timeout_seconds
            token.cancel()
            return connection

    transport = _transport(CancellingSocketBackend())
    with pytest.raises(InformationProviderCancelledError):
        transport.get(
            _request(),
            target=_target(),
            policy=retrieval,
            cancellation=token,
        )
    assert connection.sent == b""
    assert connection.closed is True


def test_live_retriever_requires_exact_stdlib_components() -> None:
    information, retrieval, live = _policies()
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=StdlibInformationDnsBackend(),
    )
    transport = DirectInformationHttpsTransport(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=StdlibInformationSocketBackend(),
    )
    retriever = LiveControlledInformationHttpRetriever(
        information_policy=information,
        retrieval_policy=retrieval,
        resolver=resolver,
        transport=transport,
        live_policy=live,
    )
    assert len(live_component_digest(resolver=resolver, transport=transport)) == 64
    assert retriever.live_policy == live

    fake_resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=ScriptedDnsBackend((InformationDnsAnswer(socket.AF_INET, ADDRESS),)),
    )
    with pytest.raises(ValueError, match="backend"):
        LiveControlledInformationHttpRetriever(
            information_policy=information,
            retrieval_policy=retrieval,
            resolver=fake_resolver,
            transport=transport,
            live_policy=live,
        )


def test_live_component_digest_contains_no_hostname_or_url() -> None:
    information, retrieval, live = _policies()
    resolver = SystemInformationNameResolver(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=StdlibInformationDnsBackend(),
    )
    transport = DirectInformationHttpsTransport(
        information_policy=information,
        retrieval_policy=retrieval,
        live_policy=live,
        backend=StdlibInformationSocketBackend(),
    )
    digest = live_component_digest(resolver=resolver, transport=transport)
    assert "example.com" not in digest
    assert "https" not in digest
