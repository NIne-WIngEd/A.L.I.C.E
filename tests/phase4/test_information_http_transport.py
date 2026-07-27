from __future__ import annotations

import pytest
from alice_information.http_transport import (
    DeterministicInformationHttpTransport,
    DeterministicInformationNameResolver,
    InformationHttpConfigurationError,
    InformationHttpExecutionError,
    InformationHttpRequest,
    InformationRawHttpResponse,
    InformationResolvedTarget,
    validate_global_address,
)
from alice_information.providers import (
    InformationCancellationToken,
    InformationProviderCancelledError,
)
from alice_information.retrieval_policy import load_information_http_retrieval_policy

POLICY = load_information_http_retrieval_policy()
URL = "https://example.com/"
PUBLIC_IP = "93.184.216.34"


def _request() -> InformationHttpRequest:
    return InformationHttpRequest(
        canonical_url=URL,
        headers=(
            ("accept", "text/html, application/xhtml+xml, text/plain;q=0.9"),
            ("accept-encoding", "gzip, deflate"),
            ("connection", "close"),
            ("user-agent", "ALICE-Information/0.3"),
        ),
        timeout_seconds=10,
    )


def _response(*, peer: str = PUBLIC_IP) -> InformationRawHttpResponse:
    return InformationRawHttpResponse(
        status_code=200,
        headers=(("content-type", "text/plain; charset=utf-8"),),
        body_chunks=(b"safe",),
        peer_address=peer,
    )


def test_global_address_gate_rejects_non_public_ranges() -> None:
    assert validate_global_address(PUBLIC_IP) == PUBLIC_IP
    for address in (
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "192.0.2.1",
        "::1",
        "fe80::1",
        "fc00::1",
    ):
        with pytest.raises(InformationHttpExecutionError) as raised:
            validate_global_address(address)
        assert raised.value.failure.code == "private_network_blocked"


def test_deterministic_resolver_rejects_private_and_localhost_fixtures() -> None:
    with pytest.raises(InformationHttpExecutionError):
        DeterministicInformationNameResolver({"example.com": ("10.0.0.1",)})
    with pytest.raises(InformationHttpConfigurationError, match="canonical"):
        DeterministicInformationNameResolver({"localhost": (PUBLIC_IP,)})


def test_resolver_enforces_default_ports_and_canonical_hostname() -> None:
    resolver = DeterministicInformationNameResolver({"example.com": (PUBLIC_IP,)})
    target = resolver.resolve(URL, policy=POLICY)
    assert target.port == 443
    with pytest.raises(InformationHttpExecutionError) as custom_port:
        resolver.resolve("https://example.com:8443/", policy=POLICY)
    assert custom_port.value.failure.code == "invalid_source_url"
    with pytest.raises(InformationHttpExecutionError):
        resolver.resolve("https://example.com./", policy=POLICY)


def test_resolved_target_rejects_mixed_private_address_set() -> None:
    target = InformationResolvedTarget(
        canonical_url=URL,
        scheme="https",
        hostname="example.com",
        port=443,
        addresses=(PUBLIC_IP, "127.0.0.1"),
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        target.validate(policy=POLICY)
    assert raised.value.failure.code == "private_network_blocked"


def test_transport_pins_peer_to_resolved_address_set() -> None:
    resolver = DeterministicInformationNameResolver({"example.com": (PUBLIC_IP,)})
    target = resolver.resolve(URL, policy=POLICY)
    transport = DeterministicInformationHttpTransport(
        {URL: _response(peer="8.8.8.8")}
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        transport.get(_request(), target=target, policy=POLICY)
    assert raised.value.failure.code == "peer_address_mismatch"


def test_request_headers_are_fixed_and_credential_free() -> None:
    bad = InformationHttpRequest(
        canonical_url=URL,
        headers=(("authorization", "Bearer secret"),),
        timeout_seconds=10,
    )
    with pytest.raises(InformationHttpConfigurationError, match="fixed"):
        bad.validate(policy=POLICY)


def test_resolver_and_transport_honor_cancellation() -> None:
    token = InformationCancellationToken()
    token.cancel()
    resolver = DeterministicInformationNameResolver({"example.com": (PUBLIC_IP,)})
    with pytest.raises(InformationProviderCancelledError):
        resolver.resolve(URL, policy=POLICY, cancellation=token)


def test_fixture_keys_cannot_collide_after_normalization() -> None:
    with pytest.raises(InformationHttpConfigurationError, match="unique"):
        DeterministicInformationNameResolver(
            {"EXAMPLE.COM": (PUBLIC_IP,), "example.com": ("8.8.8.8",)}
        )
    with pytest.raises(InformationHttpConfigurationError, match="canonical"):
        DeterministicInformationHttpTransport(
            {
                "https://example.com": _response(),
                "https://example.com/": _response(),
            }
        )


def test_multicast_addresses_are_not_accepted_as_public_targets() -> None:
    with pytest.raises(InformationHttpExecutionError) as raised:
        validate_global_address("224.0.0.1")
    assert raised.value.failure.code == "private_network_blocked"


def test_response_header_shape_uses_token_names_and_rejects_del() -> None:
    with pytest.raises(InformationHttpConfigurationError, match="token grammar"):
        InformationRawHttpResponse(
            status_code=200,
            headers=(("content:type", "text/plain"),),
            body_chunks=(b"body",),
            peer_address=PUBLIC_IP,
        ).validate_shape()
    with pytest.raises(InformationHttpConfigurationError, match="control"):
        InformationRawHttpResponse(
            status_code=200,
            headers=(("content-type", "text/plain\x7f"),),
            body_chunks=(b"body",),
            peer_address=PUBLIC_IP,
        ).validate_shape()
