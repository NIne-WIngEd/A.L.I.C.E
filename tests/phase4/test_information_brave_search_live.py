import json

import pytest

from alice_information.brave_search_live import (
    BraveSearchTransportError,
    StrictBraveSearchHttpsTransport,
)
from alice_information.live_provider_contracts import InformationLiveProviderExecutionError

from _information_live_provider_helpers import (
    FixtureResolver,
    FixtureSocketBackend,
    configuration,
    policy,
)


def _response(payload=None):
    body = json.dumps(payload or {"web": {"results": []}}).encode()
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )


def test_exact_transport_emits_one_credential_bearing_request():
    socket_backend = FixtureSocketBackend(_response())
    transport = StrictBraveSearchHttpsTransport(
        resolver=FixtureResolver(),
        retrieval_policy=object(),
        socket_backend=socket_backend,
        policy=policy(),
    )
    raw = transport.perform(
        canonical_url="https://api.search.brave.com/res/v1/web/search?q=test",
        credential_header="X-Subscription-Token",
        credential=configuration().credential,
        timeout_seconds=5,
        maximum_response_bytes=1024,
        cancellation=None,
    )
    assert raw.status_code == 200
    assert socket_backend.socket.sent.count(b"X-Subscription-Token:") == 1
    assert socket_backend.socket.closed


def test_alternate_endpoint_is_rejected_before_network():
    transport = StrictBraveSearchHttpsTransport(
        resolver=FixtureResolver(),
        retrieval_policy=object(),
        socket_backend=FixtureSocketBackend(_response()),
        policy=policy(),
    )
    with pytest.raises(BraveSearchTransportError):
        transport.perform(
            canonical_url="https://example.com/res/v1/web/search?q=test",
            credential_header="X-Subscription-Token",
            credential=configuration().credential,
            timeout_seconds=5,
            maximum_response_bytes=1024,
            cancellation=None,
        )


def test_private_peer_fails_closed_without_leaking_secret():
    secret = "super-secret-token"
    transport = StrictBraveSearchHttpsTransport(
        resolver=FixtureResolver("1.1.1.1"),
        retrieval_policy=object(),
        socket_backend=FixtureSocketBackend(_response(), peer="127.0.0.1"),
        policy=policy(),
    )
    with pytest.raises(InformationLiveProviderExecutionError) as caught:
        transport.perform(
            canonical_url="https://api.search.brave.com/res/v1/web/search?q=test",
            credential_header="X-Subscription-Token",
            credential=configuration(secret).credential,
            timeout_seconds=5,
            maximum_response_bytes=1024,
            cancellation=None,
        )
    assert secret not in str(caught.value)
