from __future__ import annotations

import gzip
from dataclasses import replace

import pytest
from alice_information.http_transport import (
    DeterministicInformationHttpTransport,
    DeterministicInformationNameResolver,
    InformationHttpExecutionError,
    InformationRawHttpResponse,
)
from alice_information.policy import load_information_policy
from alice_information.providers import (
    InformationCancellationToken,
    InformationProviderCancelledError,
)
from alice_information.retrieval import (
    ControlledInformationHttpRetriever,
    deduplicate_retrieved_resources,
)
from alice_information.retrieval_policy import load_information_http_retrieval_policy

PUBLIC_IP = "93.184.216.34"
ALT_IP = "8.8.8.8"
START = "https://example.com/start"
FINAL = "https://www.example.com/article"


def _response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
    encoding: str | None = None,
    location: str | None = None,
    peer: str = PUBLIC_IP,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> InformationRawHttpResponse:
    headers: list[tuple[str, str]] = [("content-type", content_type)]
    if encoding is not None:
        headers.append(("content-encoding", encoding))
    if location is not None:
        headers.append(("location", location))
    headers.extend(extra_headers)
    return InformationRawHttpResponse(
        status_code=status,
        headers=tuple(headers),
        body_chunks=(body,),
        peer_address=peer,
    )


def _retriever(
    fixtures: dict[str, InformationRawHttpResponse],
    *,
    hosts: dict[str, tuple[str, ...]] | None = None,
    policy=None,
) -> ControlledInformationHttpRetriever:
    return ControlledInformationHttpRetriever(
        information_policy=load_information_policy(),
        retrieval_policy=policy or load_information_http_retrieval_policy(),
        resolver=DeterministicInformationNameResolver(
            hosts
            or {
                "example.com": (PUBLIC_IP,),
                "www.example.com": (ALT_IP,),
            }
        ),
        transport=DeterministicInformationHttpTransport(fixtures),
    )


def test_plain_text_retrieval_is_deterministic_and_digest_bound() -> None:
    resource = _retriever(
        {START: _response(b"  First line\r\n\r\n Second   line  ")}
    ).retrieve(START)
    assert resource.normalized_text == "First line\n\nSecond line"
    assert resource.requested_url == START
    assert resource.final_url == START
    resource.validate()


def test_html_normalization_removes_active_and_hidden_content() -> None:
    body = b"""<html><head><title> Example Title </title><style>bad</style></head>
    <body><h1>Hello</h1><script>steal()</script><p>World &amp; friends</p></body></html>"""
    resource = _retriever(
        {START: _response(body, content_type="text/html; charset=UTF-8")}
    ).retrieve(START)
    assert resource.title == "Example Title"
    assert resource.normalized_text == "Hello\n\nWorld & friends"
    assert "steal" not in resource.normalized_text


def test_redirect_is_revalidated_and_recorded() -> None:
    retriever = _retriever(
        {
            START: _response(b"", status=302, location=FINAL),
            FINAL: _response(b"final", peer=ALT_IP),
        }
    )
    resource = retriever.retrieve(START)
    assert resource.redirect_chain == (START, FINAL)
    assert resource.final_url == FINAL


def test_https_downgrade_and_redirect_loops_are_blocked() -> None:
    downgrade = "http://www.example.com/article"
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(
            {START: _response(b"", status=302, location=downgrade)}
        ).retrieve(START)
    assert raised.value.failure.code == "redirect_blocked"

    with pytest.raises(InformationHttpExecutionError) as loop:
        _retriever(
            {
                START: _response(b"", status=302, location=FINAL),
                FINAL: _response(b"", status=302, location=START, peer=ALT_IP),
            }
        ).retrieve(START)
    assert loop.value.failure.code == "redirect_blocked"


def test_literal_private_redirect_is_blocked_before_transport() -> None:
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(
            {START: _response(b"", status=302, location="http://127.0.0.1/")}
        ).retrieve(START)
    assert raised.value.failure.code == "redirect_blocked"


def test_redirect_budget_is_enforced() -> None:
    policy = replace(load_information_http_retrieval_policy(), max_redirects=1)
    second = "https://www.example.com/second"
    fixtures = {
        START: _response(b"", status=302, location=FINAL),
        FINAL: _response(b"", status=302, location=second, peer=ALT_IP),
        second: _response(b"done", peer=ALT_IP),
    }
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(fixtures, policy=policy).retrieve(START)
    assert raised.value.failure.code == "redirect_blocked"


def test_content_type_charset_download_and_duplicate_headers_fail_closed() -> None:
    cases = (
        _response(b"pdf", content_type="application/pdf"),
        _response(b"latin", content_type="text/plain; charset=iso-8859-1"),
        _response(
            b"download",
            extra_headers=(("content-disposition", "attachment; filename=x.txt"),),
        ),
        _response(
            b"duplicate",
            extra_headers=(("content-type", "text/html"),),
        ),
    )
    expected = (
        "unsupported_content_type",
        "unsupported_content_type",
        "unsupported_content_type",
        "response_header_invalid",
    )
    for response, code in zip(cases, expected, strict=True):
        with pytest.raises(InformationHttpExecutionError) as raised:
            _retriever({START: response}).retrieve(START)
        assert raised.value.failure.code == code


def test_declared_and_streamed_byte_limits_fail_closed() -> None:
    declared = _response(
        b"small",
        extra_headers=(("content-length", "2000001"),),
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever({START: declared}).retrieve(START)
    assert raised.value.failure.code == "response_too_large"

    mismatch = _response(
        b"small",
        extra_headers=(("content-length", "4"),),
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever({START: mismatch}).retrieve(START)
    assert raised.value.failure.code == "response_header_invalid"

    small_policy = replace(
        load_information_http_retrieval_policy(),
        max_wire_bytes=4,
        max_decoded_bytes=4,
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever({START: _response(b"12345")}, policy=small_policy).retrieve(START)
    assert raised.value.failure.code == "response_too_large"


def test_gzip_is_bounded_and_malformed_streams_are_rejected() -> None:
    compressed = gzip.compress(b"hello world")
    resource = _retriever(
        {START: _response(compressed, encoding="gzip")}
    ).retrieve(START)
    assert resource.normalized_text == "hello world"

    with pytest.raises(InformationHttpExecutionError) as malformed:
        _retriever(
            {START: _response(b"not-gzip", encoding="gzip")}
        ).retrieve(START)
    assert malformed.value.failure.code in {"content_decode_failed", "response_too_large"}

    bomb_policy = replace(
        load_information_http_retrieval_policy(),
        max_decoded_bytes=10,
    )
    with pytest.raises(InformationHttpExecutionError) as bomb:
        _retriever(
            {START: _response(gzip.compress(b"x" * 100), encoding="gzip")},
            policy=bomb_policy,
        ).retrieve(START)
    assert bomb.value.failure.code == "response_too_large"


def test_peer_mismatch_blocks_dns_rebinding() -> None:
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(
            {START: _response(b"body", peer=ALT_IP)},
            hosts={"example.com": (PUBLIC_IP,)},
        ).retrieve(START)
    assert raised.value.failure.code == "peer_address_mismatch"


def test_cancellation_is_preserved() -> None:
    token = InformationCancellationToken()
    token.cancel()
    with pytest.raises(InformationProviderCancelledError):
        _retriever({START: _response(b"body")}).retrieve(
            START,
            cancellation=token,
        )


def test_exact_normalized_duplicates_are_reported() -> None:
    first = _retriever({START: _response(b"same body")}).retrieve(START)
    other_url = "https://www.example.com/other"
    second = _retriever(
        {other_url: _response(b"same   body", peer=ALT_IP)}
    ).retrieve(other_url)
    unique, duplicates = deduplicate_retrieved_resources((first, second))
    assert unique == (first,)
    assert len(duplicates) == 1
    assert duplicates[0].retained_url == START
    assert duplicates[0].duplicate_url == other_url


def test_resource_projects_to_exact_source_contract() -> None:
    resource = _retriever(
        {START: _response(b"source body")}
    ).retrieve(START)
    source = resource.to_source_document(
        source_id="source-1",
        provider="controlled-http-fixture-v1",
        retrieved_at="2026-07-27T12:00:00Z",
    )
    assert source.canonical_url == START
    assert source.normalized_text == "source body"
    assert source.content_sha256 == resource.content_sha256
    source.validate()


def test_retriever_rejects_substituted_resolver_or_transport_types() -> None:
    class ResolverSubclass(DeterministicInformationNameResolver):
        pass

    class TransportSubclass(DeterministicInformationHttpTransport):
        pass

    with pytest.raises(ValueError, match="exact deterministic resolver"):
        ControlledInformationHttpRetriever(
            information_policy=load_information_policy(),
            retrieval_policy=load_information_http_retrieval_policy(),
            resolver=ResolverSubclass({"example.com": (PUBLIC_IP,)}),
            transport=DeterministicInformationHttpTransport(
                {START: _response(b"body")}
            ),
        )
    with pytest.raises(ValueError, match="exact deterministic transport"):
        ControlledInformationHttpRetriever(
            information_policy=load_information_policy(),
            retrieval_policy=load_information_http_retrieval_policy(),
            resolver=DeterministicInformationNameResolver(
                {"example.com": (PUBLIC_IP,)}
            ),
            transport=TransportSubclass({START: _response(b"body")}),
        )


def test_retrieval_failures_do_not_echo_host_or_query_text() -> None:
    secret_url = "https://missing.example/path?token=do-not-log"
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(
            {START: _response(b"body")},
            hosts={"example.com": (PUBLIC_IP,)},
        ).retrieve(secret_url)
    rendered = str(raised.value)
    assert "missing.example" not in rendered
    assert "do-not-log" not in rendered
    assert raised.value.failure.code == "dns_resolution_failed"


def test_redirect_bodies_are_bounded_before_following() -> None:
    policy = replace(
        load_information_http_retrieval_policy(),
        max_wire_bytes=4,
        max_decoded_bytes=4,
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever(
            {START: _response(b"12345", status=302, location=FINAL)},
            policy=policy,
        ).retrieve(START)
    assert raised.value.failure.code == "response_too_large"


def test_transfer_encoding_is_rejected_at_the_boundary() -> None:
    response = _response(
        b"body",
        extra_headers=(("transfer-encoding", "chunked"),),
    )
    with pytest.raises(InformationHttpExecutionError) as raised:
        _retriever({START: response}).retrieve(START)
    assert raised.value.failure.code == "response_header_invalid"


def test_normalization_does_not_double_decode_entities() -> None:
    html_resource = _retriever(
        {START: _response(b"<p>&amp;amp;</p>", content_type="text/html")}
    ).retrieve(START)
    assert html_resource.normalized_text == "&amp;"

    plain_resource = _retriever({START: _response(b"literal &amp; text")}).retrieve(START)
    assert plain_resource.normalized_text == "literal &amp; text"
