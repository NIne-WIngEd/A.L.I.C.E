from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from alice_information.contracts import sha256_text
from alice_information.freshness_policy import load_information_freshness_policy
from alice_information.http_transport import (
    DeterministicInformationHttpTransport,
    DeterministicInformationNameResolver,
    InformationHttpExecutionError,
    InformationRawHttpResponse,
)
from alice_information.policy import load_information_policy
from alice_information.retrieval import (
    ControlledInformationHttpRetriever,
    deduplicate_retrieved_resources,
)
from alice_information.retrieval_policy import load_information_http_retrieval_policy
from alice_information.temporal_metadata import (
    DeterministicInformationTemporalMetadataAggregator,
    DeterministicInformationTemporalMetadataResolver,
    InformationTemporalMetadataCandidate,
    InformationTemporalMetadataError,
)
from alice_information.temporal_metadata_policy import (
    load_information_temporal_metadata_policy,
)

PUBLIC_IP = "93.184.216.34"
URL = "https://example.com/article"
RETRIEVED_AT = "2026-07-27T12:00:00Z"


def _response(
    body: bytes,
    *,
    content_type: str = "text/html; charset=utf-8",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> InformationRawHttpResponse:
    return InformationRawHttpResponse(
        status_code=200,
        headers=(("content-type", content_type),) + extra_headers,
        body_chunks=(body,),
        peer_address=PUBLIC_IP,
    )


def _retrieve(
    body: bytes,
    *,
    content_type: str = "text/html; charset=utf-8",
    extra_headers: tuple[tuple[str, str], ...] = (),
    url: str = URL,
):
    return ControlledInformationHttpRetriever(
        information_policy=load_information_policy(),
        retrieval_policy=load_information_http_retrieval_policy(),
        resolver=DeterministicInformationNameResolver(
            {urlsplit(url).hostname: (PUBLIC_IP,)}
        ),
        transport=DeterministicInformationHttpTransport(
            {
                url: _response(
                    body,
                    content_type=content_type,
                    extra_headers=extra_headers,
                )
            }
        ),
    ).retrieve(url)


def _resolver():
    information_policy = load_information_policy()
    freshness_policy = load_information_freshness_policy(
        information_policy=information_policy
    )
    metadata_policy = load_information_temporal_metadata_policy(
        information_policy=information_policy,
        freshness_policy=freshness_policy,
    )
    return (
        DeterministicInformationTemporalMetadataResolver(
            information_policy=information_policy,
            freshness_policy=freshness_policy,
            temporal_metadata_policy=metadata_policy,
        ),
        metadata_policy,
        freshness_policy,
    )


def _resolved(body: bytes, **kwargs):
    resolver, _, _ = _resolver()
    return resolver.resolve(_retrieve(body, **kwargs))


def test_candidate_normalizes_iso_timestamp_to_utc() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_article_published_time",
        raw_value="2026-07-27T08:30:00-04:00",
    )
    assert candidate.normalized_timestamp == "2026-07-27T12:30:00Z"
    assert candidate.valid is True


def test_candidate_rejects_unknown_local_offset() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_article_published_time",
        raw_value="2026-07-27T12:30:00-00:00",
    )
    assert candidate.normalized_timestamp is None
    assert candidate.valid is False


def test_http_last_modified_uses_rfc7231_parsing() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="http_last_modified",
        raw_value="Mon, 27 Jul 2026 12:00:00 GMT",
    )
    assert candidate.kind == "updated_at"
    assert candidate.normalized_timestamp == "2026-07-27T12:00:00Z"


def test_http_last_modified_rejects_non_imf_fixdate_forms() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="http_last_modified",
        raw_value="27 Jul 2026 12:00:00 GMT",
    )
    assert candidate.valid is False


def test_http_last_modified_rejects_mismatched_weekday() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="http_last_modified",
        raw_value="Sun, 27 Jul 2026 12:00:00 GMT",
    )
    assert candidate.valid is False


def test_html_temporal_metadata_rejects_non_rfc3339_basic_form() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_date_published",
        raw_value="20260727T120000Z",
    )
    assert candidate.valid is False


@pytest.mark.parametrize(
    "raw_value",
    (
        "2026-07-27",
        "not-a-date",
        "2026-07-27T12:00:00",
        "2026-07-27T12:00:00Z\u200b",
    ),
)
def test_invalid_candidate_values_are_preserved_but_not_accepted(raw_value: str) -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_date_published",
        raw_value=raw_value,
    )
    assert candidate.valid is False
    assert candidate.normalized_timestamp is None


def test_candidate_metadata_record_omits_raw_value() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_date_published",
        raw_value="2026-07-27T12:00:00Z",
    )
    record = candidate.metadata_record()
    assert "raw_value" not in record
    assert record["raw_value_sha256"] == sha256_text(candidate.raw_value)


def test_candidate_repr_does_not_expose_raw_value() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_date_published",
        raw_value="2026-07-27T08:00:00-04:00",
    )
    assert candidate.raw_value not in repr(candidate)


def test_candidate_forgery_is_rejected_by_rederivation() -> None:
    candidate = InformationTemporalMetadataCandidate.create(
        origin="html_meta_date_published",
        raw_value="2026-07-27T12:00:00Z",
    )
    with pytest.raises(InformationTemporalMetadataError):
        replace(candidate, normalized_timestamp="2020-01-01T00:00:00Z").validate()


def test_html_meta_and_time_candidates_are_extracted_without_visible_text_inference() -> None:
    body = b"""<html><head>
    <meta property='article:published_time' content='2026-07-20T10:00:00Z'>
    <meta itemprop='dateModified' content='2026-07-21T11:00:00Z'>
    </head><body><time itemprop='datePublished' datetime='2026-07-20T10:00:00Z'>July 20</time>
    <p>Updated yesterday at noon.</p></body></html>"""
    resource = _retrieve(body)
    assert len(resource.temporal_metadata_candidates) == 3
    assert {item.origin for item in resource.temporal_metadata_candidates} == {
        "html_meta_article_published_time",
        "html_meta_date_modified",
        "html_time_date_published",
    }
    assert all("yesterday" not in item.raw_value for item in resource.temporal_metadata_candidates)


def test_plain_text_dates_are_not_inferred() -> None:
    resource = _retrieve(
        b"Published 2026-07-20 and updated 2026-07-21.",
        content_type="text/plain; charset=utf-8",
    )
    assert resource.temporal_metadata_candidates == ()


def test_metadata_inside_template_is_not_extracted() -> None:
    resource = _retrieve(
        b"<html><head><template><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></template></head><body>Body</body></html>"
    )
    assert resource.temporal_metadata_candidates == ()


def test_mismatched_hidden_end_tag_cannot_escape_template_boundary() -> None:
    resource = _retrieve(
        b"<html><head><template></script><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></template></head><body>Body</body></html>"
    )
    assert resource.temporal_metadata_candidates == ()


def test_last_modified_header_is_extracted_as_update_evidence() -> None:
    resource = _retrieve(
        b"<p>Body</p>",
        extra_headers=(("last-modified", "Mon, 27 Jul 2026 10:00:00 GMT"),),
    )
    candidate = resource.temporal_metadata_candidates[0]
    assert candidate.origin == "http_last_modified"
    assert candidate.normalized_timestamp == "2026-07-27T10:00:00Z"


def test_explicit_html_update_takes_precedence_over_last_modified_fallback() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='dateModified' content='2026-07-27T09:00:00Z'></head><body>Body</body></html>",
        extra_headers=(("last-modified", "Mon, 27 Jul 2026 10:00:00 GMT"),),
    )
    assert resolved.resolution.verdict == "resolved"
    assert resolved.resolution.updated_at == "2026-07-27T09:00:00Z"
    assert resolved.resolution.updated_origins == ("html_meta_date_modified",)


def test_duplicate_unrelated_html_attributes_do_not_block_normalization() -> None:
    resource = _retrieve(
        b"<html><body><p class='one' class='two'>Body</p></body></html>"
    )
    assert resource.normalized_text == "Body"


def test_duplicate_metadata_named_attributes_on_other_tags_do_not_block() -> None:
    resource = _retrieve(
        b"<html><body><input name='one' name='two'><p>Body</p></body></html>"
    )
    assert resource.normalized_text == "Body"


def test_duplicate_last_modified_headers_are_rejected() -> None:
    with pytest.raises(InformationHttpExecutionError):
        _retrieve(
            b"<p>Body</p>",
            extra_headers=(
                ("last-modified", "Mon, 27 Jul 2026 10:00:00 GMT"),
                ("last-modified", "Mon, 27 Jul 2026 10:00:00 GMT"),
            ),
        )


def test_recognized_metadata_without_value_becomes_invalid_evidence() -> None:
    resolved = _resolved(
        b"<html><head><meta property='article:published_time'></head><body>Body</body></html>"
    )
    assert resolved.resolution.verdict == "invalid"
    assert resolved.resolution.supports_temporal_claims is False


def test_conflicting_markers_on_one_tag_fail_retrieval_closed() -> None:
    with pytest.raises(InformationHttpExecutionError):
        _retrieve(
            b"<html><head><meta property='article:published_time' itemprop='dateModified' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
        )


def test_exact_duplicate_candidates_resolve_to_one_timestamp() -> None:
    resolved = _resolved(
        b"""<html><head>
        <meta property='article:published_time' content='2026-07-20T10:00:00Z'>
        <meta itemprop='datePublished' content='2026-07-20T10:00:00+00:00'>
        </head><body>Body</body></html>"""
    )
    assert resolved.resolution.verdict == "resolved"
    assert resolved.resolution.published_at == "2026-07-20T10:00:00Z"
    assert len(resolved.resolution.published_origins) == 2


def test_deduplication_preserves_same_body_with_different_temporal_evidence() -> None:
    first = _retrieve(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Same body</body></html>"
    )
    second = _retrieve(
        b"<html><head><meta itemprop='datePublished' content='2026-07-21T10:00:00Z'></head><body>Same body</body></html>"
    )
    assert first.content_sha256 == second.content_sha256
    unique, duplicates = deduplicate_retrieved_resources((first, second))
    assert unique == (first, second)
    assert duplicates == ()


def test_distinct_publication_candidates_preserve_conflict() -> None:
    resolved = _resolved(
        b"""<html><head>
        <meta property='article:published_time' content='2026-07-20T10:00:00Z'>
        <meta itemprop='datePublished' content='2026-07-21T10:00:00Z'>
        </head><body>Body</body></html>"""
    )
    assert resolved.resolution.verdict == "conflict"
    assert resolved.resolution.published_at is None


def test_invalid_candidate_fails_resolution_closed() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='not-a-date'></head><body>Body</body></html>"
    )
    assert resolved.resolution.verdict == "invalid"


def test_update_before_publication_beyond_skew_is_conflict() -> None:
    resolved = _resolved(
        b"""<html><head>
        <meta itemprop='datePublished' content='2026-07-20T10:00:00Z'>
        <meta itemprop='dateModified' content='2026-07-20T09:00:00Z'>
        </head><body>Body</body></html>"""
    )
    assert resolved.resolution.verdict == "conflict"


def test_update_within_clock_skew_before_publication_is_resolved() -> None:
    resolved = _resolved(
        b"""<html><head>
        <meta itemprop='datePublished' content='2026-07-20T10:00:00Z'>
        <meta itemprop='dateModified' content='2026-07-20T09:56:00Z'>
        </head><body>Body</body></html>"""
    )
    assert resolved.resolution.verdict == "resolved"


def test_undated_resource_is_explicitly_undated() -> None:
    resolved = _resolved(b"<html><body>Body</body></html>")
    assert resolved.resolution.verdict == "undated"
    assert resolved.resolution.supports_temporal_claims is False


def test_resolved_resource_projects_verified_dates_to_source_document() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    source = resolved.to_source_document(
        source_id="source-1",
        provider="fixture",
        retrieved_at=RETRIEVED_AT,
        policy=policy,
        freshness_policy=freshness,
    )
    assert source.published_at == "2026-07-20T10:00:00Z"


def test_undated_resource_projects_without_fabricating_dates() -> None:
    resolved = _resolved(b"<html><body>Body</body></html>")
    _, policy, freshness = _resolver()
    source = resolved.to_source_document(
        source_id="source-1",
        provider="fixture",
        retrieved_at=RETRIEVED_AT,
        policy=policy,
        freshness_policy=freshness,
    )
    assert source.published_at is None and source.updated_at is None


def test_conflicted_resource_cannot_project_dates() -> None:
    resolved = _resolved(
        b"""<html><head>
        <meta itemprop='datePublished' content='2026-07-20T10:00:00Z'>
        <meta property='article:published_time' content='2026-07-21T10:00:00Z'>
        </head><body>Body</body></html>"""
    )
    _, policy, freshness = _resolver()
    with pytest.raises(InformationTemporalMetadataError):
        resolved.to_source_document(
            source_id="source-1",
            provider="fixture",
            retrieved_at=RETRIEVED_AT,
            policy=policy,
            freshness_policy=freshness,
        )


def test_raw_retrieved_resource_rejects_caller_supplied_dates() -> None:
    resource = _retrieve(b"<html><body>Body</body></html>")
    with pytest.raises(InformationTemporalMetadataError):
        resource.to_source_document(
            source_id="source-1",
            provider="fixture",
            retrieved_at=RETRIEVED_AT,
            published_at="2026-07-20T10:00:00Z",
        )


def test_resolution_tampering_is_rejected_by_rederivation() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    forged = replace(
        resolved,
        resolution=replace(
            resolved.resolution,
            published_at="2020-01-01T00:00:00Z",
        ),
    )
    with pytest.raises(InformationTemporalMetadataError):
        forged.validate(policy=policy, freshness_policy=freshness)


def test_candidate_limit_is_enforced_during_retrieval() -> None:
    metadata = "".join(
        f"<meta itemprop='datePublished' content='2026-07-{day:02d}T10:00:00Z'>"
        for day in range(1, 34)
    )
    with pytest.raises(InformationHttpExecutionError):
        _retrieve(f"<html><head>{metadata}</head><body>Body</body></html>".encode())


def test_cross_source_consensus_requires_explicit_subject_digest() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    aggregator = DeterministicInformationTemporalMetadataAggregator(policy, freshness)
    with pytest.raises(InformationTemporalMetadataError):
        aggregator.aggregate(subject_sha256="not-a-digest", observations=(resolved,))


def test_cross_source_consensus_is_consistent_for_matching_observations() -> None:
    first = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>One</body></html>"
    )
    second = _resolved(
        b"<html><head><meta property='article:published_time' content='2026-07-20T10:00:00Z'></head><body>Two</body></html>",
        url="https://example.org/article",
    )
    _, policy, freshness = _resolver()
    consensus = DeterministicInformationTemporalMetadataAggregator(
        policy, freshness
    ).aggregate(
        subject_sha256=sha256_text("same explicit temporal fact"),
        observations=(first, second),
    )
    assert consensus.verdict == "consistent"
    assert consensus.published_at == "2026-07-20T10:00:00Z"


def test_cross_source_consensus_preserves_date_conflict_without_winner() -> None:
    first = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>One</body></html>"
    )
    second = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-21T10:00:00Z'></head><body>Two</body></html>",
        url="https://example.org/article",
    )
    _, policy, freshness = _resolver()
    consensus = DeterministicInformationTemporalMetadataAggregator(
        policy, freshness
    ).aggregate(
        subject_sha256=sha256_text("same explicit temporal fact"),
        observations=(first, second),
    )
    assert consensus.verdict == "conflict"
    assert consensus.published_at is None


def test_cross_source_undated_observations_are_insufficient() -> None:
    first = _resolved(b"<html><body>One</body></html>")
    second = _resolved(
        b"<html><body>Two</body></html>",
        url="https://example.org/article",
    )
    _, policy, freshness = _resolver()
    consensus = DeterministicInformationTemporalMetadataAggregator(
        policy, freshness
    ).aggregate(
        subject_sha256=sha256_text("same explicit temporal fact"),
        observations=(first, second),
    )
    assert consensus.verdict == "insufficient"


def test_cross_source_consensus_tampering_is_rejected() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    second = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Other</body></html>",
        url="https://example.org/article",
    )
    consensus = DeterministicInformationTemporalMetadataAggregator(
        policy, freshness
    ).aggregate(
        subject_sha256=sha256_text("same explicit temporal fact"),
        observations=(resolved, second),
    )
    with pytest.raises(InformationTemporalMetadataError):
        replace(consensus, verdict="conflict").validate(
            observations=(resolved, second),
            policy=policy,
            freshness_policy=freshness,
        )


def test_cross_source_consensus_rejects_single_observation() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    with pytest.raises(InformationTemporalMetadataError):
        DeterministicInformationTemporalMetadataAggregator(
            policy, freshness
        ).aggregate(
            subject_sha256=sha256_text("same explicit temporal fact"),
            observations=(resolved,),
        )


def test_cross_source_consensus_rejects_repeated_source_url() -> None:
    first = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>One</body></html>"
    )
    second = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Two</body></html>"
    )
    _, policy, freshness = _resolver()
    with pytest.raises(InformationTemporalMetadataError):
        DeterministicInformationTemporalMetadataAggregator(
            policy, freshness
        ).aggregate(
            subject_sha256=sha256_text("same explicit temporal fact"),
            observations=(first, second),
        )


def test_duplicate_observation_resolution_is_rejected() -> None:
    resolved = _resolved(
        b"<html><head><meta itemprop='datePublished' content='2026-07-20T10:00:00Z'></head><body>Body</body></html>"
    )
    _, policy, freshness = _resolver()
    with pytest.raises(InformationTemporalMetadataError):
        DeterministicInformationTemporalMetadataAggregator(
            policy, freshness
        ).aggregate(
            subject_sha256=sha256_text("same explicit temporal fact"),
            observations=(resolved, resolved),
        )
