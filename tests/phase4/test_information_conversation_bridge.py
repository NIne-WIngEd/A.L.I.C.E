from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from alice_conversation.contracts import ModelResponse
from alice_conversation.response_validation_policy import (
    parse_conversation_response_validation_policy,
)
from alice_information.contracts import (
    InformationCitation,
    InformationClaim,
    InformationGroundingPacket,
    InformationQuery,
    InformationSourceDocument,
    sha256_text,
)
from alice_information.conversation_bridge import (
    InformationConversationBridgeError,
    project_information_grounding_to_conversation,
    validate_information_conversation_response,
)
from alice_information.conversation_bridge_policy import (
    load_information_conversation_bridge_policy,
)
from alice_information.grounding import (
    InformationSourceQualityAssessment,
    InformationVerifiedGroundingPacket,
)
from alice_information.grounding_policy import InformationGroundingPolicy


NOW = "2026-07-28T12:00:00Z"


def _grounding_policy() -> InformationGroundingPolicy:
    return InformationGroundingPolicy(
        policy_name="alice_information_grounding_policy",
        version="1.0.0",
        phase="4",
        milestone="P4.5a",
        status="deterministic_citation_grounding",
        permission_id="web.search",
        allowed_outcomes=(
            "answerable",
            "conflict",
            "uncertain",
            "insufficient_sources",
        ),
        allowed_knowledge_statuses=(
            "external_claim",
            "verified_fact",
            "uncertain",
            "disputed",
            "historical",
        ),
        max_sources=12,
        max_claims=24,
        max_support_span_characters=2000,
        min_source_characters=20,
        verified_fact_min_distinct_domains=2,
        conflict_min_distinct_domains=2,
        require_https_sources=True,
        require_clear_firewall=True,
        require_freshness_support=True,
        require_exact_support_span=True,
        require_all_packet_sources_cited=True,
        allow_unused_sources=False,
        allow_model_claim_generation=False,
        allow_semantic_entailment_inference=False,
        allow_publisher_reputation_inference=False,
        raw_support_logging_allowed=False,
        source_digest_binding_required=True,
        query_digest_binding_required=True,
        citation_token_prefix="[WEB:",
        citation_token_suffix="]",
    )


def _response_policy():
    return parse_conversation_response_validation_policy(
        {
            "policy_name": "alice_conversation_response_validation_policy",
            "version": "1.0.0",
            "phase": "3",
            "milestone": "P3.6",
            "status": "generated_response_validation",
            "boundaries": {
                "web_access_allowed": False,
                "tool_calling_allowed": False,
                "external_action_allowed": False,
                "memory_write_allowed": False,
                "memory_promotion_allowed": False,
                "highly_sensitive_grounding_allowed": False,
                "chain_of_thought_persistence_allowed": False,
                "automatic_repair_allowed": False,
                "provider_fallback_allowed": False,
            },
            "citations": {
                "require_exact_tokens": True,
                "reject_unknown_tokens": True,
                "require_grounded_personal_claims": True,
                "require_supported_factual_claims": True,
                "minimum_answerable_claims_cited": 1,
                "minimum_conflict_claims_cited": 2,
            },
            "epistemic": {
                "preserve_conflict": True,
                "preserve_uncertainty": True,
                "require_abstention_on_insufficient_evidence": True,
                "require_abstention_on_denied": True,
                "require_abstention_on_not_applicable": True,
                "reject_certainty_language_for_conflict": True,
                "reject_certainty_language_for_uncertainty": True,
            },
            "safety": {
                "reject_action_completion_claims": True,
                "reject_capability_claims": True,
                "reject_invented_personal_facts": True,
                "reject_dependency_language": True,
                "reject_hidden_reasoning_disclosure": True,
                "reject_truncated_responses": True,
            },
            "limits": {"max_response_chars": 20000, "max_issues": 64},
            "failure_codes": {
                "rejected": "response_validation_rejected",
                "internal": "response_validation_internal",
            },
        }
    )


def _source(source_id: str, host: str, text: str) -> InformationSourceDocument:
    return InformationSourceDocument.create(
        source_id=source_id,
        provider="fixture",
        url=f"https://{host}/article",
        title=f"Title {source_id}",
        normalized_text=text,
        retrieved_at=NOW,
        updated_at="2026-07-28T10:00:00Z",
    )


def _quality(source: InformationSourceDocument, freshness: str = "fresh"):
    return InformationSourceQualityAssessment(
        source_id=source.source_id,
        canonical_url=source.canonical_url,
        source_content_sha256=source.content_sha256,
        domain=source.canonical_url.split("/")[2],
        content_characters=len(source.normalized_text),
        has_temporal_metadata=True,
        freshness_verdict=freshness,
        eligible=True,
        policy_version="1.0.0",
        assessment_sha256=sha256_text(f"quality:{source.source_id}:{freshness}"),
    )


def _verified(outcome: str = "answerable"):
    query = InformationQuery.create(
        query_id="query-1",
        text="What is the launch date?",
        created_at=NOW,
    )
    if outcome == "insufficient_sources":
        packet = InformationGroundingPacket(
            packet_id="web-packet-empty",
            request_id="research-1",
            outcome=outcome,
            claims=(),
            sources=(),
            created_at=NOW,
        )
        verified = InformationVerifiedGroundingPacket(
            query_id=query.query_id,
            query_content_sha256=query.content_sha256,
            policy_version="1.0.0",
            packet=packet,
            quality_assessments=(),
            support_spans=(),
            grounding_sha256=sha256_text("grounding:empty"),
        )
        return query, verified

    source_a = _source("source-a", "alpha.example", "The launch date is July 30, 2026.")
    source_b = _source("source-b", "beta.example", "The launch date is July 30, 2026.")
    if outcome == "conflict":
        source_b = _source("source-b", "beta.example", "The launch date is August 2, 2026.")
    sources = (source_a, source_b)
    claims = []
    if outcome == "conflict":
        for index, source in enumerate(sources, start=1):
            text = source.normalized_text
            citation = InformationCitation(
                citation_id=f"web-citation-{index}",
                source_id=source.source_id,
                canonical_url=source.canonical_url,
                source_content_sha256=source.content_sha256,
                token=f"[WEB:web-citation-{index}]",
            )
            claims.append(
                InformationClaim(
                    claim_id=f"claim-{index}",
                    text=text,
                    content_sha256=sha256_text(text),
                    knowledge_status="disputed",
                    confidence=0.5,
                    citations=(citation,),
                )
            )
    else:
        text = source_a.normalized_text
        status = "uncertain" if outcome == "uncertain" else "verified_fact"
        citations = tuple(
            InformationCitation(
                citation_id=f"web-citation-{index}",
                source_id=source.source_id,
                canonical_url=source.canonical_url,
                source_content_sha256=source.content_sha256,
                token=f"[WEB:web-citation-{index}]",
            )
            for index, source in enumerate(sources, start=1)
        )
        claims.append(
            InformationClaim(
                claim_id="claim-1",
                text=text,
                content_sha256=sha256_text(text),
                knowledge_status=status,
                confidence=0.98 if outcome == "answerable" else 0.55,
                citations=citations,
            )
        )
    packet = InformationGroundingPacket(
        packet_id=f"web-packet-{outcome}",
        request_id="research-1",
        outcome=outcome,
        claims=tuple(claims),
        sources=sources,
        created_at=NOW,
    )
    verified = InformationVerifiedGroundingPacket(
        query_id=query.query_id,
        query_content_sha256=query.content_sha256,
        policy_version="1.0.0",
        packet=packet,
        quality_assessments=tuple(_quality(source) for source in sources),
        support_spans=(),
        grounding_sha256=sha256_text(f"grounding:{outcome}"),
    )
    return query, verified


def _project(outcome: str = "answerable"):
    query, verified = _verified(outcome)
    bridge_policy = load_information_conversation_bridge_policy()
    grounding_policy = _grounding_policy()
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        projection = project_information_grounding_to_conversation(
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=grounding_policy,
            bridge_policy=bridge_policy,
        )
    return query, verified, projection


def test_answerable_projection_preserves_web_tokens_and_public_classification() -> None:
    _, verified, projection = _project("answerable")
    packet = projection.conversation_packet
    assert packet.outcome == "answerable"
    assert packet.max_classification == "PUBLIC"
    assert packet.created_at == verified.packet.created_at
    assert len(packet.claims) == 1
    assert packet.claims[0].knowledge_status == "verified_fact"
    assert {citation.token for citation in packet.claims[0].citations} == {
        "[WEB:web-citation-1]",
        "[WEB:web-citation-2]",
    }
    assert all(
        citation.source_kind == "web_source"
        for citation in packet.claims[0].citations
    )
    assert all(
        "#alice-source-sha256=" in citation.source_ref
        for citation in packet.claims[0].citations
    )


@pytest.mark.parametrize(
    ("information_outcome", "conversation_outcome", "status"),
    [
        ("uncertain", "uncertain", "uncertain"),
        ("conflict", "conflict", "disputed"),
    ],
)
def test_projection_preserves_epistemic_outcomes(
    information_outcome: str,
    conversation_outcome: str,
    status: str,
) -> None:
    _, _, projection = _project(information_outcome)
    assert projection.conversation_packet.outcome == conversation_outcome
    assert all(
        claim.knowledge_status == status
        for claim in projection.conversation_packet.claims
    )


def test_insufficient_sources_maps_to_phase3_insufficient_evidence() -> None:
    _, verified, projection = _project("insufficient_sources")
    assert projection.conversation_packet.outcome == "insufficient_evidence"
    assert projection.conversation_packet.claims == ()
    assert projection.receipt.source_bindings == ()
    assert projection.receipt.citation_bindings == ()
    assert projection.state_reference.content_sha256 == verified.grounding_sha256


def test_receipt_preserves_query_source_freshness_and_citation_bindings() -> None:
    query, verified, projection = _project("answerable")
    receipt = projection.receipt
    assert receipt.query_id == query.query_id
    assert receipt.query_content_sha256 == query.content_sha256
    assert receipt.information_grounding_sha256 == verified.grounding_sha256
    assert {binding.freshness_verdict for binding in receipt.source_bindings} == {"fresh"}
    assert {binding.source_id for binding in receipt.source_bindings} == {
        "source-a",
        "source-b",
    }
    assert {binding.citation_id for binding in receipt.citation_bindings} == {
        "web-citation-1",
        "web-citation-2",
    }


def test_state_reference_is_single_metadata_only_grounding_packet_reference() -> None:
    _, verified, projection = _project("answerable")
    reference = projection.state_reference
    assert reference.source_kind == "grounding_packet"
    assert reference.source_ref == verified.packet.packet_id
    assert reference.content_sha256 == verified.grounding_sha256
    assert reference.citation_token is None
    assert "The launch date" not in repr(reference)


def test_metadata_records_do_not_include_source_bodies_or_support_spans() -> None:
    _, _, projection = _project("answerable")
    record = projection.receipt.metadata_record()
    text = repr(record)
    assert "The launch date is" not in text
    assert "normalized_text" not in text
    assert "start_character" not in text
    assert "end_character" not in text


def test_projection_is_deterministic() -> None:
    query, verified = _verified("answerable")
    bridge_policy = load_information_conversation_bridge_policy()
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        first = project_information_grounding_to_conversation(
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=bridge_policy,
        )
        second = project_information_grounding_to_conversation(
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=bridge_policy,
        )
    assert first == second


@pytest.mark.parametrize("field", ["conversation_packet", "receipt", "state_reference"])
def test_projection_validation_rejects_tampering(field: str) -> None:
    query, verified, projection = _project("answerable")
    if field == "conversation_packet":
        changed = replace(
            projection,
            conversation_packet=replace(
                projection.conversation_packet,
                packet_id="forged-packet",
            ),
        )
    elif field == "receipt":
        changed = replace(
            projection,
            receipt=replace(
                projection.receipt,
                information_request_id="forged-request",
            ),
        )
    else:
        changed = replace(
            projection,
            state_reference=replace(
                projection.state_reference,
                source_ref="forged-source",
            ),
        )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        with pytest.raises(InformationConversationBridgeError):
            changed.validate(
                verified_grounding=verified,
                query=query,
                qualified_sources=(),
                information_policy=object(),
                firewall_policy=object(),
                freshness_policy=object(),
                grounding_policy=_grounding_policy(),
                bridge_policy=load_information_conversation_bridge_policy(),
            )


def test_response_wrapper_delegates_to_p36_and_accepts_exact_web_citations() -> None:
    query, verified, projection = _project("answerable")
    tokens = " ".join(
        citation.token
        for citation in projection.conversation_packet.claims[0].citations
    )
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=f"The launch date is July 30, 2026. {tokens}",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
    assert result.report.outcome == "accepted"
    assert result.report.issues == ()
    assert result.receipt.p3_validation_outcome == "accepted"
    assert result.receipt.projection_sha256 == projection.receipt.projection_sha256


def test_response_wrapper_preserves_p36_unknown_token_rejection() -> None:
    query, verified, projection = _project("answerable")
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content="The launch date is July 30, 2026. [WEB:forged]",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
    assert result.report.outcome == "rejected"
    assert "unknown_citation_token" in {issue.code for issue in result.report.issues}


def test_response_wrapper_preserves_conflict_requirement() -> None:
    query, verified, projection = _project("conflict")
    tokens = " ".join(
        citation.token
        for claim in projection.conversation_packet.claims
        for citation in claim.citations
    )
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=f"The sources conflict on the launch date. {tokens}",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
    assert result.report.outcome == "accepted"


def test_response_wrapper_abstains_for_insufficient_sources() -> None:
    query, verified, projection = _project("insufficient_sources")
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content="I cannot determine this because there is insufficient evidence.",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
    assert result.report.outcome == "abstained"


def test_response_validation_binding_rejects_forged_receipt() -> None:
    query, verified, projection = _project("answerable")
    tokens = " ".join(
        citation.token
        for citation in projection.conversation_packet.claims[0].citations
    )
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=f"The launch date is July 30, 2026. {tokens}",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
        forged = replace(
            result,
            receipt=replace(result.receipt, p3_validation_outcome="rejected"),
        )
        with pytest.raises(InformationConversationBridgeError):
            forged.validate(
                response=response,
                projection=projection,
                verified_grounding=verified,
                query=query,
                qualified_sources=(),
                information_policy=object(),
                firewall_policy=object(),
                freshness_policy=object(),
                grounding_policy=_grounding_policy(),
                bridge_policy=load_information_conversation_bridge_policy(),
                response_validation_policy=_response_policy(),
            )


def test_response_wrapper_rejects_partial_verified_source_set_before_p36() -> None:
    query, verified, projection = _project("answerable")
    token = projection.conversation_packet.claims[0].citations[0].token
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=f"The launch date is July 30, 2026. {token}",
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        with pytest.raises(InformationConversationBridgeError) as exc:
            validate_information_conversation_response(
                response=response,
                projection=projection,
                verified_grounding=verified,
                query=query,
                qualified_sources=(),
                information_policy=object(),
                firewall_policy=object(),
                freshness_policy=object(),
                grounding_policy=_grounding_policy(),
                bridge_policy=load_information_conversation_bridge_policy(),
                response_validation_policy=_response_policy(),
            )
    assert exc.value.code == "conversation_web_citation_incomplete"

def test_response_wrapper_rejects_split_verified_source_set_across_sentences() -> None:
    query, verified, projection = _project("answerable")
    first, second = (
        citation.token
        for citation in projection.conversation_packet.claims[0].citations
    )
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=(
            f"The launch date is July 30, 2026. {first}\n"
            f"The launch date is July 30, 2026. {second}"
        ),
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        with pytest.raises(InformationConversationBridgeError) as exc:
            validate_information_conversation_response(
                response=response,
                projection=projection,
                verified_grounding=verified,
                query=query,
                qualified_sources=(),
                information_policy=object(),
                firewall_policy=object(),
                freshness_policy=object(),
                grounding_policy=_grounding_policy(),
                bridge_policy=load_information_conversation_bridge_policy(),
                response_validation_policy=_response_policy(),
            )
    assert exc.value.code == "conversation_web_citation_incomplete"


def test_response_wrapper_accepts_multiline_trailing_complete_source_set() -> None:
    query, verified, projection = _project("answerable")
    first, second = (
        citation.token
        for citation in projection.conversation_packet.claims[0].citations
    )
    response = ModelResponse(
        request_id="model-request-1",
        provider="fixture",
        model="fixture-model",
        content=(
            "The launch date is July 30, 2026.\n"
            f"{first}\n"
            f"{second}"
        ),
        finish_reason="stop",
        created_at=NOW,
    )
    with patch.object(InformationVerifiedGroundingPacket, "validate", return_value=None):
        result = validate_information_conversation_response(
            response=response,
            projection=projection,
            verified_grounding=verified,
            query=query,
            qualified_sources=(),
            information_policy=object(),
            firewall_policy=object(),
            freshness_policy=object(),
            grounding_policy=_grounding_policy(),
            bridge_policy=load_information_conversation_bridge_policy(),
            response_validation_policy=_response_policy(),
        )
    assert result.report.outcome == "accepted"
