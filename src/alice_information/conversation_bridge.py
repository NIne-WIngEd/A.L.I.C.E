"""Deterministic P4.5b projection of verified web grounding into Phase 3."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ModelResponse,
    sha256_text as conversation_sha256_text,
)
from alice_conversation.grounding_bridge import conversation_grounding_packet_sha256
from alice_conversation.response_validation import (
    ConversationResponseValidationReport,
    conversation_response_validation_report_sha256,
    validate_conversation_response,
)
from alice_conversation.response_validation_policy import (
    ConversationResponseValidationPolicy,
)
from alice_conversation.state_service import ConversationStateReference

from .contracts import InformationContractError, InformationQuery
from .freshness import InformationTemporallyQualifiedSource
from .freshness_policy import InformationFreshnessPolicy
from .grounding import (
    InformationSourceQualityAssessment,
    InformationVerifiedGroundingPacket,
)
from .grounding_policy import InformationGroundingPolicy
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy
from .conversation_bridge_policy import InformationConversationBridgePolicy

_BRIDGE_ERROR_MESSAGES = {
    "conversation_bridge_policy_invalid": "Conversation-bridge policy validation failed.",
    "conversation_projection_invalid": "Web grounding could not be projected into Phase 3.",
    "conversation_projection_binding_invalid": "Conversation grounding projection binding validation failed.",
    "conversation_response_validation_invalid": "Web-grounded response validation binding failed.",
    "conversation_web_citation_incomplete": "A web-grounded claim did not preserve its exact citation set.",
}

_RESPONSE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_RESPONSE_LEADING_CITATIONS = re.compile(
    r"^(?P<tokens>(?:\[[A-Za-z][A-Za-z0-9_.-]{0,63}:[^\]\r\n]{1,256}\]\s*)+)"
    r"(?P<rest>.*)$"
)


class InformationConversationBridgeError(InformationContractError):
    """Sanitized P4.5b conversation-bridge failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(
            _BRIDGE_ERROR_MESSAGES.get(code, "Conversation grounding bridge failed.")
        )


def conversation_bridge_failure(code: str) -> InformationConversationBridgeError:
    return InformationConversationBridgeError(code)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise conversation_bridge_failure("conversation_projection_binding_invalid")
    return value.strip()


def _require_digest(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if len(text) != 64:
        raise conversation_bridge_failure("conversation_projection_binding_invalid")
    try:
        int(text, 16)
    except ValueError as exc:
        raise conversation_bridge_failure(
            "conversation_projection_binding_invalid"
        ) from exc
    return text.lower()


def _source_ref(canonical_url: str, source_content_sha256: str) -> str:
    return f"{canonical_url}#alice-source-sha256={source_content_sha256}"


def _conversation_claim_id(
    *,
    information_grounding_sha256: str,
    information_claim_id: str,
) -> str:
    seed = f"{information_grounding_sha256}\n{information_claim_id}"
    return f"web-claim-{conversation_sha256_text(seed)[:20]}"


@dataclass(frozen=True)
class InformationConversationSourceBinding:
    """Metadata-only binding for one exact public source version."""

    source_id: str
    canonical_url: str
    source_content_sha256: str
    freshness_verdict: str
    quality_assessment_sha256: str

    def validate(
        self,
        *,
        assessment: InformationSourceQualityAssessment,
    ) -> None:
        expected = InformationConversationSourceBinding(
            source_id=assessment.source_id,
            canonical_url=assessment.canonical_url,
            source_content_sha256=assessment.source_content_sha256,
            freshness_verdict=assessment.freshness_verdict,
            quality_assessment_sha256=assessment.assessment_sha256,
        )
        if self != expected:
            raise conversation_bridge_failure(
                "conversation_projection_binding_invalid"
            )

    def metadata_record(self) -> dict[str, str]:
        _require_text(self.source_id, field="source_id")
        _require_text(self.canonical_url, field="canonical_url")
        _require_digest(
            self.source_content_sha256,
            field="source_content_sha256",
        )
        _require_text(self.freshness_verdict, field="freshness_verdict")
        _require_digest(
            self.quality_assessment_sha256,
            field="quality_assessment_sha256",
        )
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "source_content_sha256": self.source_content_sha256,
            "freshness_verdict": self.freshness_verdict,
            "quality_assessment_sha256": self.quality_assessment_sha256,
        }


@dataclass(frozen=True)
class InformationConversationCitationBinding:
    """Exact P4 citation to P3 citation projection metadata."""

    information_claim_id: str
    conversation_claim_id: str
    citation_id: str
    token: str
    source_id: str
    canonical_url: str
    source_content_sha256: str

    def metadata_record(self) -> dict[str, str]:
        for field_name in (
            "information_claim_id",
            "conversation_claim_id",
            "citation_id",
            "token",
            "source_id",
            "canonical_url",
        ):
            _require_text(getattr(self, field_name), field=field_name)
        _require_digest(
            self.source_content_sha256,
            field="source_content_sha256",
        )
        return {
            "information_claim_id": self.information_claim_id,
            "conversation_claim_id": self.conversation_claim_id,
            "citation_id": self.citation_id,
            "token": self.token,
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "source_content_sha256": self.source_content_sha256,
        }


@dataclass(frozen=True)
class InformationConversationGroundingReceipt:
    """Digest-bound metadata receipt for one exact P4-to-P3 projection."""

    policy_version: str
    information_grounding_sha256: str
    information_packet_id: str
    information_request_id: str
    query_id: str
    query_content_sha256: str
    conversation_packet_id: str
    conversation_packet_sha256: str
    information_outcome: str
    conversation_outcome: str
    source_bindings: tuple[InformationConversationSourceBinding, ...]
    citation_bindings: tuple[InformationConversationCitationBinding, ...]
    projection_sha256: str

    def metadata_record(self) -> dict[str, object]:
        for field_name in (
            "policy_version",
            "information_packet_id",
            "information_request_id",
            "query_id",
            "conversation_packet_id",
            "information_outcome",
            "conversation_outcome",
        ):
            _require_text(getattr(self, field_name), field=field_name)
        for field_name in (
            "information_grounding_sha256",
            "query_content_sha256",
            "conversation_packet_sha256",
            "projection_sha256",
        ):
            _require_digest(getattr(self, field_name), field=field_name)
        return {
            "policy_version": self.policy_version,
            "information_grounding_sha256": self.information_grounding_sha256,
            "information_packet_id": self.information_packet_id,
            "information_request_id": self.information_request_id,
            "query_id": self.query_id,
            "query_content_sha256": self.query_content_sha256,
            "conversation_packet_id": self.conversation_packet_id,
            "conversation_packet_sha256": self.conversation_packet_sha256,
            "information_outcome": self.information_outcome,
            "conversation_outcome": self.conversation_outcome,
            "source_bindings": [
                binding.metadata_record() for binding in self.source_bindings
            ],
            "citation_bindings": [
                binding.metadata_record() for binding in self.citation_bindings
            ],
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True)
class InformationConversationGroundingProjection:
    """Verified Phase 4 web grounding projected into the Phase 3 contract."""

    conversation_packet: ConversationGroundingPacket
    receipt: InformationConversationGroundingReceipt
    state_reference: ConversationStateReference

    def validate(
        self,
        *,
        verified_grounding: InformationVerifiedGroundingPacket,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
        grounding_policy: InformationGroundingPolicy,
        bridge_policy: InformationConversationBridgePolicy,
    ) -> None:
        expected = _derive_conversation_projection(
            verified_grounding=verified_grounding,
            query=query,
            qualified_sources=qualified_sources,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
            grounding_policy=grounding_policy,
            bridge_policy=bridge_policy,
        )
        if self != expected:
            raise conversation_bridge_failure(
                "conversation_projection_binding_invalid"
            )


def _projection_digest_payload(
    *,
    bridge_policy: InformationConversationBridgePolicy,
    verified_grounding: InformationVerifiedGroundingPacket,
    query: InformationQuery,
    conversation_packet: ConversationGroundingPacket,
    conversation_packet_sha256: str,
    source_bindings: tuple[InformationConversationSourceBinding, ...],
    citation_bindings: tuple[InformationConversationCitationBinding, ...],
) -> dict[str, object]:
    return {
        "policy_version": bridge_policy.version,
        "information_grounding_sha256": verified_grounding.grounding_sha256,
        "information_packet_id": verified_grounding.packet.packet_id,
        "information_request_id": verified_grounding.packet.request_id,
        "query_id": query.query_id,
        "query_content_sha256": query.content_sha256,
        "conversation_packet_id": conversation_packet.packet_id,
        "conversation_packet_sha256": conversation_packet_sha256,
        "information_outcome": verified_grounding.packet.outcome,
        "conversation_outcome": conversation_packet.outcome,
        "source_bindings": [binding.metadata_record() for binding in source_bindings],
        "citation_bindings": [
            binding.metadata_record() for binding in citation_bindings
        ],
    }


def _derive_conversation_projection(
    *,
    verified_grounding: InformationVerifiedGroundingPacket,
    query: InformationQuery,
    qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
    grounding_policy: InformationGroundingPolicy,
    bridge_policy: InformationConversationBridgePolicy,
) -> InformationConversationGroundingProjection:
    bridge_policy.validate(grounding_policy=grounding_policy)
    verified_grounding.validate(
        query=query,
        qualified_sources=qualified_sources,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
        grounding_policy=grounding_policy,
    )
    conversation_outcome = bridge_policy.map_outcome(
        verified_grounding.packet.outcome
    )
    quality_by_source = {
        assessment.source_id: assessment
        for assessment in verified_grounding.quality_assessments
    }
    if len(quality_by_source) != len(verified_grounding.quality_assessments):
        raise conversation_bridge_failure("conversation_projection_binding_invalid")
    source_bindings = tuple(
        InformationConversationSourceBinding(
            source_id=assessment.source_id,
            canonical_url=assessment.canonical_url,
            source_content_sha256=assessment.source_content_sha256,
            freshness_verdict=assessment.freshness_verdict,
            quality_assessment_sha256=assessment.assessment_sha256,
        )
        for assessment in sorted(
            verified_grounding.quality_assessments,
            key=lambda item: item.source_id,
        )
    )
    conversation_claims: list[ConversationGroundingClaim] = []
    citation_bindings: list[InformationConversationCitationBinding] = []
    seen_citation_ids: set[str] = set()
    seen_tokens: dict[str, tuple[str, str, str]] = {}
    for claim in verified_grounding.packet.claims:
        conversation_claim_id = _conversation_claim_id(
            information_grounding_sha256=verified_grounding.grounding_sha256,
            information_claim_id=claim.claim_id,
        )
        citations: list[ConversationCitation] = []
        for citation in claim.citations:
            if citation.citation_id in seen_citation_ids:
                raise conversation_bridge_failure(
                    "conversation_projection_binding_invalid"
                )
            seen_citation_ids.add(citation.citation_id)
            logical = (
                citation.source_id,
                citation.canonical_url,
                citation.source_content_sha256,
            )
            previous = seen_tokens.get(citation.token)
            if previous is not None and previous != logical:
                raise conversation_bridge_failure(
                    "conversation_projection_binding_invalid"
                )
            seen_tokens[citation.token] = logical
            assessment = quality_by_source.get(citation.source_id)
            if assessment is None:
                raise conversation_bridge_failure(
                    "conversation_projection_binding_invalid"
                )
            if (
                assessment.canonical_url != citation.canonical_url
                or assessment.source_content_sha256
                != citation.source_content_sha256
            ):
                raise conversation_bridge_failure(
                    "conversation_projection_binding_invalid"
                )
            projected = ConversationCitation(
                citation_id=citation.citation_id,
                source_kind=bridge_policy.source_kind,
                source_ref=_source_ref(
                    citation.canonical_url,
                    citation.source_content_sha256,
                ),
                token=citation.token,
                data_classification="PUBLIC",
            )
            projected.validate()
            citations.append(projected)
            citation_bindings.append(
                InformationConversationCitationBinding(
                    information_claim_id=claim.claim_id,
                    conversation_claim_id=conversation_claim_id,
                    citation_id=citation.citation_id,
                    token=citation.token,
                    source_id=citation.source_id,
                    canonical_url=citation.canonical_url,
                    source_content_sha256=citation.source_content_sha256,
                )
            )
        projected_claim = ConversationGroundingClaim(
            claim_id=conversation_claim_id,
            text=claim.text,
            content_sha256=claim.content_sha256,
            knowledge_status=claim.knowledge_status,
            confidence=float(claim.confidence),
            data_classification="PUBLIC",
            citations=tuple(citations),
        )
        projected_claim.validate()
        conversation_claims.append(projected_claim)
    packet_seed = {
        "policy_version": bridge_policy.version,
        "information_grounding_sha256": verified_grounding.grounding_sha256,
        "query_content_sha256": query.content_sha256,
    }
    conversation_packet = ConversationGroundingPacket(
        packet_id=f"web-conversation-{_digest(packet_seed)[:24]}",
        outcome=conversation_outcome,
        claims=tuple(conversation_claims),
        created_at=verified_grounding.packet.created_at,
        max_classification="PUBLIC",
    )
    conversation_packet.validate()
    conversation_packet_digest = conversation_grounding_packet_sha256(
        conversation_packet
    )
    citation_tuple = tuple(
        sorted(
            citation_bindings,
            key=lambda item: (
                item.conversation_claim_id,
                item.source_id,
                item.citation_id,
            ),
        )
    )
    payload = _projection_digest_payload(
        bridge_policy=bridge_policy,
        verified_grounding=verified_grounding,
        query=query,
        conversation_packet=conversation_packet,
        conversation_packet_sha256=conversation_packet_digest,
        source_bindings=source_bindings,
        citation_bindings=citation_tuple,
    )
    projection_sha256 = _digest(payload)
    receipt = InformationConversationGroundingReceipt(
        policy_version=bridge_policy.version,
        information_grounding_sha256=verified_grounding.grounding_sha256,
        information_packet_id=verified_grounding.packet.packet_id,
        information_request_id=verified_grounding.packet.request_id,
        query_id=query.query_id,
        query_content_sha256=query.content_sha256,
        conversation_packet_id=conversation_packet.packet_id,
        conversation_packet_sha256=conversation_packet_digest,
        information_outcome=verified_grounding.packet.outcome,
        conversation_outcome=conversation_packet.outcome,
        source_bindings=source_bindings,
        citation_bindings=citation_tuple,
        projection_sha256=projection_sha256,
    )
    state_reference = ConversationStateReference(
        reference_id=f"web-grounding-{verified_grounding.grounding_sha256[:24]}",
        source_kind=bridge_policy.state_reference_kind,
        source_ref=verified_grounding.packet.packet_id,
        citation_token=None,
        content_sha256=verified_grounding.grounding_sha256,
        data_classification="PUBLIC",
        created_at=verified_grounding.packet.created_at,
    )
    state_reference.validate()
    return InformationConversationGroundingProjection(
        conversation_packet=conversation_packet,
        receipt=receipt,
        state_reference=state_reference,
    )


def project_information_grounding_to_conversation(
    *,
    verified_grounding: InformationVerifiedGroundingPacket,
    query: InformationQuery,
    qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
    grounding_policy: InformationGroundingPolicy,
    bridge_policy: InformationConversationBridgePolicy,
) -> InformationConversationGroundingProjection:
    """Project one revalidated P4.5a packet into exact Phase 3 grounding."""

    return _derive_conversation_projection(
        verified_grounding=verified_grounding,
        query=query,
        qualified_sources=qualified_sources,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
        grounding_policy=grounding_policy,
        bridge_policy=bridge_policy,
    )


@dataclass(frozen=True)
class InformationConversationResponseValidationReceipt:
    """P4.5b receipt binding P3.6 validation to one exact projection."""

    bridge_policy_version: str
    projection_sha256: str
    information_grounding_sha256: str
    conversation_packet_sha256: str
    response_sha256: str
    p3_validation_report_sha256: str
    p3_validation_outcome: str
    validation_sha256: str

    def metadata_record(self) -> dict[str, str]:
        for field_name in (
            "bridge_policy_version",
            "p3_validation_outcome",
        ):
            _require_text(getattr(self, field_name), field=field_name)
        for field_name in (
            "projection_sha256",
            "information_grounding_sha256",
            "conversation_packet_sha256",
            "response_sha256",
            "p3_validation_report_sha256",
            "validation_sha256",
        ):
            _require_digest(getattr(self, field_name), field=field_name)
        return {
            "bridge_policy_version": self.bridge_policy_version,
            "projection_sha256": self.projection_sha256,
            "information_grounding_sha256": self.information_grounding_sha256,
            "conversation_packet_sha256": self.conversation_packet_sha256,
            "response_sha256": self.response_sha256,
            "p3_validation_report_sha256": self.p3_validation_report_sha256,
            "p3_validation_outcome": self.p3_validation_outcome,
            "validation_sha256": self.validation_sha256,
        }


@dataclass(frozen=True)
class InformationConversationResponseValidation:
    """P3.6 report plus exact P4.5b projection-binding receipt."""

    report: ConversationResponseValidationReport
    receipt: InformationConversationResponseValidationReceipt

    def validate(
        self,
        *,
        response: ModelResponse,
        projection: InformationConversationGroundingProjection,
        verified_grounding: InformationVerifiedGroundingPacket,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
        grounding_policy: InformationGroundingPolicy,
        bridge_policy: InformationConversationBridgePolicy,
        response_validation_policy: ConversationResponseValidationPolicy,
    ) -> None:
        expected = _derive_response_validation(
            response=response,
            projection=projection,
            verified_grounding=verified_grounding,
            query=query,
            qualified_sources=qualified_sources,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
            grounding_policy=grounding_policy,
            bridge_policy=bridge_policy,
            response_validation_policy=response_validation_policy,
        )
        if self != expected:
            raise conversation_bridge_failure(
                "conversation_response_validation_invalid"
            )


def _response_sentences_with_trailing_citations(text: str) -> tuple[str, ...]:
    """Mirror the P3.6 sentence grouping used for trailing citation tokens."""

    raw = [
        item.strip()
        for item in _RESPONSE_SENTENCE_SPLIT.split(text)
        if item.strip()
    ]
    grouped: list[str] = []
    for item in raw:
        match = _RESPONSE_LEADING_CITATIONS.match(item)
        if match and grouped:
            grouped[-1] = f"{grouped[-1]} {match.group('tokens').strip()}"
            rest = match.group("rest").strip()
            if rest:
                grouped.append(rest)
            continue
        grouped.append(item)
    return tuple(grouped)


def _require_complete_web_citation_sets(
    *,
    response: ModelResponse,
    grounding: ConversationGroundingPacket,
) -> None:
    """Reject a partial source set on any sentence that cites a web claim."""

    sentences = _response_sentences_with_trailing_citations(response.content)
    for claim in grounding.claims:
        tokens = tuple(citation.token for citation in claim.citations)
        for sentence in sentences:
            present = tuple(token for token in tokens if token in sentence)
            if present and len(present) != len(tokens):
                raise conversation_bridge_failure(
                    "conversation_web_citation_incomplete"
                )


def _derive_response_validation(
    *,
    response: ModelResponse,
    projection: InformationConversationGroundingProjection,
    verified_grounding: InformationVerifiedGroundingPacket,
    query: InformationQuery,
    qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
    grounding_policy: InformationGroundingPolicy,
    bridge_policy: InformationConversationBridgePolicy,
    response_validation_policy: ConversationResponseValidationPolicy,
) -> InformationConversationResponseValidation:
    bridge_policy.validate(
        grounding_policy=grounding_policy,
        response_validation_policy=response_validation_policy,
    )
    projection.validate(
        verified_grounding=verified_grounding,
        query=query,
        qualified_sources=qualified_sources,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
        grounding_policy=grounding_policy,
        bridge_policy=bridge_policy,
    )
    _require_complete_web_citation_sets(
        response=response,
        grounding=projection.conversation_packet,
    )
    report = validate_conversation_response(
        response=response,
        grounding=projection.conversation_packet,
        policy=response_validation_policy,
    )
    report_digest = conversation_response_validation_report_sha256(report)
    payload = {
        "bridge_policy_version": bridge_policy.version,
        "projection_sha256": projection.receipt.projection_sha256,
        "information_grounding_sha256": verified_grounding.grounding_sha256,
        "conversation_packet_sha256": projection.receipt.conversation_packet_sha256,
        "response_sha256": report.response_sha256,
        "p3_validation_report_sha256": report_digest,
        "p3_validation_outcome": report.outcome,
    }
    receipt = InformationConversationResponseValidationReceipt(
        bridge_policy_version=bridge_policy.version,
        projection_sha256=projection.receipt.projection_sha256,
        information_grounding_sha256=verified_grounding.grounding_sha256,
        conversation_packet_sha256=projection.receipt.conversation_packet_sha256,
        response_sha256=report.response_sha256,
        p3_validation_report_sha256=report_digest,
        p3_validation_outcome=report.outcome,
        validation_sha256=_digest(payload),
    )
    return InformationConversationResponseValidation(report=report, receipt=receipt)


def validate_information_conversation_response(
    *,
    response: ModelResponse,
    projection: InformationConversationGroundingProjection,
    verified_grounding: InformationVerifiedGroundingPacket,
    query: InformationQuery,
    qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
    grounding_policy: InformationGroundingPolicy,
    bridge_policy: InformationConversationBridgePolicy,
    response_validation_policy: ConversationResponseValidationPolicy,
) -> InformationConversationResponseValidation:
    """Revalidate P4 projection and delegate final response checks to P3.6."""

    return _derive_response_validation(
        response=response,
        projection=projection,
        verified_grounding=verified_grounding,
        query=query,
        qualified_sources=qualified_sources,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
        grounding_policy=grounding_policy,
        bridge_policy=bridge_policy,
        response_validation_policy=response_validation_policy,
    )
