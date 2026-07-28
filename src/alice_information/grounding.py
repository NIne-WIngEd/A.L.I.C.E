"""Deterministic extractive citation grounding for Phase 4 P4.5a."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

from .contracts import (
    InformationCitation,
    InformationClaim,
    InformationContractError,
    InformationGroundingPacket,
    InformationQuery,
    InformationSourceDocument,
    sha256_text,
)
from .freshness import InformationTemporallyQualifiedSource
from .freshness_policy import InformationFreshnessPolicy
from .grounding_policy import InformationGroundingPolicy
from .injection_policy import InformationInjectionFirewallPolicy
from .policy import InformationPolicy

_GROUNDING_ERROR_MESSAGES = {
    "grounding_policy_invalid": "Citation-grounding policy validation failed.",
    "grounding_source_invalid": "A source is not eligible for citation grounding.",
    "grounding_support_invalid": "A claim support span is invalid.",
    "grounding_claim_invalid": "A grounded claim is invalid.",
    "grounding_diversity_insufficient": "Source diversity is insufficient for this claim.",
    "grounding_binding_invalid": "Citation-grounding binding validation failed.",
}


class InformationGroundingError(InformationContractError):
    """Sanitized P4.5a grounding failure."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(_GROUNDING_ERROR_MESSAGES.get(code, "Citation grounding failed."))


def grounding_failure(code: str) -> InformationGroundingError:
    return InformationGroundingError(code)


def _require_text(value: object, *, maximum: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise grounding_failure("grounding_binding_invalid")
    normalized = value.strip()
    if maximum is not None and len(normalized) > maximum:
        raise grounding_failure("grounding_binding_invalid")
    return normalized


def _require_digest(value: object) -> str:
    text = _require_text(value)
    if len(text) != 64:
        raise grounding_failure("grounding_binding_invalid")
    try:
        int(text, 16)
    except ValueError as exc:
        raise grounding_failure("grounding_binding_invalid") from exc
    return text.lower()


def _require_timestamp(value: object) -> str:
    text = _require_text(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise grounding_failure("grounding_binding_invalid") from exc
    if parsed.tzinfo is None:
        raise grounding_failure("grounding_binding_invalid")
    return text


def _canonical_domain(source: InformationSourceDocument) -> str:
    host = urlsplit(source.canonical_url).hostname
    if host is None:
        raise grounding_failure("grounding_source_invalid")
    return host.lower()


def _quality_digest(
    *,
    source: InformationSourceDocument,
    domain: str,
    content_characters: int,
    has_temporal_metadata: bool,
    freshness_verdict: str,
    eligible: bool,
    policy_version: str,
) -> str:
    return sha256_text(
        "\n".join(
            (
                source.source_id,
                source.canonical_url,
                source.content_sha256,
                domain,
                str(content_characters),
                "temporal" if has_temporal_metadata else "undated",
                freshness_verdict,
                "eligible" if eligible else "ineligible",
                policy_version,
            )
        )
    )


@dataclass(frozen=True)
class InformationSourceQualityAssessment:
    """Structural source-quality metadata without reputation inference."""

    source_id: str
    canonical_url: str
    source_content_sha256: str
    domain: str
    content_characters: int
    has_temporal_metadata: bool
    freshness_verdict: str
    eligible: bool
    policy_version: str
    assessment_sha256: str

    def validate(
        self,
        *,
        qualified_source: InformationTemporallyQualifiedSource,
        query: InformationQuery,
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
        grounding_policy: InformationGroundingPolicy,
    ) -> None:
        expected = _derive_source_quality(
            qualified_source=qualified_source,
            query=query,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
            grounding_policy=grounding_policy,
        )
        if self != expected:
            raise grounding_failure("grounding_binding_invalid")

    def metadata_record(self) -> dict[str, object]:
        """Return log-safe quality metadata without raw source content."""

        _require_text(self.source_id)
        _require_digest(self.source_content_sha256)
        _require_digest(self.assessment_sha256)
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "content_characters": self.content_characters,
            "has_temporal_metadata": self.has_temporal_metadata,
            "freshness_verdict": self.freshness_verdict,
            "eligible": self.eligible,
            "policy_version": self.policy_version,
            "assessment_sha256": self.assessment_sha256,
        }


def _derive_source_quality(
    *,
    qualified_source: InformationTemporallyQualifiedSource,
    query: InformationQuery,
    information_policy: InformationPolicy,
    firewall_policy: InformationInjectionFirewallPolicy,
    freshness_policy: InformationFreshnessPolicy,
    grounding_policy: InformationGroundingPolicy,
) -> InformationSourceQualityAssessment:
    grounding_policy.validate(
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
    )
    qualified_source.validate(
        query=query,
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
    )
    source = qualified_source.inspected_source.source
    source.validate()
    domain = _canonical_domain(source)
    content_characters = len(source.normalized_text)
    has_temporal_metadata = bool(source.updated_at or source.published_at)
    eligible = (
        source.canonical_url.startswith("https://")
        and content_characters >= grounding_policy.min_source_characters
        and qualified_source.inspected_source.inspection.verdict == "clear"
        and qualified_source.assessment.supports_claim is True
    )
    digest = _quality_digest(
        source=source,
        domain=domain,
        content_characters=content_characters,
        has_temporal_metadata=has_temporal_metadata,
        freshness_verdict=qualified_source.assessment.verdict,
        eligible=eligible,
        policy_version=grounding_policy.version,
    )
    return InformationSourceQualityAssessment(
        source_id=source.source_id,
        canonical_url=source.canonical_url,
        source_content_sha256=source.content_sha256,
        domain=domain,
        content_characters=content_characters,
        has_temporal_metadata=has_temporal_metadata,
        freshness_verdict=qualified_source.assessment.verdict,
        eligible=eligible,
        policy_version=grounding_policy.version,
        assessment_sha256=digest,
    )


@dataclass(frozen=True)
class InformationSupportSpan:
    """Exact character span used as deterministic support for one claim."""

    source_id: str
    start_character: int
    end_character: int
    support_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source: InformationSourceDocument,
        start_character: int,
        end_character: int,
    ) -> "InformationSupportSpan":
        source.validate()
        if (
            not isinstance(start_character, int)
            or isinstance(start_character, bool)
            or not isinstance(end_character, int)
            or isinstance(end_character, bool)
            or start_character < 0
            or end_character <= start_character
            or end_character > len(source.normalized_text)
        ):
            raise grounding_failure("grounding_support_invalid")
        support = source.normalized_text[start_character:end_character]
        if not support.strip():
            raise grounding_failure("grounding_support_invalid")
        return cls(
            source_id=source.source_id,
            start_character=start_character,
            end_character=end_character,
            support_sha256=sha256_text(support),
        )

    def validate(
        self,
        *,
        source: InformationSourceDocument,
        policy: InformationGroundingPolicy,
    ) -> str:
        policy.validate()
        if self.source_id != source.source_id:
            raise grounding_failure("grounding_support_invalid")
        if (
            not isinstance(self.start_character, int)
            or isinstance(self.start_character, bool)
            or not isinstance(self.end_character, int)
            or isinstance(self.end_character, bool)
            or self.start_character < 0
            or self.end_character <= self.start_character
            or self.end_character > len(source.normalized_text)
        ):
            raise grounding_failure("grounding_support_invalid")
        if self.end_character - self.start_character > policy.max_support_span_characters:
            raise grounding_failure("grounding_support_invalid")
        support = source.normalized_text[self.start_character:self.end_character]
        if not support.strip() or sha256_text(support) != _require_digest(self.support_sha256):
            raise grounding_failure("grounding_support_invalid")
        return support

    def metadata_record(self) -> dict[str, object]:
        """Return support metadata without the support excerpt."""

        return {
            "source_id": self.source_id,
            "start_character": self.start_character,
            "end_character": self.end_character,
            "support_sha256": self.support_sha256,
        }


@dataclass(frozen=True)
class InformationClaimDraft:
    """Extractive claim proposal; text must equal every cited support span."""

    claim_id: str
    text: str
    content_sha256: str
    knowledge_status: str
    confidence: float
    support_spans: tuple[InformationSupportSpan, ...]

    @classmethod
    def create(
        cls,
        *,
        claim_id: str,
        text: str,
        knowledge_status: str,
        confidence: float,
        support_spans: tuple[InformationSupportSpan, ...],
    ) -> "InformationClaimDraft":
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise grounding_failure("grounding_claim_invalid")
        draft = cls(
            claim_id=claim_id,
            text=text,
            content_sha256=sha256_text(text),
            knowledge_status=knowledge_status,
            confidence=float(confidence),
            support_spans=support_spans,
        )
        draft.validate_shape()
        return draft

    def validate_shape(self) -> None:
        _require_text(self.claim_id, maximum=256)
        _require_text(self.text, maximum=10_000)
        if sha256_text(self.text) != _require_digest(self.content_sha256):
            raise grounding_failure("grounding_claim_invalid")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise grounding_failure("grounding_claim_invalid")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise grounding_failure("grounding_claim_invalid")
        if not self.support_spans:
            raise grounding_failure("grounding_claim_invalid")
        if len({span.source_id for span in self.support_spans}) != len(
            self.support_spans
        ):
            raise grounding_failure("grounding_claim_invalid")


def _citation_id(
    *,
    claim_id: str,
    source: InformationSourceDocument,
    support: InformationSupportSpan,
) -> str:
    seed = "\n".join(
        (
            claim_id,
            source.source_id,
            source.canonical_url,
            source.content_sha256,
            str(support.start_character),
            str(support.end_character),
            support.support_sha256,
        )
    )
    return f"web-{sha256_text(seed)[:20]}"


def _grounding_digest(
    *,
    packet: InformationGroundingPacket,
    query: InformationQuery,
    quality_assessments: tuple[InformationSourceQualityAssessment, ...],
    supports: tuple[InformationSupportSpan, ...],
    policy_version: str,
) -> str:
    parts = [
        packet.packet_id,
        packet.request_id,
        packet.outcome,
        packet.created_at,
        query.query_id,
        query.content_sha256,
        policy_version,
    ]
    parts.extend(
        assessment.assessment_sha256 for assessment in quality_assessments
    )
    parts.extend(
        "|".join(
            (
                support.source_id,
                str(support.start_character),
                str(support.end_character),
                support.support_sha256,
            )
        )
        for support in supports
    )
    for claim in packet.claims:
        parts.extend(
            (
                claim.claim_id,
                claim.content_sha256,
                claim.knowledge_status,
                f"{float(claim.confidence):.12g}",
            )
        )
        parts.extend(
            "|".join(
                (
                    citation.citation_id,
                    citation.source_id,
                    citation.canonical_url,
                    citation.source_content_sha256,
                    citation.token,
                )
            )
            for citation in claim.citations
        )
    return sha256_text("\n".join(parts))


@dataclass(frozen=True)
class InformationVerifiedGroundingPacket:
    """Exact P4.5a grounding plus revalidated source and support bindings."""

    query_id: str
    query_content_sha256: str
    policy_version: str
    packet: InformationGroundingPacket
    quality_assessments: tuple[InformationSourceQualityAssessment, ...]
    support_spans: tuple[InformationSupportSpan, ...]
    grounding_sha256: str

    def validate(
        self,
        *,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
        grounding_policy: InformationGroundingPolicy,
    ) -> None:
        grounding_policy.validate(
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
        )
        query.validate()
        self.packet.validate()
        if self.packet.outcome not in grounding_policy.allowed_outcomes:
            raise grounding_failure("grounding_binding_invalid")
        if len(self.packet.sources) > grounding_policy.max_sources:
            raise grounding_failure("grounding_binding_invalid")
        if len(self.packet.claims) > grounding_policy.max_claims:
            raise grounding_failure("grounding_binding_invalid")
        if self.query_id != query.query_id or self.query_content_sha256 != query.content_sha256:
            raise grounding_failure("grounding_binding_invalid")
        if self.policy_version != grounding_policy.version:
            raise grounding_failure("grounding_binding_invalid")
        source_map = {
            qualified.inspected_source.source.source_id: qualified
            for qualified in qualified_sources
        }
        if len(source_map) != len(qualified_sources):
            raise grounding_failure("grounding_binding_invalid")
        packet_source_ids = {source.source_id for source in self.packet.sources}
        if packet_source_ids != set(source_map):
            raise grounding_failure("grounding_binding_invalid")
        if tuple(source.source_id for source in self.packet.sources) != tuple(
            sorted(packet_source_ids)
        ):
            raise grounding_failure("grounding_binding_invalid")
        quality_map = {
            assessment.source_id: assessment
            for assessment in self.quality_assessments
        }
        if set(quality_map) != packet_source_ids or len(quality_map) != len(
            self.quality_assessments
        ):
            raise grounding_failure("grounding_binding_invalid")
        if tuple(
            assessment.source_id for assessment in self.quality_assessments
        ) != tuple(sorted(packet_source_ids)):
            raise grounding_failure("grounding_binding_invalid")
        for source_id, qualified in source_map.items():
            quality_map[source_id].validate(
                qualified_source=qualified,
                query=query,
                information_policy=information_policy,
                firewall_policy=firewall_policy,
                freshness_policy=freshness_policy,
                grounding_policy=grounding_policy,
            )
            if quality_map[source_id].eligible is not True:
                raise grounding_failure("grounding_source_invalid")
        support_keys = tuple(
            (
                support.source_id,
                support.start_character,
                support.end_character,
                support.support_sha256,
            )
            for support in self.support_spans
        )
        if support_keys != tuple(sorted(support_keys)) or len(set(support_keys)) != len(
            support_keys
        ):
            raise grounding_failure("grounding_binding_invalid")
        support_index = {
            key: support for key, support in zip(support_keys, self.support_spans)
        }
        if tuple(claim.claim_id for claim in self.packet.claims) != tuple(
            sorted(claim.claim_id for claim in self.packet.claims)
        ):
            raise grounding_failure("grounding_binding_invalid")
        cited_sources: set[str] = set()
        used_support_keys: set[tuple[str, int, int, str]] = set()
        citation_ids: set[str] = set()
        conflict_domains: set[str] = set()
        conflict_claim_digests: set[str] = set()
        for claim in self.packet.claims:
            claim.validate()
            if self.packet.outcome == "answerable" and claim.knowledge_status not in {
                "external_claim",
                "verified_fact",
                "historical",
            }:
                raise grounding_failure("grounding_binding_invalid")
            if self.packet.outcome == "uncertain" and claim.knowledge_status != "uncertain":
                raise grounding_failure("grounding_binding_invalid")
            if self.packet.outcome == "conflict" and claim.knowledge_status != "disputed":
                raise grounding_failure("grounding_binding_invalid")
            if tuple(
                (citation.source_id, citation.citation_id)
                for citation in claim.citations
            ) != tuple(
                sorted(
                    (citation.source_id, citation.citation_id)
                    for citation in claim.citations
                )
            ):
                raise grounding_failure("grounding_binding_invalid")
            claim_domains: set[str] = set()
            for citation in claim.citations:
                if citation.citation_id in citation_ids:
                    raise grounding_failure("grounding_binding_invalid")
                citation_ids.add(citation.citation_id)
                source = next(
                    candidate
                    for candidate in self.packet.sources
                    if candidate.source_id == citation.source_id
                )
                candidates: list[tuple[tuple[str, int, int, str], InformationSupportSpan]] = []
                for key, support in support_index.items():
                    if support.source_id != source.source_id:
                        continue
                    support_text = support.validate(
                        source=source,
                        policy=grounding_policy,
                    )
                    if support_text != claim.text:
                        continue
                    if _citation_id(
                        claim_id=claim.claim_id,
                        source=source,
                        support=support,
                    ) == citation.citation_id:
                        candidates.append((key, support))
                if len(candidates) != 1:
                    raise grounding_failure("grounding_binding_invalid")
                key, _ = candidates[0]
                used_support_keys.add(key)
                expected_token = (
                    f"{grounding_policy.citation_token_prefix}"
                    f"{citation.citation_id}"
                    f"{grounding_policy.citation_token_suffix}"
                )
                if citation.token != expected_token:
                    raise grounding_failure("grounding_binding_invalid")
                cited_sources.add(source.source_id)
                domain = _canonical_domain(source)
                claim_domains.add(domain)
                conflict_domains.add(domain)
            if claim.knowledge_status == "verified_fact" and len(
                claim_domains
            ) < grounding_policy.verified_fact_min_distinct_domains:
                raise grounding_failure("grounding_diversity_insufficient")
            conflict_claim_digests.add(claim.content_sha256)
        if cited_sources != packet_source_ids or used_support_keys != set(support_index):
            raise grounding_failure("grounding_binding_invalid")
        if self.packet.outcome == "conflict":
            if len(conflict_claim_digests) < 2:
                raise grounding_failure("grounding_binding_invalid")
            if len(conflict_domains) < grounding_policy.conflict_min_distinct_domains:
                raise grounding_failure("grounding_diversity_insufficient")
        expected_digest = _grounding_digest(
            packet=self.packet,
            query=query,
            quality_assessments=self.quality_assessments,
            supports=self.support_spans,
            policy_version=grounding_policy.version,
        )
        if self.grounding_sha256 != expected_digest:
            raise grounding_failure("grounding_binding_invalid")

    def render_for_model(
        self,
        *,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        information_policy: InformationPolicy,
        firewall_policy: InformationInjectionFirewallPolicy,
        freshness_policy: InformationFreshnessPolicy,
        grounding_policy: InformationGroundingPolicy,
    ) -> str:
        self.validate(
            query=query,
            qualified_sources=qualified_sources,
            information_policy=information_policy,
            firewall_policy=firewall_policy,
            freshness_policy=freshness_policy,
            grounding_policy=grounding_policy,
        )
        boundary = f"ALICE-WEB-GROUNDING-{self.grounding_sha256.upper()}"
        source_map = {
            qualified.inspected_source.source.source_id: qualified
            for qualified in qualified_sources
        }
        claim_lines = []
        for claim in self.packet.claims:
            tokens = " ".join(citation.token for citation in claim.citations)
            claim_lines.append(
                f"Claim {claim.claim_id} [{claim.knowledge_status}]: "
                f"{claim.text} {tokens}"
            )
        source_blocks = [
            source_map[source.source_id].render_for_model(
                query=query,
                information_policy=information_policy,
                firewall_policy=firewall_policy,
                freshness_policy=freshness_policy,
            )
            for source in self.packet.sources
        ]
        return "\n".join(
            (
                f"BEGIN VERIFIED WEB GROUNDING {boundary}",
                f"Grounding policy: {grounding_policy.policy_name}@{grounding_policy.version}",
                f"Outcome: {self.packet.outcome}",
                f"Query SHA-256: {self.query_content_sha256}",
                *claim_lines,
                *source_blocks,
                f"END VERIFIED WEB GROUNDING {boundary}",
            )
        )


@dataclass(frozen=True)
class DeterministicInformationGroundingBuilder:
    """Build exact extractive claims from already qualified source versions."""

    information_policy: InformationPolicy
    firewall_policy: InformationInjectionFirewallPolicy
    freshness_policy: InformationFreshnessPolicy
    grounding_policy: InformationGroundingPolicy

    def __post_init__(self) -> None:
        self.grounding_policy.validate(
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
        )

    def build(
        self,
        *,
        packet_id: str,
        request_id: str,
        outcome: str,
        query: InformationQuery,
        qualified_sources: tuple[InformationTemporallyQualifiedSource, ...],
        claim_drafts: tuple[InformationClaimDraft, ...],
        created_at: str,
    ) -> InformationVerifiedGroundingPacket:
        _require_text(packet_id)
        _require_text(request_id)
        _require_timestamp(created_at)
        query.validate()
        self.grounding_policy.validate(
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
        )
        if outcome not in self.grounding_policy.allowed_outcomes:
            raise grounding_failure("grounding_claim_invalid")
        if len(qualified_sources) > self.grounding_policy.max_sources:
            raise grounding_failure("grounding_source_invalid")
        if len(claim_drafts) > self.grounding_policy.max_claims:
            raise grounding_failure("grounding_claim_invalid")
        if outcome == "insufficient_sources":
            if qualified_sources or claim_drafts:
                raise grounding_failure("grounding_claim_invalid")
            packet = InformationGroundingPacket(
                packet_id=packet_id,
                request_id=request_id,
                outcome=outcome,
                claims=(),
                sources=(),
                created_at=created_at,
            )
            packet.validate()
            digest = _grounding_digest(
                packet=packet,
                query=query,
                quality_assessments=(),
                supports=(),
                policy_version=self.grounding_policy.version,
            )
            verified = InformationVerifiedGroundingPacket(
                query_id=query.query_id,
                query_content_sha256=query.content_sha256,
                policy_version=self.grounding_policy.version,
                packet=packet,
                quality_assessments=(),
                support_spans=(),
                grounding_sha256=digest,
            )
            verified.validate(
                query=query,
                qualified_sources=(),
                information_policy=self.information_policy,
                firewall_policy=self.firewall_policy,
                freshness_policy=self.freshness_policy,
                grounding_policy=self.grounding_policy,
            )
            return verified
        if not qualified_sources or not claim_drafts:
            raise grounding_failure("grounding_claim_invalid")
        source_map: dict[str, InformationTemporallyQualifiedSource] = {}
        quality_assessments: list[InformationSourceQualityAssessment] = []
        for qualified in qualified_sources:
            source_id = qualified.inspected_source.source.source_id
            if source_id in source_map:
                raise grounding_failure("grounding_source_invalid")
            source_map[source_id] = qualified
            quality = _derive_source_quality(
                qualified_source=qualified,
                query=query,
                information_policy=self.information_policy,
                firewall_policy=self.firewall_policy,
                freshness_policy=self.freshness_policy,
                grounding_policy=self.grounding_policy,
            )
            if not quality.eligible:
                raise grounding_failure("grounding_source_invalid")
            quality_assessments.append(quality)
        claims: list[InformationClaim] = []
        all_supports: list[InformationSupportSpan] = []
        cited_source_ids: set[str] = set()
        seen_claim_ids: set[str] = set()
        for draft in sorted(claim_drafts, key=lambda item: item.claim_id):
            draft.validate_shape()
            if draft.claim_id in seen_claim_ids:
                raise grounding_failure("grounding_claim_invalid")
            seen_claim_ids.add(draft.claim_id)
            if draft.knowledge_status not in self.grounding_policy.allowed_knowledge_statuses:
                raise grounding_failure("grounding_claim_invalid")
            if outcome == "answerable" and draft.knowledge_status not in {
                "external_claim",
                "verified_fact",
                "historical",
            }:
                raise grounding_failure("grounding_claim_invalid")
            if outcome == "uncertain" and draft.knowledge_status != "uncertain":
                raise grounding_failure("grounding_claim_invalid")
            if outcome == "conflict" and draft.knowledge_status != "disputed":
                raise grounding_failure("grounding_claim_invalid")
            citations: list[InformationCitation] = []
            domains: set[str] = set()
            for support in sorted(
                draft.support_spans,
                key=lambda item: (
                    item.source_id,
                    item.start_character,
                    item.end_character,
                ),
            ):
                qualified = source_map.get(support.source_id)
                if qualified is None:
                    raise grounding_failure("grounding_support_invalid")
                source = qualified.inspected_source.source
                support_text = support.validate(
                    source=source,
                    policy=self.grounding_policy,
                )
                if support_text != draft.text:
                    raise grounding_failure("grounding_support_invalid")
                citation_id = _citation_id(
                    claim_id=draft.claim_id,
                    source=source,
                    support=support,
                )
                token = (
                    f"{self.grounding_policy.citation_token_prefix}"
                    f"{citation_id}"
                    f"{self.grounding_policy.citation_token_suffix}"
                )
                citation = InformationCitation(
                    citation_id=citation_id,
                    source_id=source.source_id,
                    canonical_url=source.canonical_url,
                    source_content_sha256=source.content_sha256,
                    token=token,
                )
                citation.validate()
                citations.append(citation)
                domains.add(_canonical_domain(source))
                cited_source_ids.add(source.source_id)
                all_supports.append(support)
            citations.sort(key=lambda item: (item.source_id, item.citation_id))
            if (
                draft.knowledge_status == "verified_fact"
                and len(domains)
                < self.grounding_policy.verified_fact_min_distinct_domains
            ):
                raise grounding_failure("grounding_diversity_insufficient")
            claim = InformationClaim(
                claim_id=draft.claim_id,
                text=draft.text,
                content_sha256=draft.content_sha256,
                knowledge_status=draft.knowledge_status,
                confidence=float(draft.confidence),
                citations=tuple(citations),
            )
            claim.validate()
            claims.append(claim)
        if outcome == "conflict":
            if len(claims) < 2:
                raise grounding_failure("grounding_claim_invalid")
            conflict_domains = {
                _canonical_domain(source_map[citation.source_id].inspected_source.source)
                for claim in claims
                for citation in claim.citations
            }
            if len({claim.content_sha256 for claim in claims}) < 2:
                raise grounding_failure("grounding_claim_invalid")
            if len(conflict_domains) < self.grounding_policy.conflict_min_distinct_domains:
                raise grounding_failure("grounding_diversity_insufficient")
        if cited_source_ids != set(source_map):
            raise grounding_failure("grounding_source_invalid")
        sources = tuple(
            source_map[source_id].inspected_source.source
            for source_id in sorted(source_map)
        )
        packet = InformationGroundingPacket(
            packet_id=packet_id,
            request_id=request_id,
            outcome=outcome,
            claims=tuple(claims),
            sources=sources,
            created_at=created_at,
        )
        packet.validate()
        quality_tuple = tuple(
            sorted(quality_assessments, key=lambda item: item.source_id)
        )
        support_tuple = tuple(
            sorted(
                all_supports,
                key=lambda item: (
                    item.source_id,
                    item.start_character,
                    item.end_character,
                ),
            )
        )
        digest = _grounding_digest(
            packet=packet,
            query=query,
            quality_assessments=quality_tuple,
            supports=support_tuple,
            policy_version=self.grounding_policy.version,
        )
        verified = InformationVerifiedGroundingPacket(
            query_id=query.query_id,
            query_content_sha256=query.content_sha256,
            policy_version=self.grounding_policy.version,
            packet=packet,
            quality_assessments=quality_tuple,
            support_spans=support_tuple,
            grounding_sha256=digest,
        )
        verified.validate(
            query=query,
            qualified_sources=qualified_sources,
            information_policy=self.information_policy,
            firewall_policy=self.firewall_policy,
            freshness_policy=self.freshness_policy,
            grounding_policy=self.grounding_policy,
        )
        return verified
