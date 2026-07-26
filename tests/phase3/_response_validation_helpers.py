from __future__ import annotations

from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ModelResponse,
    sha256_text,
)


def response(content: str, *, finish_reason: str = "stop") -> ModelResponse:
    selected = ModelResponse(
        request_id="request-1",
        provider="deterministic-test",
        model="orchestration-v1",
        content=content,
        finish_reason=finish_reason,
        created_at="2026-07-26T05:10:00Z",
    )
    selected.validate()
    return selected


def claim(
    *,
    claim_id: str,
    text: str,
    token: str,
    status: str = "verified_fact",
    confidence: float = 0.9,
) -> ConversationGroundingClaim:
    citation = ConversationCitation(
        citation_id=f"citation-{claim_id}",
        source_kind="memory_source",
        source_ref=f"memory-{claim_id}",
        token=token,
        data_classification="PRIVATE",
    )
    selected = ConversationGroundingClaim(
        claim_id=claim_id,
        text=text,
        content_sha256=sha256_text(text),
        knowledge_status=status,
        confidence=confidence,
        data_classification="PRIVATE",
        citations=(citation,),
    )
    selected.validate()
    return selected


def answerable_packet() -> ConversationGroundingPacket:
    selected = ConversationGroundingPacket(
        packet_id="packet-answerable",
        outcome="answerable",
        claims=(
            claim(
                claim_id="claim-1",
                text="Rayan prefers exact deterministic workflows.",
                token="[memory:claim-1]",
            ),
        ),
        created_at="2026-07-26T05:09:00Z",
        max_classification="PRIVATE",
    )
    selected.validate()
    return selected


def conflict_packet() -> ConversationGroundingPacket:
    selected = ConversationGroundingPacket(
        packet_id="packet-conflict",
        outcome="conflict",
        claims=(
            claim(
                claim_id="claim-a",
                text="The recorded target date is August 1, 2026.",
                token="[memory:claim-a]",
                status="disputed",
                confidence=0.7,
            ),
            claim(
                claim_id="claim-b",
                text="The recorded target date is August 8, 2026.",
                token="[memory:claim-b]",
                status="disputed",
                confidence=0.7,
            ),
        ),
        created_at="2026-07-26T05:09:00Z",
        max_classification="PRIVATE",
    )
    selected.validate()
    return selected


def uncertain_packet() -> ConversationGroundingPacket:
    selected = ConversationGroundingPacket(
        packet_id="packet-uncertain",
        outcome="uncertain",
        claims=(
            claim(
                claim_id="claim-u",
                text="The application may still be under review.",
                token="[memory:claim-u]",
                status="uncertain",
                confidence=0.5,
            ),
        ),
        created_at="2026-07-26T05:09:00Z",
        max_classification="PRIVATE",
    )
    selected.validate()
    return selected


def empty_packet(outcome: str) -> ConversationGroundingPacket:
    selected = ConversationGroundingPacket(
        packet_id=f"packet-{outcome}",
        outcome=outcome,
        claims=(),
        created_at="2026-07-26T05:09:00Z",
        max_classification="PRIVATE",
    )
    selected.validate()
    return selected
