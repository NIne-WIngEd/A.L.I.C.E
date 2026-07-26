from __future__ import annotations

from dataclasses import replace

import pytest

from alice_conversation.contracts import (
    ConversationCitation,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    sha256_text,
)
from alice_conversation.grounding_bridge import (
    ConversationGroundingBridgeError,
    merge_conversation_grounding_packets,
)

NOW = "2026-07-25T22:30:00Z"


def packet(packet_id: str, outcome: str, token: str, source_ref: str) -> ConversationGroundingPacket:
    text = f"Claim from {packet_id}"
    claim = ConversationGroundingClaim(
        claim_id=f"claim-{packet_id}",
        text=text,
        content_sha256=sha256_text(text),
        knowledge_status="external_claim",
        confidence=0.5,
        data_classification="PRIVATE",
        citations=(
            ConversationCitation(
                citation_id=f"citation-{packet_id}",
                source_kind="phase1_source",
                source_ref=source_ref,
                token=token,
                data_classification="PRIVATE",
            ),
        ),
    )
    return ConversationGroundingPacket(
        packet_id=packet_id,
        outcome=outcome,
        claims=(claim,),
        created_at=NOW,
        max_classification="PRIVATE",
    )


def test_merges_answerable_packets_in_input_order() -> None:
    first = packet("p1", "answerable", "[S1]", "source-1")
    second = packet("p2", "answerable", "[memory:m source:x source_id:y]", "source-2")
    merged = merge_conversation_grounding_packets(
        (first, second), created_at=NOW, max_classification="PRIVATE"
    )
    assert merged.outcome == "answerable"
    assert [claim.claim_id for claim in merged.claims] == ["claim-p1", "claim-p2"]


def test_uncertainty_takes_precedence_over_answerable() -> None:
    merged = merge_conversation_grounding_packets(
        (
            packet("p1", "answerable", "[S1]", "source-1"),
            packet("p2", "uncertain", "[S2]", "source-2"),
        ),
        created_at=NOW,
        max_classification="PRIVATE",
    )
    assert merged.outcome == "uncertain"


def test_conflict_takes_precedence() -> None:
    conflict = packet("p2", "conflict", "[S2]", "source-2")
    conflict = replace(conflict, claims=(conflict.claims[0], replace(conflict.claims[0], claim_id="claim-p2b", citations=(replace(conflict.claims[0].citations[0], citation_id="citation-p2b", token="[S3]", source_ref="source-3"),))))
    merged = merge_conversation_grounding_packets(
        (packet("p1", "answerable", "[S1]", "source-1"), conflict),
        created_at=NOW,
        max_classification="PRIVATE",
    )
    assert merged.outcome == "conflict"


def test_rejects_denied_and_claim_bearing_mix() -> None:
    denied = ConversationGroundingPacket(
        packet_id="denied",
        outcome="denied",
        claims=(),
        created_at=NOW,
        max_classification="PRIVATE",
    )
    with pytest.raises(ConversationGroundingBridgeError):
        merge_conversation_grounding_packets(
            (denied, packet("p1", "answerable", "[S1]", "source-1")),
            created_at=NOW,
            max_classification="PRIVATE",
        )


def test_rejects_ambiguous_token_collision() -> None:
    with pytest.raises(ConversationGroundingBridgeError):
        merge_conversation_grounding_packets(
            (
                packet("p1", "answerable", "[S1]", "source-1"),
                packet("p2", "answerable", "[S1]", "source-2"),
            ),
            created_at=NOW,
            max_classification="PRIVATE",
        )


def test_empty_merge_is_insufficient_evidence() -> None:
    merged = merge_conversation_grounding_packets(
        (), created_at=NOW, max_classification="PRIVATE"
    )
    assert merged.outcome == "insufficient_evidence"
    assert merged.claims == ()
