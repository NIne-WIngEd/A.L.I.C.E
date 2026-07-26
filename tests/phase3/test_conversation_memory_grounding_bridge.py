from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from alice_conversation.contracts import sha256_text
from alice_conversation.grounding_bridge import (
    ConversationGroundingBridgeError,
    GroundingReadAuthorization,
    conversation_grounding_from_memory_submission,
    conversation_grounding_packet_sha256,
    conversation_state_references_from_grounding,
)

NOW = "2026-07-25T22:30:00Z"


@dataclass(frozen=True)
class Citation:
    memory_source_id: str
    memory_id: str
    source_ref: str
    support_relation: str = "supports"

    @property
    def token(self) -> str:
        return f"[memory:{self.memory_id} source:{self.source_ref} source_id:{self.memory_source_id}]"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    memory_id: str
    text: str
    content_sha256: str
    knowledge_status: str
    confidence: float
    data_classification: str
    citations: tuple[Citation, ...]


@dataclass(frozen=True)
class Submission:
    case_id: str
    outcome: str
    claims: tuple[Claim, ...]


def authorization(**changes) -> GroundingReadAuthorization:
    value = GroundingReadAuthorization(
        actor="rayan",
        allowed=True,
        purpose="answer personal question",
        max_classification="PRIVATE",
    )
    return replace(value, **changes)


def claim(*, memory_id: str = "memory-1", source_id: str = "source-1", classification: str = "PRIVATE") -> Claim:
    text = "Rayan is working on A.L.I.C.E. Phase 3."
    return Claim(
        claim_id=f"claim-{memory_id}",
        memory_id=memory_id,
        text=text,
        content_sha256=sha256_text(text),
        knowledge_status="verified_fact",
        confidence=0.95,
        data_classification=classification,
        citations=(Citation(source_id, memory_id, f"fixture:{source_id}"),),
    )


def test_converts_authoritative_memory_with_exact_token() -> None:
    source = claim()
    packet = conversation_grounding_from_memory_submission(
        Submission("case-1", "answerable", (source,)),
        authorization=authorization(),
        created_at=NOW,
    )
    assert packet.outcome == "answerable"
    assert packet.claims[0].text == source.text
    assert packet.claims[0].citations[0].token == source.citations[0].token
    assert packet.claims[0].citations[0].source_kind == "memory_source"


def test_preserves_conflict_outcome_and_claim_order() -> None:
    first = claim(memory_id="memory-a", source_id="source-a")
    second = claim(memory_id="memory-b", source_id="source-b")
    packet = conversation_grounding_from_memory_submission(
        Submission("case-conflict", "conflict", (first, second)),
        authorization=authorization(),
        created_at=NOW,
    )
    assert packet.outcome == "conflict"
    assert [item.claim_id for item in packet.claims] == [first.claim_id, second.claim_id]


@pytest.mark.parametrize("outcome", ["insufficient_evidence", "denied"])
def test_preserves_empty_outcomes(outcome: str) -> None:
    packet = conversation_grounding_from_memory_submission(
        Submission("case-empty", outcome, ()),
        authorization=authorization(),
        created_at=NOW,
    )
    assert packet.outcome == outcome
    assert packet.claims == ()


def test_rejects_denied_authorization() -> None:
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (claim(),)),
            authorization=authorization(allowed=False),
            created_at=NOW,
        )


def test_rejects_highly_sensitive_memory() -> None:
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (claim(classification="HIGHLY_SENSITIVE"),)),
            authorization=authorization(),
            created_at=NOW,
        )


def test_rejects_narrow_authorization() -> None:
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (claim(),)),
            authorization=authorization(max_classification="INTERNAL"),
            created_at=NOW,
        )


def test_rejects_claim_digest_tampering() -> None:
    bad = replace(claim(), content_sha256="0" * 64)
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (bad,)),
            authorization=authorization(),
            created_at=NOW,
        )


def test_rejects_cross_memory_citation() -> None:
    source = claim()
    bad = replace(source, citations=(Citation("source-1", "memory-2", "fixture:source-1"),))
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (bad,)),
            authorization=authorization(),
            created_at=NOW,
        )


def test_rejects_non_supporting_relation() -> None:
    source = claim()
    bad_citation = replace(source.citations[0], support_relation="contradicts")
    with pytest.raises(ConversationGroundingBridgeError):
        conversation_grounding_from_memory_submission(
            Submission("case-1", "answerable", (replace(source, citations=(bad_citation,)),)),
            authorization=authorization(),
            created_at=NOW,
        )


def test_packet_digest_is_deterministic() -> None:
    packet = conversation_grounding_from_memory_submission(
        Submission("case-1", "answerable", (claim(),)),
        authorization=authorization(),
        created_at=NOW,
    )
    assert conversation_grounding_packet_sha256(packet) == conversation_grounding_packet_sha256(packet)
    assert len(conversation_grounding_packet_sha256(packet)) == 64


def test_creates_metadata_only_state_references() -> None:
    packet = conversation_grounding_from_memory_submission(
        Submission("case-1", "answerable", (claim(),)),
        authorization=authorization(),
        created_at=NOW,
    )
    references = conversation_state_references_from_grounding(packet)
    assert len(references) == 1
    assert references[0].source_kind == "memory_source"
    assert references[0].content_sha256 is None
    assert references[0].citation_token == packet.claims[0].citations[0].token
