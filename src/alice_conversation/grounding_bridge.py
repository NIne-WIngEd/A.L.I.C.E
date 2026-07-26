"""Read-only adapters from Phase 1/2 grounding into Phase 3 contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .contracts import (
    ConversationCitation,
    ConversationContractError,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
    ORDINARY_CONVERSATION_CLASSIFICATIONS,
    sha256_text,
)
from .grounding_policy import ConversationGroundingPolicy
from .state_service import ConversationStateReference

if TYPE_CHECKING:
    from alice_memory.cited_answer import MemoryAnswerSubmission

_CLASSIFICATION_RANK = {
    value: index for index, value in enumerate(ORDINARY_CONVERSATION_CLASSIFICATIONS)
}
_SUPPORTING_MEMORY_RELATIONS = {"supports", "derived_from"}
_PHASE1_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ConversationGroundingBridgeError(ConversationContractError):
    """Raised when evidence cannot safely cross into Phase 3 grounding."""


@dataclass(frozen=True)
class GroundingReadAuthorization:
    actor: str
    allowed: bool
    purpose: str
    max_classification: str = "PRIVATE"

    def validate(self) -> None:
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ConversationGroundingBridgeError(
                "Grounding authorization requires a non-empty actor."
            )
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ConversationGroundingBridgeError(
                "Grounding authorization requires a non-empty purpose."
            )
        if not self.allowed:
            raise ConversationGroundingBridgeError(
                "Grounding read denied by explicit authorization."
            )
        if self.max_classification not in _CLASSIFICATION_RANK:
            raise ConversationGroundingBridgeError(
                "Ordinary grounding cannot authorize HIGHLY_SENSITIVE or SECRETS."
            )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationGroundingBridgeError(f"{field} must be non-empty text.")
    return value


def _require_digest(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field)
    if not _PHASE1_DIGEST.fullmatch(text):
        raise ConversationGroundingBridgeError(f"{field} must be a SHA-256 digest.")
    return text


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConversationGroundingBridgeError(f"{field} must be an object.")
    return dict(value)


def _require_sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConversationGroundingBridgeError(f"{field} must be an array.")
    return tuple(value)


def _classification_allowed(
    classification: str,
    authorization: GroundingReadAuthorization,
) -> None:
    rank = _CLASSIFICATION_RANK.get(classification)
    if rank is None:
        raise ConversationGroundingBridgeError(
            "Ordinary grounding cannot contain HIGHLY_SENSITIVE or SECRETS content."
        )
    if rank > _CLASSIFICATION_RANK[authorization.max_classification]:
        raise ConversationGroundingBridgeError(
            "Grounding content exceeds the authorized classification scope."
        )


def _packet_id(prefix: str, material: Any) -> str:
    return f"{prefix}-{_digest(material)[:24]}"


def conversation_grounding_from_memory_submission(
    submission: "MemoryAnswerSubmission | object",
    *,
    authorization: GroundingReadAuthorization,
    created_at: str,
    packet_id: str | None = None,
) -> ConversationGroundingPacket:
    """Convert one validated Phase 2 cited-answer submission without re-retrieval."""

    authorization.validate()
    case_id = _require_text(getattr(submission, "case_id", None), field="case_id")
    outcome = _require_text(getattr(submission, "outcome", None), field="outcome")
    raw_claims = tuple(getattr(submission, "claims", ()))
    claims: list[ConversationGroundingClaim] = []
    claim_ids: set[str] = set()
    citation_ids: dict[str, tuple[str, str]] = {}

    for raw_claim in raw_claims:
        claim_id = _require_text(getattr(raw_claim, "claim_id", None), field="claim_id")
        if claim_id in claim_ids:
            raise ConversationGroundingBridgeError(
                "Memory grounding contains duplicate claim IDs."
            )
        claim_ids.add(claim_id)
        memory_id = _require_text(
            getattr(raw_claim, "memory_id", None), field="memory_id"
        )
        text = _require_text(getattr(raw_claim, "text", None), field="claim text")
        content_sha256 = _require_digest(
            getattr(raw_claim, "content_sha256", None), field="claim content_sha256"
        )
        if sha256_text(text) != content_sha256:
            raise ConversationGroundingBridgeError(
                "Phase 2 claim text does not match its authoritative digest."
            )
        classification = _require_text(
            getattr(raw_claim, "data_classification", None),
            field="claim data_classification",
        )
        _classification_allowed(classification, authorization)
        raw_citations = tuple(getattr(raw_claim, "citations", ()))
        if not raw_citations:
            raise ConversationGroundingBridgeError(
                "Authoritative memory claims require exact source citations."
            )
        citations: list[ConversationCitation] = []
        for raw_citation in raw_citations:
            citation_memory_id = _require_text(
                getattr(raw_citation, "memory_id", None),
                field="citation memory_id",
            )
            if citation_memory_id != memory_id:
                raise ConversationGroundingBridgeError(
                    "Memory citation is bound to a different authoritative memory."
                )
            support_relation = _require_text(
                getattr(raw_citation, "support_relation", None),
                field="support_relation",
            )
            if support_relation not in _SUPPORTING_MEMORY_RELATIONS:
                raise ConversationGroundingBridgeError(
                    "Only supporting or derived-from memory citations may ground a claim."
                )
            citation_id = _require_text(
                getattr(raw_citation, "memory_source_id", None),
                field="memory_source_id",
            )
            source_ref = _require_text(
                getattr(raw_citation, "source_ref", None), field="source_ref"
            )
            token = _require_text(getattr(raw_citation, "token", None), field="token")
            logical = (source_ref, token)
            previous = citation_ids.get(citation_id)
            if previous is not None and previous != logical:
                raise ConversationGroundingBridgeError(
                    "A memory citation ID was reused for different source material."
                )
            citation_ids[citation_id] = logical
            citations.append(
                ConversationCitation(
                    citation_id=citation_id,
                    source_kind="memory_source",
                    source_ref=source_ref,
                    token=token,
                    data_classification=classification,
                )
            )
        claims.append(
            ConversationGroundingClaim(
                claim_id=claim_id,
                text=text,
                content_sha256=content_sha256,
                knowledge_status=_require_text(
                    getattr(raw_claim, "knowledge_status", None),
                    field="knowledge_status",
                ),
                confidence=float(getattr(raw_claim, "confidence", -1.0)),
                data_classification=classification,
                citations=tuple(citations),
            )
        )

    material = {
        "source": "phase2_memory_submission",
        "case_id": case_id,
        "outcome": outcome,
        "claim_ids": [claim.claim_id for claim in claims],
        "citation_ids": sorted(citation_ids),
    }
    packet = ConversationGroundingPacket(
        packet_id=packet_id or _packet_id("memory-grounding", material),
        outcome=outcome,
        claims=tuple(claims),
        created_at=created_at,
        max_classification=authorization.max_classification,
    )
    packet.validate()
    return packet


def _verify_phase1_guardrails(package: Mapping[str, Any]) -> None:
    guardrails = _require_mapping(package.get("guardrails"), field="guardrails")
    for key in (
        "memory_write_allowed",
        "answer_generation_allowed",
        "external_action_allowed",
        "contradictions_auto_resolved",
    ):
        if guardrails.get(key) is not False:
            raise ConversationGroundingBridgeError(
                f"Phase 1 guardrail {key} must remain false."
            )
    if guardrails.get("source_text_is_untrusted_data") is not True:
        raise ConversationGroundingBridgeError(
            "Phase 1 source text must remain marked as untrusted data."
        )
    if guardrails.get("private_output_only") is not True:
        raise ConversationGroundingBridgeError(
            "Phase 1 grounding must remain private-output-only."
        )


def conversation_grounding_from_phase1_package(
    package: Mapping[str, Any],
    *,
    authorization: GroundingReadAuthorization,
    policy: ConversationGroundingPolicy,
    created_at: str,
    data_classification: str = "PRIVATE",
    packet_id: str | None = None,
) -> ConversationGroundingPacket:
    """Convert a validated-style Phase 1 context package into uncertain evidence."""

    authorization.validate()
    _classification_allowed(data_classification, authorization)
    value = _require_mapping(package, field="Phase 1 context package")
    _verify_phase1_guardrails(value)
    package_id = _require_text(value.get("package_id"), field="package_id")
    evidence = _require_sequence(value.get("evidence"), field="evidence")
    if len(evidence) > policy.maximum_phase1_sources:
        raise ConversationGroundingBridgeError(
            "Phase 1 context exceeds the approved source limit."
        )
    if value.get("source_count") != len(evidence):
        raise ConversationGroundingBridgeError(
            "Phase 1 source_count does not match the evidence array."
        )

    claims: list[ConversationGroundingClaim] = []
    source_hashes: set[str] = set()
    citation_ids: set[str] = set()
    citation_to_claim: dict[str, str] = {}
    for index, raw_item in enumerate(evidence, start=1):
        item = _require_mapping(raw_item, field=f"evidence[{index - 1}]")
        local_id = _require_text(item.get("citation_id"), field="citation_id")
        expected_id = f"S{index}"
        if local_id != expected_id:
            raise ConversationGroundingBridgeError(
                "Phase 1 citation IDs must be contiguous and ordered."
            )
        if local_id in citation_ids:
            raise ConversationGroundingBridgeError(
                "Phase 1 evidence contains duplicate citation IDs."
            )
        citation_ids.add(local_id)
        source_hash = _require_digest(
            item.get("source_content_sha256"), field="source_content_sha256"
        )
        if source_hash in source_hashes:
            raise ConversationGroundingBridgeError(
                "Phase 1 evidence contains duplicate source-content hashes."
            )
        source_hashes.add(source_hash)
        provenance = _require_sequence(item.get("provenance"), field="provenance")
        if not provenance:
            raise ConversationGroundingBridgeError(
                "Phase 1 evidence must preserve source provenance."
            )
        text = _require_text(item.get("context_text"), field="context_text")
        chunk_id = str(item.get("chunk_id") or "").strip()
        source_kind = "phase1_chunk" if chunk_id else "phase1_source"
        source_ref = chunk_id or source_hash
        token = _require_text(item.get("citation"), field="citation")
        if token != f"[{local_id}]":
            raise ConversationGroundingBridgeError(
                "Phase 1 citation token does not exactly match citation_id."
            )
        global_citation_id = f"{package_id}:{local_id}"
        claim_id = hashlib.sha256(
            f"{package_id}|{local_id}|{source_hash}|{sha256_text(text)}".encode("utf-8")
        ).hexdigest()
        citation_to_claim[local_id] = claim_id
        claims.append(
            ConversationGroundingClaim(
                claim_id=claim_id,
                text=text,
                content_sha256=sha256_text(text),
                knowledge_status=policy.phase1_default_knowledge_status,
                confidence=policy.phase1_default_confidence,
                data_classification=data_classification,
                citations=(
                    ConversationCitation(
                        citation_id=global_citation_id,
                        source_kind=source_kind,
                        source_ref=source_ref,
                        token=token,
                        data_classification=data_classification,
                    ),
                ),
            )
        )

    contradiction_groups = _require_sequence(
        value.get("contradiction_groups", ()), field="contradiction_groups"
    )
    has_conflict = False
    for raw_group in contradiction_groups:
        group = _require_mapping(raw_group, field="contradiction group")
        if group.get("unresolved") is not True or group.get("resolution") is not None:
            raise ConversationGroundingBridgeError(
                "Phase 1 contradictions may not be silently resolved."
            )
        group_ids = _require_sequence(group.get("citations"), field="group citations")
        if len(group_ids) < 2:
            raise ConversationGroundingBridgeError(
                "A contradiction group requires at least two citations."
            )
        if any(str(citation_id) not in citation_to_claim for citation_id in group_ids):
            raise ConversationGroundingBridgeError(
                "Contradiction group references unknown Phase 1 evidence."
            )
        has_conflict = True

    if not claims:
        outcome = "insufficient_evidence"
    elif has_conflict:
        outcome = "conflict"
    else:
        outcome = "uncertain"
    material = {
        "source": "phase1_context_package",
        "package_id": package_id,
        "outcome": outcome,
        "source_hashes": sorted(source_hashes),
    }
    packet = ConversationGroundingPacket(
        packet_id=packet_id or _packet_id("phase1-grounding", material),
        outcome=outcome,
        claims=tuple(claims),
        created_at=created_at,
        max_classification=authorization.max_classification,
    )
    packet.validate()
    return packet


def merge_conversation_grounding_packets(
    packets: Sequence[ConversationGroundingPacket],
    *,
    created_at: str,
    max_classification: str,
    packet_id: str | None = None,
) -> ConversationGroundingPacket:
    """Merge approved packets while preserving denial, conflict, and uncertainty."""

    if max_classification not in _CLASSIFICATION_RANK:
        raise ConversationGroundingBridgeError(
            "Merged grounding requires an ordinary max classification."
        )
    selected = tuple(packets)
    if not selected:
        packet = ConversationGroundingPacket(
            packet_id=packet_id or _packet_id("merged-grounding", []),
            outcome="insufficient_evidence",
            claims=(),
            created_at=created_at,
            max_classification=max_classification,
        )
        packet.validate()
        return packet
    for packet in selected:
        packet.validate()

    claim_bearing = tuple(packet for packet in selected if packet.claims)
    denied = tuple(packet for packet in selected if packet.outcome == "denied")
    if denied and claim_bearing:
        raise ConversationGroundingBridgeError(
            "Denied grounding cannot be merged with claim-bearing packets."
        )
    if denied:
        outcome = "denied"
        claims: tuple[ConversationGroundingClaim, ...] = ()
    else:
        merged: list[ConversationGroundingClaim] = []
        claim_ids: set[str] = set()
        tokens: dict[str, tuple[str, str]] = {}
        for packet in selected:
            for claim in packet.claims:
                if claim.claim_id in claim_ids:
                    raise ConversationGroundingBridgeError(
                        "Merged grounding contains duplicate claim IDs."
                    )
                claim_ids.add(claim.claim_id)
                for citation in claim.citations:
                    logical = (citation.source_kind, citation.source_ref)
                    previous = tokens.get(citation.token)
                    if previous is not None and previous != logical:
                        raise ConversationGroundingBridgeError(
                            "Citation token collision would make merged grounding ambiguous."
                        )
                    tokens[citation.token] = logical
                merged.append(claim)
        claims = tuple(merged)
        outcomes = {packet.outcome for packet in selected}
        if "conflict" in outcomes:
            outcome = "conflict"
        elif "uncertain" in outcomes:
            outcome = "uncertain"
        elif claims:
            outcome = "answerable"
        elif "not_applicable" in outcomes and outcomes <= {
            "not_applicable",
            "insufficient_evidence",
        }:
            outcome = "not_applicable"
        else:
            outcome = "insufficient_evidence"

    material = {
        "source": "merged",
        "packet_ids": [packet.packet_id for packet in selected],
        "outcome": outcome,
        "claim_ids": [claim.claim_id for claim in claims],
    }
    packet = ConversationGroundingPacket(
        packet_id=packet_id or _packet_id("merged-grounding", material),
        outcome=outcome,
        claims=claims,
        created_at=created_at,
        max_classification=max_classification,
    )
    packet.validate()
    return packet


def conversation_grounding_packet_sha256(
    packet: ConversationGroundingPacket,
) -> str:
    packet.validate()
    return _digest(
        {
            "packet_id": packet.packet_id,
            "outcome": packet.outcome,
            "created_at": packet.created_at,
            "max_classification": packet.max_classification,
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "content_sha256": claim.content_sha256,
                    "knowledge_status": claim.knowledge_status,
                    "confidence": claim.confidence,
                    "data_classification": claim.data_classification,
                    "citations": [
                        {
                            "citation_id": citation.citation_id,
                            "source_kind": citation.source_kind,
                            "source_ref": citation.source_ref,
                            "token": citation.token,
                            "data_classification": citation.data_classification,
                        }
                        for citation in claim.citations
                    ],
                }
                for claim in packet.claims
            ],
        }
    )


def conversation_state_references_from_grounding(
    packet: ConversationGroundingPacket,
) -> tuple[ConversationStateReference, ...]:
    """Create metadata-only P3.2 references without persisting source text."""

    packet.validate()
    references: list[ConversationStateReference] = []
    by_id: dict[str, ConversationStateReference] = {}
    for claim in packet.claims:
        for citation in claim.citations:
            reference = ConversationStateReference(
                reference_id=citation.citation_id,
                source_kind=citation.source_kind,
                source_ref=citation.source_ref,
                citation_token=citation.token,
                content_sha256=None,
                data_classification=citation.data_classification,
                created_at=packet.created_at,
            )
            reference.validate()
            previous = by_id.get(reference.reference_id)
            if previous is not None:
                if previous != reference:
                    raise ConversationGroundingBridgeError(
                        "Citation ID maps to inconsistent conversation-state references."
                    )
                continue
            by_id[reference.reference_id] = reference
            references.append(reference)
    return tuple(references)
