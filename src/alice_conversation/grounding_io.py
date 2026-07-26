"""Strict loading of prebuilt read-only grounding packets for P3.7."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .cli_policy import ConversationCliPolicy
from .contracts import (
    ConversationCitation,
    ConversationContractError,
    ConversationGroundingClaim,
    ConversationGroundingPacket,
)


_CITATION_TOKEN = re.compile(
    r"^\[[A-Za-z][A-Za-z0-9_.-]{0,63}:[^\]\r\n]{1,256}\]$"
)


class ConversationGroundingFileError(ConversationContractError):
    """Raised when a prebuilt grounding file is unsafe or malformed."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationGroundingFileError(f"{field} must be an object.")
    return value


def _sequence(value: Any, *, field: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ConversationGroundingFileError(f"{field} must be an array.")
    return tuple(value)


def _exact_keys(value: dict[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise ConversationGroundingFileError(
            f"{field} fields do not match the prebuilt grounding contract."
        )


def parse_conversation_grounding_packet(
    payload: dict[str, Any],
) -> ConversationGroundingPacket:
    root = _mapping(payload, field="grounding packet")
    _exact_keys(
        root,
        {"packet_id", "outcome", "claims", "created_at", "max_classification"},
        field="grounding packet",
    )
    claims: list[ConversationGroundingClaim] = []
    for claim_index, raw_claim in enumerate(
        _sequence(root["claims"], field="claims")
    ):
        claim = _mapping(raw_claim, field=f"claims[{claim_index}]")
        _exact_keys(
            claim,
            {
                "claim_id",
                "text",
                "content_sha256",
                "knowledge_status",
                "confidence",
                "data_classification",
                "citations",
            },
            field=f"claims[{claim_index}]",
        )
        citations: list[ConversationCitation] = []
        for citation_index, raw_citation in enumerate(
            _sequence(claim["citations"], field=f"claims[{claim_index}].citations")
        ):
            citation = _mapping(
                raw_citation,
                field=f"claims[{claim_index}].citations[{citation_index}]",
            )
            _exact_keys(
                citation,
                {
                    "citation_id",
                    "source_kind",
                    "source_ref",
                    "token",
                    "data_classification",
                },
                field=f"claims[{claim_index}].citations[{citation_index}]",
            )
            token = citation["token"]
            if not isinstance(token, str) or not _CITATION_TOKEN.fullmatch(token):
                raise ConversationGroundingFileError(
                    "Citation tokens must use the exact [namespace:reference] format."
                )
            citations.append(
                ConversationCitation(
                    citation_id=citation["citation_id"],
                    source_kind=citation["source_kind"],
                    source_ref=citation["source_ref"],
                    token=token,
                    data_classification=citation["data_classification"],
                )
            )
        claims.append(
            ConversationGroundingClaim(
                claim_id=claim["claim_id"],
                text=claim["text"],
                content_sha256=claim["content_sha256"],
                knowledge_status=claim["knowledge_status"],
                confidence=claim["confidence"],
                data_classification=claim["data_classification"],
                citations=tuple(citations),
            )
        )
    packet = ConversationGroundingPacket(
        packet_id=root["packet_id"],
        outcome=root["outcome"],
        claims=tuple(claims),
        created_at=root["created_at"],
        max_classification=root["max_classification"],
    )
    packet.validate()
    return packet


def load_conversation_grounding_packet(
    path: str | Path,
    *,
    policy: ConversationCliPolicy,
    repository_root: str | Path,
) -> ConversationGroundingPacket:
    if not policy.prebuilt_grounding_file_allowed:
        raise ConversationGroundingFileError(
            "Prebuilt grounding files are disabled by CLI policy."
        )
    selected = Path(path).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=False)
    if _is_within(selected, repository):
        raise ConversationGroundingFileError(
            "Private grounding files cannot be loaded from inside the repository."
        )
    try:
        size = selected.stat().st_size
    except OSError as exc:
        raise ConversationGroundingFileError(
            "The grounding file could not be inspected."
        ) from exc
    if size <= 0 or size > policy.max_grounding_file_bytes:
        raise ConversationGroundingFileError(
            "The grounding file is empty or exceeds the approved byte limit."
        )
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConversationGroundingFileError(
            "The grounding file could not be read."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationGroundingFileError(
            "The grounding file is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ConversationGroundingFileError(
            "The grounding file root must be an object."
        )
    return parse_conversation_grounding_packet(payload)
