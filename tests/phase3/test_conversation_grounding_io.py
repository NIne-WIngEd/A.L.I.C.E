from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alice_conversation.contracts import sha256_text
from alice_conversation.grounding_io import (
    ConversationGroundingFileError,
    load_conversation_grounding_packet,
    parse_conversation_grounding_packet,
)

from _cli_helpers import cli_policy


def packet_payload():
    text = "The project is in Phase 3."
    return {
        "packet_id": "packet-1",
        "outcome": "answerable",
        "claims": [
            {
                "claim_id": "claim-1",
                "text": text,
                "content_sha256": sha256_text(text),
                "knowledge_status": "verified_fact",
                "confidence": 1.0,
                "data_classification": "PRIVATE",
                "citations": [
                    {
                        "citation_id": "citation-1",
                        "source_kind": "phase1_source",
                        "source_ref": "source-1",
                        "token": "[phase1:source-1]",
                        "data_classification": "PRIVATE",
                    }
                ],
            }
        ],
        "created_at": "2026-07-26T00:00:00Z",
        "max_classification": "PRIVATE",
    }


def test_parse_prebuilt_grounding_packet():
    packet = parse_conversation_grounding_packet(packet_payload())
    assert packet.packet_id == "packet-1"
    assert packet.claims[0].citations[0].token == "[phase1:source-1]"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra=True),
        lambda p: p.pop("outcome"),
        lambda p: p["claims"][0].update(extra=True),
        lambda p: p["claims"][0]["citations"][0].update(extra=True),
        lambda p: p["claims"][0].update(content_sha256="0" * 64),
        lambda p: p.update(max_classification="HIGHLY_SENSITIVE"),
        lambda p: p.update(outcome="conflict"),
        lambda p: p["claims"][0]["citations"][0].update(token="[S1]"),
    ],
)
def test_parse_rejects_malformed_packets(mutation):
    payload = deepcopy(packet_payload())
    mutation(payload)
    with pytest.raises(Exception):
        parse_conversation_grounding_packet(payload)


def test_loader_rejects_repository_local_file(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    path = repository / "grounding.json"
    path.write_text(json.dumps(packet_payload()), encoding="utf-8")
    with pytest.raises(ConversationGroundingFileError):
        load_conversation_grounding_packet(
            path,
            policy=cli_policy(),
            repository_root=repository,
        )


def test_loader_accepts_private_external_file(tmp_path):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    path = vault / "grounding.json"
    path.write_text(json.dumps(packet_payload()), encoding="utf-8")
    packet = load_conversation_grounding_packet(
        path,
        policy=cli_policy(),
        repository_root=repository,
    )
    assert packet.outcome == "answerable"


@pytest.mark.parametrize("content", ["", "[]", "not json"])
def test_loader_rejects_empty_or_invalid_files(tmp_path, content):
    repository = tmp_path / "repo"
    vault = tmp_path / "vault"
    repository.mkdir()
    vault.mkdir()
    path = vault / "grounding.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConversationGroundingFileError):
        load_conversation_grounding_packet(
            path,
            policy=cli_policy(),
            repository_root=repository,
        )
