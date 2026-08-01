from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from alice_information.contracts import InformationQuery, InformationResearchRequest
from alice_information.live_research import InformationLiveResearchReceipt
from alice_information.live_research_policy import InformationLiveResearchPolicy

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "policies/information_live_research_policy.json"
DIGEST = hashlib.sha256(b"p410-live-test").hexdigest()
NOW = "2026-07-30T12:00:00Z"


def policy() -> InformationLiveResearchPolicy:
    return InformationLiveResearchPolicy.load(POLICY_PATH)


def query(*, classification: str = "PUBLIC") -> InformationQuery:
    return InformationQuery.create(
        query_id="p410-live-query",
        text="OpenAI official API documentation current information",
        created_at=NOW,
        data_classification=classification,
    )


def request(*, classification: str = "PUBLIC", operations=("search", "fetch"), max_sources=2):
    return InformationResearchRequest(
        request_id="p410-live-request",
        query=query(classification=classification),
        operations=operations,
        max_search_calls=1,
        max_fetch_calls=max_sources,
        max_sources=max_sources,
        request_timeout_seconds=5.0,
        total_timeout_seconds=15.0,
    )


def command(*, grounding=None):
    return SimpleNamespace(grounding=grounding, validate=lambda: None)


def receipt(**overrides) -> InformationLiveResearchReceipt:
    values = {
        "policy_version": "1.0.0",
        "request_id": "p410-live-request",
        "query_id": "p410-live-query",
        "query_sha256": DIGEST,
        "outcome": "answerable",
        "search_result_count": 1,
        "search_receipt_sha256": DIGEST,
        "fetch_attempt_count": 1,
        "fetch_attempt_sequence_sha256": DIGEST,
        "fetch_receipt_sha256s": (DIGEST,),
        "fetch_failure_sha256s": (),
        "temporal_resolution_sha256s": (DIGEST,),
        "source_outcome_sha256": DIGEST,
        "grounded_source_sha256s": (DIGEST,),
        "grounding_sha256": DIGEST,
        "projection_sha256": DIGEST,
        "conversation_packet_sha256": DIGEST,
        "response_sha256": DIGEST,
        "validation_sha256": DIGEST,
        "citation_validation_outcome": "accepted",
        "pre_commit_validation_count": 1,
        "policy_bindings": ("p410b@1.0.0", "p410a@1.0.0"),
        "created_at": NOW,
    }
    values.update(overrides)
    return InformationLiveResearchReceipt.create(**values)
