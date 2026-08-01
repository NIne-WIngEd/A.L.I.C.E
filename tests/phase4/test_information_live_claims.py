from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from alice_information.contracts import InformationSourceDocument
from alice_information.freshness_policy import load_information_freshness_policy
from alice_information.grounding_policy import (
    InformationGroundingPolicy,
    load_information_grounding_policy,
)
from alice_information.injection_policy import (
    load_information_injection_firewall_policy,
)
from alice_information.policy import load_information_policy
from alice_information.live_claims import DeterministicLiveExtractiveClaimPlanner
from _information_live_research_helpers import NOW, query

ROOT = Path(__file__).resolve().parents[2]


def _grounding_policy() -> InformationGroundingPolicy:
    information_policy = load_information_policy(
        ROOT / "policies/information_policy.json"
    )
    firewall_policy = load_information_injection_firewall_policy(
        ROOT / "policies/information_injection_firewall_policy.json",
        information_policy=information_policy,
    )
    freshness_policy = load_information_freshness_policy(
        ROOT / "policies/information_freshness_policy.json",
        information_policy=information_policy,
        firewall_policy=firewall_policy,
    )
    return load_information_grounding_policy(
        ROOT / "policies/information_grounding_policy.json",
        information_policy=information_policy,
        firewall_policy=firewall_policy,
        freshness_policy=freshness_policy,
    )


def _qualified(source):
    return SimpleNamespace(inspected_source=SimpleNamespace(source=source))


def _source(source_id, text):
    from alice_information.contracts import sha256_text
    return InformationSourceDocument(
        source_id=source_id,
        provider="controlled-live-http-v1",
        canonical_url=f"https://example.com/{source_id}",
        title="Example",
        normalized_text=text,
        content_sha256=sha256_text(text),
        retrieved_at=NOW,
    )


def test_claim_planner_returns_exact_source_sentence_and_span():
    text = (
        "This introductory sentence contains no useful match. "
        "OpenAI official API documentation describes current public API behavior."
    )
    planner = DeterministicLiveExtractiveClaimPlanner(_grounding_policy())
    sources, drafts = planner.plan(
        query=query(),
        qualified_sources=(_qualified(_source("source-1", text)),),
        maximum_sources=1,
    )
    assert len(sources) == len(drafts) == 1
    assert drafts[0].text in text
    span = drafts[0].support_spans[0]
    assert text[span.start_character:span.end_character] == drafts[0].text
    assert drafts[0].knowledge_status == "external_claim"


def test_claim_planner_is_deterministic_and_bounded_to_two_sources():
    planner = DeterministicLiveExtractiveClaimPlanner(_grounding_policy())
    qualified = tuple(
        _qualified(_source(f"source-{i}", f"OpenAI official API documentation source {i} provides current public information."))
        for i in range(1, 4)
    )
    first = planner.plan(query=query(), qualified_sources=qualified, maximum_sources=2)
    second = planner.plan(query=query(), qualified_sources=qualified, maximum_sources=2)
    assert first == second
    assert len(first[0]) == 2
