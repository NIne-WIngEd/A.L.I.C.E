"""P4.6a deterministic research-orchestration policy tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_information.research_orchestration_policy import (
    InformationResearchOrchestrationPolicyError,
    load_information_research_orchestration_policy,
    parse_information_research_orchestration_policy,
)

POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "policies"
    / "information_research_orchestration_policy.json"
)


def _payload() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_repository_policy_is_exact_p46a_fixture_foundation() -> None:
    policy = load_information_research_orchestration_policy(POLICY_PATH)
    assert policy.phase == "4"
    assert policy.milestone == "P4.6a"
    assert policy.version == "1.0.0"
    assert policy.deterministic_fixture_only is True
    assert policy.live_provider_registration_allowed is False
    assert policy.provider_fallback_allowed is False
    assert policy.max_search_calls == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("policy_name", "lookalike", "policy_name"),
        ("version", "2.0.0", "version"),
        ("phase", "5", "milestone"),
        ("milestone", "P4.6", "milestone"),
        ("status", "live", "status"),
    ],
)
def test_policy_identity_fields_are_exact(field: str, value: object, match: str) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(InformationResearchOrchestrationPolicyError, match=match):
        parse_information_research_orchestration_policy(payload)


@pytest.mark.parametrize(
    ("section", "field", "bad_value"),
    [
        ("provider_selection", "exact_search_provider_required", False),
        ("provider_selection", "exact_fetch_provider_required", False),
        ("provider_selection", "deterministic_fixture_only", False),
        ("provider_selection", "live_provider_registration_allowed", True),
        ("provider_selection", "provider_fallback_allowed", True),
        ("execution", "foreground_only", False),
        ("execution", "query_rewriting_allowed", True),
        ("execution", "recursive_browsing_allowed", True),
        ("execution", "arbitrary_link_following_allowed", True),
        ("execution", "retries_allowed", True),
        ("execution", "canonical_url_deduplication_required", False),
        ("execution", "partial_results_preserved", False),
        ("execution", "activity_records_required", False),
        ("execution", "raw_query_in_activity_allowed", True),
        ("execution", "raw_source_content_in_activity_allowed", True),
    ],
)
def test_policy_rejects_weakened_execution_boundaries(
    section: str,
    field: str,
    bad_value: object,
) -> None:
    payload = _payload()
    payload[section][field] = bad_value
    with pytest.raises(
        InformationResearchOrchestrationPolicyError,
        match="must remain",
    ):
        parse_information_research_orchestration_policy(payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_search_calls", 2),
        ("max_fetch_calls", 7),
        ("max_sources", 9),
        ("max_response_bytes", 1_999_999),
        ("max_request_timeout_seconds", 9),
        ("max_total_timeout_seconds", 46),
        ("max_search_calls", True),
    ],
)
def test_policy_budget_constants_are_exact(field: str, bad_value: object) -> None:
    payload = _payload()
    payload["budgets"][field] = bad_value
    with pytest.raises(InformationResearchOrchestrationPolicyError, match="must equal"):
        parse_information_research_orchestration_policy(payload)


def test_policy_rejects_missing_sections() -> None:
    payload = _payload()
    del payload["execution"]
    with pytest.raises(InformationResearchOrchestrationPolicyError, match="execution"):
        parse_information_research_orchestration_policy(payload)


def test_policy_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"policy_name":"a","policy_name":"b"}',
        encoding="utf-8",
    )
    with pytest.raises(InformationResearchOrchestrationPolicyError, match="Duplicate"):
        load_information_research_orchestration_policy(path)


def test_policy_loader_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(InformationResearchOrchestrationPolicyError, match="root"):
        load_information_research_orchestration_policy(path)
