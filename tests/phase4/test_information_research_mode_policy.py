from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from alice_information.research_mode_policy import (
    APPROVED_MAX_SOURCE_SUMMARIES,
    InformationResearchModePolicyError,
    load_information_research_mode_policy,
    parse_information_research_mode_policy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "policies/information_research_mode_policy.json"


def payload() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_policy_loads_exact_p47a_contract() -> None:
    policy = load_information_research_mode_policy(POLICY_PATH)
    policy.validate()
    assert policy.version == "1.0.0"
    assert policy.milestone == "P4.7a"
    assert policy.allowed_modes == ("local_only", "research")
    assert policy.max_source_summaries == APPROVED_MAX_SOURCE_SUMMARIES


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_name", "wrong"),
        ("version", "1.0.1"),
        ("phase", "5"),
        ("milestone", "P4.7b"),
        ("status", "live_research"),
        ("permission_id", "web.fetch"),
    ],
)
def test_policy_rejects_identity_changes(field: str, value: object) -> None:
    changed = payload()
    changed[field] = value
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


@pytest.mark.parametrize(
    "field",
    [
        "explicit_mode_required",
        "unavailable_preflight_required",
        "evidence_revalidation_required",
        "exact_projection_required",
        "p36_precommit_validation_required",
        "metadata_only_source_summaries_required",
    ],
)
def test_policy_rejects_disabled_required_controls(field: str) -> None:
    changed = payload()
    changed[field] = False
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


@pytest.mark.parametrize(
    "field",
    [
        "silent_web_activation_allowed",
        "local_only_web_grounding_allowed",
        "research_without_evidence_allowed",
        "research_execution_allowed",
        "live_provider_registration_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "phase5_storage_runtime_allowed",
        "external_action_allowed",
        "retry_allowed",
        "background_execution_allowed",
    ],
)
def test_policy_rejects_enabled_forbidden_capabilities(field: str) -> None:
    changed = payload()
    changed[field] = True
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


def test_policy_rejects_vocabulary_changes() -> None:
    changed = payload()
    changed["allowed_modes"] = ["local_only", "research", "silent"]
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


def test_policy_rejects_source_summary_schema_changes() -> None:
    changed = payload()
    changed["source_summary_fields"] = ["citation_token", "raw_text"]
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


def test_policy_rejects_budget_change_without_version_change() -> None:
    changed = payload()
    changed["limits"]["max_source_summaries"] = 289  # type: ignore[index]
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


def test_policy_rejects_unknown_and_missing_keys() -> None:
    changed = payload()
    changed["unknown"] = False
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)
    changed = payload()
    changed.pop("explicit_mode_required")
    with pytest.raises(InformationResearchModePolicyError):
        parse_information_research_mode_policy(changed)


def test_policy_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    text = POLICY_PATH.read_text(encoding="utf-8")
    duplicated = text.replace(
        '"policy_name": "alice_information_research_mode_policy",',
        '"policy_name": "alice_information_research_mode_policy",\n'
        '  "policy_name": "duplicate",',
        1,
    )
    selected = tmp_path / "duplicate.json"
    selected.write_text(duplicated, encoding="utf-8")
    with pytest.raises(InformationResearchModePolicyError):
        load_information_research_mode_policy(selected)


def test_dataclass_validation_rejects_tampering() -> None:
    policy = load_information_research_mode_policy(POLICY_PATH)
    with pytest.raises(InformationResearchModePolicyError):
        replace(policy, silent_web_activation_allowed=True).validate()
