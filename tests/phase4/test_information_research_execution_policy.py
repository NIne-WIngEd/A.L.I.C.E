from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_information.research_execution_policy import (
    InformationResearchExecutionPolicyError,
    load_information_research_execution_policy,
    parse_information_research_execution_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def payload() -> dict[str, object]:
    return json.loads(
        (ROOT / "policies/information_research_execution_policy.json").read_text(
            encoding="utf-8"
        )
    )


class OrchestrationPolicy:
    policy_name = "orchestration"
    version = "1.0.0"
    deterministic_fixture_only = True
    provider_fallback_allowed = False
    live_provider_registration_allowed = False

    def validate(self) -> None:
        return None


class EvidencePolicy:
    policy_name = "evidence"
    version = "1.0.0"

    def validate(self, **kwargs) -> None:
        return None


class ModePolicy:
    policy_name = "mode"
    version = "1.0.0"

    def validate(self, **kwargs) -> None:
        return None


def dependencies():
    return {
        "orchestration_policy": OrchestrationPolicy(),
        "evidence_policy": EvidencePolicy(),
        "mode_policy": ModePolicy(),
    }


def test_public_policy_loads() -> None:
    policy = load_information_research_execution_policy(
        ROOT / "policies/information_research_execution_policy.json",
        **dependencies(),
    )
    assert policy.milestone == "P4.7b"
    assert policy.status == "governed_research_execution"


@pytest.mark.parametrize(
    "field",
    [
        "explicit_mode_required",
        "exact_provider_selection_required",
        "deterministic_fixture_execution_required",
        "orchestration_revalidation_required",
        "evidence_revalidation_required",
        "mode_adapter_revalidation_required",
        "preconversation_failure_handling_required",
    ],
)
def test_required_controls_fail_closed(field: str) -> None:
    selected = payload()
    selected[field] = False
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(selected)


@pytest.mark.parametrize(
    "field",
    [
        "local_only_provider_execution_allowed",
        "silent_web_activation_allowed",
        "provider_fallback_allowed",
        "live_provider_registration_allowed",
        "source_body_persistence_allowed",
        "memory_write_allowed",
        "phase5_storage_runtime_allowed",
        "external_action_allowed",
        "retry_allowed",
        "recursive_browsing_allowed",
        "background_execution_allowed",
    ],
)
def test_prohibited_capabilities_cannot_be_enabled(field: str) -> None:
    selected = payload()
    selected[field] = True
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(selected)


@pytest.mark.parametrize(
    "field",
    [
        "allowed_modes",
        "allowed_requested_availability_states",
        "allowed_result_availability_states",
        "allowed_result_statuses",
        "allowed_unavailable_reasons",
    ],
)
def test_vocabularies_are_exact(field: str) -> None:
    selected = payload()
    selected[field] = [*selected[field], "unexpected"]  # type: ignore[index]
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(selected)


def test_unknown_key_is_rejected() -> None:
    selected = payload()
    selected["unknown"] = False
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(selected)


def test_missing_key_is_rejected() -> None:
    selected = payload()
    selected.pop("permission_id")
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(selected)


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"policy_name":"a","policy_name":"b"}', encoding="utf-8")
    with pytest.raises(InformationResearchExecutionPolicyError, match="Duplicate"):
        load_information_research_execution_policy(path)


def test_fixture_only_dependency_is_required() -> None:
    orchestration = OrchestrationPolicy()
    orchestration.deterministic_fixture_only = False
    with pytest.raises(InformationResearchExecutionPolicyError, match="fixture-only"):
        parse_information_research_execution_policy(
            payload(),
            orchestration_policy=orchestration,
        )


def test_provider_fallback_dependency_is_rejected() -> None:
    orchestration = OrchestrationPolicy()
    orchestration.provider_fallback_allowed = True
    with pytest.raises(InformationResearchExecutionPolicyError, match="no-fallback"):
        parse_information_research_execution_policy(
            payload(),
            orchestration_policy=orchestration,
        )


def test_live_registration_dependency_is_rejected() -> None:
    orchestration = OrchestrationPolicy()
    orchestration.live_provider_registration_allowed = True
    with pytest.raises(InformationResearchExecutionPolicyError, match="fixture-only"):
        parse_information_research_execution_policy(
            payload(),
            orchestration_policy=orchestration,
        )


@pytest.mark.parametrize("dependency", ["orchestration_policy", "evidence_policy", "mode_policy"])
def test_dependency_requires_validation_contract(dependency: str) -> None:
    selected = dependencies()
    selected[dependency] = object()
    with pytest.raises(InformationResearchExecutionPolicyError):
        parse_information_research_execution_policy(payload(), **selected)
