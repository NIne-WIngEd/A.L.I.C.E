from __future__ import annotations

from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_constitution_is_ratified_and_effective() -> None:
    text = (_root() / "docs" / "ALICE_CONSTITUTION.md").read_text(encoding="utf-8")
    assert "**Status:** Ratified and effective" in text
    assert "**Ratification:** Ratified by explicit owner approval and repository merge." in text


def test_permissions_registry_is_mission_scoped_and_a0_to_a6() -> None:
    payload = yaml.safe_load(
        (_root() / "policies" / "permissions.yaml").read_text(encoding="utf-8")
    )
    assert payload["version"] == "2.0.0"
    assert payload["policy_model"] == "mission_scoped_autonomy"
    assert payload["activation_model"] == "capability_profile_plus_mission"
    assert payload["autonomy_classes"] == {
        "A0": "cognition_observation",
        "A1": "creation_experimentation",
        "A2": "reversible_operation",
        "A3": "routine_external_agency",
        "A4": "high_consequence_agency",
        "A5": "autonomous_production_and_self_evolution",
        "A6": "constitutional_and_authority_kernel",
    }
    permission_ids = {entry["id"] for entry in payload["permissions"]}
    assert "code.self_modify_candidate" in permission_ids
    assert "model.train_candidate" in permission_ids
    assert "production.deploy_or_merge" in permission_ids
    assert "model.promote_production" in permission_ids


def test_readme_surfaces_current_architecture() -> None:
    text = (_root() / "README.md").read_text(encoding="utf-8")
    for marker in (
        "P4.5a citation-bound grounding is merged",
        "Aggressive temporary capture",
        "Personal Cognitive Kernel extraction starts at Phase 5.0",
        "Friday is not the user-facing assistant name",
        "same ultimate destination capability set",
    ):
        assert marker in text


def test_evaluation_charter_covers_phase5_through_phase15() -> None:
    text = (_root() / "docs" / "EVALUATION_CHARTER.md").read_text(encoding="utf-8")
    for phase in range(5, 16):
        assert f"Phase {phase}" in text
    for marker in (
        "catastrophic forgetting",
        "retention-value prediction",
        "representative replay quality",
        "world, causal, user, social, and self models",
        "challenger/champion comparison",
        "Product-family parity",
        "backup and restore drills",
    ):
        assert marker in text


def test_v01_scope_is_historical_not_destination_architecture() -> None:
    text = (_root() / "docs" / "SCOPE_AND_NON_GOALS.md").read_text(encoding="utf-8")
    assert "HISTORICAL RELEASE SCOPE" in text
    assert "**Scope kind:** Historical compatibility document" in text
    assert "**Capability ceiling:** false" in text
    assert "intended future directions or research programs" in text
