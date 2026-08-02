from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    load_cognitive_kernel_release_attestation_policy,
)

ROOT = Path(__file__).resolve().parents[2]


def test_release_policy_loads_and_preserves_governance() -> None:
    policy = load_cognitive_kernel_release_attestation_policy(
        repository_root=ROOT
    )
    assert policy.version == "0.5.0"
    assert policy.milestone == "P5.0f"
    assert policy.required_capabilities == ("release_attestation.v1",)
    assert policy.invariants[
        "parity_contract_status_does_not_imply_alice_capability_gained"
    ] is True
    assert policy.invariants["friday_pre_phase_6_5_foundation_only"] is True
    assert policy.invariants["release_signing_implemented"] is False
    assert policy.invariants["deployment_implemented"] is False


def test_release_schema_covers_all_production_bindings() -> None:
    production = json.loads(
        (ROOT / "policies/friday_production_governance.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (ROOT / "policies/friday_release_attestation_schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == "2.0.0"
    for binding in production["required_bindings"]:
        assert binding in schema["required"]
        assert binding in schema["properties"]
    assert "alice_audit_attestation" in schema["required"]
    assert "rayan_approval" in schema["required"]
    assert "release_digest" in schema["required"]


def test_parity_status_does_not_make_friday_eligible() -> None:
    parity = json.loads(
        (ROOT / "policies/capability_parity_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    for capability, entry in parity["capabilities"].items():
        if entry.get("kernel_status") != "implemented_phase5_contract":
            continue
        assert entry["kernel_contract_implemented"] is True
        assert entry["alice_capability_gained"] is False
        assert entry["friday_eligibility_status"] == (
            "not_eligible_alice_capability_not_yet_gained"
        )
        assert entry["pre_phase_6_5_status"] == "foundation_only"
        assert isinstance(entry["evidence_bundle"], dict), capability


def test_friday_profile_is_deny_by_default_for_new_capability_work() -> None:
    profiles = json.loads(
        (ROOT / "policies/capability_profiles.json").read_text(
            encoding="utf-8"
        )
    )
    capabilities = profiles["profiles"][
        "friday.production.dual_approval"
    ]["capabilities"]
    assert capabilities["candidate_research_allowed"] is False
    assert capabilities["candidate_implementation_allowed"] is False
    assert capabilities["uneligible_capability_testing_allowed"] is False
    assert capabilities["alice_capability_precedent_required"] is True
    assert capabilities["phase_6_5_foundation_only"] is True


def test_tampered_policy_invariant_is_rejected(tmp_path: Path) -> None:
    source = json.loads(
        (
            ROOT
            / "policies/cognitive_kernel_release_attestation_policy.json"
        ).read_text(encoding="utf-8")
    )
    source["invariants"][
        "friday_eligibility_requires_alice_capability_gained"
    ] = False
    path = tmp_path / "release-policy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(CognitiveKernelContractError, match="must be true"):
        load_cognitive_kernel_release_attestation_policy(
            path=path,
            repository_root=ROOT,
        )
