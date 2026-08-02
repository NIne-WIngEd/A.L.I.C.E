from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTED = (
    "mission_graph.v1",
    "semantic_router.v1",
    "result_capsule.v1",
    "traceback_engine.v1",
    "attention_policy.v1",
    "workspace_projection.v1",
    "adaptive_compositor.v1",
    "host_window_override.v1",
    "speaker_context.v1",
    "guest_session.v1",
    "guest_grant.v1",
    "release_attestation.v1",
)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_p50f_validator_passes() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_phase5_parity_release.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_kernel_contract_and_product_eligibility_are_separate() -> None:
    parity = load("policies/capability_parity_ledger.json")
    model = parity["eligibility_model"]
    assert model[
        "kernel_contract_implementation_does_not_equal_product_capability"
    ] is True
    assert model[
        "alice_capability_gained_required_before_friday_eligibility"
    ] is True
    for capability in IMPLEMENTED:
        entry = parity["capabilities"][capability]
        assert entry["kernel_status"] == "implemented_phase5_contract"
        assert entry["alice_capability_gained"] is False
        assert entry["friday_eligibility_status"] == (
            "not_eligible_alice_capability_not_yet_gained"
        )


def test_flagship_and_pre_phase65_gates_remain_authoritative() -> None:
    flagship = load("policies/alice_friday_flagship_governance.json")
    readiness = load("policies/friday_pre_phase_6_5_gate.json")
    gate = flagship["default_new_capability_gate"]
    assert gate["friday_first_capability_allowed"] is False
    assert gate["alice_must_gain_capability_before_friday_eligibility"] is True
    assert readiness["state"] == (
        "foundation_only_waiting_for_alice_phase_6_5"
    )


def test_release_contract_is_verification_only() -> None:
    profiles = load("policies/capability_profiles.json")
    kernel = profiles["profiles"]["kernel.phase5.foundation"]["capabilities"]
    assert kernel["release_attestation_verification_implemented"] is True
    assert kernel["release_signing_implemented"] is False
    assert kernel["release_deployment_implemented"] is False
    assert kernel["approval_generation_implemented"] is False
