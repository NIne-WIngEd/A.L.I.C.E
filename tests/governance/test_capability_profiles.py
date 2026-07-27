from __future__ import annotations

import json
from pathlib import Path

from alice_capability_profiles import load_capability_profile
from alice_evolution import CapabilityRuntime, MissionAuthority

ROOT = Path(__file__).resolve().parents[2]


def test_all_profiles_are_declared_non_ceiling() -> None:
    payload = json.loads(
        (ROOT / "policies" / "capability_profiles.json").read_text(encoding="utf-8")
    )
    assert payload["profiles"]
    assert all(
        profile["capability_ceiling"] is False
        for profile in payload["profiles"].values()
    )


def test_compatibility_and_broader_profiles_can_coexist() -> None:
    compatibility = load_capability_profile("conversation.phase3.compatibility")
    integrated = load_capability_profile("conversation.integrated")
    assert compatibility.section("capabilities")["tool_calling_allowed"] is False
    assert integrated.section("capabilities")["tool_calling_allowed"] is True


def test_a5_mission_can_activate_evolution_profile() -> None:
    runtime = CapabilityRuntime(
        MissionAuthority(
            mission_id="evolution-test",
            autonomy_class="A5",
            profile_ids=("evolution.a5",),
        )
    )
    decision = runtime.decide(
        profile_id="evolution.a5",
        capability="self_modify_code",
        required_autonomy_class="A5",
    )
    assert decision.allowed is True
