from __future__ import annotations

from pathlib import Path

from cognitive_kernel import load_cognitive_kernel_tier_transition_policy


def test_tier_transition_policy_loads_from_repository() -> None:
    root = Path(__file__).resolve().parents[2]
    policy = load_cognitive_kernel_tier_transition_policy(
        repository_root=root
    )
    assert policy.version == "0.9.0"
    assert policy.milestone == "P5.1d"
    assert policy.invariants["physical_tier_movement_implemented"] is True
    assert policy.invariants["payload_deletion_implemented"] is False
    assert policy.invariants["automatic_retention_implemented"] is False
    assert policy.capability_ceiling is False
