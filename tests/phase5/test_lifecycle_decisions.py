from __future__ import annotations

import pytest

from cognitive_kernel import (
    CognitiveKernelContractError,
    LifecycleDecision,
    load_cognitive_kernel_lifecycle_policy,
)
from lifecycle_helpers import make_decision


def test_decision_identity_and_round_trip_are_deterministic() -> None:
    first = make_decision()
    second = make_decision()
    assert first == second
    assert first.decision_id.startswith("lifecycle-decision-")
    assert LifecycleDecision.from_metadata_record(
        first.metadata_record()
    ) == first


def test_decision_rejects_unknown_tier() -> None:
    with pytest.raises(CognitiveKernelContractError, match="current_tier"):
        make_decision(current_tier="orbital")


def test_decision_rejects_no_op_transition() -> None:
    with pytest.raises(CognitiveKernelContractError, match="no-op"):
        make_decision(proposed_tier="hot")


def test_retain_is_an_explicit_record_not_a_transition() -> None:
    decision = make_decision(
        decision_type="retain",
        current_tier="hot",
        proposed_tier="hot",
        authority_level="none",
        authority_decision_id=None,
        outcome="recorded",
    )
    assert decision.current_tier == decision.proposed_tier
    assert "payload" not in decision.metadata_record()


def test_lifecycle_policy_loads_from_repository() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    policy = load_cognitive_kernel_lifecycle_policy(repository_root=root)
    assert policy.version == "0.8.0"
    assert policy.milestone == "P5.1c"
    assert policy.invariants["payload_deletion_implemented"] is False
