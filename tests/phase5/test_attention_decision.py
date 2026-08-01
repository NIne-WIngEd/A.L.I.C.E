from dataclasses import replace

import pytest

from cognitive_kernel import (
    AttentionDecision,
    AttentionRankEntry,
    CognitiveKernelContractError,
)
from attention_workspace_helpers import decision, digest, entry, provenance, scope


def test_explainable_attention_decision_is_tamper_evident():
    entries = (
        entry(key="pinned", reference="node-pinned", rank=1, host_override="pinned"),
        entry(key="normal", reference="node-normal", rank=2),
        entry(
            key="background",
            reference="node-background",
            rank=3,
            selected=False,
            host_override="background",
            suppression_reason="host_background",
        ),
    )
    receipt = decision(entries, limit=2)
    assert [value.reference_id for value in receipt.selected_entries()] == [
        "node-pinned",
        "node-normal",
    ]
    assert receipt.metadata_record()["candidate_snapshot_sha256"]
    with pytest.raises(CognitiveKernelContractError):
        replace(receipt, visibility_limit=1).validate()


def test_protected_interrupt_cannot_be_suppressed_or_backgrounded():
    base = dict(
        entry_key="security",
        scope=scope(),
        reference_id="security-alert-1",
        subject_type="security_interrupt",
        observed_at="2026-08-01T10:00:00Z",
        state_digest=digest("security"),
        priority_class="protected_interrupt",
        rank=1,
        score=1.0,
        interruption_cost=0.0,
        protected_interrupt_reason="security_breach",
        reason_codes=("protected_interrupt", "security_breach"),
    )
    with pytest.raises(CognitiveKernelContractError):
        AttentionRankEntry.create(
            **base,
            selected=False,
            suppression_reason="visibility_limit",
        )
    with pytest.raises(CognitiveKernelContractError):
        AttentionRankEntry.create(
            **base,
            selected=True,
            host_override="background",
        )


def test_commercial_or_engagement_ranking_reason_is_rejected():
    with pytest.raises(CognitiveKernelContractError):
        AttentionRankEntry.create(
            entry_key="bad",
            scope=scope(),
            reference_id="node-1",
            subject_type="mission_node",
            mission_id="mission-1",
            node_id="node-1",
            observed_at="2026-08-01T10:00:00Z",
            state_digest=digest("bad"),
            priority_class="host_engaged",
            rank=1,
            score=0.8,
            interruption_cost=0.2,
            selected=True,
            reason_codes=("engagement_maximization",),
        )


def test_decision_rejects_cross_host_entries():
    other = entry(product="friday", host="host-b")
    with pytest.raises(CognitiveKernelContractError):
        AttentionDecision.create(
            decision_key="cross-host",
            scope=scope(),
            decided_at="2026-08-01T10:10:00Z",
            visibility_limit=2,
            interruption_preference="allow",
            focus_mode="automatic",
            layout_stability_weight=0.5,
            entries=(other,),
            provenance=provenance(),
        )


def test_protected_interrupts_may_exceed_nonprotected_visibility_limit():
    protected = AttentionRankEntry.create(
        entry_key="security",
        scope=scope(),
        reference_id="security-alert-1",
        subject_type="security_interrupt",
        observed_at="2026-08-01T10:00:00Z",
        state_digest=digest("security"),
        priority_class="protected_interrupt",
        rank=1,
        score=1.0,
        interruption_cost=0.0,
        protected_interrupt_reason="security_breach",
        selected=True,
        reason_codes=("protected_interrupt", "security_breach"),
    )
    normal = entry(key="normal", reference="node-normal", rank=2)
    receipt = decision((protected, normal), limit=1)
    assert len(receipt.selected_entries()) == 2
