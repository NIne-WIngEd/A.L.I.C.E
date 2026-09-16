from __future__ import annotations

import torch

from alice_personality.n0.adaptive_multi_view_latent_pool_objectives import (
    adaptive_latent_pool_objective,
    channel_coverage_loss,
    counterfactual_target_margin_loss,
    multi_slot_semantic_alignment_loss,
    slot_diversity_loss,
    view_coverage_loss,
)


def test_multi_slot_alignment_allows_any_slot_to_carry_target() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    good = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    bad = torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    assert multi_slot_semantic_alignment_loss(good, target) < multi_slot_semantic_alignment_loss(bad, target)


def test_diversity_penalizes_collapsed_slots_more_than_distinct_slots() -> None:
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    distinct = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    assert slot_diversity_loss(collapsed) > slot_diversity_loss(distinct)


def test_view_coverage_ignores_missing_view() -> None:
    attention = torch.tensor([[[0.50, 0.00, 0.50], [0.45, 0.00, 0.55]]])
    available = torch.tensor([[True, False, True]])
    assert view_coverage_loss(attention, available).item() == 0.0


def test_channel_coverage_requires_both_channels_to_remain_reachable() -> None:
    balanced = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    collapsed = torch.tensor([[[0.99, 0.01], [0.98, 0.02]]])
    assert channel_coverage_loss(balanced) < channel_coverage_loss(collapsed)


def test_counterfactual_margin_rewards_correct_evidence_state() -> None:
    target = torch.tensor([[1.0, 0.0]])
    correct = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    corrupted = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])
    assert counterfactual_target_margin_loss(correct, corrupted, target).item() == 0.0


def test_joint_objective_returns_named_components() -> None:
    torch.manual_seed(23)
    slots = torch.randn(2, 4, 8)
    pooled = torch.randn(2, 8)
    target = torch.randn(2, 8)
    view_attention = torch.softmax(torch.randn(2, 4, 3), dim=-1)
    available = torch.ones(2, 3, dtype=torch.bool)
    channel_attention = torch.softmax(torch.randn(2, 4, 2), dim=-1)
    result = adaptive_latent_pool_objective(
        latent_slots=slots,
        pooled_state=pooled,
        target_semantic=target,
        view_attention_mass=view_attention,
        view_available=available,
        channel_attention_mass=channel_attention,
    )
    assert set(result) == {
        "total",
        "semantic_alignment",
        "pooled_alignment",
        "slot_diversity",
        "view_coverage",
        "channel_coverage",
        "counterfactual_margin",
    }
    assert torch.isfinite(result["total"])
