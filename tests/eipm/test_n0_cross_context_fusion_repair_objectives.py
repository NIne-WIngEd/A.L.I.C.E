from __future__ import annotations

import pytest
import torch

from alice_personality.n0.cross_context_fusion_repair_objectives import (
    FusionRepairObjectiveWeights,
    contextualized_source_alignment_loss,
    cross_context_fusion_repair_objective,
)


def test_alignment_is_zero_for_exact_contextualized_source_match() -> None:
    source = torch.randn(2, 3, 16)
    available = torch.tensor([[True, True, True], [True, False, True]])
    loss = contextualized_source_alignment_loss(source.clone(), source, available)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_alignment_ignores_unavailable_view() -> None:
    source = torch.randn(1, 3, 8)
    contextualized = source.clone()
    contextualized[:, 2] = -source[:, 2]
    available = torch.tensor([[True, True, False]])
    loss = contextualized_source_alignment_loss(contextualized, source, available)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_repair_objective_rejects_target_mass_on_missing_view() -> None:
    fused = torch.randn(1, 8)
    semantic = torch.randn(1, 8)
    prediction = torch.tensor([[0.5, 0.5, 0.0]])
    target = torch.tensor([[0.4, 0.4, 0.2]])
    contextualized = torch.randn(1, 3, 8)
    source = torch.randn(1, 3, 8)
    available = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="available view"):
        cross_context_fusion_repair_objective(
            fused_state=fused,
            semantic_target=semantic,
            view_weights=prediction,
            target_view_distribution=target,
            contextualized_view_summaries=contextualized,
            source_view_summaries=source,
            view_available=available,
        )


def test_repair_weight_policy_keeps_routing_primary_without_capacity_ceiling() -> None:
    weights = FusionRepairObjectiveWeights()
    weights.validate()
    assert weights.view_routing > weights.contextualized_source_alignment
    assert weights.fused_semantic > 0.0
    assert weights.disagreement_geometry > 0.0
