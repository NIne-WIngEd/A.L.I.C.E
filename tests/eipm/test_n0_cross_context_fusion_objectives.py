from __future__ import annotations

import pytest
import torch

from alice_personality.n0.cross_context_fusion_objectives import (
    cross_context_fusion_objective,
    disagreement_preservation_loss,
    soft_distribution_cross_entropy,
)


def test_soft_view_routing_accepts_plural_targets() -> None:
    prediction = torch.tensor([[0.48, 0.47, 0.05]], dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    available = torch.tensor([[True, True, False]])
    loss = soft_distribution_cross_entropy(prediction, target, available=available)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_view_routing_rejects_target_mass_on_missing_view() -> None:
    prediction = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float32)
    available = torch.tensor([[True, True, False]])
    with pytest.raises(ValueError, match="available view"):
        soft_distribution_cross_entropy(prediction, target, available=available)


def test_disagreement_loss_is_zero_when_geometry_is_preserved() -> None:
    torch.manual_seed(9)
    source = torch.randn(2, 3, 16)
    available = torch.ones(2, 3, dtype=torch.bool)
    loss = disagreement_preservation_loss(source, source.clone(), available)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_full_fusion_objective_is_finite() -> None:
    torch.manual_seed(12)
    batch = 4
    width = 32
    fused = torch.randn(batch, width)
    target = torch.randn(batch, width)
    summaries = torch.randn(batch, 3, width)
    source = torch.randn(batch, 3, width)
    view_weights = torch.softmax(torch.randn(batch, 3), dim=-1)
    target_views = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    available = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )
    result = cross_context_fusion_objective(
        fused_state=fused,
        semantic_target=target,
        view_weights=view_weights,
        target_view_distribution=target_views,
        contextualized_view_summaries=summaries,
        source_view_targets=source,
        view_available=available,
    )
    assert set(result) == {
        "loss",
        "fused_semantic_loss",
        "view_routing_loss",
        "view_preservation_loss",
        "disagreement_preservation_loss",
    }
    assert all(torch.isfinite(value) for value in result.values())
