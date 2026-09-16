from __future__ import annotations

import torch

from alice_personality.n0.adaptive_multi_view_latent_pool_v0_2 import (
    AdaptiveMultiViewLatentPoolV02,
    AdaptiveMultiViewLatentPoolV02Config,
)


def _inputs(*, missing_evidence: bool = False):
    torch.manual_seed(7)
    views = [torch.randn(2, 4, 8) for _ in range(3)]
    source = [value.clone() for value in views]
    masks = [torch.ones(2, 4, dtype=torch.bool) for _ in range(3)]
    reliability = torch.tensor([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    routing = torch.tensor([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]])
    if missing_evidence:
        masks[2].zero_()
        reliability[:, 2] = 0.0
        routing[:, 2] = 0.0
        routing[:, :2] = 0.5
    return views, source, masks, reliability, routing


def test_competitive_latent_pool_shapes_and_assignment_conservation() -> None:
    config = AdaptiveMultiViewLatentPoolV02Config(
        semantic_size=8,
        latent_size=8,
        num_slots=5,
        num_layers=2,
        num_heads=2,
        feedforward_size=16,
        num_views=3,
    )
    model = AdaptiveMultiViewLatentPoolV02(config).eval()
    views, source, masks, reliability, routing = _inputs()
    with torch.inference_mode():
        out = model(
            contextualized_view_tokens=views,
            source_view_tokens=source,
            view_valid_masks=masks,
            query_semantic=torch.randn(2, 8),
            view_reliability=reliability,
            fusion_view_weights=routing,
        )
    assert out["latent_slots"].shape == (2, 5, 8)
    assert out["slot_weights"].shape == (2, 5)
    assert out["view_attention_mass"].shape == (2, 5, 3)
    assert out["channel_attention_mass"].shape == (2, 5, 2)
    assignment = out["token_slot_assignment"]
    assert torch.allclose(
        assignment.sum(dim=1),
        torch.ones_like(assignment[:, 0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_missing_view_receives_no_latent_attention() -> None:
    config = AdaptiveMultiViewLatentPoolV02Config(
        semantic_size=8,
        latent_size=8,
        num_slots=5,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        num_views=3,
    )
    model = AdaptiveMultiViewLatentPoolV02(config).eval()
    views, source, masks, reliability, routing = _inputs(missing_evidence=True)
    with torch.inference_mode():
        out = model(
            contextualized_view_tokens=views,
            source_view_tokens=source,
            view_valid_masks=masks,
            query_semantic=torch.randn(2, 8),
            view_reliability=reliability,
            fusion_view_weights=routing,
        )
    assert torch.equal(out["view_attention_mass"][:, :, 2], torch.zeros_like(out["view_attention_mass"][:, :, 2]))


def test_checkpoint_slot_count_is_not_a_capability_ceiling() -> None:
    config = AdaptiveMultiViewLatentPoolV02Config(
        semantic_size=8,
        latent_size=8,
        num_slots=17,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        num_views=4,
    )
    model = AdaptiveMultiViewLatentPoolV02(config)
    report = model.parameter_report()
    assert report["instantiated_slots"] == 17
    assert report["instantiated_view_count"] == 4
    assert report["slot_count_ceiling"] is None
    assert report["view_count_ceiling"] is None
    assert report["hard_parameter_ceiling"] is None
    assert report["competitive_cross_attention"] is True
