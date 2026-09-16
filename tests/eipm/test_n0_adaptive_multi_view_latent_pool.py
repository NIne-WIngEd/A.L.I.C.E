from __future__ import annotations

import torch

from alice_personality.n0.adaptive_multi_view_latent_pool import (
    AdaptiveMultiViewLatentPool,
    AdaptiveMultiViewLatentPoolConfig,
)


def make_inputs(*, batch: int = 2, views: int = 3, tokens: int = 4, semantic: int = 32):
    contextualized = [torch.randn(batch, tokens, semantic) for _ in range(views)]
    source = [value.clone() for value in contextualized]
    masks = [torch.ones(batch, tokens, dtype=torch.bool) for _ in range(views)]
    query = torch.randn(batch, semantic)
    reliability = torch.full((batch, views), 0.8)
    routing = torch.full((batch, views), 1.0 / views)
    return contextualized, source, masks, query, reliability, routing


def make_model(*, views: int = 3) -> AdaptiveMultiViewLatentPool:
    return AdaptiveMultiViewLatentPool(
        AdaptiveMultiViewLatentPoolConfig(
            semantic_size=32,
            latent_size=32,
            num_slots=6,
            num_layers=2,
            num_heads=4,
            feedforward_size=96,
            num_views=views,
        )
    )


def test_outputs_keep_multiple_slots_first_class() -> None:
    torch.manual_seed(7)
    model = make_model()
    contextualized, source, masks, query, reliability, routing = make_inputs()
    output = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    assert output["latent_slots"].shape == (2, 6, 32)
    assert output["slot_weights"].shape == (2, 6)
    assert output["pooled_state"].shape == (2, 32)
    assert output["view_attention_mass"].shape == (2, 6, 3)
    assert output["channel_attention_mass"].shape == (2, 6, 2)
    assert torch.allclose(output["slot_weights"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(output["view_attention_mass"].sum(dim=-1), torch.ones(2, 6), atol=1e-5)
    assert torch.allclose(output["channel_attention_mass"].sum(dim=-1), torch.ones(2, 6), atol=1e-5)


def test_missing_view_is_masked_and_cannot_smuggle_token_content() -> None:
    torch.manual_seed(11)
    model = make_model().eval()
    contextualized, source, masks, query, reliability, routing = make_inputs(batch=1)
    masks[1].zero_()
    reliability[:, 1] = 0.0
    routing[:] = torch.tensor([[0.45, 0.0, 0.55]])
    first = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    contextualized[1] = torch.randn_like(contextualized[1]) * 1000.0
    source[1] = torch.randn_like(source[1]) * 1000.0
    second = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    assert torch.allclose(first["latent_slots"], second["latent_slots"], atol=1e-6)
    assert torch.allclose(first["view_attention_mass"][..., 1], torch.zeros(1, 6), atol=1e-7)


def test_query_conditions_latent_state() -> None:
    torch.manual_seed(13)
    model = make_model().eval()
    contextualized, source, masks, query, reliability, routing = make_inputs(batch=1)
    first = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    second = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=-query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    assert not torch.allclose(first["latent_slots"], second["latent_slots"], atol=1e-5)


def test_exact_source_channel_can_change_latent_readout() -> None:
    torch.manual_seed(17)
    model = make_model().eval()
    contextualized, source, masks, query, reliability, routing = make_inputs(batch=1)
    first = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    source[2] = source[2] + 3.0
    second = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    assert not torch.allclose(first["latent_slots"], second["latent_slots"], atol=1e-5)


def test_expanded_view_checkpoint_uses_same_pool_family() -> None:
    torch.manual_seed(19)
    model = make_model(views=4)
    contextualized, source, masks, query, reliability, routing = make_inputs(views=4)
    output = model(
        contextualized_view_tokens=contextualized,
        source_view_tokens=source,
        view_valid_masks=masks,
        query_semantic=query,
        view_reliability=reliability,
        fusion_view_weights=routing,
    )
    assert output["view_attention_mass"].shape[-1] == 4
    assert model.parameter_report()["view_count_ceiling"] is None
    assert model.parameter_report()["slot_count_ceiling"] is None


def test_parameter_report_is_descriptive_not_capacity_gate() -> None:
    report = make_model().parameter_report()
    assert report["total_parameters"] > 0
    assert report["hard_parameter_ceiling"] is None
    assert report["primary_output_is_multi_slot"] is True
    assert report["single_vector_is_convenience_readout_only"] is True
    assert report["routing_weights_are_features_not_probability_targets"] is True
