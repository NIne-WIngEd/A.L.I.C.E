from __future__ import annotations

import pytest
import torch

from alice_personality.n0.cross_context_fusion import (
    CrossContextFusion,
    CrossContextFusionConfig,
)


def inputs(batch: int = 2, semantic: int = 5, structured: int = 4, evidence: int = 3):
    torch.manual_seed(20260916)
    return {
        "semantic_tokens": torch.randn(batch, semantic, 640),
        "semantic_valid_mask": torch.ones(batch, semantic, dtype=torch.bool),
        "structured_tokens": torch.randn(batch, structured, 640),
        "structured_valid_mask": torch.ones(batch, structured, dtype=torch.bool),
        "evidence_tokens": torch.randn(batch, evidence, 640),
        "evidence_valid_mask": torch.ones(batch, evidence, dtype=torch.bool),
        "query_semantic": torch.randn(batch, 640),
        "view_reliability": torch.full((batch, 3), 0.8),
    }


def test_cross_context_fusion_shapes_and_view_preservation() -> None:
    model = CrossContextFusion().eval()
    batch = inputs()
    with torch.inference_mode():
        out = model(**batch)
    assert out["semantic_tokens"].shape == (2, 5, 640)
    assert out["structured_tokens"].shape == (2, 4, 640)
    assert out["evidence_tokens"].shape == (2, 3, 640)
    assert out["view_summaries"].shape == (2, 3, 640)
    assert out["view_weights"].shape == (2, 3)
    assert out["fused_state"].shape == (2, 640)
    assert out["cross_view_cosine"].shape == (2, 3, 3)
    assert out["cross_gate_means"].shape == (2, 4, 3)
    assert torch.allclose(out["view_weights"].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_missing_evidence_view_is_safe_and_receives_zero_weight() -> None:
    model = CrossContextFusion().eval()
    batch = inputs()
    batch["evidence_valid_mask"][:] = False
    batch["view_reliability"][:, 2] = 0.0
    with torch.inference_mode():
        out = model(**batch)
    assert torch.isfinite(out["fused_state"]).all()
    assert torch.equal(out["view_available"][:, 2], torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(out["view_weights"][:, 2], torch.zeros(2), atol=1e-7)
    assert torch.allclose(out["cross_gate_means"][:, :, 2], torch.zeros(2, 4), atol=1e-7)


def test_structured_field_permutation_preserves_fused_state() -> None:
    model = CrossContextFusion().eval()
    batch = inputs(batch=1)
    with torch.inference_mode():
        original = model(**batch)["fused_state"]
        order = torch.tensor([2, 0, 3, 1])
        batch["structured_tokens"] = batch["structured_tokens"].index_select(1, order)
        batch["structured_valid_mask"] = batch["structured_valid_mask"].index_select(1, order)
        permuted = model(**batch)["fused_state"]
    assert torch.allclose(original, permuted, atol=3e-5, rtol=3e-5)


def test_evidence_field_permutation_preserves_fused_state() -> None:
    model = CrossContextFusion().eval()
    batch = inputs(batch=1)
    with torch.inference_mode():
        original = model(**batch)["fused_state"]
        order = torch.tensor([2, 0, 1])
        batch["evidence_tokens"] = batch["evidence_tokens"].index_select(1, order)
        batch["evidence_valid_mask"] = batch["evidence_valid_mask"].index_select(1, order)
        permuted = model(**batch)["fused_state"]
    assert torch.allclose(original, permuted, atol=3e-5, rtol=3e-5)


def test_reliability_prior_can_route_between_equivalent_views() -> None:
    config = CrossContextFusionConfig(reliability_prior_scale=4.0)
    model = CrossContextFusion(config).eval()
    with torch.no_grad():
        for parameter in model.view_score.parameters():
            parameter.zero_()
        for parameter in model.reliability_projection.parameters():
            parameter.zero_()
    batch = inputs(batch=1)
    shared = torch.randn(1, 4, 640)
    mask = torch.ones(1, 4, dtype=torch.bool)
    batch["semantic_tokens"] = shared
    batch["semantic_valid_mask"] = mask
    batch["structured_tokens"] = shared.clone()
    batch["structured_valid_mask"] = mask.clone()
    batch["evidence_tokens"] = shared.clone()
    batch["evidence_valid_mask"] = mask.clone()
    batch["view_reliability"] = torch.tensor([[0.95, 0.20, 0.10]])
    with torch.inference_mode():
        weights = model(**batch)["view_weights"][0]
    assert weights[0] > weights[1] > weights[2]


def test_all_missing_views_are_rejected() -> None:
    model = CrossContextFusion().eval()
    batch = inputs(batch=1)
    batch["semantic_valid_mask"][:] = False
    batch["structured_valid_mask"][:] = False
    batch["evidence_valid_mask"][:] = False
    with pytest.raises(ValueError, match="at least one fusion view"):
        model(**batch)


def test_parameter_report_declares_full_scale_frontier_architecture() -> None:
    report = CrossContextFusion().parameter_report()
    assert report["total_parameters"] > 0
    assert report["hard_parameter_ceiling"] is None
    assert report["view_count_ceiling"] is None
    assert report["private_identity_parameters"] == 0
    assert report["preserves_contextualized_views"] is True
    assert report["parent_token_views_exact_at_initialization"] is True
    assert report["full_scale_n0_candidate"] is True
    assert report["reduced_pilot_model"] is False
    assert report["bidirectional_cross_attention"] is True
    assert report["gated_cross_view_exchange"] is True
    assert report["fusion_stages"] == 4
    assert report["fusion_family"] == "multi_stream_self_refinement_plus_gated_bidirectional_cross_attention"
