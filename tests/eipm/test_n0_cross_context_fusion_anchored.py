from __future__ import annotations

import torch

from alice_personality.n0.cross_context_fusion import (
    CrossContextFusion,
    CrossContextFusionConfig,
)
from alice_personality.n0.cross_context_fusion_anchored import (
    SourceAnchoredCrossContextFusion,
)


def _inputs(batch: int = 2, width: int = 32) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260916)
    semantic = torch.randn(batch, 5, width)
    structured = torch.randn(batch, 4, width)
    evidence = torch.randn(batch, 4, width)
    semantic_mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    structured_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    evidence_mask = torch.tensor([[True, True, True, True], [False, False, False, False]])
    query = torch.randn(batch, width)
    reliability = torch.tensor([[0.9, 0.8, 0.95], [0.9, 0.8, 0.0]], dtype=torch.float32)
    source = torch.randn(batch, 3, width)
    return {
        "semantic_tokens": semantic,
        "semantic_valid_mask": semantic_mask,
        "structured_tokens": structured,
        "structured_valid_mask": structured_mask,
        "evidence_tokens": evidence,
        "evidence_valid_mask": evidence_mask,
        "query_semantic": query,
        "view_reliability": reliability,
        "source_view_summaries": source,
    }


def _config() -> CrossContextFusionConfig:
    return CrossContextFusionConfig(
        semantic_size=32,
        fusion_size=32,
        num_layers=2,
        num_heads=4,
        feedforward_size=96,
        dropout=0.0,
        num_views=3,
    )


def test_anchored_migration_loads_base_checkpoint_without_parameter_drift() -> None:
    base = CrossContextFusion(_config())
    anchored = SourceAnchoredCrossContextFusion(_config())
    anchored.load_state_dict(base.state_dict(), strict=True)
    assert anchored.parameter_report()["total_parameters"] == base.parameter_report()["total_parameters"]
    assert anchored.parameter_report()["source_anchor_parameter_cost"] == 0


def test_anchored_migration_preserves_existing_fusion_behavior() -> None:
    base = CrossContextFusion(_config()).eval()
    anchored = SourceAnchoredCrossContextFusion(_config()).eval()
    anchored.load_state_dict(base.state_dict(), strict=True)
    inputs = _inputs()

    with torch.inference_mode():
        base_out = base(
            semantic_tokens=inputs["semantic_tokens"],
            semantic_valid_mask=inputs["semantic_valid_mask"],
            structured_tokens=inputs["structured_tokens"],
            structured_valid_mask=inputs["structured_valid_mask"],
            evidence_tokens=inputs["evidence_tokens"],
            evidence_valid_mask=inputs["evidence_valid_mask"],
            query_semantic=inputs["query_semantic"],
            view_reliability=inputs["view_reliability"],
        )
        anchored_out = anchored(**inputs)

    assert torch.allclose(anchored_out["view_weights"], base_out["view_weights"], atol=1e-6)
    assert torch.allclose(anchored_out["fused_state"], base_out["fused_state"], atol=1e-6)
    assert torch.allclose(
        anchored_out["contextualized_view_summaries"],
        base_out["view_summaries"],
        atol=1e-6,
    )


def test_source_anchors_are_exact_for_available_views_and_zero_for_missing_views() -> None:
    model = SourceAnchoredCrossContextFusion(_config()).eval()
    inputs = _inputs()
    with torch.inference_mode():
        out = model(**inputs)

    expected = inputs["source_view_summaries"].clone()
    expected[1, 2] = 0.0
    assert torch.equal(out["source_view_summaries"], expected)


def test_source_anchor_fallback_uses_each_parent_token_stream_without_count_ceiling() -> None:
    model = SourceAnchoredCrossContextFusion(_config()).eval()
    inputs = _inputs()
    inputs.pop("source_view_summaries")
    with torch.inference_mode():
        out = model(**inputs)
    assert out["source_view_summaries"].shape == (2, 3, 32)
    assert out["source_view_summaries"][1, 2].abs().sum().item() == 0.0


def test_anchored_report_keeps_capability_ceiling_open() -> None:
    report = SourceAnchoredCrossContextFusion(_config()).parameter_report()
    assert report["hard_parameter_ceiling"] is None
    assert report["view_count_ceiling"] is None
    assert report["source_anchor_capacity_ceiling"] is None
    assert report["full_scale_n0_candidate"] is True
    assert report["reduced_pilot_model"] is False
