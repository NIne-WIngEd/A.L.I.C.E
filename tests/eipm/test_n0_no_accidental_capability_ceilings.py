from __future__ import annotations

import torch

from alice_personality.n0.cross_context_fusion import (
    CrossContextFusion,
    CrossContextFusionConfig,
)
from alice_personality.n0.evidence_graph import EvidenceGraphConfig, EvidenceGraphEncoder
from alice_personality.n0.evidence_view_adapter import (
    EvidenceViewAdapter,
    EvidenceViewAdapterConfig,
)
from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model


def test_structured_state_legacy_64_field_hint_is_not_a_runtime_ceiling() -> None:
    config = StructuredStateConfig(
        semantic_size=8,
        state_size=8,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        max_fields=64,
    )
    model = StructuredStateEncoder(config).eval()
    fields = 65
    with torch.inference_mode():
        out = model(
            semantic_values=torch.randn(1, fields, 8),
            field_type_ids=torch.ones(1, fields, dtype=torch.long),
            provenance_ids=torch.ones(1, fields, dtype=torch.long),
            relation_role_ids=torch.ones(1, fields, dtype=torch.long),
            temporal_scope_ids=torch.ones(1, fields, dtype=torch.long),
            confidence=torch.ones(1, fields, 1),
            missing_mask=torch.zeros(1, fields, dtype=torch.bool),
            valid_mask=torch.ones(1, fields, dtype=torch.bool),
        )
    assert out["field_states"].shape == (1, fields, 8)
    assert model.parameter_report()["field_count_limit"] is None


def test_evidence_adapter_legacy_64_field_hint_is_not_a_runtime_ceiling() -> None:
    config = EvidenceViewAdapterConfig(
        semantic_size=8,
        adapter_size=8,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        max_fields=64,
    )
    model = EvidenceViewAdapter(config).eval()
    fields = 65
    with torch.inference_mode():
        out = model(
            parent_field_states=torch.randn(1, fields, 8),
            valid_mask=torch.ones(1, fields, dtype=torch.bool),
            query_semantic=torch.randn(1, 8),
            parent_field_weights=torch.full((1, fields), 1.0 / fields),
        )
    assert out["field_states"].shape == (1, fields, 8)
    assert model.parameter_report()["field_count_limit"] is None


def test_evidence_graph_legacy_node_and_edge_hints_are_not_runtime_ceilings() -> None:
    config = EvidenceGraphConfig(
        semantic_size=8,
        graph_size=8,
        graph_layers=1,
        max_fields=64,
        max_edges=256,
    )
    model = EvidenceGraphEncoder(config).eval()
    fields = 65
    edges = 257
    edge_number = torch.arange(edges)
    source = edge_number.remainder(fields)
    target = (source + 1).remainder(fields)
    edge_index = torch.stack([source, target], dim=-1).unsqueeze(0)
    with torch.inference_mode():
        out = model(
            field_states=torch.randn(1, fields, 8),
            valid_mask=torch.ones(1, fields, dtype=torch.bool),
            edge_index=edge_index,
            edge_type_ids=torch.ones(1, edges, dtype=torch.long),
            edge_confidence=torch.ones(1, edges, 1),
            edge_valid_mask=torch.ones(1, edges, dtype=torch.bool),
            query_semantic=torch.randn(1, 8),
            base_field_weights=torch.full((1, fields), 1.0 / fields),
        )
    assert out["field_states"].shape == (1, fields, 8)
    report = model.parameter_report()
    assert report["field_count_limit"] is None
    assert report["edge_count_limit"] is None


def test_fusion_view_count_is_migratable_not_hard_coded_to_three() -> None:
    config = CrossContextFusionConfig(
        semantic_size=8,
        fusion_size=8,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        num_views=5,
    )
    model = CrossContextFusion(config).eval()
    view_tokens = [torch.randn(1, 2 + index, 8) for index in range(5)]
    masks = [torch.ones(1, tokens.size(1), dtype=torch.bool) for tokens in view_tokens]
    with torch.inference_mode():
        out = model.forward_views(
            view_tokens=view_tokens,
            view_valid_masks=masks,
            query_semantic=torch.randn(1, 8),
            view_reliability=torch.full((1, 5), 0.8),
        )
    assert len(out["contextualized_view_tokens"]) == 5
    assert out["view_weights"].shape == (1, 5)
    assert model.parameter_report()["view_count_ceiling"] is None


def test_fusion_parent_token_views_are_exact_at_initialization() -> None:
    config = CrossContextFusionConfig(
        semantic_size=8,
        fusion_size=8,
        num_layers=1,
        num_heads=2,
        feedforward_size=16,
        num_views=3,
    )
    model = CrossContextFusion(config).eval()
    views = [torch.randn(1, 3, 8) for _ in range(3)]
    masks = [torch.ones(1, 3, dtype=torch.bool) for _ in range(3)]
    with torch.inference_mode():
        out = model.forward_views(
            view_tokens=views,
            view_valid_masks=masks,
            query_semantic=torch.randn(1, 8),
            view_reliability=torch.ones(1, 3),
        )
    for source, contextualized in zip(views, out["contextualized_view_tokens"]):
        assert torch.equal(source, contextualized)


def test_semantic_model_exposes_token_level_representation_api() -> None:
    assert callable(getattr(AliceN0V02Model, "encode_tokens", None))
    assert callable(getattr(AliceN0V02Model, "encode_views", None))
