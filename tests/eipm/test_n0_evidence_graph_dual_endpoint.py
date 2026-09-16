from __future__ import annotations

import torch
from torch import nn

from alice_personality.n0.evidence_graph import (
    EvidenceGraphConfig,
    EvidenceGraphEncoder,
    EvidenceRelationType,
)
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)


def config() -> EvidenceGraphConfig:
    return EvidenceGraphConfig(
        semantic_size=32,
        graph_size=16,
        num_relation_types=16,
        graph_layers=1,
        max_fields=8,
        max_edges=8,
        base_weight_scale=0.0,
    )


def inputs(source: int = 0, target: int = 1, relation: EvidenceRelationType = EvidenceRelationType.SUPPORTS):
    torch.manual_seed(71)
    return {
        "field_states": torch.randn(1, 3, 32),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "edge_index": torch.tensor([[[source, target]]], dtype=torch.long),
        "edge_type_ids": torch.tensor([[int(relation)]], dtype=torch.long),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "query_semantic": torch.randn(1, 32),
    }


def zero_module(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_zero_initialized_source_role_preserves_v02_parent_behavior() -> None:
    torch.manual_seed(5)
    parent = EvidenceGraphEncoder(config()).eval()
    child = DualEndpointEvidenceGraphEncoder(config()).eval()
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert unexpected == []
    assert missing
    assert all(name.startswith("source_relation_pool_mlp.") for name in missing)

    batch = inputs()
    with torch.inference_mode():
        parent_out = parent(**batch)
        child_out = child(**batch)

    torch.testing.assert_close(
        parent_out["field_weights"], child_out["field_weights"], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        parent_out["pooled_state"], child_out["pooled_state"], atol=1e-6, rtol=1e-6
    )


def test_directed_relation_can_score_source_and_target_independently() -> None:
    model = DualEndpointEvidenceGraphEncoder(config()).eval()
    zero_module(model.directed_relation_pool_mlp)
    zero_module(model.conflict_pool_mlp)
    zero_module(model.source_relation_pool_mlp)
    final = model.source_relation_pool_mlp[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.bias.fill_(1.0)

    with torch.inference_mode():
        out = model(**inputs())

    torch.testing.assert_close(
        out["relation_status_bias"],
        torch.tensor([[1.0, 0.0, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_conflict_remains_symmetric_and_ignores_directed_source_head() -> None:
    model = DualEndpointEvidenceGraphEncoder(config()).eval()
    zero_module(model.directed_relation_pool_mlp)
    zero_module(model.source_relation_pool_mlp)
    zero_module(model.conflict_pool_mlp)
    source_final = model.source_relation_pool_mlp[-1]
    conflict_final = model.conflict_pool_mlp[-1]
    assert isinstance(source_final, nn.Linear)
    assert isinstance(conflict_final, nn.Linear)
    with torch.no_grad():
        source_final.bias.fill_(3.0)
        conflict_final.bias.fill_(0.5)

    with torch.inference_mode():
        forward = model(**inputs(0, 1, EvidenceRelationType.CONFLICTS_WITH))
        reverse = model(**inputs(1, 0, EvidenceRelationType.CONFLICTS_WITH))

    expected = torch.tensor([[0.5, 0.5, 0.0]])
    torch.testing.assert_close(forward["relation_status_bias"], expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(reverse["relation_status_bias"], expected, atol=1e-6, rtol=1e-6)


def test_parameter_report_is_descriptive_not_a_capacity_gate() -> None:
    report = DualEndpointEvidenceGraphEncoder().parameter_report()
    assert report["total_parameters"] > 0
    assert report["trainable_parameters"] == report["total_parameters"]
    assert report["hard_parameter_ceiling"] is None
    assert "distinct_source_and_target" in report["relation_pooling"]
