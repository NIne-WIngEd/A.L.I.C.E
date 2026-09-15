from __future__ import annotations

import torch

from alice_personality.n0.evidence_graph import (
    EvidenceGraphConfig,
    EvidenceGraphEncoder,
    EvidenceRelationType,
    relation_type_id,
)


def tiny_model() -> EvidenceGraphEncoder:
    torch.manual_seed(11)
    model = EvidenceGraphEncoder(
        EvidenceGraphConfig(
            semantic_size=32,
            graph_size=16,
            num_relation_types=16,
            graph_layers=1,
            max_fields=16,
            max_edges=16,
        )
    )
    model.eval()
    return model


def make_fields(*, batch: int = 1, fields: int = 5):
    torch.manual_seed(7)
    return {
        "field_states": torch.randn(batch, fields, 32),
        "valid_mask": torch.ones(batch, fields, dtype=torch.bool),
    }


def empty_graph(*, batch: int = 1, edges: int = 1):
    return {
        "edge_index": torch.zeros(batch, edges, 2, dtype=torch.long),
        "edge_type_ids": torch.zeros(batch, edges, dtype=torch.long),
        "edge_confidence": torch.zeros(batch, edges, 1),
        "edge_valid_mask": torch.zeros(batch, edges, dtype=torch.bool),
    }


def graph_edge(source: int, target: int, relation_type: EvidenceRelationType):
    return {
        "edge_index": torch.tensor([[[source, target]]], dtype=torch.long),
        "edge_type_ids": torch.tensor([[int(relation_type)]], dtype=torch.long),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def test_relation_vocabulary_matches_memory_core_lifecycle() -> None:
    assert relation_type_id("corrects") == int(EvidenceRelationType.CORRECTS)
    assert relation_type_id("supersedes") == int(EvidenceRelationType.SUPERSEDES)
    assert relation_type_id("conflicts_with") == int(EvidenceRelationType.CONFLICTS_WITH)


def test_consistent_node_permutation_preserves_graph_read() -> None:
    model = tiny_model()
    fields = make_fields()
    graph = {
        "edge_index": torch.tensor([[[0, 1], [2, 1], [3, 4]]], dtype=torch.long),
        "edge_type_ids": torch.tensor(
            [[
                int(EvidenceRelationType.SUPERSEDES),
                int(EvidenceRelationType.SUPPORTS),
                int(EvidenceRelationType.CONFLICTS_WITH),
            ]],
            dtype=torch.long,
        ),
        "edge_confidence": torch.ones(1, 3, 1),
        "edge_valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    query = torch.randn(1, 32)
    base_weights = torch.softmax(torch.randn(1, 5), dim=-1)

    order = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    permuted_fields = {
        "field_states": fields["field_states"][:, order, :],
        "valid_mask": fields["valid_mask"][:, order],
    }
    permuted_graph = {key: value.clone() for key, value in graph.items()}
    permuted_graph["edge_index"] = inverse[graph["edge_index"]]

    with torch.inference_mode():
        original = model(
            **fields,
            **graph,
            query_semantic=query,
            base_field_weights=base_weights,
        )
        permuted = model(
            **permuted_fields,
            **permuted_graph,
            query_semantic=query,
            base_field_weights=base_weights[:, order],
        )

    torch.testing.assert_close(original["pooled_state"], permuted["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        original["field_states"][:, order, :],
        permuted["field_states"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["field_weights"][:, order],
        permuted["field_weights"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["relation_status_bias"][:, order],
        permuted["relation_status_bias"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_supersession_penalizes_historical_target_only() -> None:
    model = tiny_model()
    fields = make_fields(fields=3)

    with torch.inference_mode():
        result = model(
            **fields,
            **graph_edge(0, 1, EvidenceRelationType.SUPERSEDES),
        )

    torch.testing.assert_close(
        result["relation_status_bias"],
        torch.tensor([[0.0, -1.0, 0.0]]),
    )
    assert float(result["relation_update_norm"][0, 1]) > 0.0
    assert float(result["relation_update_norm"][0, 0]) == 0.0
    assert float(result["relation_update_norm"][0, 2]) == 0.0


def test_conflict_is_symmetric_even_when_storage_edge_is_canonicalized() -> None:
    model = tiny_model()
    fields = make_fields(fields=3)

    with torch.inference_mode():
        forward = model(
            **fields,
            **graph_edge(0, 1, EvidenceRelationType.CONFLICTS_WITH),
        )
        reverse = model(
            **fields,
            **graph_edge(1, 0, EvidenceRelationType.CONFLICTS_WITH),
        )

    torch.testing.assert_close(forward["field_states"], reverse["field_states"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(forward["pooled_state"], reverse["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        forward["relation_status_bias"],
        torch.tensor([[-0.35, -0.35, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_directed_relation_leaves_source_and_disconnected_nodes_unchanged() -> None:
    model = tiny_model()
    fields = make_fields(fields=3)

    with torch.inference_mode():
        baseline = model(**fields, **empty_graph())
        related = model(
            **fields,
            **graph_edge(0, 1, EvidenceRelationType.SUPPORTS),
        )

    torch.testing.assert_close(
        baseline["field_states"][:, 0, :],
        related["field_states"][:, 0, :],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        baseline["field_states"][:, 2, :],
        related["field_states"][:, 2, :],
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.allclose(
        baseline["field_states"][:, 1, :],
        related["field_states"][:, 1, :],
    )


def test_query_changes_evidence_pooling_without_reencoding_fields() -> None:
    model = tiny_model()
    fields = make_fields()
    graph = empty_graph()

    with torch.inference_mode():
        first = model(
            **fields,
            **graph,
            query_semantic=torch.zeros(1, 32),
        )
        second = model(
            **fields,
            **graph,
            query_semantic=torch.arange(32, dtype=torch.float32).unsqueeze(0),
        )

    assert not torch.allclose(first["field_weights"], second["field_weights"], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first["pooled_state"], second["pooled_state"], atol=1e-6, rtol=1e-6)


def test_invalid_active_edge_fails_closed() -> None:
    model = tiny_model()
    fields = make_fields(fields=3)
    bad = graph_edge(0, 4, EvidenceRelationType.SUPPORTS)

    try:
        model(**fields, **bad)
    except ValueError as exc:
        assert "outside the valid range" in str(exc)
    else:
        raise AssertionError("out-of-range active edge must be rejected")


def test_sidecar_stays_compact_and_does_not_grow_semantic_core() -> None:
    report = EvidenceGraphEncoder().parameter_report()
    assert report["total_parameters"] < 1_000_000
    assert report["semantic_core_parameter_growth"] == 0
    assert report["private_identity_parameters"] == 0
    assert report["position_embeddings"] == 0
