from __future__ import annotations

import torch

from alice_personality.n0.structured_state import (
    EvidenceRelationType,
    StructuredStateConfig,
    StructuredStateEncoder,
    relation_type_id,
)


def make_batch(*, batch: int = 2, fields: int = 5, semantic_size: int = 32):
    torch.manual_seed(7)
    return {
        "semantic_values": torch.randn(batch, fields, semantic_size),
        "field_type_ids": torch.randint(1, 8, (batch, fields)),
        "provenance_ids": torch.randint(1, 5, (batch, fields)),
        "relation_role_ids": torch.randint(1, 8, (batch, fields)),
        "temporal_scope_ids": torch.randint(1, 5, (batch, fields)),
        "confidence": torch.rand(batch, fields, 1),
        "missing_mask": torch.zeros(batch, fields, dtype=torch.bool),
        "valid_mask": torch.ones(batch, fields, dtype=torch.bool),
    }


def make_graph(*, batch: int = 2):
    edge_index = torch.tensor(
        [
            [0, 1],
            [2, 1],
            [3, 4],
        ],
        dtype=torch.long,
    ).unsqueeze(0).repeat(batch, 1, 1)
    edge_type_ids = torch.tensor(
        [
            int(EvidenceRelationType.SUPERSEDES),
            int(EvidenceRelationType.SUPPORTS),
            int(EvidenceRelationType.CONFLICTS_WITH),
        ],
        dtype=torch.long,
    ).unsqueeze(0).repeat(batch, 1)
    return {
        "edge_index": edge_index,
        "edge_type_ids": edge_type_ids,
        "edge_confidence": torch.ones(batch, 3, 1),
        "edge_valid_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def tiny_model() -> StructuredStateEncoder:
    config = StructuredStateConfig(
        semantic_size=32,
        state_size=16,
        num_layers=1,
        num_heads=4,
        feedforward_size=32,
        num_field_types=16,
        num_provenance_classes=8,
        num_relation_roles=16,
        num_relation_types=16,
        num_temporal_scopes=8,
        graph_layers=1,
        dropout=0.0,
        max_fields=16,
        max_edges=32,
    )
    model = StructuredStateEncoder(config)
    model.eval()
    return model


def permute_fields(batch: dict[str, torch.Tensor], order: torch.Tensor):
    return {key: value[:, order, ...] for key, value in batch.items()}


def permute_graph_nodes(
    batch: dict[str, torch.Tensor],
    graph: dict[str, torch.Tensor],
    order: torch.Tensor,
):
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    permuted_batch = permute_fields(batch, order)
    permuted_graph = {key: value.clone() for key, value in graph.items()}
    permuted_graph["edge_index"] = inverse[graph["edge_index"]]
    return permuted_batch, permuted_graph


def test_relation_names_match_memory_transition_vocabulary() -> None:
    assert relation_type_id("corrects") == int(EvidenceRelationType.CORRECTS)
    assert relation_type_id("supersedes") == int(EvidenceRelationType.SUPERSEDES)
    assert relation_type_id("conflicts_with") == int(EvidenceRelationType.CONFLICTS_WITH)


def test_pooled_state_is_field_order_invariant() -> None:
    model = tiny_model()
    batch = make_batch()
    order = torch.tensor([3, 0, 4, 1, 2])

    with torch.inference_mode():
        original = model(**batch)
        permuted = model(**permute_fields(batch, order))

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


def test_graph_is_invariant_to_consistent_node_permutation() -> None:
    model = tiny_model()
    batch = make_batch()
    graph = make_graph()
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted_batch, permuted_graph = permute_graph_nodes(batch, graph, order)

    query = torch.randn(2, 32)
    with torch.inference_mode():
        original = model(**batch, **graph, query_semantic=query)
        permuted = model(**permuted_batch, **permuted_graph, query_semantic=query)

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


def test_supersession_direction_penalizes_historical_target() -> None:
    model = tiny_model()
    batch = make_batch(batch=1, fields=3)
    graph = {
        "edge_index": torch.tensor([[[0, 1]]]),
        "edge_type_ids": torch.tensor([[int(EvidenceRelationType.SUPERSEDES)]]),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    with torch.inference_mode():
        result = model(**batch, **graph)

    torch.testing.assert_close(
        result["relation_status_bias"],
        torch.tensor([[0.0, -1.0, 0.0]]),
    )
    assert float(result["relation_update_norm"][0, 1]) > 0.0
    assert float(result["relation_update_norm"][0, 0]) == 0.0


def test_conflict_is_semantically_symmetric_even_if_stored_once() -> None:
    model = tiny_model()
    batch = make_batch(batch=1, fields=3)
    common = {
        "edge_type_ids": torch.tensor([[int(EvidenceRelationType.CONFLICTS_WITH)]]),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    with torch.inference_mode():
        forward = model(**batch, edge_index=torch.tensor([[[0, 1]]]), **common)
        reverse = model(**batch, edge_index=torch.tensor([[[1, 0]]]), **common)

    torch.testing.assert_close(forward["field_states"], reverse["field_states"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(forward["pooled_state"], reverse["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        forward["relation_status_bias"],
        torch.tensor([[-0.35, -0.35, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_directed_graph_only_updates_relation_targets() -> None:
    model = tiny_model()
    batch = make_batch(batch=1, fields=3)
    graph = {
        "edge_index": torch.tensor([[[0, 1]]]),
        "edge_type_ids": torch.tensor([[int(EvidenceRelationType.SUPPORTS)]]),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    with torch.inference_mode():
        without_graph = model(**batch)
        with_graph = model(**batch, **graph)

    torch.testing.assert_close(
        without_graph["field_states"][:, 0, :],
        with_graph["field_states"][:, 0, :],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        without_graph["field_states"][:, 2, :],
        with_graph["field_states"][:, 2, :],
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.allclose(
        without_graph["field_states"][:, 1, :],
        with_graph["field_states"][:, 1, :],
    )


def test_query_semantic_changes_evidence_read() -> None:
    model = tiny_model()
    batch = make_batch(batch=1)
    first_query = torch.zeros(1, 32)
    second_query = torch.arange(32, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        first = model(**batch, query_semantic=first_query)
        second = model(**batch, query_semantic=second_query)

    assert not torch.allclose(first["field_weights"], second["field_weights"], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(first["pooled_state"], second["pooled_state"], atol=1e-6, rtol=1e-6)


def test_padding_fields_do_not_change_valid_state() -> None:
    model = tiny_model()
    batch = make_batch(batch=1)
    batch["valid_mask"][:, -1] = False

    changed = {key: value.clone() for key, value in batch.items()}
    changed["semantic_values"][:, -1, :] = 999.0
    changed["field_type_ids"][:, -1] = 15
    changed["provenance_ids"][:, -1] = 7
    changed["relation_role_ids"][:, -1] = 15
    changed["temporal_scope_ids"][:, -1] = 7
    changed["confidence"][:, -1, :] = 1.0
    changed["missing_mask"][:, -1] = True

    with torch.inference_mode():
        original = model(**batch)
        mutated_padding = model(**changed)

    torch.testing.assert_close(original["pooled_state"], mutated_padding["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        original["field_states"][:, :-1, :],
        mutated_padding["field_states"][:, :-1, :],
        atol=1e-5,
        rtol=1e-5,
    )
    assert float(original["field_weights"][0, -1]) == 0.0
    assert float(mutated_padding["field_weights"][0, -1]) == 0.0


def test_all_padding_is_rejected() -> None:
    model = tiny_model()
    batch = make_batch(batch=1)
    batch["valid_mask"].zero_()

    try:
        model(**batch)
    except ValueError as exc:
        assert "at least one valid structured field" in str(exc)
    else:
        raise AssertionError("all-padding structured state must be rejected")


def test_invalid_active_edge_is_rejected() -> None:
    model = tiny_model()
    batch = make_batch(batch=1, fields=3)
    graph = {
        "edge_index": torch.tensor([[[0, 4]]]),
        "edge_type_ids": torch.tensor([[int(EvidenceRelationType.SUPPORTS)]]),
        "edge_confidence": torch.ones(1, 1, 1),
        "edge_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    try:
        model(**batch, **graph)
    except ValueError as exc:
        assert "outside the valid range" in str(exc)
    else:
        raise AssertionError("out-of-range active edge must be rejected")


def test_default_branch_stays_under_parameter_budget() -> None:
    model = StructuredStateEncoder()
    report = model.parameter_report()
    assert report["total_parameters"] < 2_500_000
    assert report["graph_layers"] == 1
    assert report["private_identity_parameters"] == 0
    assert report["position_embeddings"] == 0
